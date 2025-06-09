import csv
import shutil
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Iterator
from enum import Enum
from pathlib import Path

import keras.src.utils.image_dataset_utils  # type: ignore
import numpy as np
import tensorflow as tf
from colorama import Fore, Style
from keras.config import image_data_format  # type: ignore
from keras.preprocessing import image_dataset_from_directory  # type: ignore
from tqdm import tqdm  # type: ignore

from celebtwin.ml_logic.preproc_face import (
    NoFaceDetectedError, preprocess_face_aligned)
from celebtwin.ml_logic.registry import try_download_dataset, upload_dataset

RAW_DATA = Path('raw_data')


class Dataset:
    """Data used for training and evaluation."""

    num_classes: int | None
    class_names: list[str] | None

    def __init__(self, num_classes: int | None):
        self.num_classes = num_classes
        self.class_names = None

    def load(self) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """Load the dataset.

        Returns:
            tuple: A tuple containing the training dataset and validation dataset.
        """
        raise NotImplementedError("Implement load in a subclass.")

    @property
    def identifier(self) -> str:
        """Unique identifier for the dataset."""
        raise NotImplementedError("Implement identifier in a subclass.")

    def params(self) -> dict:
        """Return the parameters of the dataset as a dictionary."""
        raise NotImplementedError("Implement params in a subclass.")

    def load_prediction(self, path: Path) -> np.ndarray:
        """Load an image and apply preprocessing for prediction."""
        raise NotImplementedError("Implement load_prediction in a subclass.")


class _FullDataset(ABC):
    """Abstract base class for full datasets that provide images."""

    @abstractmethod
    def iter_images(self, num_classes: int | None, undersample: bool) \
            -> Iterator[Path]:
        """Iterate over images in the dataset."""
        ...

    @abstractmethod
    def translate_path(self, path: Path) -> Path:
        """Make a path relative to the target dataset."""
        ...


class ColorMode(str, Enum):
    """Color modes for image datasets."""
    GRAYSCALE = 'grayscale'
    RGB = 'rgb'

    def id_part(self):
        """Short identifier used to build dataset names."""
        return {ColorMode.GRAYSCALE: 'g', ColorMode.RGB: 'c'}[self]

    def num_channels(self) -> int:
        """Return the number of channels for the color mode."""
        return {ColorMode.GRAYSCALE: 1, ColorMode.RGB: 3}[self]


class ResizeMode(str, Enum):
    """Resize modes for image datasets."""
    PAD = 'pad'
    CROP = 'crop'
    DISTORT = 'distort'

    def id_part(self):
        """Short identifier used to build dataset names."""
        return {
            ResizeMode.PAD: None,
            ResizeMode.CROP: 'crop',
            ResizeMode.DISTORT: 'dist'
        }[self]

    def as_crop_pad(self) -> tuple[bool, bool]:
        """Return crop and pad flags for the resize mode."""
        return {
            ResizeMode.PAD: (False, True),
            ResizeMode.CROP: (True, False),
            ResizeMode.DISTORT: (False, False)
        }[self]


def load_dataset(params: dict) -> Dataset:
    class_map = {
        'SimpleDataset': SimpleDataset,
        'AlignedDataset': AlignedDataset}
    assert params['dataset_class'] in class_map, \
        f"Unexpected dataset class: {params['dataset_class']}"
    dataset_class = class_map[params['dataset_class']]
    dataset = dataset_class(
        params['image_size'],
        params['num_classes'],
        params['undersample'],
        ColorMode[params['color_mode'].upper()],
        ResizeMode[params['resize'].upper()],
        params['batch_size'],
        params['shuffle'],
        params['validation_split'])
    dataset.class_names = params['class_names']
    return dataset


class _PinsDataset(_FullDataset):
    """The original Pins Face Recognition dataset."""

    FULL_DATASET = RAW_DATA / '105_classes_pins_dataset'
    FULL_DATA_ZIP = RAW_DATA / 'pins-face-recognition.zip'

    def iter_images(self, num_classes: int | None, undersample: bool) \
            -> Iterator[Path]:
        """Download, unzip and iterate over the Pins dataset."""
        if not self.FULL_DATASET.exists():
            RAW_DATA.mkdir(exist_ok=True)
            subprocess.run([
                'curl', '--location', '--continue-at', '-',
                '--output', str(self.FULL_DATA_ZIP),
                'https://www.kaggle.com/api/v1/datasets/download/hereisburak/pins-face-recognition'
            ], check=True)
            subprocess.run(
                ['unzip', '-q', str(self.FULL_DATA_ZIP)],
                cwd=RAW_DATA, check=True)
        return _iter_image_path(self.FULL_DATASET, num_classes, undersample)

    def translate_path(self, input_path: Path) -> Path:
        """Translate path from the original naming to the naming we use.

        Returns: Path relative to the dataset directory.

        The original images are named like 'pins_Adriana Lima/Adriana
        Lima101_3.jpg'. We rename them to 'AdrianaLima/003.jpg'.
        """
        class_name = input_path.parent.name
        assert class_name.startswith('pins_')
        class_name = class_name.removeprefix('pins_').replace(' ', '')
        assert input_path.name.startswith(class_name)
        assert input_path.name.endswith('.jpg')
        number = _image_number(input_path)
        return Path(class_name) / f"{number:03}.jpg"


class AlignedDatasetFull(_FullDataset):
    """A dataset that aligns faces in images and saves them to disk."""

    _DATASET_NAME = 'alignfull1'
    _DATASET_DIR = RAW_DATA / _DATASET_NAME
    _PARTIAL_NAME = _DATASET_NAME + '-part'
    _PARTIAL_DIR = RAW_DATA / _PARTIAL_NAME

    def preprocess_all(self) -> None:
        """Process all images, rename directory to dataset_dir, and create zip."""
        if self._DATASET_DIR.exists():
            raise ValueError(
                f'Dataset directory already exists: {self._DATASET_DIR}')
        for _ in self._iter_images_partial(None, False):
            pass
        self._PARTIAL_DIR.rename(self._DATASET_DIR)
        upload_dataset(self._DATASET_DIR)

    def iter_images(self, num_classes: int | None, undersample: bool) \
            -> Iterator[Path]:
        """Process images and yield paths to aligned faces.

        Args:
            num_classes: Number of celebrity classes to process, None for all.
            undersample: If True, use the same number of images per class.

        Yields:
            Path to the aligned face image
        """
        if not self._DATASET_DIR.exists():
            try_download_dataset(self._DATASET_DIR)
        if self._DATASET_DIR.exists():
            return self._iter_images_full(num_classes, undersample)
        else:
            return self._iter_images_partial(num_classes, undersample)

    def _iter_images_full(self, num_classes: int | None, undersample: bool) \
            -> Iterator[Path]:
        """Iterate over images in the full dataset directory."""
        for path in _iter_image_path(self._DATASET_DIR, num_classes, undersample):
            yield path

    def _iter_images_partial(
        self, num_classes: int | None, undersample: bool) \
            -> Iterator[Path]:
        """Process images from PinsDataset and yield paths to aligned faces.

        Args:
            num_classes: Number of celebrity classes to process, None for all.
            undersample: If True, use the same number of images per class.

        Yields:
            Path to the aligned face image
        """
        self._DATASET_DIR.mkdir(exist_ok=True)
        pins_dataset = _PinsDataset()
        ignored_files: set[str] = set()
        ignored_path = self._PARTIAL_DIR / 'ignore.csv'
        if ignored_path.exists():
            with open(ignored_path, 'r', encoding='utf-8', newline='') as file:
                ignored_files = {row[0] for row in csv.reader(file)}
        with _ImageWriter(RAW_DATA, self._PARTIAL_NAME, exists_ok=True) \
                as image_writer:
            for input_path in pins_dataset.iter_images(num_classes, undersample):
                output_path = pins_dataset.translate_path(input_path)
                if str(output_path) in ignored_files:
                    continue
                if image_writer.exists(output_path):
                    yield self._PARTIAL_DIR / output_path
                    continue
                try:
                    aligned_face = preprocess_face_aligned(input_path)
                except NoFaceDetectedError as error:
                    print(Fore.RED + str(error) + Style.RESET_ALL)
                    with open(ignored_path, 'a', encoding='utf-8',
                              newline='') as file:
                        csv.writer(file).writerow([str(output_path)])
                    continue
                image_writer.write_image(output_path, aligned_face)
                yield self._PARTIAL_DIR / output_path

    def translate_path(self, path: Path) -> Path:
        """Make a path relative to the target dataset."""
        return path.relative_to(self._DATASET_DIR)


class SimpleDataset(Dataset):
    """A simple dataset that reads images from a directory.

    The resized image subset is cached in a directory and a zip file.
    """

    def __init__(
            self,
            image_size: int,
            num_classes: int | None = None,
            undersample: bool = False,
            color_mode: ColorMode = ColorMode.GRAYSCALE,
            resize: ResizeMode = ResizeMode.PAD,
            batch_size: int = 32,
            shuffle: bool = True,
            validation_split: float = 0):
        super().__init__(num_classes)
        assert 32 <= image_size <= 256  # Reasonable range
        assert isinstance(num_classes, int) or num_classes is None
        assert isinstance(undersample, bool)
        assert color_mode in ColorMode
        assert resize in ResizeMode
        if validation_split <= 0 or validation_split >= 1:
            raise ValueError(
                'validation_split must a float between 0 and 1.')
        self._image_size = image_size
        self._num_classes = num_classes
        self._undersample = undersample
        self._color_mode = color_mode
        self._resize = resize
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._validation_split = validation_split
        self.class_names = None

    _identifier_version = 'v2'

    @property
    def identifier(self) -> str:
        """Unique identifier for the dataset."""
        parts = [
            self._identifier_version, str(
                self._image_size), self._color_mode.id_part(),
            f'c{self._num_classes}' if self._num_classes is not None
            else None,
            'und' if self._undersample else None,
            self._resize.id_part()]
        return '-'.join(filter(None, parts))

    def params(self) -> dict:
        """Return the parameters of the dataset as a dictionary."""
        return {
            'dataset_class': self.__class__.__name__,
            'dataset_identifier': self.identifier,
            'image_size': self._image_size,
            'num_classes': self._num_classes,
            'undersample': self._undersample,
            'color_mode': self._color_mode.value,
            'resize': self._resize.value,
            'batch_size': self._batch_size,
            'shuffle': self._shuffle,
            'validation_split': self._validation_split,
            'class_names': self.class_names,
        }

    def _dataset_path(self) -> Path:
        """Return the path to the dataset directory."""
        return RAW_DATA / self.identifier

    def load(self) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """Load the dataset.

        Returns:
            tuple: A tuple containing the training dataset and validation dataset.
        """
        if not self._dataset_path().exists():
            self._build_dataset()
        crop, pad = self._resize.as_crop_pad()
        train_data, val_data = image_dataset_from_directory(
            str(self._dataset_path()),
            label_mode='categorical',
            color_mode=self._color_mode,
            image_size=(self._image_size,) * 2,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            seed=np.random.randint(10**6),
            validation_split=self._validation_split,
            subset='both',
            crop_to_aspect_ratio=crop,
            pad_to_aspect_ratio=pad)
        self.class_names = train_data.class_names
        return train_data, val_data

    def _load_image(self, path: Path) -> np.ndarray:
        num_channels = self._color_mode.num_channels()
        return load_image(
            path, self._image_size, num_channels, self._resize)

    def _full_dataset(self) -> _FullDataset:
        """Return the full dataset to process."""
        return _PinsDataset()

    def _build_dataset(self) -> None:
        """Build a dataset directory and zip file.

        The directory is created at `self._dataset_path()`.
        """
        full_dataset = self._full_dataset()
        with _ImageWriter(RAW_DATA, self.identifier) as image_writer:
            input_paths = list(full_dataset.iter_images(
                self._num_classes, self._undersample))
            for input_path in tqdm(input_paths):
                output_path = full_dataset.translate_path(input_path)
                if image_writer.exists(output_path):
                    continue
                image_tensor = self._load_image(input_path)
                image_writer.write_image(output_path, image_tensor)
        upload_dataset(self._dataset_path())

    def load_prediction(self, path: Path) -> np.ndarray:
        return self._load_image(path)


class AlignedDataset(SimpleDataset):
    """A dataset that aligns faces in images."""

    _identifier_version = 'align3'

    def _full_dataset(self) -> _FullDataset:
        """Return the full dataset to process."""
        return AlignedDatasetFull()


def load_image(path: Path | str, image_size: int, num_channels: int,
               resize: ResizeMode = ResizeMode.PAD) -> np.ndarray:
    """Load one image as a numpy array using keras."""
    crop, pad = resize.as_crop_pad()
    return keras.src.utils.image_dataset_utils.load_image(
        str(path), (image_size,) * 2, num_channels, 'bilinear',
        image_data_format(), crop, pad)


def _image_number(path: Path) -> int:
    # Images in the input data are named like 'Adriana Lima101_3.jpg', where
    # the image number is the part after the underscore.
    return int(path.stem.split('_')[1])


def _iter_image_path(
        base_dir: Path, num_classes: int | None, undersample: bool) \
        -> Iterator[Path]:
    """Iterate over image paths in the dataset."""
    input_class_dirs = list(sorted(
        x for x in base_dir.iterdir()
        if x.is_dir() and not x.name.startswith('.')))
    if num_classes is not None:
        assert 0 < num_classes <= len(input_class_dirs)
        input_class_dirs = input_class_dirs[:num_classes]
    sample_size = None
    image_glob = '*.jpg'
    if undersample:
        sample_size = min(
            len(list(d.glob(image_glob))) for d in input_class_dirs)
    for input_dir in input_class_dirs:
        assert input_dir.name.startswith('pins_'), \
            f'unexpected directory: {input_dir.name}'
        image_paths = list(
            sorted(input_dir.glob(image_glob), key=_image_number))
        if sample_size is not None:
            image_paths = image_paths[:sample_size]
        for image_path in image_paths:
            yield image_path


class _ImageWriter:
    """Write images to a directory and a zip file."""

    def __init__(
            self, data_dir: Path, data_name: str, exists_ok: bool = False):
        self._data_dir = data_dir
        self._data_name = data_name
        self._tmp_dir = self._target_path('.tmp')
        if not exists_ok and self._target_path().exists():
            raise ValueError(f'Path exists: {self._target_path()}')

    def _target_path(self, suffix: str = '') -> Path:
        return self._data_dir / (self._data_name + suffix)

    def close(self) -> None:
        """Rename temporary directory to its final name."""
        target_dir = self._target_path()
        self._tmp_dir.rename(target_dir)

    def __enter__(self):
        if self._tmp_dir.exists():
            shutil.rmtree(self._tmp_dir)
        self._tmp_dir.mkdir()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.close()
        else:
            self._target_path('.zip.tmp').unlink(missing_ok=True)
            shutil.rmtree(self._tmp_dir)

    def exists(self, path: Path) -> bool:
        return (self._tmp_dir / path).exists()

    def write_image(self, path: Path, image_data: np.ndarray):
        """Write an image tensor to a file and add it to the zip archive.

        Args:
            path: Path to the image, relative to the data directory.
            image_data: The image data as a numpy array.
        """
        tmp_path = self._tmp_dir / path
        if tmp_path.exists():
            # Calling must check for existence to avoid repeat processing.
            raise FileExistsError(f'File already exists: {tmp_path}')
        jpeg_tensor = tf.image.encode_jpeg(tf.cast(image_data, tf.uint8))
        jpeg_bytes = jpeg_tensor.numpy()  # type: ignore

        output_dir = self._tmp_dir / path.parent
        output_dir.mkdir(exist_ok=True)
        tmp_path = output_dir / (path.name + '.tmp')
        with open(tmp_path, 'wb') as output_file:
            output_file.write(jpeg_bytes)
        tmp_path.rename(tmp_path.with_suffix(''))
