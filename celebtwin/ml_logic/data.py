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

RAW_DATA = Path('raw_data')


class Dataset(ABC):
    """Data used for training and evaluation."""

    num_classes: int | None
    class_names: list[str] | None

    def __init__(self, num_classes: int | None):
        self.num_classes = num_classes
        self.class_names = None

    @abstractmethod
    def load(self) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """Load the dataset.

        Returns:
            tuple: A tuple containing the training dataset and validation dataset.
        """
        ...

    @property
    @abstractmethod
    def identifier(self) -> str:
        """Unique identifier for the dataset."""
        ...

    @abstractmethod
    def params(self) -> dict:
        """Return the parameters of the dataset as a dictionary."""
        ...

    @abstractmethod
    def load_prediction(self, path: Path) -> np.ndarray:
        """Load an image and apply preprocessing for prediction."""
        ...


class _FullDataset(ABC):
    """Abstract base class for full datasets that provide images."""

    _dataset_dir: Path

    @abstractmethod
    def iter_images(self, num_classes: int | None, undersample: bool) \
            -> Iterator[Path]:
        """Iterate over images in the dataset."""
        ...

    def relative_path(self, path: Path) -> Path:
        """Return the relative path of an image in the dataset."""
        return path.relative_to(self._dataset_dir)


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


class PinsDataset(_FullDataset):
    """The original Pins Face Recognition dataset."""

    _original_dir = RAW_DATA / '105_classes_pins_dataset'
    _original_zip = RAW_DATA / 'pins-face-recognition.zip'
    _dataset_dir = RAW_DATA / 'rawimages1'

    def try_download(self) -> bool:
        """Try to download the dataset."""
        if self._dataset_dir.exists():
            return True
        self._download_and_unzip()
        self._make_renamed_dataset()
        shutil.rmtree(self._original_dir)
        return True

    def _download_and_unzip(self):
        self._original_zip.parent.mkdir(exist_ok=True, parents=True)
        if not self._original_zip.exists():
            tmp_path = self._original_zip.with_suffix('.part')
            # Continue downloading if the file is partially downloaded.
            subprocess.run([
                'curl', '--location', '--continue-at', '-',
                '--output', str(tmp_path),
                'https://www.kaggle.com/api/v1/datasets/download/hereisburak/'
                'pins-face-recognition'
            ], check=True)
            tmp_path.rename(self._original_zip)
        if not self._original_dir.exists():
            subprocess.run(
                ['unzip', '-q', self._original_zip.name],
                cwd=self._original_zip.parent, check=True)

    def _make_renamed_dataset(self):
        temporary_dir = self._dataset_dir.with_suffix('.tmp')
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        temporary_dir.mkdir()  # Fails if the directory already exists.
        for class_dir in self._original_dir.iterdir():
            if not class_dir.is_dir() or class_dir.name.startswith('.'):
                continue
            # Directories are named like 'pins_Adriana Lima'.
            # Some of them are not properly capitalized.
            class_name = class_dir.name.removeprefix('pins_').title()
            new_class_dir = temporary_dir / class_name
            assert not new_class_dir.exists(), \
                f'Class directory already exists: {new_class_dir}'
            new_class_dir.mkdir()  # Fails if the directory already exists.
            for image_path in class_dir.glob('*.jpg'):
                # Images in the input data are named like 'Adriana
                # Lima101_3.jpg', the biggest number is four digits long.
                image_name = f'{int(image_path.stem.split("_")[1]):04}.jpg'
                new_image_path = new_class_dir / image_name
                if new_image_path.exists():
                    raise FileExistsError(
                        f'File already exists: {new_image_path}')
                # Faster to hardlink than to copy.
                new_image_path.hardlink_to(image_path)
        temporary_dir.rename(self._dataset_dir)

    def iter_images(self, num_classes: int | None, undersample: bool) \
            -> Iterator[Path]:
        """Iterate over the dataset."""
        return _iter_image_path(self._dataset_dir, num_classes, undersample)


class AlignedDatasetFull(_FullDataset):
    """A dataset that aligns faces in images and saves them to disk."""

    _dataset_dir = RAW_DATA / 'alignfull2'

    def preprocess_all(self) -> None:
        """Process all images, rename directory to dataset_dir and upload."""
        from celebtwin.ml_logic.registry import upload_dataset
        if self._dataset_dir.exists():
            raise ValueError(
                f'Dataset directory already exists: {self._dataset_dir}')
        partial = _AlignedDatasetPartial()
        partial.try_download()
        for _ in partial.iter_images(None, False):
            pass
        partial.rename_dataset_dir(self._dataset_dir)
        upload_dataset(self._dataset_dir)

    def try_download(self) -> bool:
        """Try to download the dataset."""
        from celebtwin.ml_logic.registry import try_download_dataset
        if self._dataset_dir.exists():
            return True
        return try_download_dataset(self._dataset_dir)

    def iter_images(self, num_classes: int | None, undersample: bool) \
            -> Iterator[Path]:
        """Iterate over images in the full dataset directory."""
        if not self._dataset_dir.exists():
            raise ValueError(
                'Dataset does not exist. Use _AlignedDatasetPartial instead.')
        for path in _iter_image_path(
                self._dataset_dir, num_classes, undersample):
            yield path


class _AlignedDatasetPartial(_FullDataset):
    """A dataset that aligns faces in images.

    The dataset is built on the fly from the PinsDataset.
    """

    _dataset_dir = RAW_DATA / 'alignpartial2'

    def __init__(self):
        self._pins_dataset = PinsDataset()

    def try_download(self) -> bool:
        """Try to download the dataset."""
        return self._pins_dataset.try_download()

    def rename_dataset_dir(self, new_path: Path) -> None:
        self._dataset_dir.rename(new_path)

    def iter_images(
        self, num_classes: int | None, undersample: bool) \
            -> Iterator[Path]:
        """Process images from PinsDataset and yield paths to aligned faces.

        Args:
            num_classes: Number of celebrity classes to process, None for all.
            undersample: If True, use the same number of images per class.

        Yields:
            Path to the aligned face image
        """
        self._dataset_dir.mkdir(exist_ok=True)
        ignored_files: set[str] = set()
        ignored_path = self._dataset_dir / 'ignore.csv'
        if ignored_path.exists():
            with open(ignored_path, 'r', encoding='utf-8', newline='') as file:
                ignored_files = {row[0] for row in csv.reader(file)}
        with _ImageWriter(self._dataset_dir, continue_=True) as image_writer:
            input_paths = list(self._pins_dataset.iter_images(
                num_classes, undersample))
            for input_path in tqdm(input_paths, miniters=0):
                relative_path = self._pins_dataset.relative_path(input_path)
                if str(relative_path) in ignored_files:
                    continue
                if image_writer.exists(relative_path):
                    yield self._dataset_dir / relative_path
                    continue
                try:
                    aligned_face = preprocess_face_aligned(input_path)
                except NoFaceDetectedError as error:
                    print(Fore.RED + str(error) + Style.RESET_ALL)
                    with open(ignored_path, 'a', encoding='utf-8',
                              newline='') as file:
                        csv.writer(file).writerow([str(relative_path)])
                    continue
                image_writer.write_image(relative_path, aligned_face)
                yield self._dataset_dir / relative_path


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

    _identifier_version = 'v3'

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

    @property
    def _dataset_dir(self) -> Path:
        """Path to the dataset directory."""
        return RAW_DATA / self.identifier

    def load(self) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """Load the dataset.

        Returns:
            tuple: A tuple containing the training dataset and validation dataset.
        """
        if not self._dataset_dir.exists():
            self._build_dataset()
        crop, pad = self._resize.as_crop_pad()
        train_data, val_data = image_dataset_from_directory(
            str(self._dataset_dir),
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
        return PinsDataset()

    def _build_dataset(self) -> None:
        """Build a dataset directory and zip file.

        The directory is created at `self._dataset_dir`.
        """
        from celebtwin.ml_logic.registry import (
            try_download_dataset, upload_dataset)
        downloaded = try_download_dataset(self._dataset_dir)
        if downloaded:
            return
        base_dataset = self._make_base_dataset()
        with _ImageWriter(self._dataset_dir) as image_writer:
            input_paths = list(base_dataset.iter_images(
                self._num_classes, self._undersample))
            for input_path in tqdm(input_paths):
                relative_path = base_dataset.relative_path(input_path)
                if image_writer.exists(relative_path):
                    continue
                image_tensor = self._load_image(input_path)
                image_writer.write_image(relative_path, image_tensor)
        upload_dataset(self._dataset_dir)

    def _make_base_dataset(self) -> _FullDataset:
        """Return the base dataset to process."""
        dataset = PinsDataset()
        downloaded = dataset.try_download()
        assert downloaded
        return dataset

    def load_prediction(self, path: Path) -> np.ndarray:
        return self._load_image(path)


class AlignedDataset(SimpleDataset):
    """A dataset that aligns faces in images."""

    _identifier_version = 'align4'

    def _make_base_dataset(self) -> _FullDataset:
        """Return the base dataset to process."""
        full_dataset = AlignedDatasetFull()
        downloaded = full_dataset.try_download()
        if downloaded:
            return full_dataset
        partial_dataset = _AlignedDatasetPartial()
        downloaded = partial_dataset.try_download()
        assert downloaded
        return partial_dataset


def load_image(path: Path | str, image_size: int, num_channels: int,
               resize: ResizeMode = ResizeMode.PAD) -> np.ndarray:
    """Load one image as a numpy array using keras."""
    crop, pad = resize.as_crop_pad()
    return keras.src.utils.image_dataset_utils.load_image(
        str(path), (image_size,) * 2, num_channels, 'bilinear',
        image_data_format(), crop, pad)


def _iter_image_path(
        base_dir: Path, num_classes: int | None = None,
        undersample: bool = False) \
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
        image_paths = list(
            sorted(input_dir.glob(image_glob), key=lambda x: int(x.stem)))
        if sample_size is not None:
            image_paths = image_paths[:sample_size]
        for image_path in image_paths:
            yield image_path


class _ImageWriter:
    """Write images to a directory."""

    def __init__(self, target_dir: Path, continue_: bool = False):
        self._continue = continue_
        self._target_dir = target_dir
        self._tmp_dir = self._target_dir.with_suffix('.tmp')
        if continue_:
            self._write_dir = self._target_dir
        else:
            self._write_dir = self._tmp_dir
            if self._target_dir.exists():
                raise ValueError(f'Path exists: {self._target_dir}')

    def close(self) -> None:
        """Rename temporary directory to its final name."""
        if not self._continue:
            target_dir = self._target_dir
            self._tmp_dir.rename(target_dir)

    def __enter__(self):
        if self._continue:
            self._target_dir.mkdir(exist_ok=True)
        else:
            if self._tmp_dir.exists():
                shutil.rmtree(self._tmp_dir)
            self._tmp_dir.mkdir()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.close()
        elif not self._continue:
            shutil.rmtree(self._tmp_dir)

    def exists(self, path: Path) -> bool:
        return (self._write_dir / path).exists()

    def write_image(self, path: Path, image_data: np.ndarray):
        """Write an image tensor to a file and add it to the zip archive.

        Args:
            path: Path to the image, relative to the data directory.
            image_data: The image data as a numpy array.
        """
        output_path = self._write_dir / path
        if output_path.exists():
            # Calling must check for existence to avoid repeat processing.
            raise FileExistsError(f'File already exists: {output_path}')
        jpeg_tensor = tf.image.encode_jpeg(tf.cast(image_data, tf.uint8))
        jpeg_bytes = jpeg_tensor.numpy()  # type: ignore

        output_dir = self._write_dir / path.parent
        output_dir.mkdir(exist_ok=True)
        tmp_path = output_dir / (path.name + '.tmp')
        with open(tmp_path, 'wb') as output_file:
            output_file.write(jpeg_bytes)
        tmp_path.rename(output_path)
