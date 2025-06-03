import shutil
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import keras.src.utils.image_dataset_utils
import numpy as np
import tensorflow as tf
from keras.config import image_data_format
from keras.preprocessing import image_dataset_from_directory

RAW_DATA = Path('raw_data')
FULL_DATASET = RAW_DATA / '105_classes_pins_dataset'


def load_dataset(
        image_size: int,
        num_classes: int | None = None,
        undersample: bool = False,
        color_mode: str = 'grayscale',
        resize: str = 'pad',
        batch_size: int = 32,
        shuffle: bool = True,
        validation_split: float | None = None):
    assert 32 <= image_size <= 256  # Reasonable range
    assert isinstance(num_classes, int) or num_classes is None
    assert isinstance(undersample, bool)
    assert color_mode in ('grayscale', 'rgb')
    assert resize in ('pad', 'crop', 'distort')

    if num_classes is None and not undersample:
        # Image resizing is fast compared to model fitting. If we work on the
        # full dataset, just use image_dataset_from_directory.
        crop, pad = _get_crop_pad(resize)
        return image_dataset_from_directory(
            str(FULL_DATASET),
            label_mode='categorical',
            color_mode=color_mode,
            image_size=(image_size,) * 2,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=np.random.randint(10**6),
            validation_split=validation_split,
            subset='both' if validation_split is not None else None,
            crop_to_aspect_ratio=crop,
            pad_to_aspect_ratio=pad)
    builder = _DatasetBuilder(
        image_size, num_classes, undersample, color_mode, resize)
    if not builder.dataset_path.exists():
        builder.build_dataset()
    return builder.load_dataset(batch_size, shuffle, validation_split)


def _get_crop_pad(resize):
    return {
        'pad': (False, True), 'crop': (True, False),
        'distort': (False, False)}[resize]


class _DatasetBuilder:

    def __init__(
            self,
            image_size: int,
            num_classes: int | None = None,
            undersample: bool = False,
            color_mode: str = 'grayscale',
            resize: str = 'pad'):
        self._image_size = image_size
        self._num_classes = num_classes
        self._undersample = undersample
        self._color_mode = color_mode
        self._resize = resize

    @property
    def dataset_name(self) -> str:
        nclasses_code = \
            f'nc{self._num_classes}-' if self._num_classes is not None else ''
        undersample_code = 'und-' if self._undersample else ''
        color_code = {'grayscale': 'gray', 'rgb': 'col'}[self._color_mode]
        return (
            f'v1-{self._image_size}-{nclasses_code}{undersample_code}'
            f'{color_code}-{self._resize[:3]}')

    @property
    def dataset_path(self) -> Path:
        return RAW_DATA / self.dataset_name

    def load_dataset(
            self, batch_size: int, shuffle: bool,
            validation_split: float | None = None):
        crop, pad = _get_crop_pad(self._resize)
        return image_dataset_from_directory(
            str(self.dataset_path),
            label_mode='categorical',
            color_mode=self._color_mode,
            image_size=(self._image_size,) * 2,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=np.random.randint(10**6),
            validation_split=validation_split,
            subset='both' if validation_split is not None else None,
            crop_to_aspect_ratio=crop,
            pad_to_aspect_ratio=pad)

    def build_dataset(self):
        """Build a dataset directory and zip file.

        The directory is created at `self.dataset_path`.
        """
        dsname = self.dataset_name
        tmp_dir = RAW_DATA / (dsname + '.tmp')
        num_channels = {'grayscale': 1, 'rgb': 3}[self._color_mode]

        def inner_load_image(path):
            return load_image(
                path, self._image_size, num_channels, self._resize)

        with _ImageWriter(RAW_DATA, dsname) as image_writer:
            for input_image_path, image_tensor in _iter_read_images(
                    FULL_DATASET, inner_load_image, self._num_classes,
                    self._undersample):
                class_name = input_image_path.parent.name
                class_name = class_name.removeprefix('pins_').replace(' ', '')
                number = _image_number(input_image_path)
                image_name = f"{class_name}_{number:03}.jpg"
                image_writer.write_image(class_name, image_name, image_tensor)


def load_image(path: Path | str, image_size: int, num_channels: int,
               resize: str = 'pad') -> np.ndarray:
    """Load one image as a numpy array using keras."""
    crop, pad = _get_crop_pad(resize)
    return keras.src.utils.image_dataset_utils.load_image(
        str(path), (image_size,) * 2, num_channels, 'bilinear',
        image_data_format(), crop, pad)


def _image_number(path: Path) -> int:
    # Images in the input data are named like 'Adriana Lima101_3.jpg', where
    # the image number is the part after the underscore.
    return int(path.stem.split('_')[1])


def _iter_read_images(base_dir, inner_load_image, num_classes, undersample):
    input_class_dirs = list(sorted(
        x for x in base_dir.iterdir()
        if x.is_dir() and not x.name.startswith('.')))
    if num_classes is None:
        num_classes = len(input_class_dirs)
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
            image_tensor = inner_load_image(image_path)
            yield image_path, image_tensor


class _ImageWriter:

    def __init__(self, data_dir, data_name):
        self._data_dir = data_dir
        self._data_name = data_name
        self._tmp_dir = self._target_path('.tmp')
        self._zip_path = self._target_path('.zip.tmp')
        assert (not self._target_path().exists()), \
            f'Path exists: {self._target_path()}'
        if self._tmp_dir.exists():
            shutil.rmtree(self._tmp_dir)
        self._tmp_dir.mkdir()
        # Do not compress the zip data, jpeg files are already compressed.
        self._zipfile = ZipFile(self._zip_path, 'w', ZIP_STORED)

    def _target_path(self, suffix=''):
        return self._data_dir / (self._data_name + suffix)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._zipfile.close()
        if exc_type is None:
            self._zip_path.rename(self._target_path('.zip'))
            self._tmp_dir.rename(self._target_path())
        else:
            self._zip_path.unlink()
            shutil.rmtree(self._tmp_dir)

    def write_image(self, class_name, image_name, image_tensor):
        jpeg_tensor = tf.image.encode_jpeg(tf.cast(image_tensor, tf.uint8))
        jpeg_bytes = jpeg_tensor.numpy()
        output_dir = self._tmp_dir / class_name
        output_dir.mkdir(exist_ok=True)
        with open(output_dir / image_name, 'wb') as output_file:
            output_file.write(jpeg_bytes)
        entry_name = self._data_name + '/' + class_name + '/' + image_name
        self._zipfile.writestr(entry_name, jpeg_bytes)
