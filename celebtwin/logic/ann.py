"""Find faces with Approximate Nearest Neighbor (ANN) search."""

import csv
import pickle
import random
from itertools import groupby
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np
from annoy import AnnoyIndex
from deepface import DeepFace  # type: ignore
from tqdm import tqdm

from celebtwin.logic.data import AlignedDatasetFull, PinsDataset
from celebtwin.logic.preproc_face import (
    NoFaceDetectedError, preprocess_face_aligned)
from celebtwin.params import LOCAL_REGISTRY_PATH

annoy_dir = Path(LOCAL_REGISTRY_PATH) / "annoy"
deepface_dir = Path(LOCAL_REGISTRY_PATH) / "deepface"

validation_split = 0
annoy_metric = 'euclidean'
annoy_trees = 100


annoy_name = 'index.ann'
metadata_name = 'metadata.csv'
embedding_size_of = {
    'Facenet': 128,
    'Facenet512': 512,
    'VGG-Face': 4096,
    "OpenFace": 128,
    "DeepFace": 4096,
    "DeepID": 512,  # or 4096 depending on the version
    "Dlib": 128,
    "ArcFace": 512,
    "SFace": 512,
    "GhostFaceNet": 512,
}

normalization_of = {
    'Facenet': 'Facenet2018',
    'Facenet512': 'Facenet2018',
    'VGG-Face': 'VGGFace2',
    "OpenFace": 'Facenet2018',
}


def build_annoy_index(detector: str, model: str) -> None:
    """Build an Annoy index for the raw dataset.
    Args:
        detector (str): The face detector to use.
        model (str): The embedding model to use.
    """
    assert model in embedding_size_of, \
        f"Add model to embedding_size_of: {model}"
    builder = AnnoyIndexBuilder(detector, model, validation_split)
    builder.build_index()


class AnnoyReader:
    """Finds the class and name of the closest image in the Annoy index."""

    def __init__(self, detector: str, model: str):
        self.detector = detector
        self.model = model
        normalization = normalization_of[model]
        identifier = annoy_identifier(detector, model, normalization)
        self.dimension = embedding_size_of[model]
        self.index_dir = annoy_dir / identifier
        self.index_path = self.index_dir / annoy_name
        self.csv_path = self.index_dir / metadata_name
        self.index = None
        self.metadata = None

    def load(self) -> None:
        self.index = AnnoyIndex(self.dimension, annoy_metric)  # type: ignore
        assert self.index is not None
        print(f"Loading index from {self.index_path}")
        self.index.load(str(self.index_path))
        print(f"Loading metadata from {self.csv_path}")
        csv_file = open(self.csv_path, 'rt', encoding='utf-8')
        with csv_file:
            self.metadata = metadata = {}
            for item, class_, name in csv.reader(csv_file):
                assert int(item) not in metadata, \
                    f"Duplicate item {item} in metadata"
                metadata[int(item)] = (class_, name)

    def close(self) -> None:
        if self.index is not None:
            self.index.unload()
            self.index = None
        if self.metadata is not None:
            self.metadata = None

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def find_vector(self, vector: list[float]) -> tuple[str, str]:
        """Return the class and name of the entry closest to the vector."""
        assert self.index is not None
        assert self.metadata is not None
        neighbors = self.index.get_nns_by_vector(vector, 1)
        assert len(neighbors) == 1
        return self.metadata[neighbors[0]]

    def find_image(self, path: Path) -> tuple[str, str]:
        """Return the class and name of the entry closest to the image.

        Raise NoFaceDetectedError if no face is detected in the image.
        """
        if self.detector == "skip":
            # If the detector is "skip" use our internal face detection.
            rgb_face = preprocess_face_aligned(path)
            image_data = rgb_face[..., ::-1]  # Convert RGB to BGR
            vector = self._represent(image_data)
        else:
            vector = self._represent(str(path))
        assert len(vector) == self.dimension, \
            f"Expected vector of size {self.dimension}, got {len(vector)}"
        return self.find_vector(vector)

    def _represent(self, image: np.ndarray | str) -> list[float]:
        """Get the embedding for the given image."""
        result_list = DeepFace.represent(
            img_path=image,
            model_name=self.model,
            detector_backend=self.detector,
            enforce_detection=False,
            max_faces=1)
        assert len(result_list) == 1, \
            f"Expected exactly one face per image, got {len(result_list)}"
        result = result_list[0]
        if self.detector != "skip" and result["face_confidence"] == 0:
            raise NoFaceDetectedError()
        return result["embedding"]


class AnnoyIndexBuilder:
    """Internal use."""

    def __init__(self, detector: str, model: str, validation_split: float):
        self.detector = detector
        self.model = model
        self.normalization = normalization_of[model]
        self.validation_split = validation_split
        self.deepface_cache = DeepfaceCache(
            detector, model, self.normalization)
        self.aligned_entries: dict[tuple[str, str], Path] = {}
        self.validation_set: set[Path] = set()

    def build_index(self) -> None:
        self._build_path_lists()
        self._fill_deepface_cache()
        self._build_annoy_index()
        self._report_accuracy()
        self._report_build()

    def _build_path_lists(self) -> None:
        if self.detector == "skip":
            aligned_dataset = AlignedDatasetFull()
            downloaded = aligned_dataset.try_download()
            assert downloaded
            for path in aligned_dataset.iter_images():
                self.aligned_entries[path.parent.name, path.name] = path
        dataset = PinsDataset()
        downloaded = dataset.try_download()
        assert downloaded
        self.path_list = list(dataset.iter_images())

        if self.validation_split:
            # Build stratified validation set.
            for class_, iter_class_paths in groupby(
                    self.path_list, lambda p: p.parent.name):
                class_paths = list(iter_class_paths)
                n_validation = int(len(class_paths) * self.validation_split)
                self.validation_set.update(
                    random.sample(class_paths, n_validation))

    def _fill_deepface_cache(self) -> None:
        """Fill the Deepface cache with embeddings for all images."""
        pending_paths = [
            path for path in self.path_list
            if not self.deepface_cache.has(path)]
        if not pending_paths:
            return
        print(f"Computing embeddings for {len(pending_paths)} images...")
        path_iterator = iter(tqdm(
            pending_paths,
            initial=len(self.path_list) - len(pending_paths),
            total=len(self.path_list)))
        # Run representation once to ensure the model is downloded.
        self._fill_cache_for_path(next(path_iterator))
        with Pool(10) as pool:
            for _ in pool.imap(self._fill_cache_for_path, path_iterator, 16):
                pass

    def _fill_cache_for_path(self, path: Path) -> None:
        if self.detector == "skip":
            key = (path.parent.name, path.name)
            if key not in self.aligned_entries:
                self.deepface_cache.set(path, "noface")
                return
            result = self._represent(self.aligned_entries[key])
        else:
            result = self._represent(path)
            if result['face_confidence'] == 0:
                print(f"Warning: No face detected in {path}. Skipping.")
                self.deepface_cache.set(path, "noface")
                return
        self.deepface_cache.set(path, result["embedding"])

    def _represent(self, path: Path) -> dict:
        result_list = DeepFace.represent(
            img_path=str(path),
            model_name=self.model,
            detector_backend=self.detector,
            normalization=self.normalization,
            enforce_detection=False,
            max_faces=1)
        assert len(result_list) == 1, \
            f"Expected exactly one face per image, got {len(result_list)}"
        return result_list[0]

    def _build_annoy_index(self) -> None:
        """Build the Annoy index for the dataset."""
        # Only compute detection rate if we compute the validation accuracy.
        if not self.validation_split:
            total_count = 0
        else:
            total_count = len(self.path_list) - len(self.validation_set)
        detected_count = 0

        # Build the annoy index
        with AnnoyWriter(self.detector, self.model, self.normalization) \
                as annoy_writer:
            for path in self.path_list:
                if path in self.validation_set:
                    continue
                class_, name = path.parent.name, path.name
                vector = self.deepface_cache.get(path)
                if vector == "noface":
                    continue
                detected_count += 1
                annoy_writer.add_item(class_, name, vector)

        if total_count:
            print(f"Detected {detected_count} faces in {total_count}"
                  f" training images ({detected_count / total_count:.2%})")

    def _report_accuracy(self) -> None:
        """Report the accuracy of the Annoy index on the validation set."""
        if not self.validation_split:
            return
        total_count = len(self.validation_set)
        detected_count = 0
        correct_count = 0
        with AnnoyReader(self.detector, self.model) as annoy_reader:
            print("Computing accuracy...")
            for path in tqdm(self.validation_set):
                vector = self.deepface_cache.get(path)
                if vector == "noface":
                    continue
                detected_count += 1
                class_, name = annoy_reader.find_vector(vector)
                if class_ == path.parent.name:
                    correct_count += 1
        if total_count:
            print(f"Detected {detected_count} faces in {total_count}"
                  f" validation images ({detected_count / total_count:.2%})")
            print(
                f"Classification accuracy: {correct_count / total_count:.2%}")

    def _report_build(self) -> None:
        """Report if the index is ready for deployment."""
        if self.validation_split:
            return
        reader = AnnoyReader(self.detector, self.model)
        print(f"Index ready for deployment: {reader.index_dir}")


class DeepfaceCache:
    """Cache for Deepface results to avoid recomputing embeddings."""

    def __init__(self, detector: str, model: str, normalization: str):
        if normalization == 'base':
            identifier = f"{detector}-{model}"
        else:
            identifier = f"{detector}-{model}-{normalization}"
        self.cache_dir = deepface_dir / identifier

    def _pickle_path(self, path: Path) -> Path:
        return self.cache_dir / path.parent.name / f"{path.stem}.pickle"

    def get(self, path: Path):
        """Get the cached value for the given path, which must be present."""
        pickle_path = self._pickle_path(path)
        with open(pickle_path, 'rb') as pickle_file:
            return pickle.load(pickle_file)

    def set(self, path: Path, value):
        """Set the cache for the given path with the provided value."""
        pickle_path = self._pickle_path(path)
        tmp_path = pickle_path.with_suffix('.tmp')
        pickle_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, 'wb') as tmp_file:
            pickle.dump(value, tmp_file)
        tmp_path.rename(pickle_path)

    def has(self, path: Path) -> bool:
        """Check if the cache has an entry for the given path."""
        return self._pickle_path(path).exists()


def annoy_identifier(detector: str, model: str, normalization: str):
    return f'{detector}-{model}-{normalization}-{annoy_trees}-{annoy_metric}'


class AnnoyWriter:

    def __init__(self, detector: str, model: str, normalization: str):
        self.dimension = embedding_size_of[model]
        identifier = annoy_identifier(detector, model, normalization)
        annoy_subdir = annoy_dir / identifier
        annoy_subdir.mkdir(parents=True, exist_ok=True)
        self.index_path = annoy_subdir / annoy_name
        self.tmp_path = self.index_path.with_suffix('.tmp')
        self.csv_path = annoy_subdir / metadata_name
        self.tmp_csv_path = self.csv_path.with_suffix('.tmp')
        self.csv_file = None
        self.csv_writer = None
        self.index = None
        self.counter = 0

    def __enter__(self):
        # Delay creation of the annoy index until the first addition, so we do
        # not need to know the embedding size ahead of time.
        self.index = AnnoyIndex(self.dimension, annoy_metric)  # type: ignore
        self.index.on_disk_build(str(self.tmp_path))
        self.csv_file = open(self.tmp_csv_path, 'wt', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.csv_file is not None:
            self.csv_file.close()
        if exc_type is None:
            print(f"Built Annoy index with {self.counter} items.")
            if self.index is not None:
                self.index.build(annoy_trees)
                self.index.unload
                self.tmp_path.rename(self.index_path)
            if self.tmp_csv_path.exists():
                self.tmp_csv_path.rename(self.csv_path)
        else:
            self.tmp_path.unlink(missing_ok=True)
            self.tmp_csv_path.unlink(missing_ok=True)

    def add_item(self, class_, name, vector):
        assert self.csv_writer is not None
        assert self.index is not None
        self.csv_writer.writerow([self.counter, class_, name])
        self.index.add_item(self.counter, vector)
        self.counter += 1
