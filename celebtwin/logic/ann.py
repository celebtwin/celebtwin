"""Find faces with Approximate Nearest Neighbor (ANN) search."""

import csv
import pickle
import random
from itertools import groupby
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Iterator

import hnswlib
import numpy as np
from annoy import AnnoyIndex
from deepface import DeepFace  # type: ignore
from tqdm import tqdm

from celebtwin.logic.data import AlignedDatasetFull, PinsDataset
from celebtwin.logic.preproc_face import (
    NoFaceDetectedError, preprocess_face_aligned)
from celebtwin.params import LOCAL_REGISTRY_PATH

ann_dir = Path(LOCAL_REGISTRY_PATH) / "ann"
deepface_dir = Path(LOCAL_REGISTRY_PATH) / "deepface"


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


class ANNReader:
    """Search in a, previously created, Approximate Nearest Neighbor index."""

    def __init__(self, detector: str, model: str):
        self.detector: str = detector
        self.model: str = model
        self.strategy = get_strategy_class()(detector, model)
        self.dimension: int = embedding_size_of[model]
        self.csv_path: Path = self.strategy.metadata_path()
        self.index: ANNReaderBackend | None = None
        self.metadata: dict[int, tuple[str, str]] | None = None

    def load(self) -> None:
        self.index = self.strategy.reader(self.dimension)
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
            self.index.close()
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
        value = self.index.find_neighbor(vector)
        return self.metadata[value]

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


class ANNIndexBuilder:
    """Compute representation vectors of faces and build ANN index."""

    def __init__(self, detector: str, model: str):
        self.detector = detector
        self.model = model
        self.normalization = normalization_of[model]
        self.deepface_cache = DeepfaceCache(
            detector, model, self.normalization)
        self.aligned_entries: dict[tuple[str, str], Path] = {}

    def run(self) -> None:
        self._build_path_lists()
        self._fill_deepface_cache()
        self._build_ann_index()
        self._report()

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

    def _build_ann_index(self) -> None:
        """Build the ANN index for the dataset."""
        to_index = list(self._iter_images_to_index())
        print("Indexing images...")
        with ANNWriter(self.detector, self.model, len(to_index)) as ann_writer:
            for path in tqdm(to_index):
                vector = self.deepface_cache.get(path)
                if vector == "noface":
                    continue
                class_, name = path.parent.name, path.name
                ann_writer.add_item(class_, name, vector)
                self._item_added()

    def _iter_images_to_index(self) -> Iterator[Path]:
        yield from self.path_list

    def _item_added(self) -> None:
        pass

    def _report(self) -> None:
        """Report that the index is ready for deployment."""
        strategy = get_strategy_class()(self.detector, self.model)
        print(f"Index ready for deployment: {strategy.ann_subdir()}")


class ANNIndexEvaluator(ANNIndexBuilder):
    """Compute representation vectors and evaluate classification accuracy.

    It also report success rate for face detection in the training and
    validation sets.

    The reported accuracy depends on detector, model and ANN backend. It is
    useful to compare models and tune ANN backends.
    """

    def __init__(self, detector: str, model: str, validation_split: float):
        super().__init__(detector, model)
        self._validation_split = validation_split
        self._validation_set: set[Path] = set()
        self._detected_count = 0

    def _build_path_lists(self) -> None:
        super()._build_path_lists()
        # Build stratified validation set.
        for class_, iter_class_paths in groupby(
                self.path_list, lambda p: p.parent.name):
            class_paths = list(iter_class_paths)
            n_validation = int(len(class_paths) * self._validation_split)
            self._validation_set.update(
                random.sample(class_paths, n_validation))

    def _build_ann_index(self) -> None:
        total_count = len(self.path_list) - len(self._validation_set)
        super()._build_ann_index()
        detection_rate = self._detected_count / total_count
        print(f"Detected {self._detected_count} faces in {total_count}"
              f" training images ({detection_rate:.2%})")

    def _iter_images_to_index(self) -> Iterator[Path]:
        yield from (p for p in super()._iter_images_to_index()
                    if p not in self._validation_set)

    def _item_added(self) -> None:
        super()._item_added()
        self._detected_count += 1

    def _report(self) -> None:
        """Report the accuracy of the ANN index on the validation set."""
        total_count = len(self._validation_set)
        detected_count = 0
        correct_count = 0
        with ANNReader(self.detector, self.model) as annoy_reader:
            print("Computing accuracy...")
            for path in tqdm(self._validation_set):
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



class ANNWriter:

    def __init__(self, detector: str, model: str, size: int):
        self.dimension = embedding_size_of[model]
        self.strategy = get_strategy_class()(detector, model)
        self.strategy.ann_subdir().mkdir(parents=True, exist_ok=True)
        self.index = self.strategy.writer(self.dimension, size)
        self.csv_path = self.strategy.metadata_path()
        self.tmp_csv_path = self.csv_path.with_suffix('.tmp')
        self.csv_file = open(self.tmp_csv_path, 'wt', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        self.counter = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.csv_file is not None:
            self.csv_file.close()
        if exc_type is None:
            print(f"Built ANN index with {self.counter} items.")
            self.index.commit()
            self.tmp_csv_path.rename(self.csv_path)
        else:
            self.index.abort()
            self.tmp_csv_path.unlink()

    def add_item(self, class_: str, name: str, vector: list[float]) -> None:
        self.csv_writer.writerow([self.counter, class_, name])
        self.index.add_item(self.counter, vector)
        self.counter += 1


def get_strategy_class() -> 'type[ANNStrategy]':
    return HnswStrategy


class ANNStrategy:
    """Abstract base class for ANN index strategies."""

    _file_name: str
    _reader_type: 'type[ANNReaderBackend]'
    _writer_type: 'type[ANNWriterBackend]'

    def __init__(self, detector: str, model: str):
        self.detector = detector
        self.model = model

    def identifier(self) -> str:
        raise NotImplementedError

    def ann_subdir(self) -> Path:
        return ann_dir / self.identifier()

    def metadata_path(self) -> Path:
        return self.ann_subdir() / "metadata.csv"

    def index_path(self):
        return self.ann_subdir() / self._file_name

    def reader(self, dimension: int) -> 'ANNReaderBackend':
        return self._reader_type(self.index_path(), dimension)

    def writer(self, dimension: int, size: int) -> 'ANNWriterBackend':
        return self._writer_type(self.index_path(), dimension, size)


class ANNReaderBackend:
    """Abstract base class for ANN index reader backends."""

    def __init__(self, path: Path, dimension: int):
        self._index_path = path

    def close(self) -> None:
        raise NotImplementedError

    def find_neighbor(self, vector: list[float]) -> int:
        raise NotImplementedError


class ANNWriterBackend:
    """Abstract class for ANN index writer backends."""

    def __init__(self, path: Path, dimension: int, size: int):
        self._index_path = path
        self._tmp_path = path.with_suffix(path.suffix + '.tmp')

    def commit(self) -> None:
        self._commit()
        self._tmp_path.rename(self._index_path)

    def _commit(self) -> None:
        raise NotImplementedError

    def abort(self):
        self._abort()
        try:
            self._tmp_path.unlink()
        except FileNotFoundError:
            pass

    def _abort(self) -> None:
        raise NotImplementedError

    def add_item(self, value: int, vector: list[float]) -> None:
        raise NotImplementedError


class AnnoyReaderBackend(ANNReaderBackend):
    """Concrete ANN index reader backend using Annoy."""

    def __init__(self, path: Path, dimension: int):
        super().__init__(path, dimension)
        print(f"Loading Annoy index from {self._index_path}")
        metric = AnnoyStrategy.annoy_metric
        self._index = AnnoyIndex(dimension, metric)  # type: ignore
        self._index.load(str(self._index_path))

    def close(self) -> None:
        self._index.unload()

    def find_neighbor(self, vector: list[float]) -> int:
        neighbors = self._index.get_nns_by_vector(vector, 1)
        assert len(neighbors) == 1
        return neighbors[0]


class AnnoyWriterBackend(ANNWriterBackend):
    """Concrete ANN index writer backend using Annoy."""

    def __init__(self, path: Path, dimension: int, size: int):
        super().__init__(path, dimension, size)
        metric = AnnoyStrategy.annoy_metric
        self._index = AnnoyIndex(dimension, metric)  # type: ignore
        self._index.on_disk_build(str(self._tmp_path))

    def _commit(self) -> None:
        self._index.build(AnnoyStrategy.annoy_trees)
        self._index.unload()

    def _abort(self) -> None:
        self._index.unload()

    def add_item(self, value: int, vector: list[float]) -> None:
        self._index.add_item(value, vector)


class AnnoyStrategy(ANNStrategy):
    """Strategy for Annoy index."""

    _file_name = "index.ann"
    _reader_type = AnnoyReaderBackend
    _writer_type = AnnoyWriterBackend
    annoy_metric = "euclidean"
    annoy_trees = 100

    def identifier(self) -> str:
        normalization = normalization_of[self.model]
        return '-'.join([
            self.detector, self.model, normalization, 'annoy',
            str(self.annoy_trees), self.annoy_metric])


class HnswReaderBackend(ANNReaderBackend):
    """Concrete ANN index reader backend using hnswlib."""

    def __init__(self, path: Path, dimension: int):
        super().__init__(path, dimension)
        print(f"Loading HNSW index from {self._index_path}")
        space = HnswStrategy.hnsw_space
        self._index = hnswlib.Index(space=space, dim=dimension)
        self._index.load_index(str(self._index_path))
        self._index.set_ef(HnswStrategy.hnsw_ef)

    def close(self) -> None:
        self._index = None

    def find_neighbor(self, vector: list[float]) -> int:
        labels, distances = self._index.knn_query([vector], k=1)
        assert len(labels) == 1
        return labels[0].item()


class HnswWriterBackend(ANNWriterBackend):
    """Concrete ANN index writer backend using hnswlib."""

    def __init__(self, path: Path, dimension: int, size: int):
        super().__init__(path, dimension, size)
        space = HnswStrategy.hnsw_space
        self._index = hnswlib.Index(space=space, dim=dimension)
        self._index.init_index(size, HnswStrategy.hnsw_m, HnswStrategy.hnsw_ef)
        self._index.set_ef(HnswStrategy.hnsw_ef)

    def _commit(self) -> None:
        self._index.save_index(str(self._tmp_path))

    def _abort(self) -> None:
        self._index = None

    def add_item(self, value: int, vector: list[float]) -> None:
        self._index.add_items([vector], [value])



class HnswStrategy(ANNStrategy):
    """Strategy for HNSW index."""

    _file_name = "index.hnsw"
    _reader_type = HnswReaderBackend
    _writer_type = HnswWriterBackend
    hnsw_space = "l2"
    hnsw_ef = 100
    hnsw_m = 48

    def identifier(self) -> str:
        normalization = normalization_of[self.model]
        return '-'.join([
            self.detector, self.model, normalization, 'hnsw',
            str(self.hnsw_m), str(self.hnsw_ef)])
