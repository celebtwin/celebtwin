"""Find faces with Approximate Nearest Neighbor (ANN) search."""

import csv
import pickle
import random
from itertools import groupby
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Iterator

import hnswlib  # type: ignore
import numpy as np
from annoy import AnnoyIndex
from deepface import DeepFace  # type: ignore
from tqdm import tqdm

from celebtwin.logic.data import AlignedDatasetFull, PinsDataset
from celebtwin.logic.preproc_face import (
    NoFaceDetectedError, preprocess_face_aligned)
from celebtwin.params import LOCAL_REGISTRY_PATH
from celebtwin.logic.annenums import Detector, Model

ann_dir = Path(LOCAL_REGISTRY_PATH) / "ann"
deepface_dir = Path(LOCAL_REGISTRY_PATH) / "deepface"


class ANNReader:
    """Search in a, previously created, Approximate Nearest Neighbor index."""

    def __init__(self, strategy: 'ANNStrategy'):
        self.strategy = strategy
        self.csv_path: Path = self.strategy.metadata_path()
        self.index = self.strategy.reader()
        print(f"Loading metadata from {self.csv_path}")
        self.metadata: dict[int, tuple[str, str]] = {}
        with open(self.csv_path, 'rt', encoding='utf-8') as csv_file:
            for item, class_, name in csv.reader(csv_file):
                if int(item) in self.metadata:
                    raise ValueError(f"Duplicate item {item} in metadata")
                self.metadata[int(item)] = (class_, name)

    def close(self) -> None:
        self.index.close()
        self.metadata.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def find_vector(self, vector: list[float]) -> tuple[str, str]:
        """Return the class and name of the entry closest to the vector."""
        value = self.index.find_neighbor(vector)
        assert self.metadata is not None
        return self.metadata[value]

    def find_image(self, path: Path) -> tuple[str, str]:
        """Return the class and name of the entry closest to the image.

        Raise NoFaceDetectedError if no face is detected in the image.
        """
        if self.strategy.detector == Detector.SKIP:
            # If the detector is "skip" use our internal face detection.
            rgb_face = preprocess_face_aligned(path)
            image_data = rgb_face[..., ::-1]  # Convert RGB to BGR
            vector = self.strategy.represent(image_data)
        else:
            vector = self.strategy.represent(str(path))
        if len(vector) != self.strategy.dimension:
            raise ValueError(
                f"Expected vector of size {self.strategy.dimension}, "
                f"got {len(vector)}")
        return self.find_vector(vector)


class ANNIndexBuilder:
    """Compute representation vectors of faces and build ANN index."""

    def __init__(self, strategy: 'ANNStrategy'):
        self.strategy = strategy
        self.deepface_cache = DeepfaceCache(strategy)
        self.aligned_entries: dict[tuple[str, str], Path] = {}

    def run(self) -> None:
        self._build_path_lists()
        self._fill_deepface_cache()
        self._build_ann_index()
        self._report()

    def _build_path_lists(self) -> None:
        if self.strategy.detector == Detector.SKIP:
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
        if self.strategy.detector == Detector.SKIP:
            key = (path.parent.name, path.name)
            if key not in self.aligned_entries:
                self.deepface_cache.set(path, "noface")
                return
            result = self.strategy.represent(self.aligned_entries[key])
        else:
            try:
                result = self.strategy.represent(path)
            except NoFaceDetectedError:
                print(f"Warning: No face detected in {path}. Skipping.")
                self.deepface_cache.set(path, "noface")
                return
        self.deepface_cache.set(path, result)

    def _build_ann_index(self) -> None:
        """Build the ANN index for the dataset."""
        to_index = list(self._iter_images_to_index())
        print("Indexing images...")
        with ANNWriter(self.strategy, len(to_index)) as ann_writer:
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
        print(f"Index ready for deployment: {self.strategy.ann_subdir()}")


class ANNIndexEvaluator(ANNIndexBuilder):
    """Compute representation vectors and evaluate classification accuracy.

    It also report success rate for face detection in the training and
    validation sets.

    The reported accuracy depends on detector, model and ANN backend. It is
    useful to compare models and tune ANN backends.
    """

    def __init__(self, strategy: 'ANNStrategy', validation_split: float):
        super().__init__(strategy)
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
        with ANNReader(self.strategy) as annoy_reader:
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

    def __init__(self, strategy: 'ANNStrategy'):
        identifier = "-".join([
            strategy.detector, strategy.model, strategy.normalization])
        self.cache_dir = deepface_dir / identifier

    def _pickle_path(self, path: Path) -> Path:
        return self.cache_dir / path.parent.name / f"{path.stem}.pickle"

    def get(self, path: Path):
        """Get the cached value for the given path, which must be present."""
        pickle_path = self._pickle_path(path)
        with open(pickle_path, 'rb') as pickle_file:
            return pickle.load(pickle_file)

    def set(self, path: Path, value) -> None:
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

    def __init__(self, strategy: 'ANNStrategy', size: int):
        self.strategy = strategy
        self.strategy.ann_subdir().mkdir(parents=True, exist_ok=True)
        self.index = self.strategy.writer(size)
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


class ANNStrategy:
    """Abstract base class for ANN index strategies."""

    _file_name: str
    _reader_type: 'type[ANNReaderBackend]'
    _writer_type: 'type[ANNWriterBackend]'

    def __init__(self, detector: Detector, model: Model):
        self.detector = detector
        self.model = model

    @property
    def normalization(self) -> str:
        return self.model.normalization

    @property
    def dimension(self) -> int:
        return self.model.embedding_size

    def identifier(self) -> str:
        raise NotImplementedError

    def ann_subdir(self) -> Path:
        return ann_dir / self.identifier()

    def metadata_path(self) -> Path:
        return self.ann_subdir() / "metadata.csv"

    def index_path(self):
        return self.ann_subdir() / self._file_name

    def represent(self, image: np.ndarray | str | Path) -> list[float]:
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
        if self.detector != Detector.SKIP and result["face_confidence"] == 0:
            raise NoFaceDetectedError()
        return result["embedding"]

    def reader(self) -> 'ANNReaderBackend':
        return self._reader_type(self)

    def writer(self, size: int) -> 'ANNWriterBackend':
        return self._writer_type(self, size)


class ANNReaderBackend:
    """Abstract base class for ANN index reader backends."""

    def __init__(self, strategy: ANNStrategy):
        self.strategy = strategy

    def close(self) -> None:
        raise NotImplementedError

    def find_neighbor(self, vector: list[float]) -> int:
        raise NotImplementedError


class ANNWriterBackend:
    """Abstract class for ANN index writer backends."""

    def __init__(self, strategy: ANNStrategy, size: int):
        self.strategy = strategy
        self._index_path = strategy.index_path()
        self._tmp_path = self._index_path.with_suffix(
            self._index_path.suffix + '.tmp')

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

    def __init__(self, strategy: ANNStrategy):
        super().__init__(strategy)
        index_path = self.strategy.index_path()
        print(f"Loading Annoy index from {index_path}")
        assert isinstance(strategy, AnnoyStrategy)
        metric = strategy.annoy_metric
        self._index = AnnoyIndex(strategy.dimension, metric)  # type: ignore
        self._index.load(str(index_path))

    def close(self) -> None:
        self._index.unload()

    def find_neighbor(self, vector: list[float]) -> int:
        neighbors = self._index.get_nns_by_vector(vector, 1)
        assert len(neighbors) == 1
        return neighbors[0]


class AnnoyWriterBackend(ANNWriterBackend):
    """Concrete ANN index writer backend using Annoy."""

    def __init__(self, strategy: ANNStrategy, size: int):
        super().__init__(strategy, size)
        assert isinstance(strategy, AnnoyStrategy)
        metric = strategy.annoy_metric
        self._index = AnnoyIndex(strategy.dimension, metric)  # type: ignore
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
        return '-'.join([
            self.detector, self.model, self.model.normalization, 'annoy',
            str(self.annoy_trees), self.annoy_metric])


class HnswReaderBackend(ANNReaderBackend):
    """Concrete ANN index reader backend using hnswlib."""

    def __init__(self, strategy: ANNStrategy):
        super().__init__(strategy)
        index_path = self.strategy.index_path()
        print(f"Loading HNSW index from {index_path}")
        assert isinstance(strategy, HnswStrategy)
        space = strategy.hnsw_space
        self._index = hnswlib.Index(space=space, dim=strategy.dimension)
        self._index.load_index(str(index_path))
        self._index.set_ef(strategy.hnsw_ef)

    def close(self) -> None:
        self._index = None

    def find_neighbor(self, vector: list[float]) -> int:
        labels, distances = self._index.knn_query([vector], k=1)
        assert len(labels) == 1
        return labels[0].item()


class HnswWriterBackend(ANNWriterBackend):
    """Concrete ANN index writer backend using hnswlib."""

    def __init__(self, strategy: ANNStrategy, size: int):
        super().__init__(strategy, size)
        assert isinstance(strategy, HnswStrategy)
        space = strategy.hnsw_space
        self._index = hnswlib.Index(space=space, dim=strategy.dimension)
        self._index.init_index(size, strategy.hnsw_m, strategy.hnsw_ef)
        self._index.set_ef(strategy.hnsw_ef)

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
        return '-'.join([
            self.detector, self.model, self.model.normalization, 'hnsw',
            str(self.hnsw_m), str(self.hnsw_ef)])


class BruteForceReaderBackend(ANNReaderBackend):
    """Concrete ANN index reader backend using brute force."""

    def __init__(self, strategy: ANNStrategy):
        super().__init__(strategy)
        index_path = self.strategy.index_path()
        print(f"Loading BruteForce index from {index_path}")
        assert isinstance(strategy, BruteForceStrategy)
        space = strategy.bf_space
        self._index = hnswlib.BFIndex(space=space, dim=strategy.dimension)
        self._index.load_index(str(index_path))

    def close(self) -> None:
        self._index = None

    def find_neighbor(self, vector: list[float]) -> int:
        labels, distances = self._index.knn_query([vector], k=1)
        assert len(labels) == 1
        return labels[0].item()


class BruteForceWriterBackend(ANNWriterBackend):
    """Concrete ANN index writer backend using brute force."""

    def __init__(self, strategy: ANNStrategy, size: int):
        super().__init__(strategy, size)
        assert isinstance(strategy, BruteForceStrategy)
        space = strategy.bf_space
        self._index = hnswlib.BFIndex(space=space, dim=strategy.dimension)
        self._index.init_index(size)

    def _commit(self) -> None:
        self._index.save_index(str(self._tmp_path))

    def _abort(self) -> None:
        self._index = None

    def add_item(self, value: int, vector: list[float]) -> None:
        self._index.add_items([vector], [value])


class BruteForceStrategy(ANNStrategy):
    """Strategy for brute force index."""

    _file_name = "index.brute"
    _reader_type = BruteForceReaderBackend
    _writer_type = BruteForceWriterBackend
    bf_space = "l2"

    def identifier(self) -> str:
        return '-'.join([
            self.detector, self.model, self.model.normalization, 'brute'])
