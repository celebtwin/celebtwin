from math import atan2, degrees
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from mtcnn.mtcnn import MTCNN  # type: ignore

mtcnn_detector = MTCNN()


class NoFaceDetectedError(Exception):
    """Raised when face detection failed during image preprocessing."""

    def __init__(self, path: Path | None = None):
        if path is None:
            message = '❌ No face detected'
        else:
            message = f'❌ No face detected: {path}'
        super().__init__(message)


def preprocess_face_aligned(path: Path) -> np.ndarray:
    """Load an image, detect a face, crop and align the image.

    The image is cropped to include only the face. Resizing should be done by the caller.
    """
    # MTCNN requires a RGB image. Apply grayscale conversion later.
    print(f"Decoding image {path}...")
    image = tf.image.decode_image(
        tf.io.read_file(str(path)),
        channels=3, expand_animations=False)

    print("Detecting faces...")
    detected_faces = mtcnn_detector.detect_faces(image)
    if not detected_faces:
        raise NoFaceDetectedError(path)

    # If multiple faces were detected, just pick the first one.
    face = detected_faces[0]
    keypoints = face['keypoints']
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']

    # Angle between the eyes and the horizontal axis.
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = degrees(atan2(dy, dx))

    eyes_center = (
        (left_eye[0] + right_eye[0]) / 2,
        (left_eye[1] + right_eye[1]) / 2)

    print("Rotating image...")
    rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
    aligned_image = cv2.warpAffine(
        image.numpy(), rotation_matrix, (image.shape[1], image.shape[0]))

    # Detect face again after rotation to get the best cropping box.
    print("Detecting faces again...")
    detected_faces = mtcnn_detector.detect_faces(aligned_image)
    if not detected_faces:
        raise NoFaceDetectedError(path)
    face = detected_faces[0]

    # Crop the face from the aligned image.
    print("Cropping face...")
    x, y, w, h = face['box']
    # x, y = max(x, 0), max(y, 0)
    cropped_image = aligned_image[y:y + h, x:x + w]
    return cropped_image
