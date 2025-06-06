from math import atan2, degrees
from pathlib import Path

import cv2
import keras.ops.image  # type: ignore
import numpy as np
import tensorflow as tf
from mtcnn.mtcnn import MTCNN  # type: ignore

mtcnn_detector = MTCNN()


def preprocess_face_aligned(
        path: Path, image_size: int,
        num_channels: int) -> np.ndarray:
    """Load an image, detect a face, crop, align and resize the image.

    The image is cropped to include only the face, resizing is done without
    preserving aspect ratio.

    num_channels is 1 for grayscale, 3 for color images.
    """
    from celebtwin.ml_logic.data import load_image

    # MTCNN requires a RGB image. Apply grayscale conversion later.
    image = tf.image.decode_image(
        tf.io.read_file(str(path)),
        channels=3, expand_animations=False)

    detected_faces = mtcnn_detector.detect_faces(image)
    if not detected_faces:
        return load_image(path, image_size, num_channels)

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

    rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
    aligned_image = cv2.warpAffine(
        image.numpy(), rotation_matrix, (image.shape[1], image.shape[0]))

    # Detect face again after rotation to get the best cropping box.
    detected_faces = mtcnn_detector.detect_faces(aligned_image)
    if not detected_faces:
        raise ValueError("Face detection failed after alignment.")
    face = detected_faces[0]

    # Crop the face from the aligned image.
    x, y, w, h = face['box']
    x, y = max(x, 0), max(y, 0)
    cropped_image = aligned_image[y:y+h, x:x+w]
    resized_image = tf.image.resize_with_pad(
        cropped_image, image_size, image_size, method='bilinear')
    if num_channels == 1:
        return keras.ops.image.rgb_to_grayscale(resized_image)  # type: ignore
    return resized_image
