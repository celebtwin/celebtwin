import cv2
import numpy as np
from math import atan2, degrees
from pathlib import Path
from mtcnn.mtcnn import MTCNN

# Initialisation globale du détecteur MTCNN
detector = MTCNN()

def preprocess_face_aligned(image_input : Path, required_size=(160, 160), num_channels=3):
    """
    Détecte un visage dans l'image, aligne les yeux, recadre le visage et le redimensionne.

    Args:
        image_input (Path): Chemin vers le fichier image
        required_size : tuple (x,y), taille de l'image en sortie
        num_channels : int (1= grayscale, 3=couleur RGB)

    Returns:
        np.ndarray: Image du visage aligné, prête pour un modèles de reconnaissance faciale.

    Exceptions:
        aucune

    """
    # Gestion des différents types d'entrée
    img = cv2.cvtColor(cv2.imread(image_input), cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(img)
    if not result:
        #raise ValueError("Aucun visage détecté.")
        face_resized = cv2.resize(img, required_size)
        if num_channels==1:
            face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            face_resized = np.expand_dims(face_resized, -1)
        return face_resized

    face = result[0]
    keypoints = face['keypoints']
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']

    # Calcul de l'angle entre les yeux
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = degrees(atan2(dy, dx))

    # Calcul du centre entre les yeux
    eyes_center = (
        (left_eye[0] + right_eye[0]) / 2,
        (left_eye[1] + right_eye[1]) / 2
    )

    # on crée la matrice de rotation
    rot_mat = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)

    # on applique la rotation
    aligned_img = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)

    # Re-détection du visage après rotation
    aligned_result = detector.detect_faces(aligned_img)
    if not aligned_result:
        raise ValueError("Visage non détecté après alignement.")

    aligned_box = aligned_result[0]['box']
    x, y, w, h = aligned_box
    x, y = max(x, 0), max(y, 0)
    face_crop = aligned_img[y:y+h, x:x+w]

    if required_size:
        face_crop = cv2.resize(face_crop, required_size)
        if num_channels==1:
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            face_crop = np.expand_dims(face_crop, -1)

    return face_crop
