import cv2
import numpy as np
from math import atan2, degrees
from mtcnn.mtcnn import MTCNN

# Initialisation globale du détecteur MTCNN
detector = MTCNN()

def preprocess_face_aligned(image_input,required_size=(160, 160), show_info=False):
    """
    Détecte un visage dans l'image, aligne les yeux, recadre le visage et le redimensionne.

    Args:
        image_input (str or np.ndarray): Chemin vers l'image OU 1 tableau numpy RGB.
        show_info (bool): Afficher les informations de détection (boîte et keypoints).

    Returns:
        np.ndarray: Image du visage aligné, prête pour un modèles de reconnaissance faciale.

    Exceptions:
        ValueError: Si aucun visage n'est détecté, ou après rotation.
        TypeError: Si l'entrée n'est ni un chemin, ni une image valide.

    """
    # Gestion des différents types d'entrée
    if isinstance(image_input, str):
        img = cv2.cvtColor(cv2.imread(image_input), cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, np.ndarray):
        img = image_input
    else:
        raise TypeError("L’entrée doit être un chemin (str) ou une image RGB (np.ndarray).")

    result = detector.detect_faces(img)

    if not result:
        #raise ValueError("Aucun visage détecté.")
        return img
    face = result[0]
    keypoints = face['keypoints']
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']

    if show_info:
        print("Boîte :", face['box'])
        print("Yeux :", left_eye, right_eye)

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

    return face_crop
