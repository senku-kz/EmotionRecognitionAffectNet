import cv2
import mediapipe as mp
import numpy as np


"""
please install follow packages:
pip install opencv-python==4.5.5.64
pip install mediapipe==0.8.9.1
"""


def headPoseEstimation(path):
    # print('filename:', path)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False

    # Get the result
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])

                    # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])

            # The Distance Matrix
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360

            # print('x:y', x, y)

            # See where the user's head tilting
            if y < -10:
                text = "Camera Right (Looking Left)"
                # print("Camera Right (Looking Left)")
                # print("Camera Left (Looking Right)")
            elif y > 10:
                text = "Camera Left (Looking Right)"
                # print("Camera Left (Looking Right)")
                # print("Camera Right (Looking Left)")
            elif x < -10:
                text = "Camera up (Looking Down)"
                # print("Camera up (Looking Down)")
            elif x > 10:
                text = "Camera down (Looking up)"
                # print("Camera up (Looking Down)")
            else:
                text = "Forward"
                # print("Forward")
            print('filename: {}; \t head position: {}'.format(path, text))
            return text


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    headPoseEstimation(path='../test_class/class001/image0000002.jpg')
