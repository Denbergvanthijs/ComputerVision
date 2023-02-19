import pickle

import cv2
import numpy as np

from assignment1_online_phase import draw

if __name__ == "__main__":
    fp_video = "./videos/video.mp4"  # Set the path to the video
    fp_output = "./videos/video_run1.mp4"  # Set the path to save the output video
    fp_params = "./data/camera_params_run1.pickle"  # Set the path to load the camera parameters

    # Additional parameters
    horizontal_corners = 6  # Set the number of horizontal corners of the chessboard
    vertical_corners = 9  # Set the number of vertical corners of the chessboard
    square_size = 22  # Set the size of the squares in mm

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Load camera params from pickle
    print(f"Loading camera parameters from pickle file: {fp_params}")
    with open(fp_params, "rb") as f:
        camera_params = pickle.load(f)

    camera_mat = camera_params["camera_matrix"]
    dist_coef = camera_params["distortion_coefficients"]

    # Making the object points
    print("Making the object points")
    objp = np.zeros((vertical_corners * horizontal_corners, 3), np.float32)
    objp[:, :2] = np.mgrid[0:horizontal_corners, 0:vertical_corners].T.reshape(-1, 2)
    objp *= square_size  # Multiply by square size to get the real world coordinates, in milimeters

    n = square_size * 3  # length of the axis in mm
    axis = np.float32([[0, 0, 0],
                       [0, n, 0],
                       [n, n, 0],
                       [n, 0, 0],
                       [0, 0, -n],
                       [0, n, -n],
                       [n, n, -n],
                       [n, 0, -n]])

    print(f"Loading video: {fp_video}")
    cap = cv2.VideoCapture(fp_video)

    # Best params for YouTube
    output_video = cv2.VideoWriter(fp_output, cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))

    if not cap.isOpened():
        print("Error opening video")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Video ended")
            break

        # Resize frame to 720P
        img = cv2.resize(frame, (1280, 720))
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        pattern_found, corners = cv2.findChessboardCorners(img_grey, (horizontal_corners, vertical_corners), None)
        if pattern_found:
            corners_opt = cv2.cornerSubPix(img_grey, corners, (11, 11), (-1, -1), criteria)
            pattern_found, rot_vec, transl_vec = cv2.solvePnP(objp, corners_opt, camera_mat, dist_coef)
            imagepoints, jac = cv2.projectPoints(axis, rot_vec, transl_vec, camera_mat, dist_coef)

            img = draw(img, corners_opt, imagepoints)
        else:
            print(f"Pattern not found for frame {cap.get(cv2.CAP_PROP_POS_FRAMES)}")

        cv2.imshow("img", img)
        output_video.write(img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    output_video.release()
    cv2.destroyAllWindows()
