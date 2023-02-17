import glob
import os
import pickle

import cv2
import numpy as np
from cv2 import CALIB_CB_FAST_CHECK

points = []
points_d = {0: "Provide the left-top corner of the checkerboard",
            1: "Provide the right-top corner of the checkerboard",
            2: "Provide the left-bottom corner of the checkerboard",
            3: "Provide the right-bottom corner of the checkerboard"}
horizontal_corners = 6
vertical_corners = 9
font = cv2.FONT_HERSHEY_SIMPLEX


def on_click(event, x, y, flags, params) -> None:
    """Callback function for mouse events.

    First, collects the four corners of the chessboard provided by the user.
    Then, interpolates the points to get the corners of the chessboard.
    """
    if (len(points) < 4) and (event == cv2.EVENT_LBUTTONDOWN):  # If not all corners have been provided and the left mouse button is clicked
        text = f"{x=}, {y=}"
        print(text)

        cv2.putText(img, text, (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow("", img)

        points.append((x, y))
        print(points)

        if len(points) < 4:
            print(points_d[len(points)])

    # After the 4th cornerpoint ...
    if (len(points) == 4) and (event == cv2.EVENT_LBUTTONDOWN):
        print("All corners have been provided!")

        corners = interpolate_points(points)

        # Draw the interpolated points
        for v in range(vertical_corners):
            for h in range(horizontal_corners):
                cv2.circle(img, corners[v, h].astype(int), 5, (0, 0, 255), -1)

        cv2.imshow("", img)


def interpolate_points(points):
    """Interpolate the points to get the corners of the chessboard."""
    x1, y1 = points[0]  # left-top
    x2, y2 = points[1]  # right-top
    x3, y3 = points[2]  # left-bottom
    x4, y4 = points[3]  # right-bottom

    x1x2 = np.linspace(x1, x2, horizontal_corners)  # Interpolate the x-coordinates of the top row
    x3x4 = np.linspace(x3, x4, horizontal_corners)  # Interpolate the x-coordinates of the bottom row

    y1y3 = np.linspace(y1, y3, vertical_corners)  # Interpolate the y-coordinates of the left column
    y2y4 = np.linspace(y2, y4, vertical_corners)  # Interpolate the y-coordinates of the right column

    corners = np.zeros((vertical_corners, horizontal_corners, 2), dtype=np.float32)

    for v in range(vertical_corners):
        weight_vertical = (vertical_corners - v) / vertical_corners
        weight_vertical_inv = 1 - weight_vertical

        for h in range(horizontal_corners):
            # Apply weighting to the x and y coordinates
            # The closer the point is to the top or left, the more weight it gets
            weight_horizontal = (horizontal_corners - h) / horizontal_corners
            weight_horizontal_inv = 1 - weight_horizontal

            x = weight_vertical * x1x2[h] + weight_vertical_inv * x3x4[h]
            y = weight_horizontal * y1y3[v] + weight_horizontal_inv * y2y4[v]
            corners[v, h] = (x, y)

    # TODO: figure out why the interpolated points are not the same as the provided points
    # print(corners[0, 0], corners[0, -1], corners[-1, 0], corners[-1, -1])
    # print(points)

    corners[0, 0] = points[0]
    corners[0, -1] = points[1]
    corners[-1, 0] = points[2]
    corners[-1, -1] = points[3]

    return corners


def find_chessboard_corners_cv2(img, pattern_size: tuple) -> None:
    """Find the corners of the chessboard using OpenCV.

    Based on https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    """
    pattern_found, corners = cv2.findChessboardCorners(img, pattern_size)
    if pattern_found:
        print("Found chessboard corners")
        cv2.drawChessboardCorners(img, pattern_size, corners, pattern_found)
        cv2.imshow("", img)
    else:
        print("Could not find chessboard corners")


def check_if_corners_found(fp_folder: str) -> tuple:
    """Checks if the chessboard corners of images in a folder can be found using OpenCV.

    Part of step three in assignment 1."""
    files = os.listdir(fp_folder)

    found = []
    for file in files:
        img = cv2.imread(fp_folder + file, 1)
        pattern_found, _ = cv2.findChessboardCorners(img, (vertical_corners, horizontal_corners), None, CALIB_CB_FAST_CHECK)

        if pattern_found:
            found.append(file)
            print(f"Found chessboard corners in {file}")
        else:
            print(f"Could not find chessboard corners in {file}")

    not_found = [file for file in files if file not in found]
    return found, not_found


def calibrate_camera(fp_folder: str, horizontal_corners: int, vertical_corners: int, fp_output: str = None) -> dict:
    """Calibrate the camera using OpenCV.

    Uses a folder with images of a chessboard to calibrate the camera.

    Based on https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    """
    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(horizontal_corners-1,vertical_corners-1,0)
    objp = np.zeros((vertical_corners * horizontal_corners, 3), np.float32)
    objp[:, :2] = np.mgrid[0:vertical_corners, 0:horizontal_corners].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3D point in real world space
    imgpoints = []  # 2D points in image plane

    # Go through training images and grayscale
    images = glob.glob(fp_folder + "*.jpg")
    for file in images:
        img = cv2.imread(file)
        img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)  # Resize image to 20% of original size, to speed up processing
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        pattern_found, corners = cv2.findChessboardCorners(img_grey, (vertical_corners, horizontal_corners), None)

        # If found, add object points, image points (after refining them)
        if pattern_found == True:
            objpoints.append(objp)
            corners_improved = cv2.cornerSubPix(img_grey, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_improved)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (vertical_corners, horizontal_corners), corners_improved, pattern_found)
            cv2.imshow("", img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Return value, camera matrix, distortion coefficients, rotation and translation vectors
    return_val, camera_mat, dist_coef, rot_vec, transl_vec = cv2.calibrateCamera(objpoints, imgpoints, img_grey.shape[::-1], None, None)

    camera_mat_opt, roi = cv2.getOptimalNewCameraMatrix(camera_mat, dist_coef, img_grey.shape[::-1], 1, img_grey.shape[::-1])

    camera_params = {"camera_mat": camera_mat, "dist_coef": dist_coef, "camera_mat_opt": camera_mat_opt, "roi": roi}

    # Save the camera parameters to pickle file
    if fp_output is not None:
        with open(fp_output, "wb") as f:
            pickle.dump(camera_params, f)

    return camera_params


def undistort_image(img, camera_params) -> tuple:
    """Undistort an image using the camera matrix and distortion coefficients.

    Returns the undistorted image and the cropped image.
    """
    # Undistort the image
    img_undistorted = cv2.undistort(img, camera_params["camera_mat"], camera_params["dist_coef"], None, camera_params["camera_mat_opt"])

    # Crop the image, only keeping the region of interest
    x, y, w, h = camera_params["roi"]
    img_cropped = img_undistorted[y:y + h, x:x + w]

    return img_undistorted, img_cropped


if __name__ == "__main__":
    fp = "./images/training/01.jpg"
    img = cv2.imread(fp, 1)
    # Resize image, keeping aspect ratio
    img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)

    # Run the calibration function
    camera_params = calibrate_camera("./images/training/", horizontal_corners, vertical_corners, fp_output="./camera_params.pickle")

    # Load the camera parameters from pickle file
    with open("./camera_params.pickle", "rb") as f:
        camera_params = pickle.load(f)

    # Undistort an image
    img_undistorted, img_cropped = undistort_image(img, camera_params)
    cv2.imshow("Original", img)
    cv2.imshow("Undistorted", img_undistorted)
    cv2.imshow("Cropped", img_cropped)

    # print(points_d[len(points)])  # Prints the first instruction
    # cv2.setMouseCallback("", on_click)  # Set the callback function

    # Find chessboard corners using OpenCV
    # find_chessboard_corners_cv2(img, (vertical_corners, horizontal_corners))

    # Loop over all images, check if the corners can be found using OpenCV
    # fp = "./images/corrupt/"
    # check_if_corners_found(fp)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
