import pickle

import cv2
import numpy as np


def draw(img, corners, imagepoints):
    """Draws the axis on the image.

    Based on https://docs.opencv.org/4.6.0/d7/d53/tutorial_py_pose.html
    """
    corners = corners.astype(int)
    origin = corners[0].reshape(2)  # Flattening the array
    imagepoints = imagepoints.astype(int).reshape(-1, 2)
    colour = (0, 165, 255)  # Orange

    # Draw a cube at the origin
    for start, end in zip(range(4), range(4, 8)):
        # Draw the lines between the corners
        img = cv2.line(img, imagepoints[start], imagepoints[end], colour, 3)

    img = cv2.drawContours(img, [imagepoints[4:]], -1, colour, 3)  # Top face
    img = cv2.drawContours(img, [imagepoints[:4]], -1, colour, 3)  # Bottom face

    # Draw the three main axis
    img = cv2.line(img, origin, imagepoints[3], (255, 0, 0), 10)  # Draw x axis in blue
    img = cv2.line(img, origin, imagepoints[1], (0, 255, 0), 10)  # Draw y axis in green
    img = cv2.line(img, origin, imagepoints[4], (0, 0, 255), 10)  # Draw z axis in red

    # Draw big circle at origin
    img = cv2.circle(img, tuple(origin), 15, colour, -1)

    # Draw black circles at imagepoints
    for start in imagepoints:
        img = cv2.circle(img, tuple(start), 5, (0, 0, 0), -1)

    return img


if __name__ == "__main__":
    # Set the file paths
    fp_input_image = "./images/test/01.jpg"  # Set the path to the test image
    fp_output = "./images/cube/image_run1.png"  # Set the path to save the test image, with axis
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

    # Load test image with openCV
    print(f"Loading test image: {fp_input_image}")
    img = cv2.imread(fp_input_image)
    img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

    pattern_found, corners = cv2.findChessboardCorners(img_grey, (horizontal_corners, vertical_corners), None)
    if pattern_found:
        print("Pattern found")
        # Improve the accuracy of the corners
        print("Improve the accuracy of the corners")
        corners_opt = cv2.cornerSubPix(img_grey, corners, (11, 11), (-1, -1), criteria)

        # Find the rotation and translation vectors
        print("Find the rotation and translation vectors")
        pattern_found, rot_vec, transl_vec = cv2.solvePnP(objp, corners_opt, camera_mat, dist_coef)

        # Project 3D points to image plane
        print("Project 3D points to image plane")
        imagepoints, jac = cv2.projectPoints(axis, rot_vec, transl_vec, camera_mat, dist_coef)

        # Draw the axis on the image
        print("Draw the axis on the image")
        img = draw(img, corners_opt, imagepoints)
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.imwrite(fp_output, img)
    else:
        print("Pattern not found")

    cv2.destroyAllWindows()
