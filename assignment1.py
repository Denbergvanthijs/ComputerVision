import os

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

        if len(points) < 4:  # If not all corners have been provided
            print(points_d[len(points)])

    if (len(points) == 4) and (event == cv2.EVENT_LBUTTONDOWN):  # If all corners have been provided
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


if __name__ == "__main__":
    fp = "./images/corrupt/05.jpg"
    img = cv2.imread(fp, 1)
    # Resize image, keeping aspect ratio
    img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
    cv2.imshow("", img)

    print(points_d[len(points)])  # Prints the first instruction
    cv2.setMouseCallback("", on_click)  # Set the callback function

    # Find chessboard corners using OpenCV
    # find_chessboard_corners_cv2(img, (vertical_corners, horizontal_corners))

    # Loop over all images, check if the corners can be found using OpenCV
    # fp = "./images/corrupt/"
    # check_if_corners_found(fp)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
