import cv2
import numpy as np

points = []
points_d = {0: "Provide the left-top corner of the checkerboard",
            1: "Provide the right-top corner of the checkerboard",
            2: "Provide the left-bottom corner of the checkerboard",
            3: "Provide the right-bottom corner of the checkerboard"}
horizontal_corners = 6
vertical_corners = 9
font = cv2.FONT_HERSHEY_SIMPLEX


def on_click(event, x, y, flags, params):
    """Callback function for mouse events."""
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
                cv2.circle(img, (int(corners[v, h, 0]), int(corners[v, h, 1])), 5, (0, 0, 255), -1)

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
        for h in range(horizontal_corners):
            # Apply weighting to the x and y coordinates
            # The closer the point is to the top or left, the more weight it gets
            x = ((vertical_corners-v)/vertical_corners) * x1x2[h] + (v/vertical_corners) * x3x4[h]
            y = ((horizontal_corners-h)/horizontal_corners) * y1y3[v] + (h/horizontal_corners) * y2y4[v]
            corners[v, h] = (x, y)

    return corners


def find_chessboard_corners_cv2(img, pattern_size):
    """Find the corners of the chessboard using OpenCV.

    Based on https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    """
    pattern_found, corners = cv2.findChessboardCorners(img, pattern_size)
    if pattern_found:
        print("Found chessboard corners")
        cv2.drawChessboardCorners(img, pattern_size, corners, pattern_found)
        cv2.imshow("", img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Could not find chessboard corners")


if __name__ == "__main__":
    fp = "./images/training/02.jpg"
    img = cv2.imread(fp, 1)
    # Resize image, keeping aspect ratio
    img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
    cv2.imshow("", img)

    print(points_d[len(points)])
    cv2.setMouseCallback("", on_click)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # find_chessboard_corners_cv2(img, (vertical_corners, horizontal_corners))
