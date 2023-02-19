import glob
import json
import os
import pickle

import cv2
import numpy as np


def on_click(event, x, y, flags, param) -> None:
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

        if len(points) < 4:
            print(points_d[len(points)])

    # After the 4th cornerpoint ...
    if (len(points) == 4) and (event == cv2.EVENT_LBUTTONDOWN):
        print("All corners have been provided!")

        corners = interpolate_points(points, img)

        # Check if pickle file exists, to save annotations
        if not os.path.exists(param["fp_output"]):
            all_points = {}
        else:
            with open(param["fp_output"], "rb") as f:
                all_points = pickle.load(f)

        # Add new points, overwriting old annotations if they exist
        all_points[param["fp_image"].replace("\\", "/")] = corners

        # Save pickle dict
        with open(param["fp_output"], "wb") as f:
            pickle.dump(all_points, f)

        # Draw the interpolated points
        for corner in corners:
            cv2.circle(img, corner[0].astype(int), 5, (0, 0, 255), -1)

        cv2.imshow("", img)


def interpolate_points(points, img):
    """Interpolate the points to get the corners of the chessboard."""
    x1, y1 = points[0]  # left-top
    x2, y2 = points[1]  # right-top
    x3, y3 = points[2]  # left-bottom
    x4, y4 = points[3]  # right-bottom

    x1x2 = np.linspace(x1, x2, horizontal_corners)  # Interpolate the x-coordinates of the top row
    x3x4 = np.linspace(x3, x4, horizontal_corners)  # Interpolate the x-coordinates of the bottom row

    y1y3 = np.linspace(y1, y3, vertical_corners)  # Interpolate the y-coordinates of the left column
    y2y4 = np.linspace(y2, y4, vertical_corners)  # Interpolate the y-coordinates of the right column

    corners = np.zeros((vertical_corners, horizontal_corners, 2), dtype=np.float32)  # 2D array of all corners

    for v in range(vertical_corners):
        weight_vertical_inv = v / (vertical_corners - 1)  # From 0 to 1 (minus 1 because we start at 0)
        weight_vertical = 1 - weight_vertical_inv  # From 1 to 0

        for h in range(horizontal_corners):
            # Apply weighting to the x and y coordinates
            # The closer the point is to the top or left, the more weight it gets

            weight_horizontal_inv = h / (horizontal_corners - 1)  # From 0 to 1 (minus 1 because we start at 0)
            weight_horizontal = 1 - weight_horizontal_inv  # From 1 to 0

            x = weight_vertical * x1x2[h] + weight_vertical_inv * x3x4[h]
            y = weight_horizontal * y1y3[v] + weight_horizontal_inv * y2y4[v]
            corners[v, h] = (x, y)

    # Check if estimated corners are the same as the provided corners
    est_corners = np.array([corners[0, 0], corners[0, -1], corners[-1, 0], corners[-1, -1]])
    assert np.array_equal(np.array(points), est_corners), "The four corners are not the same!"

    # Reshape corners to a 2D array
    corners = corners.reshape(-1, 1, 2)

    # TODO turn this function back on once we fixed the interpolation
    # img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # corners = cv2.cornerSubPix(img_grey, corners, (11, 11), (-1, -1), criteria)

    return corners


def make_object_points(horizontal_corners: int, vertical_corners: int, square_size: int) -> np.ndarray:
    """Make the object points for the chessboard.q

    The object points are the 3D points in real world space.
    """
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(horizontal_corners-1,vertical_corners-1,0)
    objp = np.zeros((vertical_corners * horizontal_corners, 3), np.float32)
    objp[:, :2] = np.mgrid[0:vertical_corners, 0:horizontal_corners].T.reshape(-1, 2)
    objp *= square_size  # Multiply by square size to get the real world coordinates, in milimeters
    return objp


def calculate_calibration_stats(camera_params: dict, norm: int = cv2.NORM_L2) -> dict:
    """Calculate the calibration statistics.

    Based on https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    """
    objectpoints = camera_params["object_points"]
    imagepoints = camera_params["image_points"]

    camera_mat = camera_params["camera_matrix"]
    dist_coef = camera_params["distortion_coefficients"]
    rot_vec = camera_params["rotation_vectors"]
    transl_vec = camera_params["translation_vectors"]

    errors = []
    for image in range(len(objectpoints)):
        imagepoints_proj, _ = cv2.projectPoints(objectpoints[image], rot_vec[image], transl_vec[image], camera_mat, dist_coef)
        error = cv2.norm(imagepoints[image], imagepoints_proj, norm) / len(imagepoints_proj)
        errors.append(error)

    return {"total_images": len(objectpoints),
            "mean_error": sum(errors) / len(objectpoints),
            "individual_errors": errors}


def calibrate_camera(fp_folder: str, horizontal_corners: int, vertical_corners: int, square_size: int, fp_annotations: str, fp_output: str = None) -> dict:
    """Calibrate the camera using OpenCV.

    Uses a folder with images of a chessboard to calibrate the camera.

    Based on https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    """
    objp = make_object_points(horizontal_corners, vertical_corners, square_size)  # 3D points in real world space

    # Array to store image points from all the images
    imgpoints = []  # 2D points in image plane

    # Go through training images and grayscale
    images = glob.glob(fp_folder + "**/*.jpg", recursive=True)
    for file in images:
        file = file.replace("\\", "/")  # Replace backslashes with forward slashes, because Windows
        img = cv2.imread(file)
        img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)  # Resize image to 20% of original size, to speed up processing
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        pattern_found, corners = cv2.findChessboardCorners(img_grey, (vertical_corners, horizontal_corners), None)

        # Load pickle with annotations
        with open(fp_annotations, "rb") as f:
            annotations = pickle.load(f)

        # If found, add image points (after refining them)
        if pattern_found:
            print(f"Found chessboard corners in {file}")
            corners = cv2.cornerSubPix(img_grey, corners, (11, 11), (-1, -1), criteria)
        else:
            print(f"Could not find chessboard corners in {file}")

            # Check if annotations are available
            if file in annotations.keys():
                corners = annotations[file]
                print(f"Using annotations for {file}")
            else:
                print(f"Could not find annotations for {file}")
                continue

        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (vertical_corners, horizontal_corners), corners, pattern_found)
        cv2.imshow("", img)
        cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Since all images are taken with the same camera, the object points are the same
    objpoints = len(imgpoints) * [objp]  # 3D point in real world space

    # Return value, camera matrix, distortion coefficients, rotation and translation vectors
    return_val, camera_mat, dist_coef, rot_vec, transl_vec = cv2.calibrateCamera(objpoints, imgpoints, img_grey.shape[::-1], None, None)

    camera_mat_opt, roi = cv2.getOptimalNewCameraMatrix(camera_mat, dist_coef, img_grey.shape[::-1], 1, img_grey.shape[::-1])

    # Also save the object points and image points to calculate the calibration statistics
    camera_params = {"camera_matrix": camera_mat, "distortion_coefficients": dist_coef, "camera_matrix_optimal": camera_mat_opt, "roi": roi,
                     "rotation_vectors": rot_vec, "translation_vectors": transl_vec, "object_points": objpoints, "image_points": imgpoints}

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
    img_undistorted = cv2.undistort(img, camera_params["camera_matrix"], camera_params["distortion_coefficients"],
                                    None, camera_params["camera_matrix_optimal"])

    # Crop the image, only keeping the region of interest
    x, y, w, h = camera_params["roi"]
    img_cropped = img_undistorted[y:y + h, x:x + w]

    return img_undistorted, img_cropped


if __name__ == "__main__":
    calibration = True  # Set to True to calibrate the camera, False to annotate the images

    # Annotation mode
    fp_image = "./images/test/01.jpg"  # Set the path to the image to be annotated
    fp_annotations = "./data/annotations.pickle"  # Set the path to save the annotations

    # Calibration mode
    fp_input_images = "./images/run1/"  # Set the path to the folder with the images to be used for calibration
    fp_camera_params = "./data/camera_params_run1.pickle"  # Set the path to save the camera parameters
    fp_stats = "./data/stats_run1.json"  # Set the path to save the stats of the calibration

    # Additional parameters
    horizontal_corners = 6  # Set the number of horizontal corners of the chessboard
    vertical_corners = 9  # Set the number of vertical corners of the chessboard
    square_size = 22  # Set the size of one square in the chessboard, in milimeters

    points = []
    points_d = {0: "Provide the left-top corner of the checkerboard",
                1: "Provide the right-top corner of the checkerboard",
                2: "Provide the left-bottom corner of the checkerboard",
                3: "Provide the right-bottom corner of the checkerboard"}

    font = cv2.FONT_HERSHEY_SIMPLEX
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    img = cv2.imread(fp_image, 1)
    # Resize image, keeping aspect ratio
    img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
    cv2.imshow("", img)

    if calibration:  # Calibration mode
        # Run the calibration function
        camera_params = calibrate_camera(fp_input_images, horizontal_corners, vertical_corners, square_size,
                                         fp_annotations, fp_output=fp_camera_params)
        stats = calculate_calibration_stats(camera_params)
        print(f"The mean error is {stats['mean_error']}")

        # Load the camera parameters from pickle file
        with open(fp_camera_params, "rb") as f:
            camera_params = pickle.load(f)

        # Save stats as json file
        with open(fp_stats, "w") as f:
            json.dump(stats, f, indent=4)

        # Undistort an image
        img_undistorted, img_cropped = undistort_image(img, camera_params)
        cv2.imshow("Original", img)
        cv2.imshow("Undistorted", img_undistorted)
        cv2.imshow("Cropped", img_cropped)

    else:  # Annotation mode
        print(points_d[len(points)])  # Prints the first instruction
        cv2.setMouseCallback("", on_click, param={"fp_image": fp_image, "fp_output": fp_annotations})  # Set the callback function

    cv2.waitKey(0)
    cv2.destroyAllWindows()
