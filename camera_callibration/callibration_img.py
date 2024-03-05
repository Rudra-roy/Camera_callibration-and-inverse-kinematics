import numpy as np
import cv2
import glob

def calibrate_camera(images_folder, chessboard_size, square_size):
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ..., (6,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    # Arrays to store object points and image points from all images
    object_points = []  # 3D points in real world space
    image_points = []   # 2D points in image plane

    # Find images in the given folder
    images = glob.glob(images_folder + '/*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # If corners are found, add object points and image points
        if ret == True:
            object_points.append(objp)
            image_points.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
            cv2.imshow('Chessboard Corners', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Calibrate camera
    if object_points and image_points:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
        return ret, mtx, dist
    else:
        return False, None, None

# Example usage
images_folder = 'E:/opencv/images/'
chessboard_size = (7, 9)  # Number of inner corners of the chessboard
square_size = 0.02  # Size of each square in meters
ret, camera_matrix, distortion_coefficients = calibrate_camera(images_folder, chessboard_size, square_size)


if ret:
    np.save('camera_matrix1.npy', camera_matrix)
    np.save('dist_coeffs1.npy', distortion_coefficients)
    print("Calibration successful.")
    print("Camera matrix:")
    print(camera_matrix)
    print("Distortion coefficients:")
    print(distortion_coefficients)
else:
    print("Calibration failed.")
