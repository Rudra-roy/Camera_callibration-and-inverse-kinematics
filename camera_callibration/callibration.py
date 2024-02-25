import numpy as np
import cv2

def calibrate_camera(video_source, chessboard_size, square_size, num_frames=20):
    object_points = []  # 3D points in real world space
    image_points = []   # 2D points in image plane

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ..., (6,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    cap = cv2.VideoCapture(video_source)
    frames_found = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(frame, chessboard_size, None)

        if ret:
            object_points.append(objp)
            image_points.append(corners)

            cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)
            frames_found += 1

        cv2.imshow('Calibration', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or frames_found >= num_frames:
            break

    cap.release()
    cv2.destroyAllWindows()

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, frame.shape[::-1], None, None
    )

    return ret, camera_matrix, dist_coeffs

# Video source (0 for webcam, or video file path)
video_source = 1

# Chessboard parameters
chessboard_size = (9, 6)  # Number of internal corners (columns, rows)
square_size = 1.0  # Size of one square in real world units (e.g., centimeters)

# Calibrate camera
ret, camera_matrix, dist_coeffs = calibrate_camera(video_source, chessboard_size, square_size)

if ret:
    print("Camera calibration successful.")
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs)

    # Save calibration parameters
    np.save('camera_matrix.npy', camera_matrix)
    np.save('dist_coeffs.npy', dist_coeffs)
else:
    print("Camera calibration failed.")
