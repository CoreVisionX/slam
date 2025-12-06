import argparse
import yaml
import sys
import numpy as np
import cv2
import rerun as rr
from slam.vio import rs_sdk

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-config", type=str, default="config/checkerboard.yaml", help="Path to target.yaml")
    parser.add_argument("--vio-config", type=str, default="config/vio_d435i.yaml", help="Path to vio_config.yaml")
    parser.add_argument("--rerun-url", type=str, required=True, help="Rerun URL")
    parser.add_argument("--samples", type=int, default=10, help="Number of frames to average")
    args = parser.parse_args(argv)

    # 1. Load Configurations
    target_conf = load_yaml(args.target_config)
    vio_full_conf = load_yaml(args.vio_config)

    # Extract Target Props
    cols = target_conf['targetCols']
    rows = target_conf['targetRows']
    dx = target_conf['colSpacingMeters']
    dy = target_conf['rowSpacingMeters']

    # Extract Intrinsics (K) from VIO config
    # Note: K_left_rect is assumed to be the one corresponding to the stream's left image
    K_list = vio_full_conf['config']['K_left_rect']
    K = np.array(K_list, dtype=np.float32)
    # Rectified images usually imply 0 distortion for PnP
    D = np.zeros((4, 1), dtype=np.float32)

    # 2. Prepare 3D Object Points
    # (0,0,0), (dx,0,0), ..., (cols*dx, rows*dy, 0)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, 0] *= dx
    objp[:, 1] *= dy

    # 3. Setup Stream and Rerun
    stream = rs_sdk.D435iIterator()
    rr.init("checkerboard_pose_estimator")
    rr.connect_grpc(args.rerun_url)

    # Log Camera PinHole model once
    width = vio_full_conf['config']['width']
    height = vio_full_conf['config']['height']
    rr.log("world/camera", rr.Pinhole(image_from_camera=K, width=width, height=height))

    pose_buffer_t = []
    pose_buffer_r = []
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print(f"Looking for checkerboard ({cols}x{rows})...")

    try:
        # 4. Main Loop
        # The iterator returns: timestamp, left_img, right_img, imu_ts_list, gyro_list, accel_list
        for t_curr, left_rect, _right_rect, _, _, _ in stream:
            
            rr.set_time_seconds("stable_time", t_curr)
            
            # Ensure image is suitable for opencv (assuming SDK returns mono8 or bgr)
            if len(left_rect.shape) == 3:
                gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
            else:
                gray = left_rect

            rr.log("world/camera/image", rr.Image(left_rect))

            # Detect
            found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

            if found:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # Visualize corners
                rr.log("world/camera/image/corners", rr.Points2D(corners2, radii=3, colors=[0, 255, 0]))

                # Calculate Pose
                success, rvec, tvec = cv2.solvePnP(objp, corners2, K, D)
                tvec = tvec.reshape(3)

                if success:
                    # Log 3D Transform
                    angle = np.linalg.norm(rvec)
                    axis = rvec / angle if angle > 0 else np.array([1, 0, 0])
                    
                    rr.log(
                        "world/camera/target", 
                        rr.Transform3D(
                            translation=tvec, 
                            rotation=rr.RotationAxisAngle(axis=axis, angle=angle)
                        )
                    )

                    # Buffer for averaging
                    pose_buffer_t.append(tvec)
                    pose_buffer_r.append(rvec)

                    if len(pose_buffer_t) >= args.samples:
                        avg_t = np.mean(pose_buffer_t, axis=0)
                        # Averaging rotation vectors is a rough approximation but fine for small jitter
                        avg_r = np.mean(pose_buffer_r, axis=0)
                        
                        print(f"\n--- Average Pose ({args.samples} frames) ---")
                        print(f"Translation (m) : {avg_t.ravel()}")
                        print(f"Rotation (rvec) : {avg_r.ravel()}")
                        
                        break
            else:
                rr.log("world/camera/image/corners", rr.Clear(recursive=False))

    except KeyboardInterrupt:
        print("Stopping...")
    except Exception as err:
        raise err
    finally:
        stream.close()

if __name__ == "__main__":
    main()