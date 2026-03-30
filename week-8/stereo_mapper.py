import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json

def load_calibration():
    # Load calibration.json or camera_matrix.npy
    calib_path = os.path.join("calibration_data", "calibration.json")
    if not os.path.exists(calib_path):
        print("Calibration data not found at", calib_path)
        return None, None
        
    with open(calib_path, 'r') as f:
        data = json.load(f)
        
    camera_matrix = np.array(data["camera_matrix"])
    dist_coeffs = np.array(data["dist_coeffs"])
    return camera_matrix, dist_coeffs

class StereoMapper:
    def __init__(self, baseline_cm, focal_length_px, cx, cy):
        self.baseline_cm = baseline_cm
        self.focal_length_px = focal_length_px
        self.cx = cx
        self.cy = cy
        
        self.tables = []   # List of 3D coordinates (X, Y, Z)
        self.chairs = []   # List of 3D coordinates (X, Y, Z)
        self.current_points = []
        self.img_left = None
        self.img_right = None
        
    def compute_3d_point(self, pt_left, pt_right):
        # pt_left and pt_right are (u, v) pixel coordinates
        xl, yl = pt_left
        xr, yr = pt_right
        
        # Disparity (difference in horizontal pixel coordinate)
        disparity = abs(xl - xr)
        
        # Check for vertical movement/rotation
        y_diff = abs(yl - yr)
        if y_diff > 20: # 20 pixels difference is quite large for pure horizontal movement
            print(f"  [!] Warning: Large vertical difference ({y_diff}px) between left and right points.")
            print("      This means the camera tilted or moved up/down between pictures. Depth might be inaccurate.")

        if disparity == 0:
            print("Warning: Disparity is 0 (you clicked the exact same X coordinate). Objects need to be closer or baseline needs to be larger! Assuming small disparity of 0.1 for now.")
            # We enforce a small positive disparity to avoid div by zero
            disparity = 0.1
            
        # Z is the depth (distance perpendicular to the camera plane)
        Z = (self.focal_length_px * self.baseline_cm) / disparity
        
        # X is the horizontal distance from the camera center
        X = ((xl - self.cx) * Z) / self.focal_length_px
        
        # Y is the vertical distance from the camera center
        Y = ((yl - self.cy) * Z) / self.focal_length_px
        
        return (X, Y, Z)

    def gui_loop(self, left_path, right_path):
        self.img_left = cv2.imread(left_path)
        self.img_right = cv2.imread(right_path)
        
        if self.img_left is None or self.img_right is None:
            print("Error: Could not load the images. Make sure left.jpg and right.jpg exist!")
            return
            
        # Dynamically scale camera parameters to match the CURRENT image resolution!
        # The calibration was likely done at 640x480. If you used a phone camera 
        # (e.g., 3024x4032), we must scale focal length and centers, otherwise X and Z will be completely distorted!
        img_h, img_w = self.img_left.shape[:2]
        calib_w, calib_h = 640, 480 # Default calibration resolution
        
        scale_x = img_w / calib_w
        scale_y = img_h / calib_h
        
        if scale_x != 1.0 or scale_y != 1.0:
            print(f"[*] Adjusting camera calibration for image resolution {img_w}x{img_h}...")
            self.focal_length_px *= scale_x
            self.cx *= scale_x
            self.cy *= scale_y
            print(f"    New params: fx={self.focal_length_px:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")

        print("\n=== Stereo Mapper CLI ===")
        print("Instructions:")
        print("1. We will prompt you to select an object type in the terminal.")
        print("2. A window will pop up showing the LEFT image.")
        print("3. Click on the object in the LEFT image.")
        print("4. Then, a window will pop up showing the RIGHT image.")
        print("5. Click on the SAME EXACT POINT on the object in the RIGHT image.")
        print("=========================\n")
        
        while True:
            choice = input("Enter 't' for Table, 'c' for Chair, or 'q' to compute & quit: ").strip().lower()
            
            if choice == 'q':
                break
            elif choice not in ['t', 'c']:
                print("Invalid choice.")
                continue
                
            obj_type = "Table" if choice == 't' else "Chair"
            
            # --- Get Left Point ---
            self.current_points = []
            
            def mouse_callback_left(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.current_points.append((x, y))
                    print(f"  > Left image clicked at ({x}, {y})")

            win_name_left = f"LEFT Image - Select {obj_type} - Press Space when done"
            cv2.namedWindow(win_name_left, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name_left, 800, 600)
            cv2.setMouseCallback(win_name_left, mouse_callback_left)
            
            print(f"--> [Action Needed]: Click the {obj_type} in the LEFT image window, then press 'Spacebar' or 'Enter' in the image window.")
            while True:
                # Draw a circle where they clicked
                img_disp = self.img_left.copy()
                if len(self.current_points) > 0:
                    cv2.circle(img_disp, self.current_points[0], 5, (0, 0, 255), -1)
                
                cv2.imshow(win_name_left, img_disp)
                k = cv2.waitKey(20) & 0xFF
                if k in [32, 13]: # Space or Enter
                    if len(self.current_points) >= 1:
                        break
                    else:
                        print("Please click a point first!")
            
            pt_left = self.current_points[0]
            cv2.destroyWindow(win_name_left)
            
            # --- Get Right Point ---
            self.current_points = []
            
            def mouse_callback_right(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.current_points.append((x, y))
                    print(f"  > Right image clicked at ({x}, {y})")

            win_name_right = f"RIGHT Image - Select same {obj_type} - Press Space when done"
            cv2.namedWindow(win_name_right, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name_right, 800, 600)
            cv2.setMouseCallback(win_name_right, mouse_callback_right)
            
            print(f"--> [Action Needed]: Click the SAME {obj_type} in the RIGHT image window, then press 'Spacebar' or 'Enter' in the image window.")
            while True:
                img_disp = self.img_right.copy()
                if len(self.current_points) > 0:
                    cv2.circle(img_disp, self.current_points[0], 5, (255, 0, 0), -1)
                
                cv2.imshow(win_name_right, img_disp)
                k = cv2.waitKey(20) & 0xFF
                if k in [32, 13]: # Space or Enter
                    if len(self.current_points) >= 1:
                        break
                    else:
                        print("Please click a point first!")
            
            pt_right = self.current_points[0]
            cv2.destroyWindow(win_name_right)
            
            # Compute 3D and store
            pt_3d = self.compute_3d_point(pt_left, pt_right)
            if choice == 't':
                self.tables.append(pt_3d)
            else:
                self.chairs.append(pt_3d)
                
            print(f"Added {obj_type} at 3D location (X={pt_3d[0]:.2f}, Y={pt_3d[1]:.2f}, Z={pt_3d[2]:.2f} cm)")
            
        # Plotting
        self.plot_results()

    def plot_results(self):
        plt.figure(figsize=(8, 8))
        print(f"\nComputing Output Plot with {len(self.tables)} Tables and {len(self.chairs)} Chairs...")
        
        # We plot X (horizontal) vs Z (depth away from camera) to get a top-down floor plan.
        # This is exactly the 2D plane parallel to the floor.
        if self.tables:
            tx = [p[0] for p in self.tables] # X coordinate
            tz = [p[2] for p in self.tables] # Z coordinate (depth)
            plt.scatter(tx, tz, c='red', marker='s', s=100, label='Tables')
            
        if self.chairs:
            cx = [p[0] for p in self.chairs]
            cz = [p[2] for p in self.chairs]
            plt.scatter(cx, cz, c='blue', marker='o', s=100, label='Chairs')
            
        plt.xlabel("X - Horizontal Position (cm)")
        plt.ylabel("Z - Depth from Camera (cm)")
        plt.title("Top-Down 2D Floor Plan (X-Z Plane Parallel to Floor)")
        plt.grid(True)
        plt.legend()
        
        # Fix aspect ratio so 1cm horizontal = 1cm depth
        plt.axis('equal') 
        
        # Ensure that zero depth is at the bottom, growing upwards
        ax = plt.gca()
        if ax.get_ylim()[0] > ax.get_ylim()[1]:
            ax.invert_yaxis()
            
        plt.savefig("floor_plan.png")
        plt.show()
        print("Done! Saved plot to 'floor_plan.png'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stereo compute locations of tables and chairs.")
    parser.add_argument("--left", default="left.jpg", help="Path to left image")
    parser.add_argument("--right", default="right.jpg", help="Path to right image")
    parser.add_argument("--baseline", type=float, default=10.0, help="Horizontal distance moved between pictures in cm")
    
    args = parser.parse_args()
    
    # Load calibration parameters
    mtx, dist = load_calibration()
    if mtx is None:
        print("Cannot proceed without camera calibration matrix. Using fallback identity matrix for demo purposes.")
        # Fallback values if calibration is missing
        fx = 500.0
        cx = 640/2
        cy = 480/2
    else:
        # Assuming fx and fy are close, we'll just use fx
        fx = mtx[0, 0]
        cx = mtx[0, 2]
        cy = mtx[1, 2]
        
    print(f"Loaded camera intrinsics: focal_length={fx:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    
    mapper = StereoMapper(baseline_cm=args.baseline, focal_length_px=fx, cx=cx, cy=cy)
    
    if not os.path.exists(args.left) or not os.path.exists(args.right):
        print("\n[!] IMPORTANT: Could not find {} or {}".format(args.left, args.right))
        print("Please take two pictures of your room and place them in this folder:")
        print("  1. Place camera on a flat edge.")
        print("  2. Take the first picture and save as 'left.jpg'.")
        print(f"  3. slide the camera STRICTLY sideways (horizontally) by exactly {args.baseline} cm.")
        print(f"     Do not rotate the camera! Make sure it points straight ahead just like the first picture.")
        print("  4. Take the second picture and save as 'right.jpg'.")
        print("\nOnce you have left.jpg and right.jpg, run this script again.")
    else:
        mapper.gui_loop(args.left, args.right)
