# Stereo Vision Room Mapping Experiment Report

## Objective
The objective of this experiment is to compute and map the real-world 2D locations of tables and chairs in a classroom using a stereo camera setup. By applying epipolar geometry and stereo disparity calculations, the system extracts the physical depth (distance from camera) and horizontal spacing (left/right offset) of objects, projecting them onto a top-down X-Z floor plan view parallel to the floor.

## Procedure
To simulate a stereo camera rig using a single uncalibrated smartphone camera:
1. **Camera Positioning**: Place the camera flat on a stable edge (e.g., a desk or wall) facing the classroom.
2. **Image Capture**: 
   - Take the first picture and save it as `left.jpg`.
   - Slide the camera **strictly horizontally** by a known physical distance to form the *baseline* (e.g., exactly 10.0 cm). Do not rotate or tilt the camera during this translation.
   - Take the second picture and save it as `right.jpg`.
3. **Execution**: Run the mapping script from the terminal, specifying the physical baseline distance used:
   ```bash
   python stereo_mapper.py --baseline 10.0
   ```
4. **Interactive Mapping**: The script will prompt the user to categorize an object as either a Table (`t`) or a Chair (`c`). Using the GUI:
   - Click a distinct feature point on an object in the left image.
   - Click the exact same corresponding feature point on the object in the right image.
5. **Output Processing**: After labeling all target objects, pressing `q` will automatically triangulate the 3D coordinates and output `floor_plan.png`, a top-down scatter plot marking tables in red squares and chairs in blue circles.

## Methodology & How It Works

### Dynamic Intrinsic Calibration
The script utilizes a pre-calculated intrinsic camera matrix containing the focal length ($f$) and optical center coordinates ($c_x$, $c_y$). Because the user captures photos at a high pixel resolution (e.g., 2268x4032) but the calibration data was calculated for a different resolution (e.g., 640x480), the script dynamically computes a scale factor. It multiplies the focal length and center parameters by the image scale ratio to ensure that coordinate projection accurately matches the high-resolution smartphone sensor.

### Stereo Disparity Mathematics
When the camera translates purely horizontally along the X-axis by baseline $B$, objects in the field of view shift horizontally across the sensor.
1. **Disparity ($d$)**: The script records the pixel location $x_{left}$ and $x_{right}$ based on the user's manual clicks. The absolute difference between these coordinates is the Disparity. 
   $$d = |x_{left} - x_{right}|$$
2. **Depth ($Z$)**: Using similar triangles, the physical distance moving into the room (away from the camera plane) is calculated. Objects closer to the camera exhibit larger pixel disparity, while distant objects shift very little.
   $$Z = \frac{f \cdot B}{d}$$
3. **Horizontal Position ($X$)**: The physical horizontal distance from the camera's optical center is reconstructed by scaling the pixel offset against the object's computed depth.
   $$X = \frac{(x_{left} - c_x) \cdot Z}{f}$$

### 2D Floor Plan Projection
The above mathematical formulas construct a complete 3D spatial coordinate $(X, Y, Z)$. Since the objective asks for a structured mapping that is parallel to the floor, the vertical height component ($Y$, normal to the floor) is discarded.

The points are subsequently projected as an $X-Z$ plane onto a 2-dimensional `matplotlib` scatter graph. Depth ($Z$) provides the vertical graph plotting axis, and Horizontal offset ($X$) provides the horizontal graph axis, creating an exact to-scale bird's-eye map of the localized tables and chairs within the classroom.
