# Week 8: Stereo Vision Room Mapping

This project computes the 2D floor locations of tables and chairs in a classroom using a simple stereo camera setup. It calculates the depth using the camera calibration from `week-2`.

## Requirements
```bash
pip install -r requirements.txt
```

## How to Simulate Stereo with One Camera

If you only have one camera, you can take two pictures to simulate a stereo rig:
1. Place the camera flat on a table/desk.
2. Take the first picture and save it as `left.jpg`.
3. Slide the camera exactly horizontally by 10 centimeters (this is the *baseline*). Make sure NOT to rotate it. It should point perfectly straight ahead just like the first picture.
4. Take the second picture and save it as `right.jpg`.

*Note: The translation distance must correspond to the `--baseline` parameter passed to the script.*

## Running the Script

Place your `left.jpg` and `right.jpg` in this directory, then run:

```bash
python stereo_mapper.py --baseline 10.0
```

### Usage Instructions
1. The terminal will prompt you to enter the object type you are selecting:
   - Type `t` for a Table (will be plotted in Red).
   - Type `c` for a Chair (will be plotted in Blue).
   - Type `q` when you are finished adding objects.
2. A window will appear showing the **LEFT** image. Click exactly on the object, then press **Spacebar**.
3. A second window will appear showing the **RIGHT** image. Click exactly on the *same physical point* of the object in the right image, then press **Spacebar**.
4. Repeat this process for all tables and chairs you want mapped.
5. Once you enter `q`, the script will calculate their 3D positions in space, map them onto a 2D floor plan, and save `floor_plan.png`.
