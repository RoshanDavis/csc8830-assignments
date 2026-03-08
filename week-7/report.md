# Week 7: Optical Flow & Motion Tracking — Report

## Table of Contents

1. [Part A: Optical Flow Computation & Visualization](#part-a-optical-flow-computation--visualization)
   - [Theory](#a1-theory-of-optical-flow)
   - [Procedure](#a2-procedure)
   - [Information Inferred from Optical Flow](#a3-what-information-can-be-inferred)
   - [Evidence](#a4-evidence-from-output-videos)
2. [Part B: Motion Tracking & Bilinear Interpolation](#part-b-motion-tracking--bilinear-interpolation)
   - [Derivation of Tracking Equations](#b1-derivation-of-motion-tracking-equations-from-fundamentals)
   - [Setting Up Tracking for Two Frames](#b2-setting-up-the-tracking-problem-for-two-image-frames)
   - [Bilinear Interpolation](#b3-derivation-of-bilinear-interpolation)
   - [Validation](#b4-validation-of-tracking-with-actual-pixel-locations)

---

## Part A: Optical Flow Computation & Visualization

### A.1 Theory of Optical Flow

**Optical flow** is the pattern of apparent motion of objects, surfaces, or edges in a visual scene, caused by the relative movement between a camera and the scene. It is represented as a 2D vector field where each vector $(u, v)$ describes the displacement of a pixel from one frame to the next.

#### Brightness Constancy Assumption

The foundation of all optical flow methods is the **brightness constancy constraint** — the assumption that the intensity of a pixel does not change as it moves between frames:

$$I(x, y, t) = I(x + \Delta x,\ y + \Delta y,\ t + \Delta t)$$

where $I$ is the image intensity at spatial location $(x, y)$ and time $t$.

#### Farnebäck Dense Optical Flow

We use the **Farnebäck method** (Gunnar Farnebäck, 2003), which computes **dense** optical flow — i.e., a flow vector for *every* pixel in the image. The method works as follows:

1. **Polynomial Expansion**: Each neighborhood in the image is approximated by a quadratic polynomial:

$$f(x) \approx x^T A x + b^T x + c$$

   where $A$ is a symmetric matrix, $b$ is a vector, and $c$ is a scalar.

2. **Displacement Estimation**: Given two consecutive frames with polynomial approximations $f_1$ and $f_2$, if $f_2$ is a displaced version of $f_1$ by displacement $d$:

$$f_2(x) = f_1(x - d)$$

   Then by equating the polynomial coefficients, the displacement $d$ can be solved analytically.

3. **Iterative Refinement**: The method uses an image pyramid (multi-scale) and iterative refinement with the parameters:
   - `pyr_scale = 0.5` — each pyramid level is half the resolution of the previous
   - `levels = 3` — 3 pyramid levels
   - `winsize = 15` — 15×15 averaging window
   - `iterations = 3` — refinement iterations per level
   - `poly_n = 5` — polynomial expansion neighborhood size
   - `poly_sigma = 1.2` — Gaussian sigma for polynomial smoothing

#### HSV Color Visualization

The computed flow field is visualized using the **HSV (Hue-Saturation-Value) color space**:

| Channel | Encodes | Formula |
|---------|---------|---------|
| **Hue** | Direction of motion | $\theta = \arctan(v / u)$, mapped to [0°, 180°] |
| **Saturation** | Constant | Fixed at 255 (full saturation) |
| **Value** | Speed (magnitude) | $\|d\| = \sqrt{u^2 + v^2}$, normalized to [0, 255] |

This produces an intuitive color mapping:

| Color | Direction |
|-------|-----------|
| Red | Rightward motion |
| Cyan | Leftward motion |
| Green | Downward motion |
| Magenta | Upward motion |
| Dark / Black | No motion (static) |
| Bright | Fast motion |

---

### A.2 Procedure

1. **Input**: Two video files containing motion (≥30 seconds each).
2. **Frame-by-Frame Processing**:
   - Read consecutive frames and convert to grayscale.
   - Compute dense optical flow using `cv2.calcOpticalFlowFarneback()`.
   - Convert the flow vectors to HSV color visualization.
   - Write a side-by-side video: original frame | flow visualization.
3. **Output**: Two visualization videos (`flow_video1.mp4`, `flow_video2.mp4`).

```python
# Core optical flow computation
flow = cv2.calcOpticalFlowFarneback(
    prev_gray, curr_gray,
    flow=None, pyr_scale=0.5, levels=3,
    winsize=15, iterations=3, poly_n=5,
    poly_sigma=1.2, flags=0
)

# Convert flow to HSV visualization
magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
hsv[..., 0] = angle * 180 / np.pi / 2        # Hue = direction
hsv[..., 1] = 255                              # Full saturation
hsv[..., 2] = cv2.normalize(magnitude, ...)    # Value = speed
```

---

### A.3 What Information Can Be Inferred

Optical flow reveals the following information from video sequences:

#### 1. Direction of Object Motion
Each flow vector's angle indicates the direction a pixel is moving. The HSV hue channel makes this directly visible — different colors correspond to different motion directions. For example, a person walking left appears cyan while a car moving right appears red.

#### 2. Speed and Magnitude of Motion
The magnitude of each flow vector indicates how fast a pixel is moving (in pixels per frame). In the visualization, brighter regions correspond to faster motion. A slowly drifting cloud will appear dim, while a passing vehicle will appear bright.

#### 3. Moving vs. Static Regions (Motion Segmentation)
Regions with near-zero flow vectors are stationary (background), while regions with significant flow correspond to moving objects. This naturally segments the scene into foreground (moving) and background (static) without any learning-based approach.

#### 4. Object Boundaries
Discontinuities in the flow field often align with object boundaries. Where a moving object meets the static background, there is an abrupt change in flow vectors. These edges in the flow field can be used for object contour detection.

#### 5. Camera Motion vs. Object Motion
- **Camera pan/tilt**: Produces a globally coherent flow field (all pixels shift in the same direction).
- **Camera zoom**: Produces a radial flow pattern expanding from the center.
- **Independent object motion**: Produces localized flow patterns that differ from the global trend.
By analyzing the dominant flow pattern, camera motion can be distinguished from independent object motion.

---

### A.4 Evidence from Output Videos

After running the script on the two videos, the following evidence is observable in the output flow videos:

| Observation | Evidence |
|-------------|----------|
| **Static background is dark** | Background regions with no motion appear black/very dark in the flow visualization |
| **Moving objects are colored** | People, vehicles, or other moving objects appear as brightly colored regions |
| **Color indicates direction** | Objects moving in different directions show different hues (e.g., leftward = cyan, rightward = red) |
| **Brightness indicates speed** | Fast-moving objects appear brighter than slow-moving ones |
| **Object boundaries are visible** | Edges of moving objects show sharp color transitions against the dark background |

The mean and max flow magnitudes reported by the script quantify the overall motion level in each video. Higher mean magnitudes indicate more overall motion; spikes in max magnitude correspond to rapid movement events.

---

## Part B: Motion Tracking & Bilinear Interpolation

### B.1 Derivation of Motion Tracking Equations from Fundamentals

#### Step 1: Brightness Constancy Constraint

We begin with the fundamental assumption that a pixel's intensity does not change as it moves:

$$I(x, y, t) = I(x + \delta x,\ y + \delta y,\ t + \delta t) \quad \cdots (1)$$

where $(\delta x, \delta y)$ is the pixel's displacement in time $\delta t$.

#### Step 2: Taylor Series Expansion

Expand the right-hand side of equation (1) using a first-order Taylor series:

$$I(x + \delta x, y + \delta y, t + \delta t) \approx I(x, y, t) + \frac{\partial I}{\partial x} \delta x + \frac{\partial I}{\partial y} \delta y + \frac{\partial I}{\partial t} \delta t$$

Substituting into (1) and cancelling $I(x, y, t)$ from both sides:

$$\frac{\partial I}{\partial x} \delta x + \frac{\partial I}{\partial y} \delta y + \frac{\partial I}{\partial t} \delta t = 0$$

Dividing throughout by $\delta t$:

$$I_x \cdot u + I_y \cdot v + I_t = 0 \quad \cdots (2)$$

where:
- $I_x = \frac{\partial I}{\partial x}$ — spatial gradient in x
- $I_y = \frac{\partial I}{\partial y}$ — spatial gradient in y
- $I_t = \frac{\partial I}{\partial t}$ — temporal gradient
- $u = \frac{\delta x}{\delta t}$ — horizontal velocity (optical flow in x)
- $v = \frac{\delta y}{\delta t}$ — vertical velocity (optical flow in y)

Equation (2) is the **optical flow constraint equation**. It is a single linear equation in two unknowns $(u, v)$.

#### Step 3: The Aperture Problem

Since equation (2) has two unknowns but only one equation, it is **under-determined**. This is known as the **aperture problem** — observing a moving edge through a small aperture, we can only determine the component of motion perpendicular to the edge, not the full 2D motion.

#### Step 4: Lucas-Kanade Method — Resolving the Aperture Problem

The **Lucas-Kanade method** (1981) resolves this by assuming that *all pixels within a small local window $W$ share the same flow vector $(u, v)$*.

For each pixel $(x_i, y_i)$ in a window of $n$ pixels, we write equation (2):

$$I_x(x_i, y_i) \cdot u + I_y(x_i, y_i) \cdot v = -I_t(x_i, y_i) \quad \text{for } i = 1, 2, \ldots, n$$

This gives us an **overdetermined system** of $n$ equations in 2 unknowns, written in matrix form as:

$$A \cdot \mathbf{d} = \mathbf{b}$$

where:

$$A = \begin{bmatrix} I_x(x_1, y_1) & I_y(x_1, y_1) \\ I_x(x_2, y_2) & I_y(x_2, y_2) \\ \vdots & \vdots \\ I_x(x_n, y_n) & I_y(x_n, y_n) \end{bmatrix}, \quad \mathbf{d} = \begin{bmatrix} u \\ v \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} -I_t(x_1, y_1) \\ -I_t(x_2, y_2) \\ \vdots \\ -I_t(x_n, y_n) \end{bmatrix}$$

#### Step 5: Least-Squares Solution

The system is solved using the **normal equations**:

$$A^T A \cdot \mathbf{d} = A^T \mathbf{b}$$

$$\mathbf{d} = (A^T A)^{-1} \cdot A^T \mathbf{b} \quad \cdots (3)$$

Expanding $A^T A$:

$$A^T A = \begin{bmatrix} \sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2 \end{bmatrix}$$

$$A^T \mathbf{b} = \begin{bmatrix} -\sum I_x I_t \\ -\sum I_y I_t \end{bmatrix}$$

Therefore:

$$\begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} \sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2 \end{bmatrix}^{-1} \begin{bmatrix} -\sum I_x I_t \\ -\sum I_y I_t \end{bmatrix}$$

This matrix is invertible (and the solution is reliable) when $A^T A$ has two large eigenvalues — which corresponds to corners or textured regions. This is exactly the criterion used by the **Shi-Tomasi corner detector** (`cv2.goodFeaturesToTrack`) to select good features for tracking.

---

### B.2 Setting Up the Tracking Problem for Two Image Frames

Given two consecutive frames $I_1$ (at time $t$) and $I_2$ (at time $t+1$):

#### Procedure

| Step | Operation | Implementation |
|------|-----------|----------------|
| 1 | **Feature Detection** | Detect corner features in $I_1$ using Shi-Tomasi detector (`cv2.goodFeaturesToTrack`) |
| 2 | **Compute Spatial Gradients** | $I_x = \frac{\partial I_1}{\partial x}$, $I_y = \frac{\partial I_1}{\partial y}$ using Sobel operators or finite differences |
| 3 | **Compute Temporal Gradient** | $I_t = I_2 - I_1$ (pixel-wise difference) |
| 4 | **Build Linear System** | For each feature, gather $I_x$, $I_y$, $I_t$ over a local window to form $A$ and $\mathbf{b}$ |
| 5 | **Solve for Flow** | Compute $(u, v) = (A^T A)^{-1} A^T \mathbf{b}$ |
| 6 | **Predict New Position** | If feature is at $(x, y)$ in $I_1$, its predicted position in $I_2$ is $(x + u, y + v)$ |

#### Concrete Example

Suppose a feature is detected at position $(150, 200)$ in frame 1. Within a 15×15 window centered at that point, we compute the image gradients and form the system:

- $\sum I_x^2 = 48000$, $\sum I_y^2 = 35000$, $\sum I_x I_y = 5000$
- $-\sum I_x I_t = 12000$, $-\sum I_y I_t = -7000$

Solving:

$$\begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} 48000 & 5000 \\ 5000 & 35000 \end{bmatrix}^{-1} \begin{bmatrix} 12000 \\ -7000 \end{bmatrix}$$

The determinant: $48000 \times 35000 - 5000^2 = 1,655,000,000$

$$u = \frac{35000 \times 12000 - 5000 \times (-7000)}{1,655,000,000} = \frac{420000000 + 35000000}{1655000000} \approx 0.275$$

$$v = \frac{48000 \times (-7000) - 5000 \times 12000}{1,655,000,000} = \frac{-336000000 - 60000000}{1655000000} \approx -0.239$$

**Predicted position in frame 2**: $(150 + 0.275, 200 - 0.239) = (150.275, 199.761)$

Note that the predicted position is at **sub-pixel coordinates**, which is why bilinear interpolation is needed.

---

### B.3 Derivation of Bilinear Interpolation

#### Motivation

Optical flow tracking produces sub-pixel displacement estimates. To read image intensity at non-integer coordinates — whether for validating predictions, iterative refinement, or accuracy evaluation — we need an interpolation method. **Bilinear interpolation** achieves this by using a weighted average of the four surrounding pixels.

#### Derivation

Given a point at continuous coordinates $(x, y)$ where $x$ and $y$ are non-integer:

**Define the four neighboring integer pixel positions:**

$$x_0 = \lfloor x \rfloor, \quad x_1 = x_0 + 1$$
$$y_0 = \lfloor y \rfloor, \quad y_1 = y_0 + 1$$

**Define the fractional offsets:**

$$\alpha = x - x_0 \quad (0 \leq \alpha < 1)$$
$$\beta = y - y_0 \quad (0 \leq \beta < 1)$$

**The four surrounding pixel intensities are:**

```
Q(x₀, y₀) ────── Q(x₁, y₀)
    │                  │
    │     (x, y)       │
    │        ●         │
    │                  │
Q(x₀, y₁) ────── Q(x₁, y₁)
```

**Step 1 — Interpolate along x (horizontal):**

$$R_0 = (1 - \alpha) \cdot I(x_0, y_0) + \alpha \cdot I(x_1, y_0) \quad \text{(top row)}$$
$$R_1 = (1 - \alpha) \cdot I(x_0, y_1) + \alpha \cdot I(x_1, y_1) \quad \text{(bottom row)}$$

**Step 2 — Interpolate along y (vertical):**

$$I(x, y) = (1 - \beta) \cdot R_0 + \beta \cdot R_1$$

**Fully expanded form:**

$$\boxed{I(x, y) = (1-\alpha)(1-\beta) \cdot I(x_0, y_0) + \alpha(1-\beta) \cdot I(x_1, y_0) + (1-\alpha)\beta \cdot I(x_0, y_1) + \alpha \beta \cdot I(x_1, y_1)}$$

#### Geometric Interpretation

Each weight is the area of the rectangle **opposite** to the corresponding pixel:

| Pixel | Weight | Opposite Rectangle Area |
|-------|--------|------------------------|
| $I(x_0, y_0)$ | $(1-\alpha)(1-\beta)$ | Distance to bottom-right corner |
| $I(x_1, y_0)$ | $\alpha(1-\beta)$ | Distance to bottom-left corner |
| $I(x_0, y_1)$ | $(1-\alpha)\beta$ | Distance to top-right corner |
| $I(x_1, y_1)$ | $\alpha\beta$ | Distance to top-left corner |

The four weights always sum to 1, ensuring the interpolated value lies within the range of the surrounding pixels.

#### Concrete Example

For a predicted position $(150.275, 199.761)$:

- $x_0 = 150$, $x_1 = 151$, $\alpha = 0.275$
- $y_0 = 199$, $y_1 = 200$, $\beta = 0.761$

Suppose the four pixel intensities are: $I(150,199) = 120$, $I(151,199) = 125$, $I(150,200) = 118$, $I(151,200) = 122$

$$I(150.275, 199.761) = (0.725)(0.239)(120) + (0.275)(0.239)(125) + (0.725)(0.761)(118) + (0.275)(0.761)(122)$$
$$= 20.81 + 8.22 + 65.14 + 25.56 = 119.73$$

This sub-pixel intensity can then be compared against the original feature's intensity to validate the tracking prediction.

---

### B.4 Validation of Tracking with Actual Pixel Locations

#### Methodology

To validate that the Lucas-Kanade tracking equations produce accurate results, we compare **predicted positions** (from optical flow) against **actual positions** (found independently via template matching):

| Step | Method |
|------|--------|
| 1. **Detect features** in frame 1 | Shi-Tomasi corner detector selects strong corner features |
| 2. **Predict positions** in frame 2 | Lucas-Kanade pyramidal optical flow (`cv2.calcOpticalFlowPyrLK`) computes the displacement $(u, v)$ for each feature |
| 3. **Find actual positions** in frame 2 | Template matching (`cv2.matchTemplate` with normalized cross-correlation) searches for each feature's patch from frame 1 in a local search region of frame 2 |
| 4. **Compute error** | Euclidean distance between predicted and actual: $\epsilon = \sqrt{(x_p - x_a)^2 + (y_p - y_a)^2}$ |

#### Interpretation of Results

| Error Range | Interpretation |
|-------------|----------------|
| **< 1 px** | Excellent — sub-pixel accuracy; tracking and template match agree almost perfectly |
| **1 – 3 px** | Good — minor discrepancy, likely due to interpolation differences or slight appearance change |
| **3 – 5 px** | Moderate — some degradation, possibly due to occlusion, lighting change, or deformation |
| **> 5 px** | Poor — tracking failure at this feature, often at textureless regions or motion boundaries |

#### Expected Outcome

For frames with moderate motion and well-textured regions, the Lucas-Kanade tracker typically achieves **sub-pixel to 1–2 pixel accuracy** on corner features. The annotated output images (`tracking_video1.png`, `tracking_video2.png`) show:

- **Green dots**: Original feature positions in frame 1
- **Blue dots**: Predicted positions in frame 2 (from optical flow)
- **Red dots**: Actual positions in frame 2 (from template matching)
- **Arrows**: Motion vectors from original to predicted positions

When blue and red dots overlap closely, it confirms that the theoretical tracking equations produce accurate predictions consistent with the actual pixel locations.

---

## References

1. Lucas, B. D., & Kanade, T. (1981). *An Iterative Image Registration Technique with an Application to Stereo Vision*. IJCAI.
2. Farnebäck, G. (2003). *Two-Frame Motion Estimation Based on Polynomial Expansion*. SCIA.
3. Shi, J., & Tomasi, C. (1994). *Good Features to Track*. CVPR.
4. OpenCV Documentation — [Optical Flow](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)
