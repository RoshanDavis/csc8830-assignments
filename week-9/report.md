# Uncalibrated Stereo Vision: Distance Estimation Report

This report outlines the step-by-step matrix calculations and transformations used to estimate the distance of an object from two uncalibrated stereo images.

## 1. Intrinsic Camera Matrix ($K$)
Because the cameras are mathematically uncalibrated, we start by constructing an estimated intrinsic matrix ($K$). The focal length $f$ is approximated as the image width ($2268$ px), and the principal point ($c_x$, $c_y$) is assumed to be at the image center. 

For the tested images with width $2268$ and height $4032$, the matrix is:

$$
K = \begin{bmatrix}
f & 0 & c_x \\
0 & f & c_y \\
0 & 0 & 1
\end{bmatrix}
= \begin{bmatrix}
2268 & 0 & 1134 \\
0 & 2268 & 2016 \\
0 & 0 & 1
\end{bmatrix}
$$

## 2. Feature Matching and The Fundamental Matrix ($F$)
Using SIFT (Scale-Invariant Feature Transform), keypoints are detected in both images and matched. To map a point in the left image ($x_L$) to an epipolar line in the right image ($l_R$), we compute the **Fundamental Matrix ($F$)**.

This is done using the 8-point algorithm wrapped in a RANSAC scheme to reject outliers. The resulting matrix $F$ defines the epipolar constraint $x_R^T F x_L = 0$:

$$
F = \begin{bmatrix}
 5.089 \times 10^{-9} & -4.577 \times 10^{-7} &  1.219 \times 10^{-3} \\
 3.324 \times 10^{-7} &  3.925 \times 10^{-8} & -1.575 \times 10^{-2} \\
-1.085 \times 10^{-3} &  1.570 \times 10^{-2} &  1.000
\end{bmatrix}
$$

## 3. The Essential Matrix ($E$)
To strip away the camera internals (pixel scaling/offsets) and focus purely on the 3D geometry (Rotation and Translation) between the two cameras, we compute the **Essential Matrix ($E$)** using the known (or estimated) camera matrix $K$:

$$
E = K^T F K
$$

To enforce the geometric constraint that the Essential Matrix must be of Rank 2 with identical non-zero singular values, Singular Value Decomposition (SVD) is used on $E$, forcing the minimum singular value to $0$. The recovered matrix is:

$$
E = \begin{bmatrix}
 0.000727 & -0.068054 &  0.020418 \\
 0.048935 &  0.016045 & -0.998441 \\
-0.026241 &  0.997227 &  0.016198
\end{bmatrix}
$$

## 4. Recovering Rotation ($R$) and Translation ($\hat{t}$)
The Essential Matrix encapsulates the relative pose (Rotation $R$ and Translation direction $\hat{t}$). It is decomposed via SVD. Out of 4 possible mathematical solutions (since translation can be positive/negative and the camera can be in front/behind), `cv2.recoverPose` tests the triangulated points to find the physically valid solution where the points lie in front of both cameras.

**Rotation Matrix ($R$):**
$$
R = \begin{bmatrix}
 0.999811 &  0.005057 & -0.018747 \\
-0.004760 &  0.999863 &  0.015835 \\
 0.018824 & -0.015742 &  0.999698
\end{bmatrix}
$$
*(Note how the diagonal values are very close to 1, indicating a small angular rotation between the left and right viewpoints.)*

**Translation Vector (Normalized $\hat{t}$):**
$$
\hat{t} = \begin{bmatrix} 0.997472 \\ 0.021497 \\ 0.067725 \end{bmatrix}
$$
Because this is an *uncalibrated* pair, $\hat{t}$ is merely a unit direction vector ($||\hat{t}|| = 1$). It lacks real-world scale.

## 5. Triangulation and Distance Scaling
With $K$, $R$, and $\hat{t}$ known, camera projection matrices $P_1$ and $P_2$ are created:
*   $P_1 = K [I | \mathbf{0}]$
*   $P_2 = K [R | \hat{t}]$

The matching 2D points are triangulated via Direct Linear Transformation (DLT) into homogeneous 3D coordinates ($X, Y, Z_{au}$) in "Arbitrary Units" (au).

**Scaling to Meters:**
To convert the arbitrary depth ($Z_{au}$) into to a real-world metric depth ($Z_{mm}$), we apply the physical baseline between the two captures. Based on our adjusted baseline setup:

$$
\text{Scale Factor} = \frac{\text{Baseline}_{mm}}{||\hat{t}||} = \frac{28.16}{1.0} = 28.16
$$

The distance to the object is taken as the median $Z$ value of the triangulated strictly-positive depth points, multiplied by the scale factor:

$$
\text{Distance}_{mm} = \text{Median}(Z_{au}) \times \text{Scale Factor}
$$

With the arbitrary units outputting roughly $142.06$ au, the final real-world estimated depth comes out to:
$$
\text{Distance}_{mm} \approx 142.06 \times 28.16 \approx 4000.41 \text{ mm} \approx 4.0 \text{ meters}
$$