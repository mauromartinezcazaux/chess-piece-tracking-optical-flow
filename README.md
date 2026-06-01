# Chess Piece Detection and Tracking using Template Matching and Optical Flow

## Overview

This project implements a computer vision system capable of detecting and tracking chess pieces on a chessboard using template matching and optical flow techniques.

The system identifies pieces from video frames, tracks their movements throughout the game and updates the board state automatically. The project combines traditional computer vision methods with motion estimation techniques to achieve robust tracking under real-world conditions.

---

## Objectives

* Detect chess pieces on a chessboard.
* Track piece movements across consecutive frames.
* Maintain an updated representation of the board state.
* Handle piece movement, captures and position updates.
* Evaluate the performance of different computer vision techniques.

---

## Technologies

* Python
* OpenCV
* NumPy
* Optical Flow (Farnebäck)
* Template Matching
* Feature Detection

---

## Methodology

The system follows the following pipeline:

1. Chessboard detection and calibration.
2. Piece identification using template matching.
3. Motion estimation using optical flow.
4. Board state update after each detected move.
5. Validation of detected movements.

The combination of template matching and optical flow improves robustness compared to using either technique independently.

---

## Repository Structure

```text
dataset/        -> Input videos and test data
src/            -> Source code
test/           -> Validation scripts
results/        -> Generated outputs
docs/           -> Project report and documentation
```

---

## Results

The system successfully detects and tracks chess pieces during gameplay, maintaining the board state throughout the match.

Key achievements:

* Automatic piece detection.
* Continuous tracking across video frames.
* Move recognition and board update.
* Robust performance under moderate camera movement and lighting variations.

---

## Example Output

The output consists of an annotated video showing:

* Detected chess pieces.
* Piece trajectories.
* Updated board state after each move.

Example:

```text
White Pawn: e2 → e4
Black Knight: g8 → f6
White Bishop: f1 → c4
```

---

## Skills Demonstrated

* Computer Vision
* Image Processing
* Motion Tracking
* Algorithm Design
* Python Development
* Data Analysis

---

## Author

**Mauro Martínez Cazaux**

Data Engineering Student – Universidad Politécnica de Cartagena (UPCT)
