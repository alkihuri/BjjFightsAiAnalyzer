## Project: Video Pose Analysis and Tracking

### Overview

This project is a comprehensive solution for analyzing and tracking human poses in video footage using MediaPipe, a powerful computer vision library developed by Google. The project includes a main script (`main.py`) that orchestrates the processing of input videos, a primary script (`video_processing.py`) that handles the core logic of pose detection, feature extraction, and tracking, and a utility script (`utils.py`) that contains shared functions and constants used across the project.

### Key Features

1. **Pose Detection**: Utilizes MediaPipe's Pose Landmarker to detect human poses in video frames.
2. **Feature Extraction**: Extracts torso HSV histograms and positions to track athletes across frames.
3. **Identity Tracking**: Tracks athletes by comparing the current frame's torso features and positions with previously detected frames.
4. **Output Generation**: Saves the processed video with pose skeletons drawn on each frame and generates JSON and TXT files containing detailed pose data.

### Dependencies

- Python 3.8 or later
- OpenCV (`cv2`)
- MediaPipe (`mediapipe`)
- NumPy
- Tqdm

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/your-repo.git
   cd your-repo
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

To run the project, execute the `main.py` script with the appropriate arguments:

```bash
python main.py --input input_video.mp4 --output output_video.mp4 --json output_data.json --model mediapipe_model.task --sample-rate 10.0 --max-poses 2
```

### Detailed Analysis

#### `video_processing.py`

This script contains the core logic for processing videos, including pose detection, feature extraction, and tracking.

- **Pose Detection**: Uses MediaPipe's Pose Landmarker to detect human poses in each frame of the video.
- **Feature Extraction**: Extracts torso HSV histograms and positions to identify athletes.
- **Identity Tracking**: Tracks athletes by comparing current frame features and positions with previously detected frames.
- **Output Generation**: Saves the processed video with pose skeletons and generates JSON and TXT files containing detailed pose data.

#### `main.py`

This script serves as the entry point for the project, parsing command-line arguments and orchestrating the video processing workflow.

#### `utils.py`

This script contains utility functions and constants used across the project, including shared functions for pose detection and feature extraction.

### README.md

```markdown
# Video Pose Analysis and Tracking

## Overview

This project is a comprehensive solution for analyzing and tracking human poses in video footage using MediaPipe, a powerful computer vision library developed by Google. The project includes a main script (`main.py`) that orchestrates the processing of input videos, a primary script (`video_processing.py`) that handles the core logic of pose detection, feature extraction, and tracking, and a utility script (`utils.py`) that contains shared functions and constants used across the project.

## Key Features

1. **Pose Detection**: Utilizes MediaPipe's Pose Landmarker to detect human poses in video frames.
2. **Feature Extraction**: Extracts torso HSV histograms and positions to track athletes across frames.
3. **Identity Tracking**: Tracks athletes by comparing the current frame's torso features and positions with previously detected frames.
4. **Output Generation**: Saves the processed video with pose skeletons drawn on each frame and generates JSON and TXT files containing detailed pose data.

## Dependencies

- Python 3.8 or later
- OpenCV (`cv2`)
- MediaPipe (`mediapipe`)
- NumPy
- Tqdm

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/your-repo.git
   cd your-repo
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the project, execute the `main.py` script with the appropriate arguments:

```bash
python main.py --input input_video.mp4 --output output_video.mp4 --json output_data.json --model mediapipe_model.task --sample-rate 10.0 --max-poses 2
```

## Detailed Analysis

### `video_processing.py`

This script contains the core logic for processing videos, including pose detection, feature extraction, and tracking.

- **Pose Detection**: Uses MediaPipe's Pose Landmarker to detect human poses in each frame of the video.
- **Feature Extraction**: Extracts torso HSV histograms and positions to identify athletes.
- **Identity Tracking**: Tracks athletes by comparing current frame features and positions with previously detected frames.
- **Output Generation**: Saves the processed video with pose skeletons and generates JSON and TXT files containing detailed pose data.

### `main.py`

This script serves as the entry point for the project, parsing command-line arguments and orchestrating the video processing workflow.

### `utils.py`

This script contains utility functions and constants used across the project, including shared functions for pose detection and feature extraction.
```

This README.md file provides a comprehensive overview of the project, including its key features, dependencies, installation instructions, and usage instructions. It also includes a detailed analysis of the core logic implemented in `video_processing.py`.


run

python3 openmatanalyzer.py --model pose_landmarker_lite.task --source 0
python3 openmatanalyzer.py --model pose_landmarker_lite.task --source 0 \
    --record-dir recordings --record-fps 30 --record-fourcc mp4v \
    --record-max-minutes 20
python3 openmatanalyzer.py --model pose_landmarker_lite.task --source 0 --no-record





python2 main.py \
  --input fight.mp4 \
  --output analyzed_fight.mp4 \
  --json fight_data.json \
  --model pose_landmarker_lite.task \
  --sample-rate 10 \
  --max-poses 2