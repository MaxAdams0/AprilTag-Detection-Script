# Apriltag-Detection-Script
~~Apriltag detection script for use in [6695](https://github.com/AlphaKnights)'s 2023 FRC Robot.~~ <br>
Alpha Knighs will be using [PhotonVision](https://docs.photonvision.org/en/latest/) after recognizing its benefits, this is now a personal project for learning visual recognition.

# Usage
For CLI-only OSs, use `main-CLI.py`, and for OSs with GUIs, use `debug.py`<br>

# Requirements
* opencv-python
* pupil-apriltags

## Install using
(Windows)
```
py -m pip install <package_name>
```
(Linux)
```
sudo apt install <package_name>
```

## Tags
[AprilRobotics/apriltag-imgs](https://github.com/AprilRobotics/apriltag-imgs)

# Recources
* [Apriltag Specifics](https://optitag.io/blogs/news/designing-your-perfect-apriltag)
* [pupil-apriltag Documentation](https://pupil-apriltags.readthedocs.io/en/stable/api.html)
* [Major Programming Aid - Kazuhito00](https://github.com/Kazuhito00/AprilTag-Detection-Python-Sample)

# Feature Goals; Highest priority at top
* Support Multithreading to run two cameras at once
* Improve Detection
* Hold individual tag location (if past # tags's centers were in the same area ±# pixels)
* Thresholding (if beneficial)

# Liscence
Apriltag-Detection-Script is licensed under the [MIT License](https://github.com/MaxAdams0/Apriltag-Detection-Script/blob/main/LICENSE)
