# Video Object Detection with YOLOv3

This project implements object detection on video files using the YOLOv3 (You Only Look Once) model. It processes video frames, detects objects in each frame, and draws bounding boxes around them with labels and confidence scores. The project is built using Python, TensorFlow, OpenCV, and PIL (Python Imaging Library).

---

## Features

- **Video Object Detection**: Detects objects in video files frame by frame.
- **YOLOv3 Model**: Utilizes a pre-trained YOLOv3 model for accurate and fast object detection.
- **Bounding Boxes and Labels**: Draws bounding boxes around detected objects and labels them with their class names and confidence scores.
- **Customizable Thresholds**: Allows you to adjust the object detection threshold (`obj_thresh`) and non-maximum suppression threshold (`nms_thresh`).
- **Output Video**: Saves the processed video with detected objects to a new file.

---

## Requirements

To run this project, you need the following dependencies:

- Python 3.x
- TensorFlow (for loading and running the YOLOv3 model)
- OpenCV (for video processing)
- PIL (Python Imaging Library) for image manipulation
- NumPy (for numerical operations)

You can install the required packages using `pip`:

```
pip install tensorflow opencv-python pillow numpy
```

---

## How to Use

1. ### Clone the Repository:
```
git clone https://github.com/Ashish-Reddy-T/Videos-Object-Detection.git
cd video-object-detection
```

2. ### Download the YOLOv3 Model:
- Ensure you have the pre-trained YOLOv3 model (`YOLO_model.h5`) in the project directory. You can download it from a reliable source or train your own model.

3. ### Run the Video Detection:
- Execute the script to process a video file:
```
python video_detection.py
```
- The script will process the video frame by frame, detect objects, and save the output to a new video file.

4. ### Customize Input and Output Paths:
- Modify the `video_path` and `output_path` variables in the script to specify the input video file and the desired output file path.

---

## Code Overview

### Key Functions

- `detect_image(image_pil)`:
  - Performs object detection on the input image using the YOLOv3 model.
  - Returns the image with bounding boxes and labels drawn.

- `detect_video(video_path, output_path)`:
  - Processes a video file frame by frame.
  - Detects objects in each frame using the `detect_image` function.
  - Saves the processed frames to a new video file.

### Customization

- #### Object Detection Threshold (`obj_thresh`):
  - Adjust the confidence threshold for object detection. Default is `0.4`.
  - Example: `detect_image(image_pil, obj_thresh=0.5)`
- #### Non-Maximum Suppression Threshold (`nms_thresh`):
  - Adjust the threshold for suppressing overlapping boxes. Default is `0.45`.
  - Example: `detect_image(image_pil, nms_thresh=0.5)`

---

## Example Output

When you run the script, it will process the input video and save a new video file with bounding boxes and labels for detected objects. For example:

- __Person__: 0.92
- __Car__: 0.88
- __Traffic Signals__: 0.85

---

## Troubleshooting

1. #### Video Not Opening:
- Ensure the video file exists and the path is correct.
- Check if OpenCV can read the video file by running a simple OpenCV script.

2. #### Model Not Found:
- Ensure the YOLOv3 model file (`YOLO_model.h5`) is in the correct directory.
- Download or train the model if it is missing.

3. #### Low FPS:
- Reduce the input resolution or use a lighter model for better performance.
- Ensure your system has sufficient resources (CPU/GPU).

---

## Future Improvements

- #### Support for GPU Acceleration:
  - Modify the code to leverage GPU for faster inference.
- #### Multi-Object Tracking:
  - Implement object tracking to maintain IDs for detected objects across frames.
- #### Custom Object Detection:
  - Train the YOLOv3 model on a custom dataset for specific use cases.

---

## Credits

- __YOLOv3 Model__: Joseph Redmon and Ali Farhadi (https://pjreddie.com/darknet/yolo/)
- __OpenCV__: Open Source Computer Vision Library (https://opencv.org/)
- __TensorFlow__: Machine Learning Framework (https://www.tensorflow.org/)

---

## License

This project is licensed under the MIT License.

---

#### Enjoy video object detection! Contributions and feedback are welcome. 😊
  
