Installation

To get started, follow these steps:

1. Install Required Dependencies
   pip install -r requirements.txt

2. Install Tkinter
  sudo apt-get install python3-tk

Project Structure

The project requires the following inputs:

    Video File: The video file can be in .mp4, .avi, or .mov format.
    Image Directory: A directory containing subdirectories of images. Each subdirectory should be named after the person it contains images of, as this name will be used as the label for face recognition. The images within these subdirectories can be in .png, .jpg, or .jpeg format.

Example Directory Structure

Images (Directory name can be anything)
├── Donald Trump (Subdirectory name used as label)
│   ├── img1.png (Image names can be anything)
│   ├── img2.png
│   ├── donald.jpeg
│   └── etc.jpg
├── Person Two
│   ├── img1.png
│   ├── img2.png
│   └── img.png
└── Person Three
    └── ...


CPU Load and Accuracy

By default, the script processes every 10th frame of the video. This setting strikes a balance between performance and accuracy. If you require more precise results and are willing to trade off processing time and computational power, you can adjust the frame_skip variable. Lowering this value will increase accuracy but also increase the load on your CPU and extend processing time.

To adjust the frame processing interval, search for the frame_skip variable in the code (approximately line 90) and set it to your desired value. For example:
frame_skip = 10  # Change this value as needed

