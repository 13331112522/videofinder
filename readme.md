# Video Finder Tool

## Overview

The Video Finder Tool is designed to analyze video frames and provide detailed descriptions of scenes, objects, and events within the video. It utilizes advanced AI models to generate frame descriptions and allows users to perform semantic searches on the generated data.

## Features

- **Frame Analysis**: Automatically analyzes video frames and generates detailed descriptions in JSON format.
- **Semantic Search**: Allows users to search through frame descriptions using natural language queries.
- **Grounded Segmentation**: Provides bounding boxes and labels for detected objects in the video frames.
- **User Interface**: A graphical user interface (GUI) for easier interaction and visualization of results.
- **Multi-language Support**: The GUI is available in both English and Chinese.

## Requirements

Make sure to install the required packages listed in `requirements.txt`:

bash
pip install -r requirements.txt

## Environment Variables

Before running the application, set up the following environment variables in a `.env` file:


ZHIPU_API_KEY=your_zhipu_api_key

DEEPDATASPACE_API_TOKEN=your_deepdataspace_api_token

VIDEO_PATH=path_to_your_video_file.mp4

FRAME_INTERVAL=120 # Interval for frame analysis

## Usage

### Command Line Interface (CLI)

To run the video analysis using the command line interface, execute the following command:


bash
python video_finder_cli.py

### Graphical User Interface (GUI)

To run the graphical user interface, execute the following command:

bash
python video_finder_gui.py

### GUI in Chinese

For a Chinese version of the GUI, execute:


bash
python video_finder_gui_zh.py

## How It Works

1. **Video Input**: The tool takes a video file as input, specified by the `VIDEO_PATH` environment variable.
2. **Frame Processing**: The video is processed frame by frame, with descriptions generated based on the specified `FRAME_INTERVAL`.
3. **Description Storage**: Frame descriptions are saved in a text file, and a backup is created for recovery.
4. **Semantic Indexing**: A semantic index is created from the frame descriptions to facilitate efficient searching.
5. **User Interaction**: Users can input queries to search for specific frames, and the results are displayed with bounding boxes around detected objects.

## Troubleshooting

If the analysis stops unexpectedly (e.g., at frame 420), consider the following:

- **Video File Integrity**: Ensure that the video file is not corrupted and can be played without issues.
- **Frame Processing Errors**: Check the console output for any error messages related to frame processing or API responses.
- **API Limits**: Verify that you are not exceeding any API usage limits set by the ZhipuAI or DeepDataSpace services.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the developers of the libraries used in this project, including OpenCV, NumPy, and others.
- Special thanks to the ZhipuAI and DeepDataSpace teams for their powerful APIs.

