# Video Finder

A Python-based video analysis tool that combines semantic search, object detection, and object tracking capabilities. The tool provides both GUI and CLI interfaces for searching and tracking objects in video content.

## Features

- **Semantic Search**: Search for specific objects, people, or scenes in videos using natural language queries
- **Object Detection**: Precise object localization using grounding-based detection
- **Object Tracking**: Real-time tracking of detected objects through video sequences
- **Dual Interfaces**: 
  - GUI mode for user-friendly interaction
  - CLI mode for command-line operations
- **Frame Analysis**: Detailed scene description including people, vehicles, and environmental details

## Installation

1. Clone the repository:

bash
git clone [repository-url]
cd video-finder

2. Install the required dependencies:

bash
pip install -r requirements.txt


3. Additional dependencies:
- ZhipuAI API key (for semantic analysis)
- DeepDataSpace API key (for object detection)

## Usage

### GUI Mode

Run the graphical interface:

bash
python video_finder_gui.py


The GUI provides:
- Video input selection
- Text-based search queries
- Real-time object detection visualization
- Object tracking capabilities
- Progress monitoring

### CLI Mode

Run the command-line interface:

bash
python video_finder_cli.py


CLI Navigation:
- `n`: Next frame
- `p`: Previous frame
- `t`: Start tracking
- `q`: Quit current view

## File Structure

- `video_finder_gui.py`: Graphical user interface implementation
- `video_finder_cli.py`: Command-line interface implementation
- `requirements.txt`: Required Python packages

## Dependencies

- requests: HTTP library for API calls
- opencv-python & opencv-contrib-python: Video processing and object tracking
- numpy: Numerical computations
- Pillow: Image processing
- IPython: Interactive computing
- scikit-learn: Machine learning utilities
- matplotlib: Visualization
- typing: Type hints

## API Requirements

The application requires two API keys:
1. ZhipuAI API key for semantic analysis
2. DeepDataSpace API key for object detection

Please ensure you have valid API keys before running the application.

## License

[Your chosen license]

## Contributing

[Your contribution guidelines]
## Acknowledgements

This project utilizes the following APIs:

- [ZhipuAI API](https://open.bigmodel.cn/) - For semantic analysis and natural language processing capabilities
- [DeepDataSpace (DDS) Cloud API Platform](https://deepdataspace.com/) - For object detection and computer vision functionality

We thank both platforms for providing the API services that power core features of this application.

