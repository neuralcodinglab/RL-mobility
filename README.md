# RL-mobility
phosphene-based mobility in virtual envrironments


## About the project
This module was developed to evaluate and optimize image processing for prosthetic vision using reinforcement learning. Many credits for the code go to Sam Danen.

#### Todo:
- Integrate code and replicate experiments by Sam Danen
- Image processing: frame stacking, limited depth rendering
- Task-based optimization experiment: obstacle avoidance with reward cues of different gradient intensity
- Build outdoor environment


## Getting Started

### Prerequisites

Demo code:
- cv2 (pip install opencv-python) 
- socket (pip install socket) 

RL-optimization:
- torch (see: https://pytorch.org/)
- skimage (pip install scikit-image)
- pandas (pip install pandas)

Visualization:
- matplotlib
- seaborn

Unity project:
- Unity 2019.3.7f1 (other versions should work as well)

### Usage

Step 1: 
Start the environment server by running the Unity server application which can be found in the Unity Build folder. In the GUI, press 'start' to accept the default environment parameters.

Step 2: 
For a demonstration of navigation in the environment run:

  ```sh
  python PythonScrips/demoNavigation.py
  ```
Controls:
- 1: normal vision
- 2: low resolution prosthethic vision
- 3: high resolution prosthetic vision
- w: forward
- a: left
- d: right
- r: reset
- q: quit

Step 3: 
For the usage and implementation, refer to PythonScrips/demoUsage.py
