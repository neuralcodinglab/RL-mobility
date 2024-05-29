# RL-mobility
Code repository for experiments and analyses described in the thesis chapter "Towards a task-based computational benchmark for the
evaluation of prosthetic vision". (see [this doctoral thesis](https://www.publicatie-online.nl/uploaded/flipbook/174411-Jaap-de-Ruyter/36/) by Jaap de Ruyter van Steveninck). 

This chapter is in preparation for submission as:

de Ruyter van Steveninck, J., Danen, S., Küçükogl u, B., Güçlü,
U., van Wezel, R., & van Gerven, M. (2024). Deep reinforcement learning for evaluation and optimization of prosthetic vision.


## About the project
Neuroprosthetic visual implants are a promising technology to restore some form of visual perception for persons with blindness. As research is actively investigating different prototype designs, the field is looking for easy, cost-effective and non-invasive simulation paradigms to speed up the experimental cycle of hypothesis
testing. In this study, we propose a deep-reinforcement-learning-based computational framework for that purpose. 

#### Experimental setup
A virtual implant user (powered by a PyTorch double-Q RL network) performs a mobility task in a 3D visual environment (in the Unity game engine). The experiments are modeled
modeled after a previously published simulation study with sighted human participants ([de Ruyter van Steveninck *et al.*, 2022](https://doi.org/10.1167/jov.22.2.1)). The agent moves through a virtual hallway, taking actions based on simulated prosthetic visual input. Several experimental parameters can be freely adjusted, such as the complexity of the hallway, the resolution of the phosphene vision, the image processing, etc. 

#### Contributors
- Jaap de Ruyter
- Sam Danen
- Umut Güçlü

## Usage

### Start a pre-build Unity environment
To run the pre-build Unity environment:
1. Start the application: `Unity/Build/Windows/RL_Hallway.exe` or `Unity/Build/Linux/RL-Hallway.x86_64` depending on your operating system. 
2. Specify environment variables, IP address, etc. and press 'start'. 


### Interacting with the environment from Python
Basic example (see Demo directory for more):
```python
import pyClient
import matplotlib.pyplot as plt 

# Connect to the environment
environment = pyClient.Environment(ip = "127.0.0.1", port = 13000)

# Reset the environment 
_, _, _ = environment.reset(kind=0) # (0: plain, 1: complex)

# Move the agent in the environment
end, reward, state = environment.step(action=0) # (0: forward, 1: left, 2: right)

# Visualize the current state observation
plt.imshow(state['colors'] )
``` 

### RL training pipeline 
The pipeline can be trained using a YAML configuration file or a CSV file that specifies multiple training instances. The following command initiates the training with the demo configuration:

`$ python training.py -c _config.yaml`

### Adjusting the Unity environment
To use and adapt the source code for the Unity environment, simply:
1. Install Unity (we used v2019.3.7f1, but other versions should work as well)
2. Launch Unity Hub and press 'ADD' to add the source code (directory: `Unity/indoor-mobility`) as a new project. 
3. (optional) Build the application to enable launching it outside Unity.


## Reproduce the experiments

### Data analysis
To reproduce the analysis for the thesis chapter:
1. Download the experimental results from [here](https://surfdrive.surf.nl/files/index.php/s/m7575dnzNT7ouTe) and save them to the output directory: `Python/Experiments/Out/`.  
2. Run the jupyter notebook for the data analysis: `Python/Experiments/data_analysis_phd_thesis_dec2023.ipynb`.

### Experiments with pre-build Unity application
To run experiments yourself without the need of installing Unity:
1. Run the pre-built Unity application hosting the virtual environment. 
   - Start the application: `Unity/Build/Windows/RL_Hallway.exe` or `Unity/Build/Linux/RL-Hallway.x86_64` depending on your operating system. 
   - Specify environment variables, IP address, etc. and press 'start'. 
2. Choose a training configuration:
   - Either adjust the _config.yaml according to your own preference
   - Or use the train specifications of the experiments described in the chapter, which are located in `Experiments/_train_specs/`
3. Run the python reinforcement learning client:
   - `$ python Python/training.py -c _config.yaml`
   - or `$ python Python/training.py -s Experiments/_train_specs/<csv train specification>`

## Demo
For the usage and implementation, refer to the demo notebook `PythonScrips/demoUsage.ipynb`.

To run a demo navigation run through the environment:
1. Start the environment server by running the Unity server application which can be found in the Unity Build folder. In the GUI, press 'start' to accept the default environment parameters.
2. run `python Python/demoNavigation.py`

Controls:
- 1: normal vision
- 2: low resolution prosthethic vision
- 3: high resolution prosthetic vision
- w: forward
- a: left
- d: right
- r: reset
- q: quit
