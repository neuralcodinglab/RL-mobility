## Dataset creation
For the RL-experiments in this repository no dataset was used. Each frame of visual information provided to the agent, is rendered 'on-the-fly' in the Unity game engine while the agent moves through the virtual environment. 

If you want, however, it is easy to store frames that are rendered in Unity for creating a synthetic dataset. This `dataCollection.py` script is provided  as a utility for that purpose. The main advantages of creating a pre-stored synthetic dataset are speed and ease of use. Reading image files will be faster than obtaining the frames through live communication between Python and Unity. Furthermore, having a synthetic dataset allows you to train models on a remote machine that doesn't have Unity installed. 

Note that working with a synthetic dataset for this task is only possible because of the limited action space. For free exploration in virtual environments, the number of camera poses will be unconstrained, and live rendering remains necessary. 

### Usage
1. Adjust environment parameters, save directory, etc. (hardcoded in the file...)
2. Start a virtual environment in Unity (from the build directory).
3. Run the script to generate data. 
4. The generated `_labels.csv` file can be used to read the appropriate image frames based on x and y positions in the environment. 

