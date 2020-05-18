| WARNING: This code is designed for use on a specific UR5 robotic arm, if using the physical robot environments, workframes defined in ur5_with_tactip.py will need to be altered before use! (for both cont and disc robot envs)... It is reccomended that you use the robot-jogger tool provided in the required [Common Robot Interface](https://github.com/jlloyd237/cri "Common Robot Interface") to find the neccesary coordinates for your setup, details are given in the guide directory. |
| --- |


# Deep Reinforcement Learning for Tactile Robotics: Learning to Type on a Braille Keyboard

![](figures/all_tasks_short.gif)

### Structure ###
    ├── braille_rl
        ├── algos                    
            ├── cont_sac_algo 
                ├── train_cont_model.py
                ├── sac.py 
                ├── core.py
                ├── resume_training.py
                └── evaluate_model.py  
            ├── disc_sac_algo
                └── ...  
            ├── dd_dqn_algo  
                └── ...
            ├── td3_algo
                └── ...
            ├── saved_modles
            ├── image_utils.py
            ├── rl_utils.py
            └── plot_progress.py
        ├── envs   
            ├── robot
                ├── discrete
                └── continuous
            └── sim
                ├── discrete
                └── continuous
    ├── data  
    ├── guide
    └── CAD
   
### Contents ###

* guide: Details how to use this repository as a starting point for re-running experiments on a physical robot.

* CAD: This directory contains the STL files needed to 3D print the Cherry MX braille keycaps. SLDPRT Files are also included incase minor adjustments are required.

* data: This directory contains tactile data, collected by exhaustively sampling the braille keyboard with a TacTip sensor. This data is used to create the simulated environments.

* envs: This directory contains the environments used in this project. Simulated environments are approximations to the real physical environments, created using data collected with a physical robot. 

* algos: This directory contains the reinforcement learning algorithms used to train agents across the provided environments, these are based off the implementations given in [Spinning Up](https://spinningup.openai.com/en/latest/ "Spinning Up").
  * Each algorithm has a train_model script, this is where experiment hyper-parameters are set.
  * The algorithm logic is provided in the algorithm titled script (e.g. sac.py) and network logic is provided in core.py.
  * When the save_freq parameter is set, trained models will be saved in the saved_models directory (Warning, uncompressed replay_buffers are saved which means file sizes will be large).
  * resume_training.py can be used to continue learning from a previously saved model.
  * evaluate_model.py will evaluate the accuracy and speed of a saved model.
  * Shared utility functions are provided in image_utils.py and rl_utils.py.
  
  
### Dependencies:
```
tensorflow 1.15.0
gym
cv2
numpy
cri
vsp
```

### Installation ###

```
# requires python>=3.6
# It is reccomended that you use a virtual environment for this set up

# clone and install the repo (this may take a while)
git clone https://github.com/ac-93/braille_rl.git
cd braille_rl
pip install -e .

# install Spinningup from openAI
git clone https://github.com/openai/spinningup.git
cd spinningup
pip install -e .

# install common robot interface
git clone https://github.com/jlloyd237/cri.git
cd cri
python setup.py install

# install video stream processor
git clone https://github.com/jlloyd237/vsp.git
cd vsp
python setup.py install

# install python3-v4l2capture
git clone https://github.com/atareao/python3-v4l2capture.git
cd python3-v4l2capture
python setup.py install

# test the installation by running a training script in simulation, from the base directory run
python algos/dd_dqn_algo/train_discrete_model.py

```

### Setting up Experiments ###

Check the guide for details on how to setup experiments on a physical robot.
