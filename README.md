| WARNING: This code is designed for use on a specific UR5 robotic arm, if using the physical robot environments, workframes defined in ur5_with_tactip.py will need to be altered before use! (for both cont and disc robot envs)... It is reccomended that you use the robot-jogger tool provided in the required [Common Robot Interface](https://github.com/jlloyd237/cri "Common Robot Interface") to find the neccesary coordinates for your setup, details are given in the guide directory. |
| --- |


# Deep Reinforcement Learning for Tactile Robotics: Learning to Type on a Braille Keyboard

![](figures/all_tasks_short.gif)

### Structure ###

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
    └── CAD
   
### Contents ###

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

### Setting up experiments on a Physical Robot ###
Please view the guide directory.
