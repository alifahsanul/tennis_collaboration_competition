# Deep Reinforcement Learning Project 3
# Unity Tennis
![alt text](https://raw.githubusercontent.com/alifahsanul/banana_navigation/master/image/environment.PNG)
## Project Details
* This project is the third mandatory project in Udacity Deep Reinforcement Learning Nanodegree.
* The goal is to train an agent to play a tennis game where the agent has to collect as many yellow banana as possible and avoiding blue banana. The rewards are as follows:
  * `+1` for interaction with yellow banana
  * `-1` for interaction with blue banana
  * `0` for else
* There are 4 possible actions:
  * `0` move forward
  * `1` move backward
  * `2` turn left
  * `3` turn right
* The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction
### Task requirements
* Code must be written in Python 3
* PyTorch must be used
* Agent must be able to get average score of 13 from 100 consecutive episodes

## Getting Started
1. Clone the DRLND Repository https://github.com/udacity/deep-reinforcement-learning/#dependencies
2. Download the Unity Environment
	* For this project, you will not need to install Unity - this is because Udacity have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:
		* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
		* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
		* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
		* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
3. Place the Unity Environment file from step 2 in the p1_navigation/ folder in the DRLND GitHub repository, and unzip (or decompress) the file. p1_navigation/ folder is from DRLND (step 1).

## Instruction
1. Open Navigation.ipynb to start training the agent.
2. Open Visualizing Agent.ipynb to watch agent plays.
