# Imitation-Learning---Lunar-Lander
Using imitation learning to train a Deep Q network that can control the lunar vehicle to softly land on the lunar surface.


![out](https://user-images.githubusercontent.com/51358612/167246686-9dc16dad-9e3e-400c-b54b-1b112ba26b0b.gif)

## Training Sequence
Iterative approach is used to train the model.
For each iteration there are two phases. 1. Expert phase  2. Trial phase
1. Expert phase
      I run the simulation using the Expert as controller (here a heuristic function is used) that is sure to succeed in landing the lunar vehicle on         the lunar surface safely without crashing and making sure that it lands on exact marked spot.
      While running it I store the state,action,next_state and Reward for each step.
      At the end I will train the Deep Q network with the above stored data.
2. Trial phase
      I run the simulation using the Deep Q network as controller, such that Deep Q network is used to predict the next state given present state.
      
![train](https://user-images.githubusercontent.com/51358612/167430113-d6866cad-7b4f-4ff1-bde2-05e9e029af66.gif)

## Testing Sequence
In the Testing Sequence I just run the simulation a given number of times using the trained model.

![test](https://user-images.githubusercontent.com/51358612/167430220-e5bd8eb5-f0dc-4a92-ad5d-7480946a4c90.gif)
