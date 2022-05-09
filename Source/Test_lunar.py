import gym
import random
import time
import numpy as np
import keras
from keras.layers import Dropout
from keras.utils import to_categorical
from keras_visualizer import visualizer
from termcolor import cprint,colored

import sys

n = len(sys.argv)
if n !=2:
	cprint("Command Test_Lunar.py <NUM_EPISODES>","red")
	exit()
Num_episodes = 0
try:
	Num_episodes = int(sys.argv[1])
except:
	cprint("Error invalid argument","red")
	exit()


def create_model():
	model = keras.Sequential(name="my_sequential")
	model.add(keras.layers.Dense(20,input_shape=(8,), activation="relu", name="layer1"))
	model.add(Dropout(0.2))
	model.add(keras.layers.Dense(16, activation="relu", name="layer2"))
	model.add(Dropout(0.2))
	model.add(keras.layers.Dense(16, activation="relu", name="layer3"))
	model.add(Dropout(0.2))
	model.add(keras.layers.Dense(4,activation ="sigmoid", name="layer4"))
	opt = keras.optimizers.Adam(learning_rate=0.01)
	model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
	model.build()
	
	return model
	
actions = {0:"nothing",1:"right",2:"main",3:"left"}
def run_game_predict(env,model,render=False,episode=1):
	Replay = []
	Reward_total = 0
	Done = False
	for i in range(episode):
		state = env.reset()
		if render:
			env.render()
			#time.sleep(2)
		done = False
		n = 0
		print("\n"*30)
		
		while n<1000 and not done:
			#reshape state to match the input for model
			p_state = np.array(state)
			p_state = np.reshape(p_state,[1,8])
			#predict using the model
			predict_actions = list(model.predict(p_state))
			predict_actions = np.reshape(predict_actions,[4])
			predict_actions= predict_actions.tolist()
			#get the action with max prediction value
			action = predict_actions.index(max(predict_actions))
			  
			next_state,reward,done,info = env.step(action)
			Replay.append((state,action,next_state,reward))
			Reward_total+=reward
			state = next_state

			text = str(n)+"---"+str(actions[action])
			cprint(text,"yellow","on_blue")
			print("\n")
			if reward == 100:
				Done = True
			if render:
				env.render()
			if done:
		        	#env.close()
		        	break
            
			n+=1
	#env.close()
	return Replay,Reward_total,Done
	
env = gym.make("LunarLander-v2")
try:
	model = keras.models.load_model('Lunar_model')
	model.load_weights('my_model_weights.h5')
except:
	cprint("Error loading the Model ","red")
	cprint("Run the Train_Lunar.py first to generate the Model and then run Test_Lunar.py","red")
	exit()


for i in range(Num_episodes):
	cprint("       Running Model     ","white","on_yellow")
	print("\n"*10)
	Replay,Reward_total ,Done=  run_game_predict(env,model,render=True)
	print("\n" *30)
	cprint("      Trained Model       ","white","on_yellow")
	text = "     Episode : " + str(i)
	text = colored(text,"yellow")
	print(text)
	text = "Total reward = " + str(Reward_total)
	text = colored(text,"green")
	print(text)
	if Done:
		cprint("     Landed safely       ","blue")
	else:
		cprint("     Crash landed        ","red")
	print("\n"*5)
	time.sleep(2)
