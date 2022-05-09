import keras
from keras.layers import Dropout
from keras.utils import to_categorical
import gym
import random
import time
import numpy as np
import pickle
from termcolor import cprint,colored
import sys
n = len(sys.argv)
if n !=2:
	cprint("Command Train_Lunar.py <NUM_EPISODES>","red")
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
def heuristic(s):
	angle_targ = s[0] * 0.5 + s[2] * 1.0  # angle should point towards center
	if angle_targ > 0.4:
		angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
	if angle_targ < -0.4:
		angle_targ = -0.4
	hover_targ = 0.55 * np.abs(s[0])  # target y should be proportional to horizontal offset
	angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
	hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5
	
	if s[6] or s[7]:  # legs have contact
		angle_todo = 0
		hover_todo = (
		-(s[3]) * 0.5
		)  # override to reduce fall speed, that's all we need after contact
	        
	a = 0
	if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
		a = 2
	elif angle_todo < -0.05:
		a = 3
	elif angle_todo > +0.05:
		a = 1
	return a
	
def run_game(env,render=False,episode=1):
	Replay = []
	for i in range(episode):
		state = env.reset()
		if render:
			env.render()
			time.sleep(2)
		done = False
		n = 0
		while n<1000 and not done:
			action = 0 
			"""
			if n%10 ==0:
				action = random.randrange(env.action_space.n)
				next_state,reward,done,info = env.step(action)
			else:
				action = heuristic(state)
				next_state,reward,done,info = env.step(action)
			"""
			action = heuristic(state)
			next_state,reward,done,info = env.step(action)
			Replay.append((state,action,next_state,reward))
			state = next_state
			if render:
				env.render()
			if done:
				env.close()
			n+=1
	if render:
		env.close()
	return Replay
    
    
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
    
def train_episodes(num_episodes,model,Replay_model,counter,render=False):
	env = gym.make("LunarLander-v2")
	class_weight = {0: 20.0,
                1: 100.0,
                2: 30.0,
                3:100.0}
	for i in range(num_episodes):
	    	print("\n"* 30)
	    	print("     Episode : ",i)
	    	print("\n" * 10)
	    	counter+=1
	    	
	    	#Get values from True sample
	    	cprint("    Taking inputs from Expert     ","white","on_yellow")
	    	print("\n"*10)
	    	Train_Replay = run_game(env,render=render)
	    	states = []
	    	actions = []
	    	for e in Train_Replay:
	    		states.append(list(e[0]))
	    		actions.append(e[1])
	    	
	    	#train the model for current episode

	    	x_train = np.empty((0,env.observation_space.shape[0]))
	    	y_train = np.empty((0))
	    	x_train = np.append(x_train,np.asarray(states),axis=0)
	    	y_train = np.append(y_train,np.asarray(actions),axis=0)
	    	y_train_cat = to_categorical(y_train, num_classes=env.action_space.n)
	    	metric = model.fit(x_train, y_train_cat, class_weight=class_weight,epochs=100, shuffle=True,verbose=0)
	    	
	    	cprint("       Running Model     ","white","on_yellow")
	    	print("\n"*10)
	    	Replay,Reward_total ,Done=  run_game_predict(env,model,render=render)
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
	    	Replay_model  = Replay_model + Replay
	if render:
		env.close()
	return Replay_model,counter


model = create_model()
Replay_model = []

counter = 0
Replay,counter = train_episodes(Num_episodes,model,Replay_model,counter,render = True)
print("\n" * 10)
with open("model_replay.txt","wb" ) as fp:
	pickle.dump(Replay,fp)
model.save("Lunar_model")
model.save_weights('my_model_weights.h5')

