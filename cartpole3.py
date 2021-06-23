# Q learning (func approx)

import gym
from gym import envs
import time
from my_cartpole import CartPoleEnv
import os
print(os.getcwd())
os.chdir("C:/Users/Naman/Documents/Transfer Learning Survey/code")
import math
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
import random
import pandas as pd

# This function samples action according the e-greedy policy
def get_action(w,obs,epsilon):
    if(random.random()<epsilon):
        return env.action_space.sample()
    else:
       if(w[0].T@obs> w[1].T@obs):
           return 0
       else:
           return 1

# weight vector for linear function 
w=np.random.uniform(0,1,(2,4,1))

#parameters
gamma=0.1
epsilon=1.0
alpha=0.01
n_episodes=1001

#code for training agent
env= CartPoleEnv()
total_reward=0
plot_ep=np.zeros(n_episodes)
plot_reward=np.zeros(n_episodes)

for epi in range(1,n_episodes):
    print("Episode::",epi)
    obs=env.reset()
    episode=[]    
    epsilon*=0.99
    alpha*=0.99
    gamma*=0.98
    plot_ep[epi]=epsilon
    try:
        while(True):
#            env.render()
            obs[2]=1.05*obs[2]
            ca=get_action(w, obs, epsilon) # current action A
# state variable contains 4 varaible of env and 1 variable for action
            cs=obs #current state S
            obs,reward,done,info=env.step(ca)
            total_reward+=reward
            obs[2]=1.5*obs[2]            
            ns=obs #next state S'
            w[ca]+=(alpha*(reward+gamma*max([ns.T@w[0],ns.T@w[1]]) - cs.T@w[ca])*cs).reshape(-1,1) # Q-learning update equation
            if done:
                break
            plot_reward[epi]+=reward
        env.close()
        print("Total reward:",total_reward/epi)
        if(total_reward/epi>200):
         break
    except Exception as e:
        env.close()
        print(e)
        break

#Reward value plot
def total_average(data):
    result=[]
    sum=0
    for i in range(len(data)):
        sum+=data[i]
        result.append(sum/(i+1))
    return np.array(result)

def moving_average(data,window):
    result=[]
    for i in range(1,window-1):
        result.append(np.mean(data[:i]))
    for i in range(window,len(data)):
        result.append(np.mean(data[i-window:i]))        
    return np.array(result)        

    
_ , plot=plt.subplots(1,2,sharex=True)
plot[0].plot(plot_reward)
plot[1].plot(total_average(plot_reward))
plot[1].plot(moving_average(plot_reward,50))
plt.setp(plot[0], ylabel='Reward')
plt.setp(plot[0], xlabel='Episodes')
plt.setp(plot[1], xlabel='Episodes')
plt.tight_layout()

np.save("funcapprox",plot_reward)

#code for testing agent after training 
env=CartPoleEnv()
epsilon=0
for i in range(15):
    obs=env.reset()
    total_reward=0
    for _ in range(300):
        env.render()
        ca=get_action(w, obs, epsilon)
        obs,reward,done,info=env.step(ca)
        total_reward+=reward
        time.sleep(0.02)
        if done:
            break
    print(total_reward)    
    env.close()

temp=[]
for i in range(2001):
    temp.append(abs(math.sin(0.005*i)))
plt.plot(temp)