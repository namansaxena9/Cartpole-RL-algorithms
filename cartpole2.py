#Q learning and SARSA (tabular)
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

def pos_bin(x):
    bins=[-2.4,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.4]
    for i in range(len(bins)):
        if(x<bins[i]):
            return i
    return len(bins)

def vel_bin(x):
    bins=[-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1]
    for i in range(len(bins)):
        if(x<bins[i]):
            return i
    return len(bins)

def angle_bin(x):
    bins=[-60,-40,-30,-20,-10,-5,-2,0,2,5,10,20,30,40,60]
    for i in range(len(bins)):
        if(x<bins[i]):
            return i
    return len(bins)

def angle_vel_bin(x):
    bins=[-3,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,3]
    for i in range(len(bins)):
        if(x<bins[i]):
            return i
    return len(bins)


def state(obs):
    obs[0]=pos_bin(obs[0])
     
    obs[1]=vel_bin(obs[1])
    
    obs[2]=obs[2]*360/(2*math.pi)
    obs[2]=angle_bin(math.ceil(obs[2]))
  
    obs[3]=angle_vel_bin(obs[3])
    
    return obs.astype(int)
    
def get_action(state_action,cs,epsilon):
    if(random.random()<epsilon):
        return env.action_space.sample()
    else:
       if(state_action[cs[0]][cs[1]][cs[2]][cs[3]][0]>state_action[cs[0]][cs[1]][cs[2]][cs[3]][1]):
           return 0
       else:
           return 1


def build_state(features):
    ''' Build state by concatenating features (bins) into 4 digit int. '''
    return list(map(lambda feature: int(feature), features))

def create_state(obs):
    ''' Create state variable from observation.

    Args:
        obs: Observation list with format [horizontal position, velocity,
             angle of pole, angular velocity].
    Returns:
        state: State tuple
    '''
    cart_position_bins = pd.cut([-2.4, 2.4], bins=10, retbins=True)[1][1:-1]
    pole_angle_bins = pd.cut([-1.1, 1.1], bins=10, retbins=True)[1][1:-1]
    cart_velocity_bins = pd.cut([-1, 1], bins=10, retbins=True)[1][1:-1]
    angle_rate_bins = pd.cut([-3.5, 3.5], bins=10, retbins=True)[1][1:-1]
    state = build_state([np.digitize(x=[obs[0]], bins=cart_position_bins)[0],
                             np.digitize(x=[obs[1]], bins=pole_angle_bins)[0],
                             np.digitize(x=[obs[2]], bins=cart_velocity_bins)[0],
                             np.digitize(x=[obs[3]], bins=angle_rate_bins)[0]])
    return state


#state_action=np.random.uniform(0,1,(12,12,16,12,2))
state_action=np.random.uniform(0,1,(10,10,10,10,2))
env=CartPoleEnv()
gamma=0.9
epsilon=1
alpha=0.9
n_episodes=1001
env=CartPoleEnv()
plot_reward=np.zeros(n_episodes)
total_reward=0
for epi in range(1,n_episodes):
    print("Episode::",epi)
    obs=env.reset()
    episode=[]    
    epsilon*=0.99
    try:
        while(True):
#            env.render()
            cs=create_state(obs)
            ca=get_action(state_action, cs, epsilon)
            obs,reward,done,info=env.step(ca)
            total_reward+=reward
            plot_reward[epi]+=reward
            ns=create_state(obs)
            na=get_action(state_action,ns,epsilon)
#            future=state_action[ns[0]][ns[1]][ns[2]][ns[3]][na]
            future=max(state_action[ns[0]][ns[1]][ns[2]][ns[3]])
            state_action[cs[0]][cs[1]][cs[2]][cs[3]][ca]+=alpha*(reward+gamma*future -state_action[cs[0]][cs[1]][cs[2]][cs[3]][ca])
            if done:
                break
        env.close()
        print("Total reward:",total_reward/epi)
        if(total_reward/epi>200):
         break
    except Exception as e:
        env.close()
        print(e)
        break

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
plot[1].plot(moving_average(plot_reward,25))
plt.setp(plot[0], ylabel='Reward')
plt.setp(plot[0], xlabel='Episodes')
plt.setp(plot[1], xlabel='Episodes')
plt.tight_layout()
np.save("qlearning",plot_reward)

env=CartPoleEnv()
for i in range(15):
    obs=env.reset()
    total_reward=0
    for _ in range(100):
        env.render()
        cs=create_state(obs)        
        action=get_action(state_action, cs, epsilon)
        obs,reward,done,info=env.step(action)
        total_reward+=reward
        if done:
            break
    print(total_reward)    
    env.close()
           
