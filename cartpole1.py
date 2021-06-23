#Monte Carlo method 
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

def pos_bin(x):
    bins=[-2.4,-2.0,-1.6,-1.2,-0.8,-0.4,0,0.4,0.8,1.2,1.6,2.0,2.4]
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
    bins=[-18,-14,-10,-8,-6,-4,-2,0,2,4,6,8,10,14,18]
    for i in range(len(bins)):
        if(x<bins[i]):
            return i
    return len(bins)

def angle_vel_bin(x):
    bins=[-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1]
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
    
   

state_action=np.zeros((14,12,16,12,2))
state_action_n=np.zeros((14,12,16,12,2))
action_table=np.random.uniform(0,1,(14,12,16,12))

env=CartPoleEnv()
gamma=0.1
epsilon=0.99
n_episodes=1001
env=CartPoleEnv()
plot_reward=np.zeros(n_episodes)
for epi in range(n_episodes):
    print("Episode::",epi)
    obs=env.reset()
    episode=[]
    epsilon*=0.99
    G=0
    try:
          while(True):
#            env.render()
            cs=state(obs)
            action=np.random.binomial(1,action_table[cs[0]][cs[1]][cs[2]][cs[3]],1)[0]
            obs,reward,done,info=env.step(action)
            episode.append((state(obs),action,reward))
            plot_reward[epi]+=reward
            if done:
                break
          env.close()
    except Exception as e:
        env.close()
        print(e)
        break
    for t in range(len(episode)-1,-1,-1):
        G=episode[t][2]+gamma*G
        cs=episode[t][0]
        action=episode[t][1]
        state_action_n[cs[0]][cs[1]][cs[2]][cs[3]][action]+=1
        state_action[cs[0]][cs[1]][cs[2]][cs[3]][action]+=(G-state_action[cs[0]][cs[1]][cs[2]][cs[3]][action])/state_action_n[cs[0]][cs[1]][cs[2]][cs[3]][action]
        a_star=1
        if(state_action[cs[0]][cs[1]][cs[2]][cs[3]][0]>state_action[cs[0]][cs[1]][cs[2]][cs[3]][1]):
            a_star=0
        if(a_star==1):
            action_table[cs[0]][cs[1]][cs[2]][cs[3]]=1-epsilon/2
        else:
            action_table[cs[0]][cs[1]][cs[2]][cs[3]]=epsilon/2

def totol_average(data):
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


plt.plot(plot_reward)
plt.plot(totol_average(plot_reward))
plt.plot(moving_average(plot_reward,25))
plt.xlabel('Episodes')
plt.ylabel('reward')
plt.tight_layout()    

np.save("montecarlo",plot_reward)

env=CartPoleEnv()
for i in range(10):
    obs=env.reset()
    for _ in range(100):
        env.render()        
        action=np.random.binomial(1,action_table[cs[0]][cs[1]][cs[2]][cs[3]],1)[0]
        obs,reward,done,info=env.step(action)
        print(obs)
        time.sleep(0.02)
        if done:
            break
    env.close()
            


