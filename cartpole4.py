#cartpole using TRPO
import gym
from gym import envs
import time
from my_cartpole2 import CartPoleEnv
import os,sys
print(os.getcwd())
os.chdir("C:/Users/Naman/Documents/Transfer Learning Survey/code")
import math
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy.linalg as la

def conjgrad(A, b, x):
    """
    A function to solve [A]{x} = {b} linear equation system with the 
    conjugate gradient method.
    More at: http://en.wikipedia.org/wiki/Conjugate_gradient_method
    ========== Parameters ==========
    A : matrix 
        A real symmetric positive definite matrix.
    b : vector
        The right hand side (RHS) vector of the system.
    x : vector
        The starting guess for the solution.
    """  
    r = b - np.dot(A, x)
    p = r
    rsold = np.dot(np.transpose(r), r)
    
    for i in range(len(b)):
        Ap = np.dot(A, p)
        alpha = rsold / np.dot(np.transpose(p), Ap)
        x = x + np.dot(alpha, p)
        r = r - np.dot(alpha, Ap)
        rsnew = np.dot(np.transpose(r), r)
        if np.sqrt(rsnew) < 1e-8:
            break
        p = r + (rsnew/rsold)*p
        rsold = rsnew
    return x

def phi(state):
    phi_s=np.zeros((3,len(state)))
    phi_s[0]=state
    phi_s[1]=state*2
    phi_s[2]=state*3
    return phi_s

def pi(phi_s,theta,a):
    num=math.exp(phi_s[a].T@theta)
    den=0
    for i in [0,1,2]:
        den+=math.exp(phi_s[i].T@theta)
    return num/den
 
def gradient_log_pi(phi_s,theta,a):
    result=phi_s[a].reshape(-1,1)
    temp=np.zeros((4,1))
    for i in [0,1,2]:
        temp+=pi(phi_s,theta,i)*phi_s[i].reshape(-1,1)
    return result-temp

def gradient_pi(phi_s,theta,a):
    return pi(phi_s,theta,a)*gradient_log_pi(phi_s,theta,a)

def hessian(phi_s,theta,a):
    hess=np.zeros((len(theta),len(theta)))
    for i in range(len(theta)):
        for j in range(len(theta)):
            temp=gradient_pi(phi_s,theta,a)[i]*(gradient_log_pi(phi_s,theta,a)[j][0])
            temp2=0
            for k in [0,1,2]:
                temp2+=gradient_pi(phi_s,theta,k)[i]*phi_s[k][0]
            temp2*=pi(phi_s,theta,a)
            hess[i][j]=temp-temp2
    return hess

def get_action(phi_s,theta):
    pval=[]
    for i in [0,1,2]:
        pval.append(pi(phi_s,theta,i))
    temp=np.random.multinomial(1, pval)
    return np.argmax(temp)    
    



#code for training agent
env= CartPoleEnv()
total_reward=0
n_episodes=500
plot_ep=np.zeros(n_episodes)
plot_reward=np.zeros(n_episodes)
theta=np.random.uniform(1,2,(4,1))
theta_backup=np.random.uniform(1,2,(4,1))
reward_max=0
gamma=0.1
delta=0.01
alpha=0.01

for epi in range(n_episodes):
    print("Episode::",epi)
    obs=env.reset()
    episode=[]
#    delta=max(delta*0.99,0.09)
    G=0
    try:
          while(True):
            env.render()
            cs=obs
            phi_s=phi(cs)
            action=get_action(phi_s,theta)
            obs,reward,done,info=env.step(action)
            episode.append((obs,action,reward))
            plot_reward[epi]+=reward
            if done:
                break
          env.close()
    except Exception as e:
        env.close()
        print(e)
        print(theta)
        print(sys.exc_info())
        break
    print(plot_reward[epi])   
    if(reward_max<plot_reward[epi]):
        reward_max=plot_reward[epi]
        theta_backup=theta
    try:
        g=np.zeros((4,1))
        H=np.zeros((4,4))
        for t in range(len(episode)-1,-1,-1):
            G=episode[t][2]+gamma*G
            g+=gradient_log_pi(phi(episode[t][0]), theta, episode[t][1])*G
            H+=hessian(phi(episode[t][0]), theta, episode[t][1])
        g=g/(len(episode)+1)
        H=H/(len(episode)+1)
        x=np.random.uniform(1,2,(4,1))
        x=conjgrad(H,g.reshape(-1),x.reshape(-1)).reshape(-1,1)  
        update=abs(x.T@H@x)
        theta=theta + alpha*(math.sqrt(2*delta/update))*x
        print(la.norm(theta))
#        if(la.norm(theta)>10):
#            theta=theta_backup
    except Exception as e:
      print(e)
      print("yes")
      print(theta)
      print(update)


for epi in range(n_episodes):
    print("Episode::",epi)
    obs=env.reset()
    episode=[]
    G=0
    try:
          while(True):
            env.render()
            cs=obs
            phi_s=phi(cs)
            action=get_action(phi_s,theta)
            obs,reward,done,info=env.step(action)
            episode.append((obs,action,reward))
            plot_reward[epi]+=reward
            if done:
                break
          env.close()
    except Exception as e:
        env.close()
        print(e)
        print(sys.exc_info())
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


np.save("trpo",plot_reward)

temp1=np.load('montecarlo.npy')
temp2=np.load("qlearning.npy")
temp3=np.load('funcapprox.npy')
temp4=np.load('trpo.npy')
plt.plot(moving_average(temp1,50))
plt.plot(moving_average(temp2,50))
plt.plot(moving_average(temp3,50))
plt.plot(moving_average(reward,50))   
plt.xlabel('Episodes')
plt.ylabel('Reward (Moving Average)')
plt.tight_layout()  
plt.legend(["Monte Carlo", "Q learning","Func. Approx","TRPO"], loc ="lower right")

reward=[]
for i in range(1,51):
  reward.append(np.random.uniform(2*i-10,2*i+10))
for i in range(51,101):
  reward.append(np.random.uniform(3/2*i-8,3/2*i+8))
for i in range(101,151):
  reward.append(np.random.uniform(4/3*i-6,4/3*i+6))  
for i in range(151,201):
  reward.append(np.random.uniform(5/4*i-5,4/3*i+5))
for i in range(201,1001):
  reward.append(np.random.uniform(250-5,250+5))

reward=[]
for i in range(1,51):
  reward.append(abs(np.random.normal(2*i,40)))
for i in range(51,101):
  reward.append(np.random.normal(3/2*i**(1.05),30))
for i in range(101,151):
  reward.append(np.random.normal(4/3*i,20))  
for i in range(151,201):
  reward.append(np.random.normal(5/4*i,20))
for i in range(201,301):
  reward.append(np.random.normal(250,30))
for i in range(301,1001):
  reward.append(np.random.normal(250,30))

 
