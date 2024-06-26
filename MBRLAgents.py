#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning policies
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
By Thomas Moerland
"""
import numpy as np
from queue import PriorityQueue
from MBRLEnvironment import WindyGridworld


class DynaAgent:


    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        # TO DO: Initialize count tables, and reward sum tables.
        self.means = np.zeros((n_states,n_actions))
        self.transition_counts = np.zeros((n_states,n_actions,n_states))
        self.reward_sums = np.zeros((n_states,n_actions,n_states))
       
    def select_action(self, s, epsilon):
        # TO DO: Change this to e-greedy action selection
        if np.random.rand() > epsilon:
            a = np.argmax(self.Q_sa[s])
        else:
            a = np.random.choice(self.n_actions)
        return a
       
    def update(self,s,a,r,done,s_next,n_planning_updates):
        # TO DO: Add Dyna update
        self.transition_counts[s][a][s_next] += 1
        self.reward_sums[s][a][s_next] += r
        self.Q_sa[s, a] += self.learning_rate * (r + self.gamma * np.max(self.Q_sa[s_next]) - self.Q_sa[s,a])

        for k in range(n_planning_updates):
            s = np.random.choice(self.n_states)
            a = np.random.choice(self.n_actions)
            while np.sum(self.transition_counts[s][a]) == 0:
                s = np.random.choice(self.n_states)
                a = np.random.choice(self.n_actions)

            prob_transition = (self.transition_counts[s][a])/(np.sum(self.transition_counts[s][a]))
            s_next = np.random.choice(self.n_states, p=prob_transition)
            r = self.reward_sums[s][a][s_next]/self.transition_counts[s][a][s_next]

            self.Q_sa[s,a] += self.learning_rate * (r + self.gamma * np.max(self.Q_sa[s_next]) - self.Q_sa[s,a])
        pass


    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = np.argmax(self.Q_sa[s]) # greedy action selection
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return


class PrioritizedSweepingAgent:


    def __init__(self, n_states, n_actions, learning_rate, gamma, max_queue_size=200, priority_cutoff=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.priority_cutoff = priority_cutoff
       
        self.Q_sa = np.zeros((n_states,n_actions))
        # TO DO: Initialize count tables, reward sum tables, priority queue
        self.transition_counts = np.zeros((n_states,n_actions,n_states))
        self.reward_sums = np.zeros((n_states,n_actions,n_states))
        self.queue = PriorityQueue()
       
    def select_action(self, s, epsilon):
        # TO DO: Change this to e-greedy action selection
        if np.random.rand() > epsilon:
            a = np.argmax(self.Q_sa[s])
        else:
            a = np.random.choice(self.n_actions)
        return a
       
    def update(self,s,a,r,done,s_next,n_planning_updates):
        self.transition_counts[s][a][s_next] += 1
        self.reward_sums[s][a][s_next] += r
        # Compute the priority of the state
        prioriri = np.abs(r+(self.gamma*np.max(self.Q_sa[s_next])-self.Q_sa[s,a]))
        if prioriri > self.priority_cutoff:
            self.queue.put((-prioriri,(s,a)))
        
        for k in range(n_planning_updates):
            if self.queue.empty():
                break
            _, (s, a) = self.queue.get()
            prob_trans= self.transition_counts[s][a] / (np.sum(self.transition_counts[s][a]))
            s_next = np.argmax(prob_trans)
            r_new = self.reward_sums[s][a][s_next] / self.transition_counts[s][a][s_next]
            r = r_new
            self.Q_sa[s,a] = self.Q_sa[s,a] + self.learning_rate*(r + self.gamma * np.max(self.Q_sa[s_next]) - self.Q_sa[s,a])
            for previous_s, previous_a in zip(*np.nonzero(self.transition_counts[:,:,s])[:2]):
                r_bar = self.reward_sums[previous_s][previous_a][s] / self.transition_counts[previous_s][previous_a][s]
                r_new = r_bar
                prioriri = np.abs(r_new+(self.gamma*np.max(self.Q_sa[s])-self.Q_sa[previous_s, previous_a]))
                if prioriri > self.priority_cutoff:
                    self.queue.put((-prioriri,(previous_s,previous_a)))
            
        

    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = np.argmax(self.Q_sa[s]) # greedy action selection
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return        


def test():


    n_timesteps = 10001 #10001
    gamma = 1.0


    # Algorithm parameters
    policy = 'ps' # or 'ps' 'dyna
    epsilon = 0.1
    learning_rate = 0.2
    n_planning_updates = 3


    # Plotting parameters
    plot = True #True  
    plot_optimal_policy = True
    step_pause = 0.0001
   
    # Initialize environment and policy
    env = WindyGridworld()
    if policy == 'dyna':
        pi = DynaAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize Dyna policy
    elif policy == 'ps':    
        pi = PrioritizedSweepingAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize PS policy
    else:
        raise KeyError('Policy {} not implemented'.format(policy))
   
    # Prepare for running
    s = env.reset()  
    continuous_mode = False
   
    for t in range(n_timesteps):            
        # Select action, transition, update policy
        a = pi.select_action(s,epsilon)
        s_next,r,done = env.step(a)
        pi.update(s=s,a=a,r=r,done=done,s_next=s_next,n_planning_updates=n_planning_updates)
       
        # Render environment
        if plot:
            env.render(Q_sa=pi.Q_sa,plot_optimal_policy=plot_optimal_policy,
                       step_pause=step_pause)
           
        #Ask user for manual or continuous execution
        if not continuous_mode:
           key_input = input("Press 'Enter' to execute next step, press 'c' to run full algorithm")
           continuous_mode = True if key_input == 'c' else False


        # Reset environment when terminated
        if done:
            s = env.reset()
        else:
            s = s_next
           
   
if __name__ == '__main__':
    test()

