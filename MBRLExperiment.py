#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning experiments
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
By Thomas Moerland
"""
import numpy as np
from MBRLEnvironment import WindyGridworld
from MBRLAgents import DynaAgent, PrioritizedSweepingAgent
from Helper import LearningCurvePlot, smooth

def experiment():
    n_timesteps = 10001
    eval_interval = 250
    n_repetitions = 10
    gamma = 1.0
    learning_rate = 0.2
    epsilon=0.1
    
    wind_proportions=[0.9,1.0]
    n_planning_updatess = [1,3,5] 
    
    # IMPLEMENT YOUR EXPERIMENT HERE
    WG_env = WindyGridworld()
    Dyna_agent = DynaAgent(WG_env.n_states, WG_env.n_actions, learning_rate, gamma)
    eval = Dyna_run_repetitions(Dyna_agent, WG_env, n_timesteps, eval_interval, epsilon)
    print(eval)

def Dyna_run_repetitions(agent, env, n_timesteps, eval_interval, epsilon):
    eval_list = []
    s = env.reset()
    for i in range(1, n_timesteps):
        if i % eval_interval == 0:
            eval = agent.evaluate(env)
            eval_list.append(eval)
        a = agent.select_action(s, epsilon)
        s_next, r, done = env.step(a)
        agent.update(s, a, r, done, s_next, n_planning_updates=3)
        if done:
            s_next = env.reset()
        s = s_next
    return eval_list


if __name__ == '__main__':
    experiment()
