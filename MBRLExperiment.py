#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning experiments
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
By Thomas Moerland
"""
import numpy as np
import matplotlib.pyplot as plt
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
    n_planning_updates = [1,3,5]
    
    # IMPLEMENT YOUR EXPERIMENT HERE
    all_evals = []
    for planning_update in n_planning_updates:
        rep_evals = []
        for n in range(n_repetitions):
            WG_env = WindyGridworld(wind_proportion=wind_proportions[0])
            Dyna_agent = DynaAgent(WG_env.n_states, WG_env.n_actions, learning_rate, gamma)
            eval = Dyna_run_repetitions(Dyna_agent, WG_env, WG_env, n_timesteps, eval_interval, epsilon, planning_update)
            rep_evals.append(eval)
        all_evals.append(rep_evals)
    all_avg_evals = []
    for rep_evals in all_evals:
        avg_eval = np.mean(rep_evals, axis=0)
        all_avg_evals.append(avg_eval)
    plt.figure()
    for i in range(len(all_avg_evals)):
        smoothed_eval = smooth(all_avg_evals[i], 8)  # Adjust window_size as needed
        plt.plot(range(1, len(all_avg_evals[i]) + 1), smoothed_eval, label=f"{n_planning_updates[i]} planning updates")
    plt.legend()
    plt.savefig(f"Dyna plot with {wind_proportions[0]} as wind proportion")
    plt.show()


def Dyna_run_repetitions(agent, env, eval_env, n_timesteps, eval_interval, epsilon, planning_update):
    eval_list = []
    s = env.reset()
    for i in range(1, n_timesteps):
        if i % eval_interval == 0:
            eval = agent.evaluate(eval_env)
            eval_list.append(eval)
        a = agent.select_action(s, epsilon)
        s_next, r, done = env.step(a)
        agent.update(s, a, r, done, s_next, planning_update)
        if done:
            s_next = env.reset()
        s = s_next
    return eval_list


if __name__ == '__main__':
    experiment()
