import numpy as np
from collections import defaultdict
import random

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.policy = None
        self.eps = 1
        self.i_episode = 1
        self.gamma = 0.8
        self.alpha = 0.1

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return self.action_epsilon_greedy(state)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        old_value = self.Q[state][action]
        if done:
            self.Q[state][action] = old_value + self.alpha * (reward - old_value)
            self.update_policy_from_q_greedy()
            self.end_of_episode()
            return
        self.Q[state][action] = old_value + \
            self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - old_value)
        self.update_policy_from_q_greedy()
        

    def update_policy_from_q_greedy(self):
        self.policy = dict((k,np.argmax(v)) for k, v in self.Q.items())
    
    def action_epsilon_greedy(self, state):
        if random.random()< self.eps or self.policy == None or state not in self.policy:
            action = np.random.choice(np.arange(self.nA))
        else:
            action = self.policy[state]
        return action

    def end_of_episode(self):
        self.i_episode += 1
        self.eps = 1 / self.i_episode
        #self.eps = max (np.interp(self.i_episode,[1.0,5000.0], [1, 0.1]) , 0.000001)
        self.prev_state = None
        self.prev_action = None