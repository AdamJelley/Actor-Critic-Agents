import numpy as np

class Agent():
    def __init__(self, lr, gamma, n_actions, state_space, eps_start, eps_end,
            eps_desc):
        self.lr=lr
        self.gamma=gamma
        self.epsilon=eps_start
        self.eps_end=eps_end
        self.eps_desc=eps_desc
        self.n_actions = n_actions

        self.state_space = state_space
        self.actions = [i for i in range(self.n_actions)]

        self.Q = {}
        self.init_Q()

    def init_Q(self):
        for state in self.state_space:
            for action in self.actions:
                self.Q[(state, action)] = 0.0

    def max_action(self, state):
        actions = np.array([self.Q[(state, a)] for a in self.actions])
        action = np.argmax(actions)
        return action

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            action = self.max_action(state)
        return action

    def decremement_epsilon(self):
        self.epsilon = self.epsilon-self.eps_desc if self.epsilon > self.eps_end else self.eps_end

    def learn(self, state, action, reward, new_state):
        a_max = self.max_action(new_state)

        self.Q[(state, action)] = self.Q[(state, action)] + self.lr*(reward + \
            self.gamma*self.Q[(new_state, a_max)] -
            self.Q[(state, action)])


