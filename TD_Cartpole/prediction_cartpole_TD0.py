import numpy as np
import gym

def simple_policy(state):
    action = 0 if state < 5 else 1
    return action

if __name__=='__main__':
    env = gym.make('CartPole-v0')
    alpha = 0.1
    gamma = 0.99

    states = np.linspace(-0.2094, 0.2094, 10)
    V = {}
    for state in range(len(states)+1):
        V[state] = 0

    for i in range(5000):
        observation = env.reset()
        done = False
        while not done:
            state = int(np.digitize(observation[2], states))
            action = simple_policy(state)
            new_observation, reward, done, info = env.step(action)
            new_state = int(np.digitize(new_observation[2], states))
            V[state] = V[state] + alpha*(reward + gamma*V[new_state] - V[state])
            observation = new_observation

    for state in V:
        print(state, '%3f' % V[state])