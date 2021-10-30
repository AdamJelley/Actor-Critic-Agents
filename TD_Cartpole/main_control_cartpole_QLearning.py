import gym
import matplotlib.pyplot as plt
import numpy as np
from control_cartpole_QLearning import Agent

class CartPoleStateDigitiser():
    def __init__(self, bounds=(2.4, 4, 0.209, 4), n_bins=10):
        self.position_space = np.linspace(-1*bounds[0], bounds[0], n_bins)
        self.velocity_space = np.linspace(-1*bounds[1], bounds[1], n_bins)
        self.pole_angle_space = np.linspace(-1*bounds[2], bounds[2], n_bins)
        self.pole_velocity_space = np.linspace(-1*bounds[3], bounds[3], n_bins)
        self.states = self.get_state_space()

    def get_state_space(self):
        states = []
        for i in range(len(self.position_space)+1):
            for j in range(len(self.velocity_space)+1):
                for k in range(len(self.pole_angle_space)+1):
                    for l in range(len(self.pole_velocity_space)+1):
                        states.append((i, j, k, l))
        return states

    def digitize(self, observation):
        x, x_dot, theta, theta_dot = observation
        cart_x = int(np.digitize(x, self.position_space))
        cart_x_dot = int(np.digitize(x_dot, self.velocity_space))
        pole_theta = int(np.digitize(theta, self.pole_angle_space))
        pole_theta_dot = int(np.digitize(theta_dot, self.pole_velocity_space))

        return (cart_x, cart_x_dot, pole_theta, pole_theta_dot)

def plot_learning_curve(scores, x):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] += np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.show()

if __name__=='__main__':
    env = gym.make('CartPole-v0')
    n_games = 50000
    eps_desc = 2 / n_games
    digitiser = CartPoleStateDigitiser()
    agent = Agent(lr=0.01, gamma=0.99, n_actions=2, \
        eps_start=1.0, eps_end=0.01, eps_desc=eps_desc, state_space=digitiser.states)

    scores = []
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        state = digitiser.digitize(observation)
        while not done:
            action = agent.choose_action(state)
            new_observation, reward, done, info = env.step(action)
            new_state = digitiser.digitize(new_observation)
            agent.learn(state, action, reward, new_state)
            state = new_state
            score += reward

        if i % 5000 == 0:
            print('episode ', i, 'score %.1f' % score, 'epsilon %.2f' % agent.epsilon)

        agent.decremement_epsilon()
        scores.append(score)

    x = [i+1 for i in range(n_games)]
    plot_learning_curve(scores, x)