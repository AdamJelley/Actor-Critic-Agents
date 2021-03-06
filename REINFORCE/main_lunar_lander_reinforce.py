import gym
import numpy as np
from reinforce import PolicyGradientAgent
from utils import plot_learning_curve

if __name__=='__main__':
    env = gym.make('LunarLander-v2')
    n_games = 3000
    agent = PolicyGradientAgent(gamma=0.99, lr=0.0005, input_dims=[8], n_actions=4)

    fname = 'REINFORCE_' + 'lunar_lander_lr_' + str(agent.lr) + '_' \
        + str(n_games) + '_games'
    figure_file = 'plots/' + fname + '.png'

    scores = []
    for i in range(n_games):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            new_observation, reward, done, info = env.step(action)
            score += reward
            agent.store_rewards(reward)
            observation = new_observation
        agent.learn()
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score)

    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(scores, x, figure_file)