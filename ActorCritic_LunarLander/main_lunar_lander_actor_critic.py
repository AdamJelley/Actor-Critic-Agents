import gym
import numpy as np
from actor_critic import ActorCriticAgent
from utils import plot_learning_curve

if __name__=='__main__':
    env = gym.make('LunarLander-v2')
    agent = ActorCriticAgent(gamma=0.99, lr=5e-6, input_dims=[8], n_actions=4, \
        fc1_dims=2048, fc2_dims=1536)
    n_games = 2000

    fname = 'ActorCritic_' + 'lunar_lander_' + str(agent.fc1_dims) + \
        '_fc1_dims_' + str(agent.fc2_dims) + '_fc2_dims_lr' + str(agent.lr) + \
        '_' + str(n_games) + 'games'
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
            agent.learn(observation, reward, new_observation, done)
            observation = new_observation
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('epsiode ', i, 'score %.1f' % score, 'average score %.1f' % avg_score)

    x = [i+1 for i in range(n_games)]
    plot_learning_curve(scores, x, figure_file)
