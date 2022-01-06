import gym
import numpy as np
import time
from ddpg_agent import Agent
from utils import plot_learning_curve

load_checkpoint = True

if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2')
    agent = Agent(alpha=0.0001, beta=0.001, input_dims=env.observation_space.shape, tau=0.001,
                    batch_size=64, fc1_dims=400, fc2_dims=300, n_actions=env.action_space.shape[0])
    n_games = 10
    filename = 'DDPG_lunar_lander_alpha' + str(agent.alpha) + '_beta_' + str(agent.beta) + '_' + str(n_games) + '_games'
    figure_file = 'plots/' + filename + '.png'

    best_score = env.reward_range[0]
    score_history = []

    if load_checkpoint:
        agent.load_models()

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            action = agent.choose_action(observation)
            new_observation, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, new_observation, done)
            if load_checkpoint:
                env.render(mode='human')
                time.sleep(0.01)
            else:
                agent.learn()
            score += reward
            observation = new_observation
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('Episode ', i, 'score %.1f' % score, 'average score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(score_history, x, figure_file)