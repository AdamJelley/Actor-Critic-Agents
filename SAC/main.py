import pybullet_envs
import gym
import numpy as np
from sac_agent import Agent
from utils import plot_learning_curve
import time

load_checkpoint = True

if __name__ == '__main__':
    env_id = 'InvertedPendulumBulletEnv-v0'
    env = gym.make(env_id)
    agent = Agent(alpha=0.0003, beta=0.0003, reward_scale=2, env_id=env_id, input_dims=env.observation_space.shape, tau=0.005,
                    env=env, batch_size=256, layer1_size=256, layer2_size=256, n_actions=env.action_space.shape[0])

    n_games = 10
    filename = 'SAC_' + env_id + '_' + str(n_games) + 'games_scale' + str(agent.scale) + '.png'
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    steps = 0
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            new_observation, reward, done, info = env.step(action)
            steps += 1
            score += reward
            agent.remember(observation, action, reward, new_observation, done)
            if load_checkpoint:
                time.sleep(0.01)
            else:
                agent.learn()
            observation = new_observation
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('Episode', i, 'Score %.1f' % avg_score, 'Steps %d' % steps, env_id, 'Scale', agent.scale)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(score_history, x, figure_file)