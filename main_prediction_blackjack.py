import gym
from prediction_blackjack import Agent

if __name__ == '__main__':
    env = gym.make('Blackjack-v0')
    agent = Agent()
    n_episodes = 500000
    for i in range(n_episodes):
        if i%50000==0:
            print('Starting episode ', i)
        observation = env.reset()
        done = False
        while not done:
            action = agent.policy(observation)
            new_observation, reward, done, info = env.step(action)
            agent.memory.append((observation, reward))
            observation = new_observation
        agent.update_V()
    print(agent.V[21,3,True])
    print(agent.V[4,1,False])