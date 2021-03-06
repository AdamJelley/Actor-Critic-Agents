import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, env, env_id, gamma=0.99, n_actions=2, max_size=int(1e6),
                    layer1_size=256, layer2_size=256, batch_size=100, reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions=n_actions,
                                    name=env_id+'_actor', max_action=env.action_space.high)
        self.critic_1 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions, name=env_id+'_critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions, name=env_id+'_critic_2')
        self.value = ValueNetwork(beta, input_dims, layer1_size, layer2_size, name=env_id+'_value')
        self.target_value = ValueNetwork(beta, input_dims, layer1_size, layer2_size, name=env_id+'_target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation, evaluation=False):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        actions, _ =  self.actor.sample_normal(state, evaluation=evaluation, reparameterize=False)
        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_state_dict = dict(self.target_value.named_parameters())
        value_state_dict = dict(self.value.named_parameters())

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + (1-tau)*target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('... Saving Models ...')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('... Loading Models ...')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)
        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)

        # Value network update
        value = self.value(state).view(-1)
        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        value_target = critic_value - log_probs
        value_loss = 0.5*F.mse_loss(value, value_target)
        self.value.optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        # Actor network update
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        actor_loss = T.sum(log_probs - critic_value)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        # Critic network update
        new_value_target = self.target_value(new_state).view(-1)
        new_value_target[done] = 0.0
        critic_target = self.scale*reward + self.gamma*new_value_target
        # Note we are using the sampled action 'action' (not 'actions') here corresponding to the old policy
        q1 = self.critic_1.forward(state, action).view(-1)
        q2 = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5*F.mse_loss(q1, critic_target)
        critic_2_loss = 0.5*F.mse_loss(q2, critic_target)
        self.critic_1.optimizer.zero_grad()
        critic_1_loss.backward(retain_graph=True)
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.zero_grad()
        critic_2_loss.backward(retain_graph=True)
        self.critic_2.optimizer.step()

        self.update_network_parameters()