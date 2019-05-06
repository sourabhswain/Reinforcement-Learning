#
# cartpole.py, exercise sheet 2, Advanced Machine Learning course, RWTH Aachen University, summer term 2019, Sourabh Swain
#

import gym
# render a dummy environment before importing tensorflow to circumvent tensorflow/openai-gym integration bug
# g_env = gym.make('CartPole-v0')
# g_env.render()
import math
import matplotlib.pyplot as plt
import numpy as np
import collections

num_training_episodes = 2000
episode_length = 300
bins = (1, 1, 10, 10)

class QLEARNING():
    def __init__(
            self,
            n_states,
            n_actions,
            discount_rate=1.0,
            min_alpha = 0.1,
            min_epsilon = 0.1,
            annealing_rate = 25
            ):

        self.n_states = n_states
        self.n_actions = n_actions
        self.discount_rate = discount_rate
        self.min_alpha = min_alpha
        self.min_epsilon = min_epsilon
        self.bins = bins
        self.q_value_table = np.zeros(self.bins + (env.action_space.n,))
        self.annealing_rate = annealing_rate

    def get_alpha(self, eps_number):
        return max(self.min_alpha, min(1.0, 1.0 - np.log10((eps_number + 1) / self.annealing_rate)))

    def get_epsilon(self, eps_number):
        return max(self.min_epsilon, min(1, 1.0 - np.log10((eps_number + 1) / self.annealing_rate)))

    def update_qtable(self, cur_state, action, reward, new_state, alpha):
        self.q_value_table[cur_state][action] += alpha * (reward +
                                                          self.discount_rate * np.max(self.q_value_table[new_state]) - self.q_value_table[cur_state][action])


    def choose_action(self, state, epsilon):
        return env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.q_value_table[state])

    def get_qtable(self):
        return self.q_value_table

    def digitalize(self, obs):
        upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
        lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.bins[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.bins[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]

        return tuple(new_obs)


def run_episode(env, eps_number):
    observation = env.reset()
    episode_return = 0

    cur_state = QL.digitalize(observation)
    alpha = QL.get_alpha(eps_number)
    epsilon = QL.get_epsilon(eps_number)

    for _ in range(episode_length):
        #env.render()
        action = QL.choose_action(cur_state, epsilon)

        observation, reward, done, info = env.step(action)
        new_state = QL.digitalize(observation)
        QL.update_qtable(cur_state, action, reward, new_state, alpha)
        cur_state = new_state
        episode_return += reward


        if done:
            break

    print("Episode No. {} - Return: {}".format(eps_number, episode_return))
    return episode_return


env = gym.make('CartPole-v0')
#env.render()
monitor = gym.wrappers.Monitor(env, 'cartpole/', force=True)

QL = QLEARNING(
    n_states = env.observation_space.shape[0],
    n_actions = env.action_space.n
)


rewards = collections.deque(maxlen=100)
for i in range(num_training_episodes):
    episode_return = run_episode(env, i)
    rewards.append(episode_return)
    avg_rewards = np.mean(rewards)

    if(avg_rewards >= 195 and i >= 100):
        print("Solved after {} episodes.".format(i))
        break

plt.plot(rewards)
plt.show()
monitor.close()

