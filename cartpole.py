#
# cartpole.py, exercise sheet 2, Advanced Machine Learning course, RWTH Aachen University, summer term 2019, Sourabh Swain
#

import gym
# render a dummy environment before importing tensorflow to circumvent tensorflow/openai-gym integration bug
#g_env = gym.make('CartPole-v0')
#g_env.render()

import tensorflow as tf
import random
import matplotlib.pyplot as plt
import numpy as np

num_training_episodes = 1000
episode_length = 200

class REINFORCE():
    def __init__(
            self,
            n_states,
            n_hidden,
            n_actions,
            learning_rate=1e-3,
            discount_rate=0.99
            ):

        self.n_states = n_states
        self. n_actions = n_actions
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate

        self.episode_states, self.episode_actions, self.episode_rewards = [], [], []
        self.build_net()
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())


    def build_net(self):
        with tf.variable_scope("policy"):
            self.state = tf.placeholder(tf.float32, [None, self.n_states], name="State")
            self.actions = tf.placeholder(tf.int32, [None, 1], name="Action")
            self.discounted_reward = tf.placeholder(tf.float32, [None, 1],  name="Reward")

            with tf.variable_scope("weights"):
                params_w1 = tf.get_variable("policy_parameters_w1", [self.n_states, self.n_hidden])
                params_b1 = tf.get_variable("policy_parameters_b1", [self.n_hidden])
                params_w2 = tf.get_variable("policy_parameters_w2", [self.n_hidden, self.n_actions])
                params_b2 = tf.get_variable("policy_parameters_b2", [self.n_actions])

            hidden = tf.nn.relu(tf.matmul(self.state, params_w1) + params_b1)
            self.probabilities = tf.nn.softmax(tf.matmul(hidden, params_w2) + params_b2)

            neg_log_prob = tf.reduce_sum(-tf.log(self.probabilities) * tf.one_hot(self.actions, self.n_actions), axis = 0)
            loss = tf.reduce_mean(neg_log_prob * self.discounted_reward)

            with tf.variable_scope("optimizer"):
                self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        #return self.probabilities, self.state, self.actions, self.discounted_reward, self.train_op


    def choose_action(self, state):
        prob_weights = self.sess.run(self.probabilities, feed_dict={self.state: np.reshape(state, (1, self.n_states))})
        #print("Prob_weights inside choose_action", prob_weights.shape)
        action = np.random.choice(range(len(prob_weights[0])), p=prob_weights[0])
        return action

    def learn(self, ep_states, ep_actions, discounted_reward):
        #print("TESTING", self.discounted_reward)
        #print("Test", len(ep_states), len(ep_actions), len(discounted_reward))
        self.sess.run(self.train_op, feed_dict={self.state: np.vstack(ep_states), self.actions: np.vstack(ep_actions),
                                                self.discounted_reward: np.vstack(discounted_reward)})


    def discount_rewards(self, ep_rewards):
        discounted_rewards = np.zeros_like(ep_rewards)
        cur_sum = 0

        for t in reversed(range(0, len(ep_rewards))):
            cur_sum = cur_sum * self.discount_rate + ep_rewards[t]
            discounted_rewards[t] = cur_sum

        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        return discounted_rewards


def run_episode( env):

    observation = env.reset()
    episode_return = 0

    ep_states, ep_actions, ep_rewards = [], [], []

    #pl_prob, pl_state, pl_action, pl_reward, pl_optimizer = RL.build_net()
    #RL.build_net()

    for _ in range( episode_length ):

        state = observation

        action = RL.choose_action(state)
        ep_states.append(state)
        ep_actions.append(action)

        observation, reward, done, info = env.step(action)

        episode_return += reward
        ep_rewards.append(reward)
        
        if done:
            print("Episode ended early Length:{}".format(_+1))
            print("Episode return: {}".format(episode_return))

            discounted_reward = RL.discount_rewards(ep_rewards)

            RL.learn(ep_states, ep_actions, discounted_reward)
            break



    return episode_return



env = gym.make('CartPole-v0')
#env.render()
monitor = gym.wrappers.Monitor(env, 'cartpole/', force=True)
rewards = []
#sess = tf.InteractiveSession()
#sess.run(tf.initialize_all_variables())

RL = REINFORCE(
        n_states=env.observation_space.shape[0],
        n_hidden=20,
        n_actions=env.action_space.n
        )

for i in range( num_training_episodes ):
    
    episode_return = run_episode( env)
    print("****episode no.{} return:**** %f".format(i) % (episode_return,))
    rewards.append(episode_return)


plt.plot(rewards)
plt.show()
monitor.close()

