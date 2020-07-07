import gym
import numpy as np
import matplotlib.pyplot as plt


def policy(state, theta):
    """ TODO: return probabilities for actions under softmax action selection """
    weight_dim, num_action = theta.shape
    policy = np.zeros(num_action)
    e_h = [np.exp(action_preference(state, theta[:, i])) for i in range(num_action)]
    for action in range(num_action):
        policy[action] = e_h[action]/sum(e_h)
    return policy

def action_preference(state, parameters):
    return np.dot(parameters, state)

def generate_episode(env, theta, display=False):
    """ enerates one episode and returns the list of states, the list of rewards and the list of actions of that episode """
    state = env.reset()
    states = [state]
    actions = []
    rewards = []
    for t in range(500):
        if display:
            env.render()
        p = policy(state, theta)
        action = np.random.choice(len(p), p=p)

        state, reward, done, info = env.step(action)
        rewards.append(reward)
        actions.append(action)
        if done:
            break
        states.append(state)
    return states, rewards, actions, p


def REINFORCE(env):
    theta = np.random.rand(4, 2)  # policy parameters
    weight_dim, num_action = theta.shape
    last_ep_lens = []
    gamma = 0.9
    alpha = 0.0001
    for e in range(10000):
        if e % 300 == 0:
            states, rewards, actions, probs = generate_episode(env, theta, True)  # display the policy every 300 episodes
        else:
            states, rewards, actions, probs = generate_episode(env, theta, False)

        print("episode: " + str(e) + " length: " + str(len(states)))
        # TODO: keep track of previous 100 episode lengths and compute mean
        # avg_len = avg_len + (len(states) - avg_len)/(e+1)
        last_ep_lens.append(len(states))
        if e >= 100:
            last_ep_lens.pop(0)
            avg_len = sum(last_ep_lens)/(100)
            print(avg_len)
        # TODO: implement the reinforce algorithm to improve the policy weights
        T = len(rewards)
        for i in range(T):
            gammas = [gamma**j for j in range(T-i)]
            G = sum(np.multiply(np.array(gammas), rewards[i:T]))
            # update:
            for dim in range(num_action):
                # compute partial derivative for dimension, see .pdf
                grad = (states[i]*int(dim==actions[i]) - states[i]*probs[dim])
                # apply the part derivative on this dimension
                theta[:, dim] = theta[:, dim] + alpha * G * grad


def main():
    env = gym.make('CartPole-v1')
    REINFORCE(env)
    env.close()


if __name__ == "__main__":
    main()
