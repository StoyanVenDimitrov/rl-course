import gym
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import copy

env = gym.make('FrozenLake-v0')
n_states = env.observation_space.n
n_actions = env.action_space.n
dynamics = env.P
mu = np.random.dirichlet(np.ones(n_states), size=1)[0]
rewards = np.zeros(n_states)
gamma = 1.0


def naive_policy(env, trajs):
    """policy from state-action co-occurancies"""
    action_values = np.zeros((n_states, n_actions))
    for traj in trajs:
        for state_action in traj:
            action_values[state_action[0]][state_action[1]] += 1
    return np.argmax(action_values, axis=1)


def generate_demonstrations(env, expertpolicy, epsilon=0.1, n_trajs=100):
    """ This is a helper function that generates trajectories using an expert policy """
    demonstrations = []
    for d in range(n_trajs):
        traj = []
        state = env.reset()
        for i in range(100):
            if np.random.uniform() < epsilon:
                action = np.random.randint(env.action_space.n)
            else:
                action = expertpolicy[state]
            traj.append((state, action))  # one trajectory is a list with (state, action) pairs
            state, _, done, info = env.step(action)
            if done:
                traj.append((state, 0))
                break
        demonstrations.append(traj)
    return demonstrations  # return list of trajectories


def plot_rewards(rewards, env):
    """ This is a helper function to plot the reward function"""
    fig = plt.figure()
    dims = env.desc.shape
    plt.imshow(np.reshape(rewards, dims), origin='upper', 
               extent=[0,dims[0],0,dims[1]], 
               cmap=plt.cm.RdYlGn, interpolation='none')
    for x, y in product(range(dims[0]), range(dims[1])):
        plt.text(y+0.5, dims[0]-x-0.5, '{:.3f}'.format(np.reshape(rewards, dims)[x,y]),
                horizontalalignment='center', 
                verticalalignment='center')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def reward_function(state, psi):
    """compute linear reward function"""
    one_hot = np.array([int(i == state) for i in range(n_states)])
    # use one_hot directly since for linear function this is the derivative
    return np.dot(psi, one_hot), one_hot


def value_iteration(env, rewards):
    """ Computes a policy using value iteration given a list of rewards (one reward per state) """
    V_states = np.zeros(n_states)
    theta = 1e-8
    gamma = .9
    maxiter = 1000
    policy = np.zeros(n_states, dtype=np.int)
    for iter in range(maxiter):
        delta = 0.
        for s in range(n_states):
            v = V_states[s]
            v_actions = np.zeros(n_actions) # values for possible next actions
            for a in range(n_actions):  # compute values for possible next actions
                v_actions[a] = rewards[s]
                for tuple in env.P[s][a]:  # this implements the sum over s'
                    v_actions[a] += tuple[0]*gamma*V_states[tuple[1]]  # discounted value of next state
            policy[s] = np.argmax(v_actions)
            V_states[s] = np.max(v_actions)  # use the max
            delta = max(delta, abs(v-V_states[s]))

        if delta < theta:
            break

    return policy

def learn_from_demonstration(env, trajectories, psi):
    """apply steps 2 to 5 of MaxEntropy IRL"""
    traj_gradients = np.zeros(n_states)
    for traj in trajectories:
        for s, a in traj:
            rewards[s], gradient = reward_function(s, psi)
            traj_gradients += gradient
    traj_derivatives = traj_gradients/len(trajectories)
    policy = value_iteration(env, rewards=rewards)
    print(policy)
    def get_new_mu(T):
        new_mu = np.zeros([n_states, T])
        for step in range(T):
            for state in range(n_states):
                if step == 0:
                    new_mu[state][step] = mu[state]
                else:
                    for s in range(n_states):
                        # policy is deterministic, no sum over a:
                        action = policy[s]
                        for p_s_s in dynamics[s][action]:
                            if p_s_s[1] == state:
                                new_mu[state][step] += p_s_s[0]*new_mu[s][step-1]
        return np.sum(new_mu, axis=1)/T
    state_freq = get_new_mu(100)
    #print(state_freq)
    state_derivatives = np.array([[int(i == state) for i in range(n_states)] for state in range(n_states)])
    gradient_psi = traj_derivatives - np.matmul(state_derivatives, state_freq)
    psi = psi + gamma*gradient_psi
    return psi, rewards



def main():
    # env.render()
    env.seed(0)
    np.random.seed(0)
    expertpolicy = [0, 3, 0, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0]
    trajs = generate_demonstrations(env, expertpolicy, 0.1, 20)  # list of trajectories
    env.render()
    psi = np.random.rand(n_states)
    for i in range(100):
        psi, rewards = learn_from_demonstration(env, trajs, psi)
    plot_rewards(rewards, env)
    print(naive_policy(env, trajs))


if __name__ == "__main__":
    main()
