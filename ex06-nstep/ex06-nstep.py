import gym
import numpy as np
import matplotlib.pyplot as plt
import random


def _print_policy(Q, env):
    """ This is a helper function to print a nice policy from the Q function"""
    moves = [u'←', u'↓',u'→', u'↑']
    if not hasattr(env, 'desc'):
        env = env.env
    dims = env.desc.shape
    policy = np.chararray(dims, unicode=True)
    policy[:] = ' '
    for s in range(len(Q)):
        idx = np.unravel_index(s, dims)
        policy[idx] = moves[np.argmax(Q[s])]
        if env.desc[idx] in ['H', 'G']:
            policy[idx] = u'·'
    print('\n'.join([''.join([u'{:2}'.format(item) for item in row])
        for row in policy]))

def nstep_sarsa(env, n=3, alpha=0.1, gamma=0.9, epsilon=0.1, num_ep=int(1e4)):
    Q = np.zeros((env.observation_space.n,  env.action_space.n))

    def choose_a(Q, s):
        if random.random() > epsilon:
            # breaking the ties at random!
            a = np.random.choice(np.where(Q[s] == Q[s].max())[0])
        else:
            a = random.randrange(len(Q[s]))
        return a

    for i in range(num_ep):
        s = env.reset()
        a = choose_a(Q, s)
        t = 0
        T = 100000
        R = []
        S = []
        A = []
        A.append(a)
        print('########')
        end = False
        while not end:
            if t < T:
                s_, r, done, _ = env.step(a)
                S.append(s_)
                R.append(r)
                A.append(a)
                if not done:
                    a_ = choose_a(Q, s_)
                    a = a_
                else:
                    T = t +1
            tau = t - n + 1
            if tau == T-1:
                end = True
            else:

                print(t, tau)
                print(S)
                print(A)
                print(T)
                if tau >= 0:
                    sum_up_to = min((tau + n), T)
                    G = 0
                    for i in range(sum_up_to):
                        G = G + (gamma**i)*R[i]
                    if (tau + n) < T:
                        G = G + (gamma**n)*Q[S[tau + n]][A[tau + n]]
                    Q[S[tau]][A[tau]] = Q[S[tau]][A[tau]] + alpha*(G-Q[S[tau]][A[tau]])
            t = t + 1
    return Q
    # Plot the average episode length as training continues.
    # fig, ax = plt.subplots()
    #
    # ax.set(xlabel='episode', ylabel='episode length',
    #        title='average episode length as training continues')
    # ax.grid()


env=gym.make('FrozenLake-v0')
# TODO: run multiple times, evaluate the performance for different n and alpha
Q = nstep_sarsa(env)
print(Q.shape)
_print_policy(Q, env)