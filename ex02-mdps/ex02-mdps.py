import gym
import numpy as np
import itertools

# Init environment
# Lets use a smaller 3x3 custom map for faster computations
custom_map3x3 = [
    'SFFF',
    'FHFH',
    'FFFH',
    'HFFG'
]
env = gym.make("FrozenLake-v0", desc=custom_map3x3)
# TODO: Uncomment the following line to try the default map (4x4):
#env = gym.make("FrozenLake-v0")

# Uncomment the following lines for even larger maps:
#random_map = generate_random_map(size=5, p=0.8)
#env = gym.make("FrozenLake-v0", desc=random_map)

# Init some useful variables:
n_states = env.observation_space.n
n_actions = env.action_space.n
r = np.zeros(n_states) # the r vector is zero everywhere except for the goal state (last state)
r[-1] = 1.

gamma = 0.8


""" This is a helper function that returns the transition probability matrix P for a policy """
def trans_matrix_for_policy(policy):
    transitions = np.zeros((n_states, n_states))
    for s in range(n_states):
        probs = env.P[s][policy[s]]
        for el in probs:
            transitions[s, el[1]] += el[0]
    return transitions


""" This is a helper function that returns terminal states """
def terminals():
    terms = []
    for s in range(n_states):
        # terminal is when we end with probability 1 in terminal:
        if env.P[s][0][0][0] == 1.0 and env.P[s][0][0][3] == True:
            terms.append(s)
    return terms


def value_policy(policy):
    P = trans_matrix_for_policy(policy)
    identity = (np.identity(len(P)))
    substr = np.subtract(identity, P*gamma)
    inverse = np.linalg.inv(substr)
    v_n = inverse.dot(r)
    return v_n


def bruteforce_policies():
    terms = terminals()
    # policy = np.zeros(n_states, dtype=np.int)  # in the discrete case a policy is just an array with action = policy[state]
    # optimalvalue = np.zeros(n_states)
    optimalpolicies = []

    # generate all policies like [0, 0, 0, 0, 0, 2, 3, 0, 0], skipping the terminal states:
    # first, build the arrays permutations, using only one action, e.g. 0, for the terminal state:
    iterables = []
    for state in range(n_states):
        if state in terms:
            iterables.append([0])
        else:
            iterables.append([0, 1, 2, 3])
    # that build all permutations:
    policies = np.asarray([np.asarray(p, dtype=np.int) for p in itertools.product(*iterables)])
    value_functions = [value_policy(p) for p in policies]
    optimalvalue = np.amax(value_functions, axis=0)

    # find the policies with v_\pi = v_*
    for (i, arr) in enumerate(value_functions):
        if np.array_equal(arr, optimalvalue):
            optimalpolicies.append(policies[i])

    print ("Optimal value function:")
    print(optimalvalue)
    print ("number optimal policies:")
    print (len(optimalpolicies))
    print ("optimal policies:")
    print (np.array(optimalpolicies))
    return optimalpolicies



def main():
    # print the environment
    print("current environment: ")
    env.render()
    print("")

    # Here a policy is just an array with the action for a state as element
    policy_left = np.zeros(n_states, dtype=np.int)  # 0 for all states
    policy_right = np.ones(n_states, dtype=np.int) * 2  # 2 for all states

    # Value functions:
    print("Value function for policy_left (always going left):")
    print (value_policy(policy_left))
    print("Value function for policy_right (always going right):")
    print (value_policy(policy_right))

    optimalpolicies = bruteforce_policies()


    #This code can be used to "rollout" a policy in the environment:
    print ("rollout policy:")
    maxiter = 100
    state = env.reset()
    for i in range(maxiter):
        new_state, reward, done, info = env.step(optimalpolicies[0][state])
        env.render()
        state=new_state
        if done:
            print ("Finished episode")
            break


if __name__ == "__main__":
    main()
