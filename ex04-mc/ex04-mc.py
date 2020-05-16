import gym
import numpy as np

def mc_predictions():
    """
    First-visit MC prediction, for estimating V of the policy below
    :return: value for each state
    """
    env = gym.make('Blackjack-v0')
    gamma = 1.0
    values = np.zeros((10, 10, 2))  # len(sum>11) x len(dealer_card) x (usable, not_usable)
    times_visited = np.zeros((10, 10, 2))  # len(sum>11) x len(dealer_card) x (usable, not_usable)

    for i in range(500000):
        obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
        done = False
        state = None
        episode = []
        while not done:
            # don't add episodes when hit is the only choice: at sum<12
            if obs[0] >= 12:
                state = (obs[0] - 12, obs[1] - 1, int(obs[2]))
            #print("observation:", obs)
            if obs[0] >= 20:
                #print("stick")
                action = 0
                obs, reward, done, _ = env.step(0)
                # find position of state value from the new state:
            else:
                #print("hit")
                action = 1
                # don't add episodes when hit is the only choice: at sum<12
                obs, reward, done, _ = env.step(1)
            #print("reward:", reward)
            #print("")
            if state is not None:
                times_visited[state[0]][state[1]][state[2]] += 1
                episode.append([state, action, reward])
        #print(episode)
        G = 0
        for step in episode:
            G = gamma*G + step[2]  # update G with the reward
            state = step[0]
            old_value = values[state[0]][state[1]][state[2]]
            visited = times_visited[state[0]][state[1]][state[2]]
            values[state[0]][state[1]][state[2]] = old_value + (G - old_value)/visited
        i += 1

    no_usable_ace_values = np.take(values, [0], axis=2)
    usable_ace_values = np.take(values, [1], axis=2)

    return no_usable_ace_values, usable_ace_values


def mc_exploring_starts():
    """
    :return: optimal policy
    """
    env = gym.make('Blackjack-v0')
    gamma = 1.0
    policy = np.zeros((10, 10, 2), dtype=np.int8)  # len(sum>11) x len(dealer_card) x (usable, not_usable)
    action_values = np.zeros((10, 10, 2, 2))  # len(sum>11) x len(dealer_card) x (usable, not_usable) x [stick, hit]
    times_visited = np.zeros((10, 10, 2, 2))  # len(sum>11) x len(dealer_card) x (usable, not_usable) x [stick, hit]

    for i in range(10000):
        obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
        done = False
        episode = []
        while not done:
            # for all sums < 12: hit. Cannot loose.
            if obs[0] < 12:
                obs, reward, done, _ = env.step(1)
                continue
            # print("observation:", obs)
            state = (obs[0] - 12, obs[1] - 1, int(obs[2]))
            action = policy[state[0]][state[1]][state[2]]
            obs, reward, done, _ = env.step(action)
            # print("reward:", reward)
            # print("")
            if state is not None:
                times_visited[state[0]][state[1]][state[2]] += 1
                episode.append([state, action, reward])
        G = 0
        for step in episode:
            G = gamma * G + step[2]  # update G with the reward
            state = step[0]
            action = step[1]
            old_value = action_values[state[0]][state[1]][state[2]][action]
            visited = times_visited[state[0]][state[1]][state[2]][action]
            action_values[state[0]][state[1]][state[2]][action] = old_value + (G - old_value) / visited
            policy[state[0]][state[1]][state[2]] = np.argmax(action_values[state[0]][state[1]][state[2]])
        i += 1
    # sum over the action values for each state:
    optimal_values = np.sum(action_values, axis=3)
    no_usable_ace_policy = np.take(policy, [0], axis=2)
    usable_ace_policy = np.take(policy, [1], axis=2)

    return no_usable_ace_policy, usable_ace_policy, optimal_values


def main():
    # This example shows how to perform a single run with the policy that hits for player_sum >= 20
    env = gym.make('Blackjack-v0')
    obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
    done = False
    while not done:
        print("observation:", obs)
        if obs[0] >= 20:
            print("stick")
            obs, reward, done, _ = env.step(0)
        else:
            print("hit")
            obs, reward, done, _ = env.step(1)
        print("reward:", reward)
        print("")


if __name__ == "__main__":
    # main()
    print(mc_predictions()[0])
