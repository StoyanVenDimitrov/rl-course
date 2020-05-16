import gym
import numpy as np

def mc_predictions():
    """
    First-visit MC prediction, for estimating V of the policy below
    :return: value for each state
    """
    env = gym.make('Blackjack-v0')
    gamma = 0.8
    values = np.zeros((10, 10, 2))  # len(sum>11) x len(dealer_card) x (usable, not_usable)
    times_visited = np.zeros((10, 10, 2))  # len(sum>11) x len(dealer_card) x (usable, not_usable)

    for i in range(10000):
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

    print(values)



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
    mc_predictions()
