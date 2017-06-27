import gym

from baselines import deepq

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
def main():
    env = gym.make("Wavefollower-v0")
    act = deepq.load("wavefollower_model.pkl")

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            #env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            #plt.scatter(obs[0],obs[1], color='b')
            plt.scatter(obs[0],obs[2], color = 'r')
            plt.pause(0.00001)
            episode_rew += rew
            #print("Observation = {}".format(obs))
            print("Action = {}".format(act(obs[None])[0]))
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
