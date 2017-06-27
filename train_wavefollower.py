import gym

from baselines import deepq


def main():
    env = gym.make("Wavefollower-v0")
    model = deepq.models.mlp([64,64])
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=2500000,
        buffer_size=50000,
        exploration_fraction=0.4,
        exploration_final_eps=0.02,
        print_freq=1
    )
    print("Saving model to wavefollower_model.pkl")
    act.save("wavefollower_model.pkl")


if __name__ == '__main__':
    main()
