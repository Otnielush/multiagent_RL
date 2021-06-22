from pettingzoo.sisl import multiwalker_v7
import numpy as np
from PPO_Agent import Agent
from utils import plot_learning_curve
from copy import deepcopy

if __name__ == '__main__':
    env = multiwalker_v7.env()
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003

    agent = Agent(n_actions=env.action_spaces['walker_0'].shape[0], batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs,
                  input_dims=env.observation_spaces['walker_0'].shape)

    n_games = 100

    figure_file = 'plots/walker.png'

    best_score = -100
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        env.reset()
        done = False
        score = 0
        observation = {'walker_0':0, 'walker_1':0, 'walker_2':0}
        observation_ = {'walker_0': 0, 'walker_1': 0, 'walker_2': 0}
        reward = {'walker_0': 0, 'walker_1': 0, 'walker_2': 0}
        action = {'walker_0': 0, 'walker_1': 0, 'walker_2': 0}
        prob = {'walker_0': 0, 'walker_1': 0, 'walker_2': 0}
        val = {'walker_0': 0, 'walker_1': 0, 'walker_2': 0}
        done = {'walker_0': False, 'walker_1': False, 'walker_2': False}
        first = 0

        while not any(done.values()):
            for walker in env.agent_iter():
                observation_[walker], reward[walker], done[walker], info = env.last()
                if done[walker]:
                    break
                    env.step(None)
                    continue

                action[walker], prob[walker], val[walker] = \
                    agent.choose_action(observation_[walker])
                # print(f"{action = } {val = }")
                # print(f"{observation_= }")
                env.step(action[walker])
                env.render()

                score += reward[walker]

                if first >= 3:
                    agent.remember(observation[walker], action[walker], prob[walker],
                               val[walker], reward[walker], done[walker])
                else: first += 1

                observation[walker] = observation_[walker]

                n_steps += 1

                if n_steps % N == 0:
                    agent.learn()
                    learn_iters += 1

        env.close()
        score_history.append(score)
        avg_score = np.mean(score_history[-30:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print(f"episode {i} score {score:.1f} avg ({avg_score:.1f}) "
              f"time_steps {n_steps} learning_steps {learn_iters}")

    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file, 30)

