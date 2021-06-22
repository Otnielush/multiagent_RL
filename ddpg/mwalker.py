import gym
from pettingzoo.sisl import multiwalker_v7
import numpy as np
from ddpg_torch import Agent
from utils import plot_learning_curve


if __name__ == '__main__':
    env = multiwalker_v7.env(max_cycles=500)
    agent = Agent(alpha=0.0001, beta=0.001,
                    input_dims=env.observation_spaces['walker_0'].shape, tau=0.001,
                    batch_size=64, fc1_dims=400, fc2_dims=300,
                    n_actions=env.action_spaces['walker_0'].shape[0])
    try:
        agent.load_models()
    except: print('No saved models')

    n_games = 5000
    filename = 'multiwalker_' + str(agent.alpha) + '_beta_' + \
                str(agent.beta) + '_' + str(n_games) + '_games'
    figure_file = 'plots/' + filename + '.png'

    best_score = -100
    score_history = []
    for i in range(n_games):
        env.reset()
        score = 0
        agent.noise.reset()
        moves = 0

        observation = {'walker_0': 0, 'walker_1': 0, 'walker_2': 0}
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
                moves += 1
                action[walker] = agent.choose_action(observation_[walker])
                env.step(action[walker])
                env.render()

                # observation_, reward, done, info = env.step(action)
                if first >= 3:
                    agent.remember(observation[walker], action[walker],
                                reward[walker], observation_[walker], done[walker])
                else:
                    first += 1


                score += sum(reward.values())
                observation[walker] = observation_[walker]

            agent.learn()
        env.close()

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()



        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score, f"{moves = }")
    env.close()
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)