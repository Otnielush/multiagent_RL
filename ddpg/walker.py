import gym
import numpy as np
from ddpg_torch import Agent
from utils import plot_learning_curve


if __name__ == '__main__':
    env = gym.make('BipedalWalker-v3')
    agent = Agent(alpha=0.0001, beta=0.001,
                    input_dims=env.observation_space.shape, tau=0.001,
                    batch_size=128, fc1_dims=400, fc2_dims=300,
                    n_actions=env.action_space.shape[0])
    agent.load_models()
    n_games = 1000
    filename = 'BipedalWalker_alpha_' + str(agent.alpha) + '_beta_' + \
                str(agent.beta) + '_' + str(n_games) + '_games'
    figure_file = 'plots/' + filename + '.png'

    best_score = env.reward_range[0]
    score_history = []
    for i in range(n_games):
        observation = env.reset()
        done = False
        steps = 0
        score = 0
        agent.noise.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            # env.render()
            steps += 1
            if steps >= 500:
                done = True
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.1f' % score,
              'average score %.1f' % avg_score, f"{steps = }")
    env.close()
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)