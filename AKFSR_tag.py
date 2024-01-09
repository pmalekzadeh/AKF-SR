import AKFSR
# To test the 1v2 environment
from Multi_Agent_SR.AKFSR_simple1v2_initials import Mu, Sigma
from numpy.linalg import inv
import numpy as np
import argparse
from make_env import make_env
import simple_tag_utilities
import general_utilities
import os
from numpy import inf

# Train
class TrainAKFSR(AKFSR):
    def __init__(self, env, Sigma, Mu, initial_theta, episodes=10, number_samples=200, gamma=0.95,
                 checkpoint_interval=20, csv_filename_prefix='../save/statistics-MAAKFSR_1v1-Train'):
        MAAKFSR.__init__(self, env, observation=None, Sigma=None, Mu=None)
        self.env = env
        self.Sigma = Sigma
        self.Mu = Mu
        self.gamma = gamma
        self.episodes = episodes
        self.csv_filename_prefix = csv_filename_prefix
        self.number_samples = number_samples
        self.checkpoint_interval = checkpoint_interval
        self.n_actions = [self.env.action_space[agentNumber].n for agentNumber in range(self.env.n)]
        self.state_sizes = [self.env.observation_space[i].shape[0] for i in range(env.n)]
        # self.initial_theta = np.zeros((self.n_actions[0]*(len(self.Mu[0])+1), 1))     # Theta0 -- 50*1
        self.initial_theta = initial_theta
        self.P_initial = 10 * np.identity(self.n_actions[0]*(len(self.Mu[0])+1), float)
        self.F = np.identity(self.n_actions[0]*(len(self.Mu[0])+1))
        self.Q = (10 ^ (-7)) * np.identity(self.n_actions[0]*(len(self.Mu[0])+1))
        self.lambda_Mu = 100
        self.lambda_Sigma = 200
        self.R = [.1, .5, 1, 5, 10, 50, 100]
        self.I = np.identity(self.n_actions[0]*(len(self.Mu[0])+1))
        self.episode_indexes = []
        self.sample_indexes = []
        self.theta_k = np.zeros((self.n_actions[0]*(len(self.Mu[0])+1), 1))
        self.Q_theta = (10 ^ (-3)) * np.identity(self.n_actions[0]*(len(self.Mu[0])+1), float)
        self.P_theta = 0.1 * np.identity(self.n_actions[0]*(len(self.Mu[0])+1), float)
        self.R_theta = 1
        self.I_theta = np.identity(self.n_actions[0]*(len(self.Mu[0])+1), float)

        self.m_k = np.zeros((self.n_actions[0]*(len(self.Mu[0])+1) * self.n_actions[0]*(len(self.Mu[0])+1), 1))  # 50*50
        self.Q_m = (10 ^ (-3)) * np.identity(self.n_actions[0]*(len(self.Mu[0])+1) * self.n_actions[0]*(len(self.Mu[0])
                                                                                                        + 1), float)
        self.P_m = .1 * np.identity(self.n_actions[0]*(len(self.Mu[0])+1) * self.n_actions[0]*(len(self.Mu[0])+1), float
                                    )
        self.R_m = 1 * np.identity(self.n_actions[0]*(len(self.Mu[0])+1), float)
        self.I_m = np.identity(self.n_actions[0]*(len(self.Mu[0])+1) * self.n_actions[0]*(len(self.Mu[0])+1), float)

        self.W = np.zeros((self.n_actions[0] * self.n_actions[0], 1))
        self.Q_W = (10 ^ (-3)) * np.identity(self.n_actions[0] * self.n_actions[0], float)
        self.P_W = .1 * np.identity(self.n_actions[0] * self.n_actions[0], float)
        self.F_W = np.identity(self.n_actions[0] * self.n_actions[0])
        self.R_W = 1
        self.I_W = np.identity(self.n_actions[0] * self.n_actions[0])

    def learn(self):
        statistics_header = ["episode"]
        statistics_header.append("steps")
        statistics_header.extend(["reward_{}".format(i) for i in range(env.n)])
        statistics_header.extend(["loss_{}".format(i) for i in range(env.n)])
        statistics_header.extend(["collisions_{}".format(i) for i in range(env.n)])
        print("Collecting statistics {}:".format(" ".join(statistics_header)))
        statistics = general_utilities.Time_Series_Statistics_Store(
            statistics_header)

        episode_losses = np.zeros(env.n)
        episode_rewards = np.zeros(env.n)
        collision_count = np.zeros(env.n)
        steps = 0
        State_k = env.reset()
        el_previous = np.float(1.0)
        # S_previous = np.float(0.0)
        # weight_previous = np.float(1.0)
        Weight_previous = 1

        for episode in range(self.episodes):
            episode_index = []
            sample_index = []
            step = 0
            for agent in range(self.env.n):
                State_k = np.reshape(State_k[agent], (self.state_sizes[agent], 1))  # 10*1
                M_k = np.reshape(self.m_k, (self.n_actions[0]*(len(self.Mu[0])+1), self.n_actions[0] * (len(self.Mu[0])
                                                                                                        + 1)))

                theta_m = np.zeros((self.n_actions[agent]*(len(self.Mu[agent])+1), len(self.R)))    # 50 * 7
                Pi = np.zeros((self.n_actions[agent]*(len(self.Mu[agent])+1),
                               self.n_actions[agent]*(len(self.Mu[agent])+1), len(self.R)))
                weight = np.zeros(len(self.R))
                weight_buf = [0]
                Loss_buf = [0]
                # for t in range(self.number_samples):
                done = [False, False]
                while not any(done):
                    act_test, Phi_current, g = MAAKFSR(self.env, State_k, self.Sigma, self.Mu).Policy_Logic(agent)

                    # env.render()
                    # action
                    actions = []
                    actions_onehot = []
                    for i in range(self.env.n):
                        _action_ = act_test
                        _speed_ = 0.9 if env.agents[i].adversary else 1
                        _onehot_action_ = np.zeros(self.n_actions[i])
                        _onehot_action_[_action_] = _speed_
                        actions_onehot.append(_onehot_action_)
                        actions.append(_action_)
                    State_next, reward, done, info = env.step(actions_onehot)

                    for Agent in range(env.n):
                        if done[Agent]:
                            reward[Agent] -= 50

                    State_next = np.reshape(State_next[agent], (self.state_sizes[agent], 1))

                    # Kalman Filter for theta
                    self.P_theta = self.P_theta + self.Q_theta
                    H = Phi_current.T  # phi_estimated
                    temp = np.dot(np.dot(H, self.P_theta), H.T) + self.R_theta
                    temp = temp.astype(np.int32)
                    temp = np.linalg.pinv(temp)
                    # print("temp: ", temp)
                    K_theta = np.dot(np.dot(self.P_theta, H.T), temp)
                    self.theta_k = self.theta_k + np.dot(K_theta, (reward - np.dot(H, self.theta_k)))
                    self.P_theta = np.dot(self.I_theta - np.dot(K_theta, H), self.P_theta)

                    # m\k
                    self.P_m = self.P_m + self.Q_m  # 900*900

                    #  next action
                    g_final = np.kron(g.T, np.identity(self.n_actions[0]*(len(self.Mu[0])+1), float))  # 50*2500
                    temp = np.dot(np.dot(g_final, self.P_m), g_final.T) + self.R_m
                    temp = temp.astype(np.int32)
                    temp = np.linalg.pinv(temp)
                    K_m = np.dot(np.dot(self.P_m, g_final.T), temp)  # 2500*50
                    self.m_k = self.m_k + np.dot(K_m, (Phi_current - np.dot(g_final, self.m_k)))  # 2500*1
                    self.P_m = np.dot(self.I_m - np.dot(K_m, g_final), self.P_m)

                    # Update RBFs
                    loss = np.dot(Phi_current.T, self.theta_k) - reward
                    # print("loss: ", loss)
                    L_r = loss ** 2

                    if (L_r.any() != -inf) and (L_r.any() != inf):
                        Loss_buf.append(L_r)
                    if (L_r.any() == -inf) or (L_r.any() == inf):
                        L_r = np.mean(Loss_buf)
                    episode_losses += L_r[0]
                    # 50*50
                    M_k = np.reshape(self.m_k, (self.n_actions[0]*(len(self.Mu[0])+1), self.n_actions[0]*(len(self.Mu[0]
                                                                                                              )+1)))
                    for j in range(len(self.Mu[agent])):
                        mu = np.reshape(self.Mu[agent][j], (len(self.Mu[agent][j]), 1))  # # 12*1 or 10*1 or 8*1
                        temp1 = 2 * self.lambda_Mu * loss[0][agent] * np.dot(self.theta_k.T, Phi_current)[agent] * np.linalg.pinv(
                            Sigma[agent][:, :, j])
                        temp2 = np.dot(temp1, State_k - mu)
                        temp3 = 2 * self.lambda_Sigma * loss[0][agent] * np.dot(self.theta_k.T, Phi_current)[agent] * np.linalg.pinv(
                            Sigma[agent][:, :, j])  # 2*2
                        temp4 = np.dot(temp3, State_k - mu)
                        temp5 = np.dot(temp4, np.dot((State_k - mu).T, np.linalg.pinv(Sigma[agent][:, :, j])))
                        if np.dot(self.theta_k.T, Phi_current)[agent] * loss[0][agent] > 0:
                            Sigma[agent][:, :, j] = Sigma[agent][:, :, j] - temp5
                        else:
                            Mu[agent][j] = [we for we in mu - temp2]

                    Q_current = np.dot(self.theta_k.T, np.dot(M_k, Phi_current))

                    episode_rewards += reward
                    State_k = State_next
                    steps += 1
                    step += 1
                    collision_count += np.array(simple_tag_utilities.count_agent_collisions(env))

                    if any(done):
                        episode_rewards = episode_rewards / steps
                        episode_losses = episode_losses / steps
                        #
                        statistic = [episode]
                        statistic.append(step)
                        statistic.extend([episode_rewards[i] for i in range(env.n)])
                        statistic.extend([episode_losses[i] for i in range(env.n)])
                        statistic.extend(collision_count.tolist())
                        statistics.add_statistics(statistic)

                        print("Episode:", episode, ",finished at:", steps, ',reward', reward)
                        # if episode % 5 == 0:
                        #     print("Episode:", episode, ",finished at:", steps, ',reward', reward)
                        episode_index.append(episode)
                        sample_index.append(step)
                        State_k = env.reset()
                        break

                if episode % self.checkpoint_interval == 0:
                    statistics.dump("{}_{}.csv".format(self.csv_filename_prefix,
                                                       episode))
                    # if episode >= self.checkpoint_interval:
                    #     os.remove("{}_{}.csv".format(self.csv_filename_prefix,
                    #                                  episode - self.checkpoint_interval))

            # self.episode_indexes.append(episode_index)
            # self.sample_indexes.append(sample_index)

        return self.Sigma, self.Mu, self.theta_k


# test
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='simple_tag_guided', type=str)
    parser.add_argument('--episodes', default=102, type=int)  # 500000
    parser.add_argument('--episodesTest', default=1002, type=int)  # 100000
    parser.add_argument('--render', default=True, action="store_true")
    parser.add_argument('--benchmark', default=False, action="store_true")
    parser.add_argument('--experiment_prefix', default="..",
                        help="directory to store all experiment data")
    parser.add_argument('--weights_filename_prefix', default='/save/tag-MAAKFSR_1v1',
                        help="where to store/load network weights")
    parser.add_argument('--csv_filename_prefix', default='../save/statistics-MAAKFSR_1v1',
                        help="where to store statistics")
    parser.add_argument('--checkpoint_interval', default=10,
                        help="how often to checkpoint")
    parser.add_argument('--testing', default=False, action="store_true",
                        help="reduces exploration substantially")
    parser.add_argument('--random_seed', default=2, type=int)
    parser.add_argument('--memory_size', default=10000, type=int)

    args = parser.parse_args()

    general_utilities.dump_dict_as_json(vars(args),
                                        args.experiment_prefix + "/save/run_parameters.json")

    # init env
    env = make_env(args.env, args.benchmark)
    _n_actions = [env.action_space[agentNumber].n for agentNumber in range(env.n)]
    state_sizes = [env.observation_space[i].shape[0] for i in range(env.n)]
    initial_theta = np.zeros((_n_actions[0] * (len(Mu[0]) + 1), 1))
    # Train
    Sigma, Mu, _theta = TrainAKFSR(env, Sigma, Mu, initial_theta, episodes=args.episodes,
                                          checkpoint_interval=args.checkpoint_interval).learn()

    # print("statistics: ", statistics)
    # statistics.dump(args.experiment_prefix + args.csv_filename_prefix + ".csv")

    episode_index_test = []
    sample_index_test = []
    episode_index_test_final = []
    sample_index_test_final = []
    Theta_total_test = []
    velocity_total_test = []
    Q_total_test = []
    m = 0

    statistics_header = ["episode"]
    statistics_header.append("steps")
    statistics_header.extend(["reward_{}".format(i) for i in range(env.n)])
    statistics_header.extend(["loss_{}".format(i) for i in range(env.n)])
    statistics_header.extend(["collisions_{}".format(i) for i in range(env.n)])
    print("Collecting statistics {}:".format(" ".join(statistics_header)))
    statistics = general_utilities.Time_Series_Statistics_Store(
        statistics_header)

    _episode_losses = 0
    # _episode_losses = np.reshape(_episode_losses, (2, 1))
    _episode_rewards = np.zeros(env.n)
    _collision_count = np.zeros(env.n)

    print("Starting TESTING Approach")
    for _episode in range(args.episodesTest):
        _State_k = env.reset()
        # State_k = np.reshape(State_k, (2, 1))  # 2*1
        # for t in range(number_sample_test):
        _done = [False, False]
        _step = 0
        while not any(_done):
            _step += 1
            for _agent in range(env.n):
                action_test, _phi_current, g = MAAKFSR(env, _State_k, Sigma, Mu).Policy_Logic(_agent)

                if args.render:
                    env.render()
                actions_ = []
                actions_onehot_ = []
                for i_ in range(env.n):
                    action = action_test
                    speed = 0.9 if env.agents[i_].adversary else 1
                    onehot_action = np.zeros(_n_actions[i_])
                    onehot_action[action] = speed
                    actions_onehot_.append(onehot_action)
                    actions_.append(action)
                # step
                _State_next, _reward, _done, _info = env.step(actions_onehot_)
                _Loss = np.dot(_phi_current.T, _theta) - _reward
                _Loss = np.sqrt(np.power(_Loss, 2))

                _episode_losses += _Loss.T

                for _i in range(env.n):
                    if _done[_i]:
                        _reward[_i] -= 50

                _State_next = np.reshape(_State_next[_agent], (state_sizes[_agent], 1))  # 10*1 or 12*1
                _State_k = _State_next
                _episode_rewards += _reward
                _collision_count += np.array(
                    simple_tag_utilities.count_agent_collisions(env))

                if any(_done):
                    _episode_rewards = _episode_rewards / _step
                    _episode_losses = _episode_losses / _step

                    statistic = [_episode]
                    statistic.append(_step)
                    statistic.extend([_episode_rewards[i] for i in range(env.n)])
                    statistic.extend([_episode_losses[i] for i in range(env.n)])
                    statistic.extend(_collision_count.tolist())
                    statistics.add_statistics(statistic)
                    if _episode % 25 == 0:
                        print(statistics.summarize_last())

                    print("Episode_test:", _episode, ",finished at:", _step, ',reward_test', _reward)
                    episode_index_test.append(_episode)
                    # sample_index_test.append(t)
                    break
            episode_index_test_final.append(episode_index_test)

        if _episode % args.checkpoint_interval == 0:
            statistics.dump("{}_{}.csv".format(args.csv_filename_prefix,
                                               _episode))
            if _episode >= args.checkpoint_interval:
                os.remove("{}_{}.csv".format(args.csv_filename_prefix,
                                             _episode - args.checkpoint_interval))

        # statistics.dump(args.experiment_prefix + args.csv_filename_prefix + ".csv")
    # print(episode_index_test_final)
    # print(sample_index_test_final)
