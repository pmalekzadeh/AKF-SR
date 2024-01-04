from mpmath import *
import numpy as np
import random
from make_env import make_env
from numpy import inf

mp.dps = 300

# init env
# env = make_env("simple_tag_guided_1v2", False)
# n_actions = [env.action_space[i].n for i in range(env.n)]
# state_sizes = [env.observation_space[i].shape[0] for i in range(env.n)]
# print(state_sizes)

class MAAKFSR:
    def __init__(self, env, observation, Sigma, Mu):
        self.observation = observation
        self.Sigma = Sigma
        self.Mu = Mu
        # self.states = states
        self.env = env
        self.n_actions = [self.env.action_space[i].n for i in range(self.env.n)]
        self.state_sizes = [self.env.observation_space[i].shape[0] for i in range(env.n)]
        self.MaxActions = np.prod(self.n_actions)
        # self.learning_rate = learning_rate


    def choose_action(self):
        # p = np.random.random()
        # if p < self.eps_greedy:
        #     action_probs = self.eval_network.predict(state[np.newaxis, :])
        #     return np.argmax(action_probs[0])
        # else:
        return random.randrange(self.n_actions[0])

    def PHI_function(self, observation):
        S = observation
        L = len(self.Mu[0])
        phi_F = []
        phi_list_F = []
        # previous_Sigma_n = 0
        for n in range(self.env.n):
            phi_list = []
            phi_list.append(1)
            for i in range(L):
                fg = np.reshape(self.Mu[n][i], (self.state_sizes[n], 1))  # 10*1
                for el in self.Sigma[n][:, :, i]:
                    el = np.array(el)
                    if np.any(el == -inf) or np.any(el == inf):
                        print("I AM HERE MAN .......")
                        # print("el: ", el)
                        self.Sigma[n][:, :, i] = np.zeros((self.state_sizes[n], self.state_sizes[n]))
                        break
                    # self.Sigma[n][:, :, i] = previous_Sigma_n
                    break
                # previous_Sigma_n = self.Sigma[n][:, :, i]
                # print("self.Sigma[n][:, :, i]: ", self.Sigma[n][:, :, i])
                Sigma_inverse = np.linalg.pinv(self.Sigma[n][:, :, i])

                A = np.dot((S[n] - fg).T, Sigma_inverse)
                append_value = np.exp(-.5 * np.dot(A, (S[n] - fg))[0, 0],  dtype=np.float128)

                # print("append_value: ", append_value)
                if (append_value == -inf) or (append_value == inf) or (append_value == np.nan):
                    append_value = 0
                    # print("THIS_append_value: ", append_value)
                phi_list.append(append_value)

            phi_F.append(phi_list)
        return phi_F

    def Policy_Logic(self, agents):
        # H_k = np.zeros((self.n_actions[agent], self.n_actions[agent]*self.Mu[0]))
        Buffer = []
        bufferNotFull = True
        # a_max_final = []
        # a_max_test_final = []
        memory_dict = {}
        next_memory_dict = {}
        action_memory = []
        A = []
        Action_test = []
        # phi_previous = 0
        phi_list_previous = 0
        phi_list_next_previous = 0
        # phi_next_previous = 0
        while bufferNotFull:
            features_list_ = []
            features_next_list = []
            actions = []
            actions_onehot = []
            for i in range(self.env.n):
                action = self.choose_action()
                speed = 0.9 if self.env.agents[i].adversary else 1
                onehot_action = np.zeros(self.n_actions[i])
                onehot_action[action] = speed
                actions_onehot.append(onehot_action)
                actions.append(action)

            if actions not in Buffer:
                Buffer.append(actions)
                # step
                states_next, rewards, done, info = self.env.step(actions_onehot)
                # phi, phi_list = self.PHI_function(self.observation)    # Prev
                phi_list = self.PHI_function(self.observation)
                if (phi_list == -inf) or (phi_list == inf):
                    phi_list = phi_list_previous
                phi_list_previous = phi_list

                phi_list_next = self.PHI_function(states_next[agents].reshape(self.state_sizes[agents], 1))
                if (phi_list_next == -inf) or (phi_list_next == inf):
                    phi_list_next = phi_list_next_previous
                phi_list_next_previous = phi_list_next

                for el in actions_onehot[agents]:
                    if el == 0:
                        features_list_.extend([0]*(len(self.Mu[0]) + 1))
                        features_next_list.extend([0]*(len(self.Mu[0]) + 1))
                    else:
                        # action_number = actions_onehot[agent].index(el)
                        if (phi_list[agents] == -inf) or (phi_list[agents] == inf):
                            phi_list[agents] = [0]*(len(self.Mu[0]) + 1)
                        features_list_.extend(phi_list[agents])
                        if (phi_list_next[agents] == -inf) or (phi_list_next[agents] == inf):
                            phi_list_next[agents] = [0]*(len(self.Mu[0]) + 1)
                        features_next_list.extend(phi_list_next[agents])

                memory_dict[str(actions)] = features_list_
                next_memory_dict[str(actions)] = features_next_list
                action_memory.append(actions)


            if len(Buffer) >= self.MaxActions:
                bufferNotFull = False

        # print("action memory: ", action_memory)
        # print("Length of action memory: ", len(action_memory))
        B = []
        H_k_Buffer = []
        for feature in memory_dict.values():
            for feature_next in next_memory_dict.values():
                H_k = [a_i - .95 * b_i for a_i, b_i in zip(feature, feature_next)]
                if len(H_k_Buffer) == 1:
                    g_total = np.vstack((H_k_Buffer[-1], H_k))
                if len(H_k_Buffer) > 1:
                    g_total = np.vstack((g_total, H_k))
                H_k_Buffer.append(H_k)
                C = np.reshape(H_k, (1, self.n_actions[agents]*(len(self.Mu[0]) + 1)))  # 1*50
                B.append(np.dot(C, C.T))

        a_final_max = B.index(max(B))
        g_return = g_total[a_final_max, :]
        g_final = np.reshape(g_return, (self.n_actions[agents]*(len(self.Mu[0]) + 1), 1))

        action_idx = int(a_final_max/(np.power(5, self.env.n)))
        # print("action_idx: ", action_idx)
        # print("action_memory: ", action_memory, len(action_memory))
        Final_action = action_memory[action_idx]
        phi_current = memory_dict[str(Final_action)]
        phi_current = np.reshape(phi_current, (self.n_actions[agents]*(len(self.Mu[0]) + 1), 1))

        return Final_action, phi_current, g_final


# Test
# from kalman_algorithms.MAMMKTD_simple1v1_initials import Mu, Sigma
# from kalman_algorithms.MAMMKTD_1v2_initials import Mu, Sigma
#
# env = make_env("simple_tag_guided_1v2", False)
# # env = make_env("simple_tag_guided", False)
# n_actions = [env.action_space[i].n for i in range(env.n)]
# state_sizes = [env.observation_space[i].shape[0] for i in range(env.n)]
# observation = env.reset()
#
# for agent in range(env.n):
#     print("######################################")
#     g_final, phi_current, Final_action = MAMMKTD(env, observation, Sigma, Mu).Policy_Logic(agent)
#     print("g_final SHAPE: ", g_final.shape)
#     print("g_final: ", g_final)
#     print("phi_current SHAPE: ", phi_current.shape)
#     print("phi_current: ", phi_current)
#     print("Final_action: ", Final_action)
#
#
#
# features_list = []
# features_next_list = []
# actions = []
# actions_onehot = []
# for i in range(env.n):
#     action = MAMMKTD(env, observation, Sigma, Mu).choose_action()
#     print(action)
#     speed = 0.9 if env.agents[i].adversary else 1
#     onehot_action = np.zeros(n_actions[i])
#     onehot_action[action] = speed
#     actions_onehot.append(onehot_action)
#     actions.append(action)
#
# print(actions_onehot)
# print(actions)
#
# State_next, reward, done, info = env.step(actions_onehot)
# for agent in range(env.n):
#     print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
#     print(MAMMKTD(env, State_next, Sigma, Mu).Policy_Logic(agent))
