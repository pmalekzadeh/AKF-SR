"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
    env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""


def make_env(scenario_name, benchmark=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation, done_callback=scenario.done)
    return env


# # #
# import numpy as np
# import gym
# from memory import Memory
# from dqn import DQN
# import random
# import simple_tag_utilities
# memory_size = 10000
# batch_size = 32
# env = make_env('simple_tag_guided_2v1')
#
# #
# # episode_losses = np.zeros(env.n)
# # episode_rewards = np.zeros(env.n)
# # collision_count = np.zeros(env.n)
# # steps = 0
# #
# n_actions = [env.action_space[i].n for i in range(env.n)]
# # print(n_actions[0])
# class MAMMKTD:
#     def __init__(self, env, observation, Sigma, Mu, eps_greedy=0.5, learning_rate=0.001):
#         self.observation = observation
#         self.Sigma = Sigma
#         self.Mu = Mu
#         # self.states = states
#         self.env = env
#         # self.n_actions = [self.env.action_space[i].n for i in range(self.env.n)]
#         self.n_actions = [env.action_space[i].n for i in range(env.n)]
#         self.state_sizes = [self.env.observation_space[i].shape[0] for i in range(env.n)]
#         # self.eval_network = self.build_network()
#         self.eps_greedy = eps_greedy
#         self.learning_rate = learning_rate
#         self.Theta = np.zeros((self.n_actions[0]*len(self.Mu[0]) + 1, 1))
#
#
#     def choose_action(self):
#         return random.randrange(self.n_actions[0])
#
# from kalman_algorithms.MMKTD_Multi_initials import Mu, Sigma
# observation = env.reset()
# # print(TrainMMKTD(env, Mu).Q)
# print(MAMMKTD(env, observation, Sigma, Mu).n_actions)
#
# print(MAMMKTD(env, observation, Sigma, Mu).choose_action())
# actions_onehot = []
# actions = []
# for i in range(env.n):
#     action = MAMMKTD(env, observation, Sigma, Mu).choose_action()
#     speed = 0.9 if env.agents[i].adversary else 1
#     onehot_action = np.zeros(n_actions[i])
#     onehot_action[action] = speed
#     actions_onehot.append(onehot_action)
#     actions.append(action)
# print(actions)
# states_next, rewards, done, info = env.step(actions_onehot)
# print(states_next)
# state_sizes = [env.observation_space[i].shape[0] for i in range(env.n)]
# print(state_sizes)
# memories = [Memory(memory_size) for i in range(env.n)]
# epsilon_greedy = [0.5 for i in range(env.n)]
# dqns = [DQN(n_actions[i], state_sizes[i], eps_greedy=epsilon_greedy[i])
#         for i in range(env.n)]
#
# for episode in range(30):  # '--episodes', default=10000,
#     states = env.reset()
#     # Sample State:
#     # [array([0., 0., -0.95826908, 0.63890378, 1.49756682,-0.33700142, 0.50534522, -0.41117881, 0., 0.,0., 0.]),
#     # array([0., 0., 0.53929774, 0.30190236, -1.49756682, 0.33700142, -0.9922216, -0.07417739, 0., 0.]),
#     # array([0., 0., -0.45292386, 0.22772497, -0.50534522, 0.41117881, 0.9922216, 0.07417739, 0., 0.])]
#     #
#     episode_losses = np.zeros(env.n)
#     episode_rewards = np.zeros(env.n)
#     collision_count = np.zeros(env.n)
#     steps = 0
#
#     print("#####################")
#     print("states: ", states)
#     for i in range(350):
#         actions = []
#         actions_onehot = []
#         for i in range(env.n):
#             action = dqns[i].choose_action(states[i])
#             speed = 0.9 if env.agents[i].adversary else 1
#
#             onehot_action = np.zeros(n_actions[i])
#             onehot_action[action] = speed
#             actions_onehot.append(onehot_action)
#             actions.append(action)
#             print("#####################")
#         print("actions: ", actions)
#         print("actions_onehot: ", actions_onehot)
#         # step
#         states_next, rewards, done, info = env.step(actions_onehot)
#         print("done: ", done)
#         print("rewards: ", rewards)
#         print("next states: ", states_next)
#
#         size = memories[0].pointer
#         print("size: ", size)
#         batch = random.sample(range(size), size) if size < batch_size else random.sample(
#             range(size), batch_size)
#         print("batch: ", batch)
#
#         for i in range(env.n):
#             if done[i]:
#                 rewards[i] -= 50
#
#             memories[i].remember(states[i], actions[i],
#                                  rewards[i], states_next[i], done[i])
#
#             if memories[i].pointer > batch_size * 10:
#                 history = dqns[i].learn(*memories[i].sample(batch))
#                 episode_losses[i] += history.history["loss"][0]
#             else:
#                 episode_losses[i] = -1
#         states = states_next
#         episode_rewards += rewards
#         collision_count += np.array(
#             simple_tag_utilities.count_agent_collisions(env))
#         # reset states if done
#         if any(done):
#             print("done Happened for BREAK: ", done)
#             episode_rewards = episode_rewards / steps
#             episode_losses = episode_losses / steps
#
#             break
#         print("collision_count", collision_count)
# env = gym.wrappers.Monitor(env, "./vid", video_callable=lambda episode_id: True,force=True)
#
# from multiagent.policy import InteractivePolicy
# # env.configure(remotes=1)
# # observation_n = env.reset()
# print(env.n)
# print(np.array([env.world.dim_p + env.world.dim_c, 1]))
# print(env.action_space)
# print(env.world.dim_c)
# #
# # # import numpy as np
# # # env.reset()
# # # # env.step()
# # # while True:
# # #     env.render()
# # #
# for i_episode in range(100):
#     observation = env.reset()
#     for t in range(10000):
#         env.render()
#         print(observation)
#         # action = env.action_space.sample()
#         # action = np.zeros([env.world.dim_p + env.world.dim_c, 1])
#         # action_n = env.action_space(action)
#         n_actions = [env.action_space[i].n for i in range(env.n)]
#         observation, reward, done, info = env.step(n_actions)
#         # if done:
#         #     print("{} timesteps taken for the episode".format(t + 1))
#         #     break
#         # act_n = []
#         # for i, policy in enumerate(policies):
#         #     act_n.append(policy.action(obs_n[i]))
#         # # step environment
#         # obs_n, reward_n, done_n, _ = env.step(act_n)
#         # render all agent views
#         # env.render()