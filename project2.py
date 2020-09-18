import argparse
import json
import logging
import os
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import cityflow
import sys
import argparse
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf

class CityFlowEnvM(object):
    '''
    multi inersection cityflow environment
    '''
    def __init__(self,
                lane_phase_info,
                intersection_id,
                num_step=2000,
                thread_num=1,
                cityflow_config_file='example/config_1x2.json',
                cityflow_out_dir='replay'
                ):
        self.eng = cityflow.Engine(cityflow_config_file, thread_num=thread_num)
        self.num_step = num_step
        self.intersection_id = intersection_id # list, [intersection_id, ...]
        self.state_size = None
        self.lane_phase_info = lane_phase_info # "intersection_1_1"

        self.current_phase = {}
        self.current_phase_time = {}
        self.start_lane = {}
        self.end_lane = {}
        self.phase_list = {}
        self.phase_startLane_mapping = {}
        self.intersection_lane_mapping = {} #{id_:[lanes]}
        self.cityflow_out_dir = cityflow_out_dir

        for id_ in self.intersection_id:
            self.start_lane[id_] = self.lane_phase_info[id_]['start_lane']
            self.end_lane[id_] = self.lane_phase_info[id_]['end_lane']
            self.phase_startLane_mapping[id_] = self.lane_phase_info[id_]["phase_startLane_mapping"]

            self.phase_list[id_] = self.lane_phase_info[id_]["phase"]
            self.current_phase[id_] = self.phase_list[id_][0]
            self.current_phase_time[id_] = 0
        self.get_state() # set self.state_size
        
    def reset(self):
        self.eng.reset()

    def step(self, action):
        '''
        action: {intersection_id: phase, ...}
        '''
        for id_, a in action.items():
            if self.current_phase[id_] == a:
                self.current_phase_time[id_] += 1
            else:
                self.current_phase[id_] = a
                self.current_phase_time[id_] = 1
            self.eng.set_tl_phase(id_, self.current_phase[id_]) # set phase of traffic light
        self.eng.next_step()
        return self.get_state(), self.get_reward()

    def get_state(self):
        state =  {id_: self.get_state_(id_) for id_ in self.intersection_id}
        return state

    def get_state_(self, id_):
        state = self.intersection_info(id_)
        state_dict = state['start_lane_waiting_vehicle_count']
        sorted_keys = sorted(state_dict.keys())
        return_state = [state_dict[key] for key in sorted_keys] + [state['current_phase']]
        return self.preprocess_state(return_state)

    def log(self):
        if not os.path.exists(self.cityflow_out_dir):
            os.makedirs(self.cityflow_out_dir)
        self.eng.print_log(self.cityflow_out_dir + "/replay_")

    def intersection_info(self, id_):
        '''
        info of intersection 'id_'
        '''
        state = {}

        get_lane_vehicle_count = self.eng.get_lane_vehicle_count()
        get_lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()
        get_lane_vehicles = self.eng.get_lane_vehicles()
        vehicle_speed = self.eng.get_vehicle_speed()

        state['start_lane_vehicle_count'] = {lane: get_lane_vehicle_count[lane] for lane in self.start_lane[id_]}
        state['end_lane_vehicle_count'] = {lane: get_lane_vehicle_count[lane] for lane in self.end_lane[id_]}
        
        state['start_lane_waiting_vehicle_count'] = {lane: get_lane_waiting_vehicle_count[lane] for lane in self.start_lane[id_]}
        state['end_lane_waiting_vehicle_count'] = {lane: get_lane_waiting_vehicle_count[lane] for lane in self.end_lane[id_]}
        
        state['start_lane_vehicles'] = {lane: get_lane_vehicles[lane] for lane in self.start_lane[id_]}
        state['end_lane_vehicles'] = {lane: get_lane_vehicles[lane] for lane in self.end_lane[id_]}
        
        state['start_lane_speed'] = {lane: np.sum(list(map(lambda vehicle:vehicle_speed[vehicle], get_lane_vehicles[lane]))) / (get_lane_vehicle_count[lane]+1e-5) for lane in self.start_lane[id_]} # compute start lane mean speed
        state['end_lane_speed'] = {lane: np.sum(list(map(lambda vehicle:vehicle_speed[vehicle], get_lane_vehicles[lane]))) / (get_lane_vehicle_count[lane]+1e-5) for lane in self.end_lane[id_]} # compute end lane mean speed
        
        state['current_phase'] = self.current_phase[id_]
        state['current_phase_time'] = self.current_phase_time[id_]

        return state


    def preprocess_state(self, state):
        return_state = np.array(state)
        if self.state_size is None:
            self.state_size = len(return_state.flatten())
        return_state = np.reshape(return_state, [1, self.state_size])
        return return_state

    def get_reward(self):
        reward = {id_: self.get_reward_(id_) for id_ in self.intersection_id}
        return reward

    def get_reward_(self, id_):
        '''
        every agent/intersection's reward
        '''
        state = self.intersection_info(id_)
        start_lane_speed = state['start_lane_speed']
        reward = np.mean(list(start_lane_speed.values())) * 100
        return reward

    def get_score(self):
        score = {id_: self.get_score_(id_) for id_ in self.intersection_id}
        return score
    
    def get_score_(self, id_):
        state = self.intersection_info(id_)
        start_lane_waiting_vehicle_count = state['start_lane_waiting_vehicle_count']
        end_lane_waiting_vehicle_count = state['end_lane_waiting_vehicle_count']
        x = -1 * np.sum(list(start_lane_waiting_vehicle_count.values()) + list(end_lane_waiting_vehicle_count.values()))
        score = ( 1/(1 + np.exp(-1 * x)) )/self.num_step
        return score

class DQNAgent(object):
    def __init__(self,
            intersection_id,
            state_size=9,
            action_size=8,
            batch_size=32,
            phase_list=[],
            env=None
            ):
        self.env = env
        self.intersection_id = intersection_id
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=3000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001
        self.batch_size = batch_size
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()

        self.phase_list = phase_list

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(40, input_dim=self.state_size, activation='relu'))
        model.add(Dense(40, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_network(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)

    def remember(self, state, action, reward, next_state):
        action = self.phase_list.index(action) # index
        self.memory.append((state, action, reward, next_state))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def choose_action_(self, state):
        '''
        choose phase with max waiting count in its start lanes
        '''
        intersection_info = self.env.intersection_info(self.intersection_id)
        phase_waiting_count = self.phase_list.copy()
        for index, phase in enumerate(self.phase_list):
            phase_start_lane = self.env.phase_startLane_mapping[self.intersection_id][phase] # {0:['road_0_1_1_1',], 1:[]...}
            phase_start_lane_waiting_count = [intersection_info["start_lane_vehicle_count"][lane] for lane  in phase_start_lane] # [num1, num2, ...]
            sum_count = sum(phase_start_lane_waiting_count)
            phase_waiting_count[index] = sum_count

        action = np.argmax(phase_waiting_count)
        return action

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        states = []
        q_targets = []
        for state, action, reward, next_state in minibatch:
            target = (reward + self.gamma *
                      np.amax(self.target_model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target # action is a action_list index

            states.append(state)
            q_targets.append(target_f)

            # self.model.fit(state, target_f, epochs=1, verbose=0)

        states = np.reshape(np.array(states), [-1, self.state_size])
        q_targets = np.reshape(np.array(q_targets), [-1, self.action_size])
        self.model.fit(state, target_f, epochs=2, verbose=0) # batch training

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
        # print("model saved:{}".format(name))

class DDQNAgent(object):
    def __init__(self,
            intersection_id,
            state_size=9,
            action_size=8,
            batch_size=32,
            phase_list=[],
            env=None
            ):
        self.env = env
        self.intersection_id = intersection_id
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=3000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001
        self.batch_size = batch_size
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()

        self.phase_list = phase_list

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(40, input_dim=self.state_size, activation='relu'))
        model.add(Dense(40, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_network(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)

    def remember(self, state, action, reward, next_state):
        action = self.phase_list.index(action) # index
        self.memory.append((state, action, reward, next_state))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def choose_action_(self, state):
        '''
        choose phase with max waiting count in its start lanes
        '''
        intersection_info = self.env.intersection_info(self.intersection_id)
        phase_waiting_count = self.phase_list.copy()
        for index, phase in enumerate(self.phase_list):
            phase_start_lane = self.env.phase_startLane_mapping[self.intersection_id][phase] # {0:['road_0_1_1_1',], 1:[]...}
            phase_start_lane_waiting_count = [intersection_info["start_lane_vehicle_count"][lane] for lane  in phase_start_lane] # [num1, num2, ...]
            sum_count = sum(phase_start_lane_waiting_count)
            phase_waiting_count[index] = sum_count

        action = np.argmax(phase_waiting_count)
        return action

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
        # print("model saved:{}".format(name))

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        states = []
        q_targets = []
        for state, action, reward, next_state in minibatch:
            # compute target value, this is the key point of Double DQN

            # target
            target_f = self.model.predict(state)

            # choose best action for next state using current Q network
            actions_for_next_state = np.argmax(self.model.predict(next_state)[0])

            # compute target value
            target = (reward + self.gamma *
                      self.target_model.predict(next_state)[0][actions_for_next_state] )

            target_f[0][action] = target

            states.append(state)
            q_targets.append(target_f)
            # self.model.fit(state, target_f, epochs=1, verbose=0) # train on single sample

        states = np.reshape(np.array(states), [-1, self.state_size])
        q_targets = np.reshape(np.array(q_targets), [-1, self.action_size])
        self.model.fit(state, target_f, epochs=1, verbose=0) # batch training

        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon_decay

class MDQNAgent(object):
    def __init__(self, config):
        self.intersection = config['intersection_id']
        self.agents = {}
        self.make_agents(config['intersection_id'], 
                         config['state_size'], 
                         config['batch_size'], 
                         config['phase_list'], 
                         config['env'])

    def make_agents(self, intersection, state_size, batch_size, phase_list, env):
        for id_ in self.intersection:
            self.agents[id_] = DQNAgent(id_,
                                state_size=state_size,
                                action_size=len(phase_list[id_]),
                                batch_size=batch_size,
                                phase_list=phase_list[id_],
                                env=env
                                )

    def update_target_network(self):
        for id_ in self.intersection:
            self.agents[id_].update_target_network()

    def remember(self, state, action, reward, next_state):
        for id_ in self.intersection:
            self.agents[id_].remember(state[id_],
                                    action[id_],
                                    reward[id_],
                                    next_state[id_])

    def choose_action(self, state):
        action = {}
        for id_ in self.intersection:
            action[id_] = self.agents[id_].choose_action_(state[id_])
        return action

    def replay(self):
        for id_ in self.intersection:
            self.agents[id_].replay()

    def load(self, name):
        for id_ in self.intersection:
            #assert os.path.exists(name + '.' + id_), "Wrong checkpoint, file not exists!"
            self.agents[id_].load(name + '.' + id_)

    def save(self, name):
        for id_ in self.intersection:
            self.agents[id_].save(name + '.' + id_)

class MDDQNAgent(object):
    def __init__(self, config):
        self.intersection = config['intersection_id']
        self.agents = {}
        self.make_agents(config['intersection_id'], 
                         config['state_size'], 
                         config['batch_size'], 
                         config['phase_list'], 
                         config['env'])

    def make_agents(self, intersection, state_size, batch_size, phase_list, env):
        for id_ in self.intersection:
            self.agents[id_] = DDQNAgent(id_,
                                state_size=state_size,
                                action_size=len(phase_list[id_]),
                                batch_size=batch_size,
                                phase_list=phase_list[id_],
                                env=env
                                )

    def update_target_network(self):
        for id_ in self.intersection:
            self.agents[id_].update_target_network()

    def remember(self, state, action, reward, next_state):
        for id_ in self.intersection:
            self.agents[id_].remember(state[id_],
                                    action[id_],
                                    reward[id_],
                                    next_state[id_])

    def choose_action(self, state):
        action = {}
        for id_ in self.intersection:
            action[id_] = self.agents[id_].choose_action_(state[id_])
        return action

    def replay(self):
        for id_ in self.intersection:
            self.agents[id_].replay()

    def load(self, name):
        for id_ in self.intersection:
            #assert os.path.exists(name + '.' + id_), "Wrong checkpoint, file not exists!"
            self.agents[id_].load(name + '.' + id_)

    def save(self, name):
        for id_ in self.intersection:
            self.agents[id_].save(name + '.' + id_)

def parse_roadnet(roadnetFile):
    roadnet = json.load(open(roadnetFile))
    lane_phase_info_dict ={}

    # many intersections exist in the roadnet and virtual intersection is controlled by signal
    for intersection in roadnet["intersections"]:
        if intersection['virtual']:
            continue
        lane_phase_info_dict[intersection['id']] = {"start_lane": [],
                                                     "end_lane": [],
                                                     "phase": [],
                                                     "phase_startLane_mapping": {},
                                                     "phase_roadLink_mapping": {}}
        road_links = intersection["roadLinks"]

        start_lane = []
        end_lane = []
        roadLink_lane_pair = {ri: [] for ri in
                              range(len(road_links))}  # roadLink includes some lane_pair: (start_lane, end_lane)

        for ri in range(len(road_links)):
            road_link = road_links[ri]
            for lane_link in road_link["laneLinks"]:
                sl = road_link['startRoad'] + "_" + str(lane_link["startLaneIndex"])
                el = road_link['endRoad'] + "_" + str(lane_link["endLaneIndex"])
                start_lane.append(sl)
                end_lane.append(el)
                roadLink_lane_pair[ri].append((sl, el))

        lane_phase_info_dict[intersection['id']]["start_lane"] = sorted(list(set(start_lane)))
        lane_phase_info_dict[intersection['id']]["end_lane"] = sorted(list(set(end_lane)))

        for phase_i in range(1, len(intersection["trafficLight"]["lightphases"])):
        # for phase_i in range(0, len(intersection["trafficLight"]["lightphases"])): # change for test_roadnet_1*1.json file, intersection id: intersection_1*1
            p = intersection["trafficLight"]["lightphases"][phase_i]
            lane_pair = []
            start_lane = []
            for ri in p["availableRoadLinks"]:
                lane_pair.extend(roadLink_lane_pair[ri])
                if roadLink_lane_pair[ri][0][0] not in start_lane:
                    start_lane.append(roadLink_lane_pair[ri][0][0])
            lane_phase_info_dict[intersection['id']]["phase"].append(phase_i)
            lane_phase_info_dict[intersection['id']]["phase_startLane_mapping"][phase_i] = start_lane
            lane_phase_info_dict[intersection['id']]["phase_roadLink_mapping"][phase_i] = lane_pair

    return lane_phase_info_dict

def getAgent(config):
    if config['algo'] == 'DQN':
        return MDDQNAgent(config)
    elif config['algo'] == 'DDQN':
        return MDQNAgent(config)
    else:
        print ("Wrong input algo")
        sys.exit(0)

def trainModel2(config):
    print ("Training model")
    total_step = 0
    epoch_rewards = {id_:[] for id_ in config['intersection_id']}
    epoch_scores = {id_:[] for id_ in config['intersection_id']}
    env = config['env']
    Agent = config['Agent']
    phase_list = config['phase_list']
    learning_start = 300
    update_model_freq = config['batch_size']//3
    update_target_model_freq = 300//config['phase_step']

    with tqdm(total=config['epochs'] * config['num_step']) as pbar:
        for i in range(config['epochs']):
            # print("episode: {}".format(i))
            env.reset()
            state = env.get_state()

            epoch_length = 0
            epoch_reward = {id_:0 for id_ in config['intersection_id']} # for every agent
            epoch_score = {id_:0 for id_ in config['intersection_id']} # for everg agent
            while epoch_length < config['num_step']:
                
                action = Agent.choose_action(state) # index of action
                action_phase = {}
                for id_, a in action.items():
                    action_phase[id_] = phase_list[id_][a]
                
                next_state, reward = env.step(action_phase) # one step
                score = env.get_score()

                # consistent time of every phase
                for _ in range(config['phase_step'] - 1):
                    next_state, reward_ = env.step(action_phase)
                    score_ = env.get_score()
                    for id_ in config['intersection_id']:
                        reward[id_] += reward_[id_]
                        score[id_] += score_[id_]

                for id_ in config['intersection_id']:
                    reward[id_] /= config['phase_step']
                    score[id_] /= config['phase_step']

                for id_ in config['intersection_id']:
                    epoch_reward[id_] += reward[id_]
                    epoch_score[id_] += score[id_]

                epoch_length += 1
                total_step += 1
                pbar.update(1)

                # store to replay buffer
                if epoch_length > learning_start:
                    Agent.remember(state, action_phase, reward, next_state)

                state = next_state

                # training
                if epoch_length > learning_start and total_step % update_model_freq == 0 :
                    if len(Agent.agents[config['intersection_id'][0]].memory) > args.batch_size:
                        Agent.replay()

                # update target Q netwark
                if epoch_length > learning_start and total_step % update_target_model_freq == 0 :
                    Agent.update_target_network()

                # logging.info("\repisode:{}/{}, total_step:{}, action:{}, reward:{}"
                #             .format(i+1, EPISODES, total_step, action, reward))
                print_reward = {'_'.join(k.split('_')[1:]):v for k, v in reward.items()}
                pbar.set_description(
                    "t_st:{}, epi:{}, st:{}, r:{}".format(total_step, i+1, epoch_length, print_reward))

            # compute episode mean reward
            for id_ in config['intersection_id']:
                epoch_reward[id_] /= config['num_step']
            
            # save episode rewards
            for id_ in config['intersection_id']:
                epoch_rewards[id_].append(epoch_reward[id_])
                epoch_scores[id_].append(epoch_score[id_])
            
            print_episode_reward = {'_'.join(k.split('_')[1:]):v for k, v in epoch_reward.items()}
            print_episode_score = {'_'.join(k.split('_')[1:]):v for k, v in epoch_score.items()}
            print('\n')
            print("Episode:{}, Mean reward:{}, Score: {}".format(i+1, print_episode_reward, print_episode_score))

            # save model
            if (i + 1) % config['save_freq'] == 0:
                if config['algo'] == 'MDQN':
                    # Magent.save(model_dir + "/{}-ckpt".format(args.algo), i+1)
                    Magent.save(config['model_out'] + "/{}-{}.h5".format(config['algo'], i+1))
                    
                # save reward to file
                df = pd.DataFrame(epoch_rewards)
                df.to_csv(config['result_out'] + '/rewards.csv', index=None)

                df = pd.DataFrame(epoch_scores)
                df.to_csv(config['result_out'] + '/scores.csv', index=None)

                # save figure
                plot_data_lists([epoch_rewards[id_] for id_ in config['intersection_id']], config['intersection_id'], figure_name=config['result_out'] + '/rewards.pdf')
                plot_data_lists([epoch_scores[id_] for id_ in config['intersection_id']], config['intersection_id'], figure_name=config['result_out'] + '/scores.pdf')
    
    df = pd.DataFrame(epoch_rewards)
    df.to_csv(config['result_out'] + "/rewards.csv", index=None)

    df = pd.DataFrame(epoch_scores)
    df.to_csv(config['result_out'] + "/scores.csv", index=None)

    plot_data_lists([epoch_rewards[id_] for id_ in config['intersection_id']], config['intersection_id'], figure_name=config['result_out'] + '/rewards.pdf')
    plot_data_lists([epoch_scores[id_] for id_ in config['intersection_id']], config['intersection_id'], figure_name=config['result_out'] + '/scores.pdf')

def trainModel(config):
    epochs = config['epochs']
    total_step = 0
    env = config['env']
    Agent = config['Agent']
    phase_list = config['phase_list']
    learning_start = 300
    update_model_freq = config['batch_size']//3
    update_target_model_freq = 300//config['phase_step']
    epoch_rewards = {id_:[] for id_ in config['intersection_id']}
    epoch_scores = {id_:[] for id_ in config['intersection_id']}
    num_step = config['num_step']
    phase_step = config['phase_step']
    intersection_id = config['intersection_id']

    with tqdm(total=epochs * num_step) as pbar:
        for epoch in range(epochs):
            env.reset()
            state = env.get_state()

            epoch_length = 0
            epoch_reward = {id_:0 for id_ in intersection_id} # for every agent
            epoch_score = {id_:0 for id_ in intersection_id} # for everg agent 

            while epoch_length < num_step:
                action = Agent.choose_action(state)
                action_phase = {}
                for id_, a in action.items():
                    action_phase[id_] = phase_list[id_][a]

                next_state, reward_ = env.step(action_phase)
                score_ = env.get_score()

                for _ in range(phase_step - 1):
                    next_state, reward = env.step(action_phase)
                    score = env.get_score()
                    for id_ in intersection_id:
                        reward[id_] = reward[id_] + reward_[id_]
                        score[id_] = score[id_] + score_[id_]

                for id_ in intersection_id:
                    reward[id_] = reward[id_] / phase_step
                    score[id_] = score[id_] / phase_step

                for id_ in intersection_id:
                    epoch_reward[id_] += reward[id_]
                    epoch_score[id_] += score[id_]

                epoch_length = epoch_length + 1
                total_step = total_step + 1
                pbar.update(1)

                if epoch_length > learning_start:
                    Agent.remember(state, action_phase, reward, next_state)

                state = next_state

                if epoch_length > learning_start and total_step % update_model_freq == 0 :
                    if len(Agent.agents[intersection_id[0]].memory) > config['batch_size']:
                        Agent.replay()

                if epoch_length > learning_start and total_step % update_target_model_freq == 0 :
                    Agent.update_target_network()

                print_reward = {'_'.join(k.split('_')[1:]):v for k, v in reward.items()}
                pbar.set_description("t_st:{}, epi:{}, st:{}, r:{}".format(total_step, epoch+1, epoch_length, print_reward))

            for id_ in intersection_id:
                epoch_reward[id_] = epoch_reward[id_] / num_step

            for id_ in intersection_id:
                epoch_rewards[id_].append(epoch_reward[id_])
                epoch_scores[id_].append(epoch_score[id_])

            print_episode_reward = {'_'.join(k.split('_')[1:]):v for k, v in epoch_reward.items()}
            print_episode_score = {'_'.join(k.split('_')[1:]):v for k, v in epoch_score.items()}
            print('\n')
            print("Episode:{}, Mean reward:{}, Score: {}".format(epoch +1, print_episode_reward, print_episode_score))

def inferModel(config):
    config['saveReplay'] = True
    Agent = config['Agent']
    env = config['env']

    Agent.load(config['model_out'] + '/' + config['algo'] + '_' + str(config['epochs']) + '.h5')

    epoch_reward = {id_:[] for id_ in config['intersection_id']}
    epoch_score = {id_:[] for id_ in config['intersection_id']}

    env.reset()

    state = env.get_state()

    for i in range(config['num_step']):
        action = Agent.choose_action(state)
        action_phase = {}

        for id_, a in action.items():
            action_phase[id_] = config['phase_list'][id_][a]

        next_state, reward = env.step(action_phase)
        score = env.get_score()

        for _ in range(config['phase_step'] - 1):
            next_state, reward_ = env.step(action_phase)
            score_ = env.get_score()
            for id_ in config['intersection_id']:
                reward[id_] += reward_[id_]
                score[id_] += score_[id_]

        for id_ in config['intersection_id']:
            reward[id_] /= config['phase_step']
            score[id_] /= config['phase_step']

        for id_ in config['intersection_id']:
            epoch_reward[id_].append(reward[id_])
            epoch_score[id_].append(score[id_])

        state = next_state

def plot_data_lists(data_list, label_list, length=10, height=6, x_label='x', y_label='y', label_fsize=14, save=True, figure_name='temp'):
    import matplotlib
    if save:
        matplotlib.use('PDF')
    import matplotlib.pyplot as plt
    

    fig, ax = plt.subplots(figsize=(length, height))
    ax.grid(True)

    for data, label in zip(data_list, label_list):
        ax.plot(data, label=label)
    
    ax.plot()
    ax.set_xlabel(x_label, fontsize=label_fsize)
    ax.set_ylabel(y_label, fontsize=label_fsize)
    ax.legend()
    ax.grid(True)
    
    if save:
        plt.savefig(figure_name)
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Input config file')
    args = parser.parse_args()

    infer = False

    config = json.load(open(args.config))

    cityflow_config = json.load(open(config['cityflow_config_file']))
    cityflow_config['saveReplay'] = True
    roadnetFile = cityflow_config['dir'] + cityflow_config['roadnetFile']
    json.dump(cityflow_config, open(config["cityflow_config_file"], 'w'))

    config["lane_phase_info"] = parse_roadnet(roadnetFile)
    intersection_id = list(config['lane_phase_info'].keys())
    config["intersection_id"] = intersection_id
    phase_list = {id_:config["lane_phase_info"][id_]["phase"] for id_ in intersection_id}
    config["phase_list"] = phase_list

    if not os.path.exists(config['model_out']):
        os.makedirs(config['model_out'])
    if not os.path.exists(config['result_out']):
        os.makedirs(config['result_out'])

    env = CityFlowEnvM(config["lane_phase_info"],
                       intersection_id,
                       num_step=config["num_step"],
                       thread_num=config["thread_num"],
                       cityflow_config_file=config["cityflow_config_file"]
                       )

    config["env"] = env

    config["state_size"] = env.state_size

    config['Agent'] = getAgent(config)

    if not infer:
        trainModel(config)
    else:
        inferModel(config)
    

if __name__ == '__main__':
    main()