import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pandas as pd
import json
import time
import sys
import os
import re
import yaml
from collections import defaultdict

# Add mcts to path
sys.path.append('/root/mcts')
try:
    from config_sdk import ConfigSDK, ConfigSDKError
except ImportError:
    print("Warning: Could not import config_sdk")

class AigisEnv(gym.Env):
    def __init__(self, booted=True, act_importance=53):
        self.sdk = ConfigSDK(base_url="http://192.168.0.96:8321/")
        self.action_max_file = "/root/Athena/action.max.yaml"
        
        # Keep internal actions list
        self._internal_actions = [
            'CORE_PEER_GOSSIP_STATE_BLOCKBUFFERSIZE',
            'CORE_PEER_GOSSIP_PUBLISHCERTPERIOD',
            'CORE_PEER_GOSSIP_PROPAGATEITERATIONS',
            'PreferredMaxBytes',
            'CORE_PEER_KEEPALIVE_MININTERVAL',
            'CORE_PEER_DISCOVERY_AUTHCACHEMAXSIZE',
            'CORE_PEER_DELIVERYCLIENT_CONNTIMEOUT',
            'CORE_PEER_KEEPALIVE_CLIENT_INTERVAL',
            'CORE_PEER_CLIENT_CONNTIMEOUT',
            # 'CORE_LEDGER_TOTALQUERYLIMIT',
            'CORE_PEER_GOSSIP_SENDBUFFSIZE',
            'CORE_PEER_KEEPALIVE_CLIENT_TIMEOUT',
            'CORE_PEER_GOSSIP_MAXPROPAGATIONBURSTLATENCY',
            'CORE_PEER_GOSSIP_REQUESTWAITTIME',
            'CORE_PEER_GOSSIP_STATE_MAXRETRIES',
            'CORE_PEER_KEEPALIVE_DELIVERYCLIENT_TIMEOUT',
            'ORDERER_GENERAL_KEEPALIVE_SERVERTIMEOUT',
            'CORE_PEER_GOSSIP_PUBLISHSTATEINFOINTERVAL',
            # 'CORE_CHAINCODE_EXECUTETIMEOUT',
            'CORE_PEER_GOSSIP_MAXBLOCKCOUNTTOSTORE',
            'ORDERER_GENERAL_KEEPALIVE_SERVERMININTERVAL',
            'CORE_PEER_GOSSIP_REQUESTSTATEINFOINTERVAL',
            'CORE_PEER_GOSSIP_ALIVEEXPIRATIONTIMEOUT',
            'CORE_PEER_GOSSIP_STATE_CHECKINTERVAL',
            'CORE_PEER_KEEPALIVE_DELIVERYCLIENT_INTERVAL',
            'CORE_PEER_GOSSIP_RECONNECTINTERVAL',
            'CORE_PEER_GOSSIP_PULLPEERNUM',
            'CORE_PEER_GOSSIP_STATE_BATCHSIZE',
            'CORE_PEER_GOSSIP_RECVBUFFSIZE',
            'CORE_PEER_GOSSIP_RESPONSEWAITTIME',
            'CORE_PEER_GOSSIP_PROPAGATEPEERNUM',
            'CORE_PEER_GOSSIP_PULLINTERVAL',
            'CORE_PEER_GOSSIP_STATE_RESPONSETIMEOUT',
            'AbsoluteMaxBytes',
            'CORE_PEER_GOSSIP_MAXPROPAGATIONBURSTSIZE',
            'BatchTimeout',
            'MaxMessageCount',
            'ORDERER_GENERAL_AUTHENTICATION_TIMEWINDOW',
            'ORDERER_GENERAL_CLUSTER_SENDBUFFERSIZE',
            'ORDERER_GENERAL_KEEPALIVE_SERVERINTERVAL',
            'ORDERER_METRICS_STATSD_WRITEINTERVAL',
            'ORDERER_RAMLEDGER_HISTORYSIZE',
            # 'CORE_METRICS_STATSD_WRITEINTERVAL',
            'CORE_PEER_GOSSIP_MEMBERSHIPTRACKERINTERVAL',
            'CORE_PEER_AUTHENTICATION_TIMEWINDOW',
            'CORE_PEER_DELIVERYCLIENT_RECONNECTTOTALTIMETHRESHOLD',
            'CORE_PEER_DISCOVERY_AUTHCACHEPURGERETENTIONRATIO',
            'CORE_PEER_GOSSIP_ALIVETIMEINTERVAL',
            'CORE_PEER_GOSSIP_CONNTIMEOUT',
            'CORE_PEER_GOSSIP_DIALTIMEOUT',
            'CORE_PEER_GOSSIP_DIGESTWAITTIME',
            'CORE_PEER_GOSSIP_ELECTION_LEADERALIVETHRESHOLD',
            'CORE_PEER_GOSSIP_ELECTION_STARTUPGRACEPERIOD',
            'CORE_PEER_DELIVERYCLIENT_RECONNECTBACKOFFTHRESHOLD',
            'CORE_PEER_GOSSIP_ELECTION_MEMBERSHIPSAMPLEINTERVAL'
        ]

        self._act_importance = act_importance
        self._action_limits = self._get_action_limits()
        
        # This call populates self._act_range_dict and returns max values
        self._max_list = self._action_dict2list(self._action_limits, index=True)
        
        self._obs_limits = None
        self._obs_shape_dict = None
        
        self.initial_reward_params = {"TPS": 100.0, "Latency": 1.0, "SuccessRate": 1.0}
        self.current_reward_params = self.initial_reward_params.copy()
        self.last_reward_params = self.initial_reward_params.copy()

        # Agents: orderer, peer, peer-net -> 3 agents
        self.n = 3
        print("number of agent: ", self.n)

        # Initialize CDT and get baseline
        if not booted:
            self._init_cdt()
        else:
            self._init_cdt()

        
        print("obs: 3 (TPS,Lat,SR)\t current action: %d\t total action: %d" % (
        len(self._act_range_dict['orderer']) + len(self._act_range_dict['peer']) + len(self._act_range_dict['peer-net']),
        self._act_range_dict['length']))
        
        # Action Space
        self.action_space = [
            spaces.Box(low=0.0, high=1.0, shape=(len(self._act_range_dict['orderer']),), dtype=np.float32),
            spaces.Box(low=0.0, high=1.0, shape=(len(self._act_range_dict['peer']),), dtype=np.float32),
            spaces.Box(low=0.0, high=1.0, shape=(len(self._act_range_dict['peer-net']),), dtype=np.float32)
        ]
        
        # Observation Space
        # We use [TPS, Latency, SuccessRate] -> shape (3,) for all agents
        self.obs_dim = 3
        self.observation_space = [spaces.Box(low=0, high=1, shape=(self.obs_dim,), dtype=np.float32) for _ in range(self.n)]
        
        self._obs_shape_dict = {
            "orderer": (self.obs_dim,),
            "peer": (self.obs_dim,),
            "peer-net": (self.obs_dim,)
        }

        self.c_T = 0.8
        self.c_L = 0.2
        self.err_count = 0
        
        print("initial_reward:\t", self.initial_reward_params)
        print("Aigis init success!")

    def _get_action_limits(self):
        if not os.path.exists(self.action_max_file):
            print(f"Error: {self.action_max_file} not found.")
            return {}
        # yaml_data = athena_utils.load_config(self.action_max_file)
        with open(self.action_max_file) as f:
            yaml_data = yaml.load(f, Loader=yaml.FullLoader)
        return self._yaml2inter(yaml_data)

    def _yaml2inter(self, yaml_data):
        res = defaultdict(dict)
        for item in yaml_data:
            for key in yaml_data[item]:
                if type(yaml_data[item][key]) in (int, float):
                    res[item][key] = {"value": yaml_data[item][key], "unit": None}
                else:
                    temp_str = str(yaml_data[item][key])
                    value = re.findall(r'\d', temp_str)
                    if not value: continue
                    value_num = "".join(value)
                    unit = temp_str.replace(value_num, "")
                    res[item][key] = {"value": int(value_num), "unit": unit}
        return dict(res)

    def _action_dict2list(self, dict_data, index=False):
        res = []
        _index_range = {
            "orderer": [],
            "peer" : [],
            "peer-net": [],
            "length": 0 
        }
        
        flag = 0
        orderer_acts = []
        for item in ('configtx', 'orderer'):
            if item in dict_data:
                for key in dict_data[item]:
                    if key in self._internal_actions[:self._act_importance]:
                        _index_range['orderer'].append(flag)
                        orderer_acts.append(dict_data[item][key]['value'])
                    flag += 1
        res.append(orderer_acts)

        peer_acts = []
        net_acts = []

        if 'peer' in dict_data:
            for key in dict_data['peer']:
                if key in self._internal_actions[:self._act_importance]:
                    if "GOSSIP" in key:
                        _index_range['peer-net'].append(flag)
                        net_acts.append(dict_data['peer'][key]['value'])
                    else:
                        _index_range['peer'].append(flag)
                        peer_acts.append(dict_data['peer'][key]['value'])
                flag += 1
        
        res.append(peer_acts)
        res.append(net_acts)
        _index_range['length'] = flag

        if index:
            self._act_range_dict = _index_range
            print("set internal action range dict: ", self._act_range_dict)
        return res

    def _action_list2dict(self, list_data):
        flatten_list = [0] * self._act_range_dict['length']
        dict_data = self._action_limits 
        
        for (index, item) in enumerate(list_data[0]):
            flatten_list[self._act_range_dict['orderer'][index]] = item
        for (index, item) in enumerate(list_data[1]):
            flatten_list[self._act_range_dict['peer'][index]] = item
        for (index, item) in enumerate(list_data[2]):
            flatten_list[self._act_range_dict['peer-net'][index]] = item
            
        output_configs = {}
        idx = 0
        
        for item in ('configtx', 'orderer'):
            if item in dict_data:
                for key in dict_data[item]:
                    val = flatten_list[idx]
                    unit = dict_data[item][key]['unit']
                    val = int(val)
                    CONST_ORDERER_PARAMS = ("BatchTimeout", "MaxMessageCount", "AbsoluteMaxBytes", "PreferredMaxBytes")
                    if key in CONST_ORDERER_PARAMS:
                         val = val if val > 0 else (val + 1)
                    val_str = str(val)
                    if unit:
                        val_str += unit
                    output_configs[key] = val_str
                    idx += 1
        
        if 'peer' in dict_data:
            for key in dict_data['peer']:
                val = flatten_list[idx]
                unit = dict_data['peer'][key]['unit']
                val = int(val)
                val_str = str(val)
                if unit:
                    val_str += unit
                output_configs[key] = val_str
                idx += 1
                
        return output_configs

    def _init_cdt(self):
        print("Initializing CDT...")
        try:
            try:
                with self.sdk.session(configs={}) as sess:
                    result = sess.test()
                    self._collect_state(result)
            except ConfigSDKError as e:
                if e.status == 409:
                    print("Session conflict detected. Cleaning up previous session...")
                    payload = e.payload
                    if isinstance(payload, dict) and 'session_id' in payload:
                        try:
                            self.sdk.session_end(payload['session_id'])
                        except Exception as cleanup_err:
                            print(f"Cleanup warning: {cleanup_err}")
                    # Retry once
                    with self.sdk.session(configs={}) as sess:
                        result = sess.test()
                        self._collect_state(result)
                else:
                    raise e
            
            self.initial_reward_params = self.current_reward_params.copy()
            self.last_reward_params = self.current_reward_params.copy()
            
            init_tps = self.initial_reward_params['TPS']
            init_lat = self.initial_reward_params['Latency']
            
            self.limits = {
                "TPS": max(init_tps * 5, 5000),
                "Latency": max(init_lat * 5, 30),
                "SuccessRate": 1.0
            }
            self._init_obs = self._handle_state(self.current_reward_params)
        except Exception as e:
            print(f"Error initializing: {e}")
            self.initial_reward_params = {"TPS": 100.0, "Latency": 1.0, "SuccessRate": 1.0}
            self.last_reward_params = self.initial_reward_params.copy()
            self.limits = {"TPS": 5000, "Latency": 30, "SuccessRate": 1.0}
            self.current_reward_params = self.initial_reward_params.copy()
            self._init_obs = self._handle_state(self.current_reward_params)

    def _collect_state(self, test_result=None):
        if test_result is not None:
            tps = 0.0
            latency = 0.0
            succ = 0.0
            if test_result and 'result' in test_result:
                for name, metrics in test_result['result'].items():
                    try:
                        tps = float(metrics.get('throughput', 0))
                        latency = float(metrics.get('avg-lat', 0))
                        s = float(metrics.get('succ', 0))
                        f = float(metrics.get('fail', 0))
                        total = s + f
                        succ = (s / total) if total > 0 else 0.0
                        break
                    except:
                        pass
            
            self.current_reward_params = {
                "TPS": tps,
                "Latency": latency,
                "SuccessRate": succ
            }
        return self.current_reward_params

    def _handle_state(self, state_dict):
        norm_tps = state_dict['TPS'] / self.limits['TPS']
        norm_lat = state_dict['Latency'] / self.limits['Latency']
        norm_sr = state_dict['SuccessRate'] / self.limits['SuccessRate']
        
        obs_vec = np.array([norm_tps, norm_lat, norm_sr]).clip(0, 1)
        return [obs_vec.copy() for _ in range(self.n)]

    def _deploy_cdt(self, config):
        input_config = []
        for i, agent_actions in enumerate(config):
            input_config.append(np.array(agent_actions) * np.array(self._max_list[i]))
        configs_dict = self._action_list2dict(input_config)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with self.sdk.session(configs=configs_dict) as sess:
                    result = sess.test()
                    return result
            except ConfigSDKError as e:
                if e.status == 409:
                    print(f"Conflict in deploy (attempt {attempt+1}/{max_retries}). Cleaning up...")
                    payload = e.payload
                    if isinstance(payload, dict) and 'session_id' in payload:
                        try:
                            self.sdk.session_end(payload['session_id'])
                        except Exception as cleanup_err:
                            print(f"Cleanup warning: {cleanup_err}")
                    
                    # Wait a bit before retrying
                    time.sleep(2)
                    continue
                else:
                    print(f"Deployment error: {e}")
                    return None
            except Exception as e:
                print(f"Deployment exception: {e}")
                return None
        
        print("Deployment failed after retries.")
        return None

    def step(self, action_n, with_stop=True):
        start_time = time.time()
        test_result = self._deploy_cdt(action_n)
        
        if test_result:
            self._collect_state(test_result)
            obs_n = self._handle_state(self.current_reward_params)
            reward = self._cal_reward()
            rew_n = [reward] * self.n
            self.last_reward_params = self.current_reward_params.copy()
        else:
            obs_n = [np.zeros(self.obs_dim) for _ in range(self.n)]
            rew_n = [-100.0] * self.n
            reward = -100.0
        
        seconds = int(time.time() - start_time)
        print("\n  Step time: {}s".format(seconds))
        print(f"Obs: TPS={self.current_reward_params['TPS']}, Lat={self.current_reward_params['Latency']}, SR={self.current_reward_params['SuccessRate']}, Rew={reward}")
        
        return obs_n, rew_n, [False]*self.n, {}, self.current_reward_params['TPS'], self.current_reward_params['Latency']

    def _cal_reward(self):
        r_T = 0
        r_L = 0
        init_tps = self.initial_reward_params['TPS'] if self.initial_reward_params['TPS'] > 0 else 1
        last_tps = self.last_reward_params['TPS'] if self.last_reward_params['TPS'] > 0 else 1
        curr_tps = self.current_reward_params['TPS']
        deltaT_0 = (curr_tps - init_tps) / init_tps
        deltaT_1 = (curr_tps - last_tps) / last_tps
        r_T = self._cal_delta_T(deltaT_0, deltaT_1, 10)
        
        init_lat = self.initial_reward_params['Latency'] if self.initial_reward_params['Latency'] > 0 else 0.1
        last_lat = self.last_reward_params['Latency'] if self.last_reward_params['Latency'] > 0 else 0.1
        curr_lat = self.current_reward_params['Latency']
        deltaL_0 = -(curr_lat - init_lat) / init_lat
        deltaL_1 = -(curr_lat - last_lat) / last_lat
        r_L = self._cal_delta_L(deltaL_0, deltaL_1, 10)
        
        return self.c_L * r_L + self.c_T * r_T

    def _cal_delta_T(self, delta0, delta1, eta):
        if delta1 > 0:
            return np.exp(eta * delta0*delta1)
        else:
            return -np.exp(-eta*delta0*delta1)
    
    def _cal_delta_L(self, delta0, delta1, eta):
        if delta1 > 0:
            return -np.exp(eta*delta0*delta1)
        else:
            return np.exp(-eta*delta0*delta1)

    def reset(self):
        return self._init_obs

    def close(self):
        pass
