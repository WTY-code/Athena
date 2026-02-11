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
from collections import defaultdict

# Add mcts to path to import config_sdk
sys.path.append('/root/mcts')
try:
    from config_sdk import ConfigSDK, ConfigSDKError
except ImportError:
    # Fallback if running from a different directory structure
    print("Warning: Could not import config_sdk. Make sure /root/mcts is in python path.")

# Add Athena to path to import utils
sys.path.append('/root/Athena')
try:
    from utils import utils as athena_utils
except ImportError:
    print("Warning: Could not import athena_utils. Make sure /root/Athena is in python path.")

class AigisEnv(gym.Env):

    def __init__(self, boot=False):
        # Initialize ConfigSDK
        # Assuming the server is running at the address specified in the prompt/task
        self.sdk = ConfigSDK(base_url="http://192.168.0.96:8321/")
        
        self.action_max_file = "/root/Athena/action.max.yaml"
        self._action_limits = self._get_action_limits()
        
        # We define observation space as [TPS, Latency, SuccessRate]
        # instead of Prometheus metrics which are unavailable
        self.obs_num = 3
        self.n = 3
        
        # Initialize CDT (Get initial baseline)
        self._init_cdt()
        
        self.act_num = len(self._action_dict2list(self._action_limits))
        print("obs: %d\t action: %d" % (self.obs_num, self.act_num))
        
        obs_high = np.ones(self.obs_num, dtype='float32')
        self.observation_space = spaces.Box(
            low=0,
            high=obs_high,
            dtype=np.float32
        )
        
        action_high = np.ones(self.act_num, dtype=np.float32)
        self.action_space = spaces.Box(
            low=0,
            high=action_high,
            dtype=np.float32
        )

        self.c_T = 0.5
        self.c_L = 0.5
        print("initial_reward:\t", self.initial_reward_params)
        print("Aigis init success!")

    def _get_action_limits(self):
        # Parse action.max.yaml
        if not os.path.exists(self.action_max_file):
            print(f"Error: {self.action_max_file} not found.")
            return {}
        yaml_data = athena_utils.load_config(self.action_max_file)
        return self._yaml2inter(yaml_data)

    def _yaml2inter(self, yaml_data):
        res = defaultdict(dict)
        for item in yaml_data:
            for key in yaml_data[item]:
                if type(yaml_data[item][key]) in (int, float):
                    res[item][key] = {
                        "value": yaml_data[item][key],
                        "unit": None
                    }
                else:
                    temp_str = str(yaml_data[item][key])
                    value = re.findall(r'\d', temp_str)
                    if not value:
                        continue
                    value_num = "".join(value)
                    # unit is the rest of the string
                    unit = temp_str.replace(value_num, "")
                    res[item][key] = {
                        "value": int(value_num),
                        "unit": unit
                    }
        return dict(res)

    def _action_dict2list(self, dict_data):
        res = []
        for item in dict_data:
            for key in dict_data[item]:
                res.append(dict_data[item][key]['value'])
        return res

    def _action_list2dict(self, list_data):
        output_configs = {}
        index = 0
        for item in self._action_limits:
            for key in self._action_limits[item]:
                if index >= len(list_data):
                    break
                max_val = self._action_limits[item][key]['value']
                unit = self._action_limits[item][key]['unit']
                
                # Scale 0-1 to actual value
                # Assuming list_data is numpy array or list of floats 0-1
                val = int(list_data[index] * max_val)
                
                # Special handling for constant parameters (from deployer.py logic)
                CONST_ORDERER_PARAMS = ("BatchTimeout", "MaxMessageCount", "AbsoluteMaxBytes", "PreferredMaxBytes")
                if key in CONST_ORDERER_PARAMS:
                     val = val if val > 0 else (val + 1)
                
                val_str = str(val)
                if unit:
                    val_str += unit
                
                output_configs[key] = val_str
                index += 1
        return output_configs

    def _collect_state(self, test_result):
        tps = 0.0
        latency = 0.0
        succ = 0.0
        
        if test_result and 'result' in test_result:
            res_dict = test_result['result']
            for name, metrics in res_dict.items():
                try:
                    tps = float(metrics.get('throughput', 0))
                    latency = float(metrics.get('avg-lat', 0))
                    s = float(metrics.get('succ', 0))
                    f = float(metrics.get('fail', 0))
                    total = s + f
                    succ = (s / total) if total > 0 else 0.0
                    break # Take first result
                except:
                    pass
        
        self.current_reward_params = {
            "TPS": tps,
            "Latency": latency,
            "SuccessRate": succ
        }
        
        return np.array([tps, latency, succ])

    def _init_cdt(self):
        print("Initializing CDT with default config...")
        try:
            # Use empty config to trigger defaults
            with self.sdk.session(configs={}) as sess:
                result = sess.test()
                self._collect_state(result)
                
                self.initial_reward_params = self.current_reward_params.copy()
                self.last_reward_params = self.current_reward_params.copy()
                
                # Heuristic limits
                init_tps = self.initial_reward_params['TPS']
                init_lat = self.initial_reward_params['Latency']
                
                self.limits = {
                    "TPS": max(init_tps * 5, 5000),
                    "Latency": max(init_lat * 5, 30),
                    "SuccessRate": 1.0
                }
        except Exception as e:
            print(f"Error initializing CDT: {e}")
            self.initial_reward_params = {"TPS": 100.0, "Latency": 1.0, "SuccessRate": 1.0}
            self.last_reward_params = self.initial_reward_params.copy()
            self.limits = {"TPS": 5000, "Latency": 30, "SuccessRate": 1.0}
            self.current_reward_params = self.initial_reward_params.copy()

    def _deploy_cdt(self, action):
        configs = self._action_list2dict(action)
        # print("Deploying with configs:", configs)
        try:
            with self.sdk.session(configs=configs) as sess:
                result = sess.test()
                return result
        except Exception as e:
            print(f"Deployment/Test failed: {e}")
            return None

    def _cal_reward(self):
        r_T = 0
        r_L = 0
        
        # Avoid div by zero
        init_tps = self.initial_reward_params['TPS'] if self.initial_reward_params['TPS'] > 0 else 1
        last_tps = self.last_reward_params['TPS'] if self.last_reward_params['TPS'] > 0 else 1
        
        curr_tps = self.current_reward_params['TPS']
        
        deltaT_0 = (curr_tps - init_tps) / init_tps
        deltaT_1 = (curr_tps - last_tps) / last_tps
        
        if deltaT_0 > 0:
            r_T = (np.square((1 + deltaT_0)) - 1) * abs(1 + deltaT_1)
        else:
            r_T = -(np.square((1 - deltaT_0)) - 1) * abs(1 - deltaT_1)

        init_lat = self.initial_reward_params['Latency'] if self.initial_reward_params['Latency'] > 0 else 0.1
        last_lat = self.last_reward_params['Latency'] if self.last_reward_params['Latency'] > 0 else 0.1
        
        curr_lat = self.current_reward_params['Latency']
        
        deltaL_0 = (curr_lat - init_lat) / init_lat
        deltaL_1 = (curr_lat - last_lat) / last_lat
        
        # Preserving original Athena logic
        if deltaL_0 > 0:
            r_L = (np.square((1 + deltaL_0)) - 1) * abs(1 + deltaL_1)
        else:
            r_L = -(np.square((1 - deltaL_0)) - 1) * abs(1 - deltaL_1)
            
        return self.c_L * r_L + self.c_T * r_T

    def step(self, action, with_stop=True):
        start_time = time.time()
        
        test_result = self._deploy_cdt(action)
        
        obs_raw = self._collect_state(test_result)
        obs = self._normalization(obs_raw)
        reward = self._cal_reward()
        
        self.last_reward_params = self.current_reward_params.copy()
        
        seconds = int(time.time() - start_time)
        print("\n  Step complete time cost {:>02d}:{:>02d}:{:>02d}".format(seconds // 3600, (seconds % 3600) // 60, seconds % 60))
        print("Obs: tps: %s, latency: %s, reward: %f" % (str(self.current_reward_params['TPS']), str(self.current_reward_params['Latency']), reward))
        
        # stop_cdt is handled by the session context manager in _deploy_cdt
        
        return obs, reward, True, {}

    def _normalization(self, state):
        # state: [TPS, Latency, SuccessRate]
        res = []
        res.append(state[0] / self.limits['TPS'])
        res.append(state[1] / self.limits['Latency'])
        res.append(state[2] / self.limits['SuccessRate'])
        return np.array(res).clip(0, 1)

    def reset(self):
        # Return current state normalized
        obs_raw = np.array([
            self.current_reward_params['TPS'],
            self.current_reward_params['Latency'],
            self.current_reward_params['SuccessRate']
        ])
        return self._normalization(obs_raw)

    def render(self, mode='human'):
        pass
    def close(self):
        pass

    def stop_cdt(self):
        pass
