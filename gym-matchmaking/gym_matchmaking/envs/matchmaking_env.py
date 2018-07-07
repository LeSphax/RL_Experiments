import gym
import math
import random
import numpy as np
import time
from gym import error, spaces, utils
from gym.utils import seeding


class MatchmakingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    padding_value = -1

    def __init__(self):
        self.state_size = 10
        self.max_history_size = 10
        # A list of player ratings
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10,))
        # Indexes for the two players we want to match
        self.action_space = spaces.Tuple((spaces.Discrete(self.state_size+1), spaces.Discrete(self.state_size+1)))
        self.viewer = None
        self.state = None
        self.padded_state = None
        self.history = None
        self.error_last_step = False

    def seed(self,seed):
        random.seed(seed)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self.error_last_step = False
        if action[0] != action[1] and len(self.state) > action[0] and len(self.state) > action[1]:

            p1 = self.state[action[0]]
            p2 = self.state[action[1]]

            self.history.insert(0, [p1, p2])
            if(len(self.history) > self.max_history_size):
                self.history = self.history[:-1]

            self.state = np.delete(self.state, action)

            paddingLen = self.state_size - len(self.state)
            self.padded_state = np.pad(np.array(self.state), (0, paddingLen), 'constant', constant_values=(self.padding_value, self.padding_value))

            reward = 1 - (pow((p1 - p2), 2) * 2)
        elif action[0] == self.state_size and action[1] == self.state_size:
            reward = 0
        else:
            reward = -1
            self.error_last_step = True

        return self.padded_state, reward, False, {}

    def reset(self):
        self.history = []

        self.state = np.random.rand(self.state_size)

        self.state.sort()
        paddingLen = self.state_size - len(self.state)
        self.padded_state = np.pad(np.array(self.state), (0, paddingLen), 'constant', constant_values=(self.padding_value, self.padding_value))

        return self.padded_state

    def render(self, mode='human', close=False):
        screen_width = 600
        screen_height = 400

        tile_w = 20.0
        tile_h = 20.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            tiles = [None] * 10
            self.transforms = [None] * 10
            self.colors = [None] * 10

            l, r, t, b = -tile_w/2, tile_w/2, tile_h/2, -tile_h/2
            for idx in range(self.state_size):
                tiles[idx] = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                self.transforms[idx] = rendering.Transform()
                tiles[idx].add_attr(self.transforms[idx])
                self.colors[idx] = tiles[idx].attrs[0]
                self.viewer.add_geom(tiles[idx])
            for _ in range(self.max_history_size * 2 + 2):
                tile = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                tiles.append(tile)
                transform = rendering.Transform()
                self.transforms.append(transform)
                tile.add_attr(transform)
                self.colors.append(tile.attrs[0])
                self.viewer.add_geom(tile)

        if self.state is None:
            return None

        for idx in range(self.state_size):
            self.transforms[idx].set_translation(50 + 50 * idx, 50)
            self.colors[idx].vec4 = ((0, self.padded_state[idx], 0, 1.0 if self.padded_state[idx] != -1 else 0))
        for idx in range(len(self.history)):
            p1Idx = idx * 2 + self.state_size
            p2Idx = p1Idx + 1
            self.transforms[p1Idx].set_translation(100, 150 + 30 * idx)
            self.transforms[p2Idx].set_translation(150, 150 + 30 * idx)
            self.colors[p1Idx].vec4 = ((0, self.history[idx][0], 0, 1.0))
            self.colors[p2Idx].vec4 = ((0, self.history[idx][1], 0, 1.0))
        
        #Tile to show time passing
        self.transforms[self.state_size + self.max_history_size *2].set_translation(500, 300)
        self.colors[self.state_size + self.max_history_size *2].vec4 = ((0, 0, 0, 1.0 if self.colors[self.state_size + self.max_history_size *2].vec4[3] == 0 else 0))
        #Tile to show errors
        self.transforms[self.state_size + self.max_history_size *2 +1].set_translation(500, 200)
        self.colors[self.state_size + self.max_history_size *2 +1].vec4 = ((1.0, 0, 0, 1.0 if self.error_last_step else 0))
        time.sleep(0.3)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
