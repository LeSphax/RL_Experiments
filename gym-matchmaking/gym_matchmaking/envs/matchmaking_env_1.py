import gym
import math
import random
import numpy as np
import time
from gym import error, spaces, utils
from gym.utils import seeding


class MatchmakingEnv1(gym.Env):
    metadata = {'render.modes': ['human']}

    padding_value = -1

    def __init__(self):
        self.state_size = 10
        self.max_history_size = 10
        # A list of player ratings
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(11,), dtype=np.float32)
        # Indexes for the two players we want to match
        self.action_space = spaces.Discrete(self.state_size+1)
        self.room = None
        self.viewer = None
        self.state = None
        self.padded_state = None
        self.history = None
        self.error_last_step = False

    def seed(self, seed):
        random.seed(seed)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        self.error_last_step = False
        self.timestep += 1

        if len(self.state) > action:
            if self.room == -1:
                self.room = self.pop_player(action)

                reward = 0
            else:

                player1 = self.room
                player2 = self.pop_player(action)

                self.history.insert(0, [player1, player2])
                if(len(self.history) > self.max_history_size):
                    self.history = self.history[:-1]

                reward = 1 - (pow((player1 - player2), 2) * 2)
               
                self.room = -1
        elif action == self.state_size:
            reward = 0
        else:
            reward = -0.1
            self.error_last_step = True

        return self.get_return_state(), reward, self.timestep == 20, {}

    def reset(self):
        self.timestep = 0
        self.history = []

        self.room = -1
        self.state = np.random.rand(self.state_size)

        self.refresh_padding()

        return self.get_return_state()

    def pop_player(self, index):
        player = self.state[index]
        self.state = np.delete(self.state, index)
        self.refresh_padding()
        return player

    def refresh_padding(self):
        paddingLen = self.state_size - len(self.state)
        self.padded_state = np.pad(np.array(self.state), (0, paddingLen), 'constant', constant_values=(self.padding_value, self.padding_value))

    def get_return_state(self):
        return np.insert(self.padded_state, 0, self.room)

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
            for _ in range(self.max_history_size * 2 + 3):
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


        # Tile to show the room
        index = self.state_size + self.max_history_size * 2
        if self.room != -1:
            self.transforms[index].set_translation(300, 150)
            self.colors[index].vec4 = ((0, self.room, 0, 1.0))
        else:
            self.colors[index].vec4 = ((0, 0, 0, 0))
        # Tile to show time passing
        index = index + 1
        self.transforms[index].set_translation(500, 300)
        self.colors[index].vec4 = ((0, 0, 1.0, 1.0 if self.colors[index].vec4[3] == 0 else 0))
        # Tile to show errors
        index = index + 1
        self.transforms[index].set_translation(500, 200)
        self.colors[index].vec4 = ((1.0, 0, 0, 1.0 if self.error_last_step else 0))
        time.sleep(0.3)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
