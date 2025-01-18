from collections import deque
import cv2
import gymnasium as gym
import numpy as np
from torch.multiprocessing import Process
import src.flag as flag
import time
import ale_py
import traceback

cv2.ocl.setUseOpenCL(False)
gym.register_envs(ale_py)


class KingKong(Process):
    def __init__(self, env_id, child, action_re, p, max_steps):
        super(KingKong, self).__init__()
        self.env = self.make_env()
        self.env.reset()
        self.child = child
        self.env_id = env_id
        self.action_re = action_re
        self.p = p
        self.last_action = 0
        self.ep_num = 0
        self.steps = 0
        self.max_steps = max_steps

    def make_env(self):
        env = gym.make("ALE/KingKong-v5", render_mode="human" if flag.SHOW_GAME else None)
        env = PreprocessFrame(env)
        return env

    def run(self):
        try:
            lives = 3
            while True:
                obs, done = None, None
                action = self.child.recv()
                if action is None:
                    print(f"Child process {self.env_id} received termination signal.")
                    break

                reward = 0
                if flag.STICKY_ACTION:
                    if np.random.rand() <= self.p:
                        action = self.last_action
                    self.last_action = action

                for i in range(self.action_re):
                    obs, rew, done, trunc, info = self.env.step(action)
                    reward += rew

                    if info["lives"] < lives:
                        penalty_value = 50
                        reward = reward - penalty_value
                        lives = info["lives"]
                        break
                    if done or trunc:
                        lives = 3
                        self.ep_num += 1
                        self.steps = 0
                        obs, _ = self.env.reset()
                        break

                if flag.SHOW_GAME:
                    self.env.render()
                    time.sleep(0.05)
                self.steps += 1
                self.child.send([obs, reward, done])
        except EOFError:
            print(f"Child process {self.env_id}: EOFError - Parent process closed pipe.")
        except Exception as e:
            print(f"Error in child process {self.env_id}: {e}")
            print(traceback.format_exc())
        finally:
            self.child.close()


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8)
        self.frame_deque = deque(
            [
                np.zeros((self.height, self.width)),
                np.zeros((self.height, self.width)),
                np.zeros((self.height, self.width)),
                np.zeros((self.height, self.width)),
            ],
            maxlen=4,
        )

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = frame[:, :, None]

        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return self.stack_frames(frame)

    def stack_frames(self, new_frame):
        self.frame_deque.append(new_frame)
        return np.stack(self.frame_deque)
