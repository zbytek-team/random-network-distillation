from model import Model
from torch.multiprocessing import Pipe
import montezuma_revenge_env
import freeway_env
import torch
import flag
import numpy as np


class Player:
    def __init__(self, load_path):
        flag.SHOW_GAME = True
        print("loaded model weigths from checkpoint")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            checkpoint = torch.load(load_path, map_location=self.device)
        else:
            checkpoint = torch.load(load_path)
        if flag.ENV == "MR":
            num_action = 18
        elif flag.ENV == "FW":
            num_action = 3
            
        self.model = Model(num_action).to(self.device)
        self.model.load_state_dict(checkpoint["new_model_state_dict"])

        self.model.eval()

    def play(self):
        parent, child = Pipe()
        if flag.ENV == "MR":
            env = montezuma_revenge_env.MontezumaRevenge(0, child, 1, 0, 18000)
        elif flag.ENV == "FW":
            env = freeway_env.Freeway(0, child, 1, 0, 18000)
        env.start()
        self.current_observation = np.zeros((4, 84, 84))

        while True:
            observation_tensor = (
                torch.from_numpy(np.expand_dims(self.current_observation, 0))
                .float()
                .to(self.device)
            )
            predicted_action, value1, value2 = self.model.step(observation_tensor / 255)
            parent.send(predicted_action[0])
            self.current_observation, rew, done = parent.recv()
