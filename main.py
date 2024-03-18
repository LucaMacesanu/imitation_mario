import gym_super_mario_bros
# print(dir(gym_super_mario_bros))
import gym
import nes_py
from nes_py.wrappers import JoypadSpace
from nes_py.app.play_human import play_human
from gym.wrappers.gray_scale_observation import GrayScaleObservation
import cv2
import numpy as np
import sys
from datetime import datetime

# state is a 240x256 grayscale image 0-255
# reward is some reward function that rewards you for getting closer to the end of the level
# done is a dummy variable that we don't need
# action is a hard one, there are many actions that don't do anything, we will have to put a wrapper around those
def my_call(state,action,reward,done,next_state):
    # print("State shape: ", state.shape)
    # print("max: ", state.max())
    # print("min: ", state.min())
    print("Action: ",action)
    # print("Done: ", done)
    # print("Reward: ",reward)
    # print("next_state:",len(next_state))

    action_history.append(action)
    state_history.append(state)



if __name__ == "__main__":
    global action_history, state_history
    action_history = []
    state_history = []
    print(gym.__version__)
    env = GrayScaleObservation(gym.make("SuperMarioBros-v3"))
    play_human(env,callback=my_call)

    # wrap them as arrays rather than list for saving
    action_history = np.array(action_history)
    state_history = np.array(state_history)
    history = [action_history,state_history]
    day_time = datetime.today().strftime("%m%d%y_%H%M%S")
    np.savez("./recordings/imitation_mario_rec_" + day_time,state_history,action_history)
    



