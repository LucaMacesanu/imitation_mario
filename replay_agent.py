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
import argparse


class ReplayAgent:
    def __init__(self,action_history):
        self.index = 0
        self.max = len(action_history)
        self.action_history = action_history
        self.done = False
        self.name = "replay_agent"

    def get_action(self,state):
        if self.index == self.max - 1:
            print("End of recording reached")
            self.done = True
            return 0
        else:
            action = self.action_history[self.index]
            self.index += 1
            return action

def run_agent(agent):

    # Initialize environment without GrayScaleObservation wrapper
    env = gym.make("SuperMarioBros-v3")
    env.reset()  # Reset the environment before starting playback

    # setup histories for recordin
    action_history = []
    state_history = []
    reward_history = []

    action = 0
    done = False
    while not done:

        
        # Apply action to environment
        state, reward, done, _ = env.step(action)
        action_history.append(action)
        state_history.append(state)
        reward_history.append(reward)

        action = agent.get_action(state)
        
        # Render the environment
        cv2.imshow("Playback", state)
        cv2.waitKey(5)  # Adjust the delay between frames if needed

        # Check if the episode is done
        # if reward == -15:
        #     print("Agent Died")
        #     break

        if done:
            print("Done")
            break
        if agent.done:
            print("Agent done")
            done = True
            break

    # Save the data
    action_history = np.array(action_history)
    state_history = np.array(state_history)
    reward_history = np.array(reward_history)
    day_time = datetime.today().strftime("%m%d%y_%H%M%S")

    np.savez(".\\agent_recordings\\" + agent.name + "_" + day_time,state_history,action_history,reward_history)
    print("Recording saved")

    # Close the environment
    env.close()
    cv2.destroyAllWindows()
    print("Windows closed")



#Specifically for the recording agent
if __name__ == "__main__":
    record_file = "agent_recordings\mlp_agent_040424_160131.npz"  # Path to your recorded data
    data = np.load(record_file)
    #print(sorted(data))
    rec_state_history = data['arr_0']
    rec_action_history = data['arr_1']
    rec_reward_history = data['arr_2']
    agent = ReplayAgent(rec_action_history)
    print("max reward: ",rec_reward_history.max())
    run_agent(agent)