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
from sklearn.neural_network import MLPClassifier
import pickle


class MLPAgent:
    def __init__(self, model = None):
        self.done = False
        self.name = "mlp_agent"
        if model is None:
            self.model = MLPClassifier(hidden_layer_sizes=10)
        else:
            self.model = model


    def get_action(self,state):
        state = state.flatten().reshape(1,-1)
        return self.model.predict(state)
        
    def train(self, state_history,action_history):
        print("Training...")
        nsamples, nx, ny = state_history.shape
        d2_train_dataset = state_history.reshape((nsamples,nx*ny))
        self.model.fit(d2_train_dataset,action_history)
        pickle.dump(self.model,open( "models/mlp_model.p", "wb" ))
        print("Training Complete.")


def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def run_agent(agent):

    # Initialize environment without GrayScaleObservation wrapper
    env = GrayScaleObservation(gym.make("SuperMarioBros-v3"))
    env.reset()  # Reset the environment before starting playback

    # setup histories for recordin
    action_history = []
    state_history = []
    reward_history = []

    action = [0]
    done = False
    while not done:

        
        # Apply action to environment
        state, reward, done, _ = env.step(action)
        
        action_history.append(action)
        state_history.append(state)
        reward_history.append(reward)

        action = agent.get_action(state)
        
        # Render the environment
        # cv2.imshow("Playback", state)
        # cv2.waitKey(5)  # Adjust the delay between frames if needed

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
    record_file = "recordings\imitation_mario_rec_luca_032724_181035.npz"  # Path to your recorded data
    data = np.load(record_file)
    #print(sorted(data))
    rec_state_history = data['arr_0']
    rec_action_history = data['arr_1']
    rec_reward_history = data['arr_2']
    pickled_model = open("models/mlp_model.p", "rb")
    mlp_model = pickle.load(pickled_model)
    agent = MLPAgent(mlp_model)
    pickled_model.close()
    # agent.train(rec_state_history,rec_action_history)
    run_agent(agent)