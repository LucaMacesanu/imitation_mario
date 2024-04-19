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
import pickle
from playback import get_state_action_pairs
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
def eliminate_unnecessary_action(data):
    action_space = data
    
    #This might not be necessary but im on a time crunch (Theres 100% a better way to do this)
    action_map = {
        64: 0,
        65: 1,
        66: 2,
        67: 3,
        128: 4,
        129: 5,
        130: 6,
        131: 7
    }
    
    # Map original action values to new values
    action_space = np.array([action_map.get(action, 0) for action in action_space])
    #print(action_space.size())
    
    return action_space


def create_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

class CNNAgent:
    def __init__(self, model=None, state_history=None, action_history=None):
        self.done = False
        self.name = "cnn_agent"
        if model is None:
            num_classes = np.max(action_history) + 1
            self.model = create_cnn(input_shape=(state_history.shape[1], state_history.shape[2], 1), num_classes=num_classes)
        else:
            self.model = model

    def get_action(self, state):
        state = np.expand_dims(state, axis=0)
        state = np.expand_dims(state, axis=-1)
        return np.argmax(self.model.predict(state))
        
    def train(self, state_history, action_history):
        print("Training...")
        train1, test1, train2, test2 = train_test_split(state_history, action_history, test_size=0.2, random_state=42)
        train1 = np.expand_dims(train1, axis=-1)
        test1 = np.expand_dims(test1, axis=-1)
        train1 = train1 / 255.0
        test1 = test1 / 255.0
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history = self.model.fit(train1, train2, epochs=10, validation_data=(test1, test2))
        self.model.save("models/cnn_model.h5")
        print("Training complete.")

        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.show()

def run_agent(agent):

    env = GrayScaleObservation(gym_super_mario_bros.make("SuperMarioBros-v3"))
    env.reset()

    action_history = []
    state_history = []
    reward_history = []

    action = [0]
    done = False
    while not done:
        state, reward, done, _ = env.step(action)
        action_history.append(action)
        state_history.append(rgb2gray(state))  # Convert state to grayscale
        reward_history.append(reward)
        action = agent.get_action(state)
        if done:
            print("Done")
            break
        if agent.done:
            print("Agent done")
            done = True
            break

    day_time = datetime.today().strftime("%m%d%y_%H%M%S")
    np.savez(f".\\agent_recordings\\{agent.name}_{day_time}", np.array(state_history), np.array(action_history), np.array(reward_history))
    print("Recording saved")
    env.close()
    cv2.destroyAllWindows()
    print("Windows closed")

#Specifically for the recording agent
if __name__ == "__main__":
    rec_state_history, rec_action_history = get_state_action_pairs(3)
    rec_action_history = eliminate_unnecessary_action(rec_action_history)
    # pickled_model = open("models/mlp_model.p", "rb")
    # mlp_model = pickle.load(pickled_model)
    # agent = MLPAgent(mlp_model)
    # pickled_model.close()
    agent = CNNAgent(model=None, state_history=rec_state_history, action_history=rec_action_history)
    agent.train(rec_state_history, rec_action_history)
    run_agent(agent)
