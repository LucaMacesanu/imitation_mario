import gym
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import pickle
from datetime import datetime
from tqdm import tqdm
from gym.wrappers import GrayScaleObservation
import os
from playback import get_state_action_pairs

class EnsembleAgent:
    def __init__(self, model=None):
        self.done = False
        self.name = "ensemble_agent"
        if model is None:
            self.model = RandomForestClassifier()
        else:
            self.model = model

    def get_action(self, state):
        state = state.flatten().reshape(1, -1)
        return self.model.predict(state)

    def train(self, state_space, action_space):
        self.model.fit(state_space, action_space)
        pickle.dump(self.model, open("ensemble_model.p", "wb"))
        print("Training Complete.")

    def evaluate_and_tune(self, features, labels):
        parameters = {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        cv = 5
        grid_search = GridSearchCV(self.model, parameters, cv=cv, scoring='accuracy', return_train_score=True)
        grid_search.fit(features, labels)

        print('Best Parameters:', grid_search.best_params_)
        return grid_search

def plot_accuracy(grid_search):
    results = grid_search.cv_results_
    plt.figure(figsize=(10, 5))
    plt.title("Training vs Validation Accuracy")
    plt.plot(results['mean_train_score'], label='training accuracy')
    plt.plot(results['mean_test_score'], label='validation accuracy')
    plt.xlabel('parameter combination')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def run_agent(agent):
    env = GrayScaleObservation(gym.make("SuperMarioBros-v3"))
    env.reset()
    done = False
    state_space= []
    action_space = []

if __name__ == "__main__":
    record_path = "recordings"

    if not os.path.isdir(record_path):
        print(f"Error: {record_path} is not a valid directory.")
    else:
        rec_state_space, rec_action_space = get_state_action_pairs(2)

        if rec_state_space.ndim == 3:
            nsamples, height, width = rec_state_space.shape
            rec_state_space= rec_state_space.reshape((nsamples, height * width))

        print("Reshaped state history shape:", rec_state_space.shape)

        train_states, test_states, train_actions, test_actions = train_test_split(rec_state_space, rec_action_space, test_size=0.2, random_state=42)

        agent = EnsembleAgent()
        agent.train(train_states, train_actions)

        grid_search = agent.evaluate_and_tune(train_states, train_actions)
        plot_accuracy(grid_search)

        run_agent(agent)
