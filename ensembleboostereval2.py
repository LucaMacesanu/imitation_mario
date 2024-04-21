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

# Custom imports assuming these are correctly implemented
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

    def train(self, state_history, action_history):
        self.model.fit(state_history, action_history)
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
    plt.plot(results['mean_train_score'], label='Train Accuracy')
    plt.plot(results['mean_test_score'], label='Validation Accuracy')
    plt.xlabel('Parameter Combination')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def run_agent(agent):
    env = GrayScaleObservation(gym.make("SuperMarioBros-v3"))
    env.reset()
    done = False
    state_history = []
    action_history = []

    with tqdm(total=100) as pbar:
        while not done:
            state, reward, done, _ = env.step(0)  # Example using a constant action
            action = agent.get_action(state)
            state_history.append(state)
            action_history.append(action)
            pbar.update(1)

    env.close()



if __name__ == "__main__":
    record_path = "recordings"
    
    if not os.path.isdir(record_path):
        print(f"Error: {record_path} is not a valid directory.")
    else:
        rec_state_history, rec_action_history = get_state_action_pairs(2)

        if rec_state_history.ndim == 3:
            nsamples, height, width = rec_state_history.shape
            rec_state_history = rec_state_history.reshape((nsamples, height * width))

        print("Reshaped state history shape:", rec_state_history.shape)

        train_states, test_states, train_actions, test_actions = train_test_split(
            rec_state_history, rec_action_history, test_size=0.2, random_state=42)

        agent = EnsembleAgent()
        agent.train(train_states, train_actions)

        grid_search = agent.evaluate_and_tune(train_states, train_actions)
        plot_accuracy(grid_search)

        run_agent(agent)
