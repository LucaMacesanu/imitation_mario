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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import pickle
from playback import get_state_action_pairs
from sklearn.model_selection import *
import sklearn as sk

class knn:
    def __init__(self, model = None, n_n= 10):
        self.done = False
        self.name = "boosting_agent"
        if model is None:
            pline = Pipeline([('scaler', sk.preprocessing.StandardScaler()), ('pca', sk.decomposition.PCA()), 
                  ('knn', KNeighborsClassifier(n_neighbors= n_n))])
            self.model = pline
        else:
            self.model = model

    def get_action(self,state):
        state = state.flatten().reshape(1,-1)
        return self.model.predict(state)
    
    def train(self, state_history, action_history):
        print("Training...")
        nsamples, nx, ny = state_history.shape
        d2_train_dataset = state_history.reshape((nsamples,nx*ny))
        self.model.fit(d2_train_dataset,action_history)
        pickle.dump(self.model,open( "models/knn.p", "wb" ))
        print("Training Complete.")

    def evaluate(self, features, labels, cv, parameters):
        # parameters = {'max_depth':[35+i for i in range(11)], 'min_samples_leaf':[8, 10, 12], 'max_features':['sqrt', 'log2']}
        gridsearch = GridSearchCV(self.model, parameters, cv = cv, scoring = 'accuracy')
        gridsearch.fit(features, labels)
        nested_cross_validation_score = cross_val_score(gridsearch, features, labels, cv = cv)

        print('Best Parameters for boosting agent:', gridsearch.best_params_)
        print('Tuned, nested accuracy for boosting agent using best parameters:', nested_cross_validation_score.mean()*100)
        return(gridsearch.best_params_)
    

    
    
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
    record_file = "recordings\imitation_mario_rec_carson_032124_142721.npz"  # Path to your recorded data
    rec_state_history, rec_action_history = get_state_action_pairs(3)
    # rec_reward_history = data['arr_2']
    # pickled_model = open("models/ensemble_model.p", "rb")
    # ensemble_model = pickle.load(pickled_model)
    # agent = EnsembleAgent(ensemble_model) #Can set the ensembling algorithm as AdaBoostClassifier()
    agent = knn()
    # pickled_model.close()
    agent.train(rec_state_history,rec_action_history)
    run_agent(agent)