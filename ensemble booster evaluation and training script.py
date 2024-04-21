
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
from sklearn.ensemble import *
import pickle
from playback import get_state_action_pairs
from sklearn.model_selection import *
from ensemble_agent import EnsembleAgent
from boosting_agent import BoostingAgent
from dataloader import dataloader

#################################################################
#################################################################
#################################################################
#Algo to find best hyperparameters###############################
#################################################################
#################################################################
################################################################# 


boosting_parameters = {'n_estimators':[50+10*i for i in range(100)]}
RFC_parameters = {'n_estimators':[100+10*i for i in range(100)], 'criterion':['gini','entropy','log_loss'], 'max_depth':[0+i for i in range(100)]}
cv = 5
tuning_ADABooster = BoostingAgent()
tuning_RFC = EnsembleAgent()


tuned_RFC_parameters = tuning_RFC.evaluate(features, labels, cv, RFC_parameters)
tuned_ADABooster_parameter = tuning_ADABooster.evaluate(features, labels, cv, boosting_parameters)


#################################################################
#################################################################
#################################################################
#Algo to load data into training for the RFC Agent###############
#################################################################
#################################################################
#################################################################

RFC = EnsembleAgent(n_estimators = tuned_RFC_parameters['n_estimators'], criterion = tuned_RFC_parameters['criterion'], max_depth = tuned_RFC_parameters['max_depth'])
ADABooster = BoostingAgent(n_estimators = tuned_ADABooster_parameter['n_estimators'])
alldata = dataloader(filepath="./recordings", batch_size=999999999) #Note: arbitrarily set the batch size
states, actions = alldata.get_next_batch()
#States is a np array, actions is a np array
RFC.train(states, actions)
ADABooster.train(states, actions)


# for i in range(batchsize):
#     alldata = dataloader(filepath="./recordings", batch_size=100) #Note: arbitrarily set the batch size
#     states, actions = alldata.get_next_batch()

    #States is a np array, actions is a np array
    # RFC.fit(states[i], actions[i])
    # ADABooster.fit(states[i], actions[i])

    








