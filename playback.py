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
import os

def judge_run(data):
    # a reward of -15 == death
    # a reward of 4 == success

    reward_history = data['arr_2']
    
    total_reward_score = sum(reward_history)
    average_score_per_time = total_reward_score/reward_history.size

    print("Average reward per time:",average_score_per_time )
    print("Total Reward for run: ", total_reward_score)

    start_index = 0
    end_index = None
    subarray = []
    result = []
    for i in range(reward_history.size):
        if (reward_history[i] > 3):
            end_index = i
            subarray.extend(reward_history[start_index:end_index + 1])
            result.append(np.array(subarray))
            subarray = []
            start_index = i
        if(reward_history[i] == -15):
            start_index = i

    sum2 = 0
    for i in range(len(result)):
        sum2 = 0
        for j in range(len(result[0])):
            sum2 += result[i][j]

        avg_2 = sum2/ len(result[i])
        avg_2 = "{:.3f}".format(avg_2)
        print("Run #: ",i+1, " = ", "Total Score:", sum2, "Average score reward per time: ",avg_2 )


    
    
    
    print("Number of successful runs:",len(result))

    return total_reward_score

def eliminate_unecessary_action(data):

    #Given an array of actions, replace all actions that do not move mario in a specific way
    action_space = data['arr_1']
    for i in range (action_space.size):
        if (action_space[i] != 64 or 65 or 66 or 67 or 128 or 129 or 130 or 131 or 1):
            action_space[i] == 0



    return action_space
        



    

def playback(record_file):
    # Load recorded data
    data = np.load(record_file)
    #print(sorted(data))
    state_history = data['arr_0']
    action_history = data['arr_1']
    reward_history = data['arr_2']
    #print("Reward min",reward_history.min())
    #print("Reward max",reward_history.max())
    
    
    

    
    #print("State shape ",state_history.shape)
    #print("Action shape ",action_history.shape)
    #print("state 0" ,state_history[0])

    


    
    for i in range(state_history.shape[0]):
        action = action_history[i]
        reward = reward_history[i]
        print("Action:", action)
        print("Reward:",reward)
        cv2.imshow("staet 0" , state_history[i])
        cv2.waitKey(1)
    
    

    judge_run(data)


    return








    # Initialize environment without GrayScaleObservation wrapper
    env = gym.make("SuperMarioBros-v3")
    env.reset()  # Reset the environment before starting playback

    # Wrap environment with JoypadSpace
    for i in range(len(action_history)):
        action_img = action_history[i]
        state = state_history[i]
        
        # Convert grayscale image to binary image using a threshold
        _, binary_action = cv2.threshold(action_img, 127, 255, cv2.THRESH_BINARY)
        
        # Find indices of nonzero elements, representing activated buttons
        action_indices = np.nonzero(binary_action)
        
        # Convert indices to integer action
        action = action_indices[1][0]  # Assuming only one button is activated
        
        # Apply action to environment
        state, reward, done, _ = env.step(action)
        
        # Render the environment
        cv2.imshow("Playback", state)
        cv2.waitKey(100)  # Adjust the delay between frames if needed
        print("Action:", action)

        # Check if the episode is done
        if done:
            print("Episode done.")
            break

        
        if done:
            break

    # Close the environment
    env.close()
    cv2.destroyAllWindows()

def get_state_action_pairs(n):
    recordings_dir = "./recordings"
    state_history = np.array([])
    action_history = np.array([])
    filenames = os.listdir(recordings_dir)
    for i in range(min(n,len(filenames))):
        filename = filenames[i]
        if filename.endswith(".npz"):
            print("Adding ", filename)
            record_file = os.path.join(recordings_dir, filename)
            data = np.load(record_file)
            if (state_history.size == 0):
                state_history = data['arr_0']
                action_history = data['arr_1']
            else:
                state_history = np.concatenate([state_history, data['arr_0']], axis=0)
                action_history = np.concatenate([action_history, data['arr_1']], axis=0)
    return state_history, action_history



if __name__ == "__main__":
    #record_file = "./recordings/imitation_mario_rec_samik_032824_144523.npz"  # Path to your recorded data
    #playback(record_file)
    # count = 0
    # recordings_dir = "./recordings"
    # for filename in os.listdir(recordings_dir):
    #     if filename.endswith(".npz"):
    #         record_file = os.path.join(recordings_dir, filename)
    #         print("----------------------------------------")
    #         print("Processing file:", record_file)
    #         try:
    #             playback(record_file)
    #         except KeyError:
    #             count += 1
    #             # print(f"Data in file {record_file} does not have key 'arr_2'")
    state_history, action_history = get_state_action_pairs(3)
                

    # print("invalid count:",count)eci
            

            
               


