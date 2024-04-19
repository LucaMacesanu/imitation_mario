import numpy as np
import tensorflow as tf
import gym
import os
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

#create env
env = gym.make('SuperMarioBros-v3')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

#not sure how to access dataloader CAN REMOVE WHEN INTEGRATED
class dataloader:
    # Batch size is defined in number of state action pairs.
    def __init__(self,filepath, batch_size, requires_reward = False):
        self.filepath = filepath
        self.batch_size = batch_size
        self.requires_reward = requires_reward

        # get the total number of files available
        self.file_names = os.listdir(filepath)
        self.n_files = len(self.file_names)
        self.index = 0
        self.subindex = 0
        self.done = False

        print("num files: ", self.n_files)

        

    

    def get_next_batch(self):
        path = self.filepath + "/" + self.file_names[self.index]
        print("Path: ", path)
        data = np.load(path)

        if self.index >= self.n_files:  # Check if all files have been processed
            self.done = True
            return None  # No more data to process

        # if this recording is invalid move on to the next one if it exists
        if len(data.keys()) != 3 and self.requires_reward:
            if(self.index == self.n_files- 1):
                # print("oof, last file was invalid")
                self.done = True
                return None
            else:
                # print("recursing down")
                self.index += 1
                self.subindex = 0
                return self.get_next_batch()
            
        
        stop = min(self.subindex + self.batch_size, len(data['arr_0']))
        

        # if weve reached the end of this recording move on to the next recording


        # print("indexes: %3d - %3d" % (self.subindex, stop))
        states =  data['arr_0'][self.subindex: stop]
        #normalize the observation space
        states = states / 255
        actions = data['arr_1'][self.subindex: stop]
        if self.requires_reward:
            rewards = data['arr_2'][self.subindex: stop]

        self.subindex += self.batch_size
        if stop == len(data['arr_0']):
            # print("eof, going to next index")
            self.index += 1
            self.subindex = 0
            if(self.index == self.n_files - 1):
                # print("oof, returning end of last file")
                self.done = True
        
        if self.requires_reward:
            return (states, actions, rewards)
        else:
            return (states,actions)

#Define Model
num_actions = 300
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(240, 256)),  # Assuming input dimensions are correct
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_actions, activation='softmax')  # num_actions should match the number of unique labels in your dataset
])


optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


# Custom BC training function
def train_on_batch(states, actions):
    with tf.GradientTape() as tape:
        predictions = model(states, training=True)
        loss = loss_fn(actions, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

#save the model

def save_model(epoch, model, save_path='model_checkpoints'):
    model.save(os.path.join(save_path, f'model_epoch_{epoch}.h5'))

# Initialize dataloader CAN REMOVE WHEN INTEGRATED
dl = dataloader(filepath="./rec_test", batch_size=100, requires_reward=False)


#EDIT WITH INTEGRATION, IDEALLY GET DATA AND THEN USE TUPLES
n_epochs = 10
for epoch in range(n_epochs):
    i = 0
    while not dl.done:
        data = dl.get_next_batch()
        loss = train_on_batch(data[0], data[1])
        print(f"Epoch {epoch}, Loss: {loss.numpy()}")
        if data is not None:
            st,ac = dl.get_next_batch()
            print("%3d: index: %3d len: %3d" % (i,dl.index, len(st)))
            i += 1
    save_model(epoch, model)
    dl.reset()

