import numpy as np
import sys
import os

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
        
if __name__ == "__main__":
    dl = dataloader(filepath="./recordings", batch_size=100, requires_reward=False)
    i = 0
    while not dl.done:
        data = dl.get_next_batch()
        if data is not None:
            st,ac = dl.get_next_batch()
            print("%3d: index: %3d len: %3d" % (i,dl.index, len(st)))
            i += 1
