import numpy as np
import sys
import os

class dataloader:
    # Batch size is defined in number of state action pairs.
    def __init__(self,filepath, batch_size):
        self.filepath = filepath
        self.batch_size = batch_size

        # get the total number of files available
        self.file_names = os.listdir(filepath)
        self.n_files = len(self.file_names)
        self.index = 0
        self.subindex = 0
        self.done = False

        

    

    def get_next_batch(self):
        path = self.filepath + "/" + self.file_names[self.index]
        print("Path: ", path)
        data = np.load(path)


        if len(data.keys()) != 3:
            if(self.index == self.n_files):
                self.done = True
                return None
            else:
                self.index += 1
                return self.get_next_batch()
            
        
        
        states =  data['arr_0'][self.subindex: self.subindex + self.batch_size]
        actions = data['arr_1'][self.subindex: self.subindex + self.batch_size]
        rewards = data['arr_2'][self.subindex: self.subindex + self.batch_size]
        return (states, actions, rewards)
        
if __name__ == "__main__":
    dl = dataloader(filepath="./recordings", batch_size=100)
    breakpoint()
