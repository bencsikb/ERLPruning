import os
import numpy as np
from utils.LR_utils import denormalize

""" To run without import issues, move this file to the root folder. """


def differ_in_one_row(array1, array2):
    # Check if the arrays have the same shape
    if array1.shape != array2.shape:
        return False

    # Count the number of differing rows
    differing_rows = np.sum(np.any(array1 != array2, axis=1))

    # Return True if there is exactly one differing row, otherwise False
    return differing_rows == 1

def add_dmap(path):
    
    state_path = os.path.join(path, 'states')
    label_path = os. path.join(path, 'labels')    
    out_state_path = os.path.join(path, 'exp_states')
    
    state_files = os.listdir(state_path)
    label_files = os.listdir(label_path)     
    
    int_state_files  = [int(fname.replace(".txt", "")) for fname in state_files]
    int_label_files = [int(fname.replace(".txt", "")) for fname in label_files]

    int_state_files = sorted(int_state_files)
    int_label_files = sorted(int_label_files)
    #print(int_state_files)
 
    int_shifted_label_files =  [0] + int_label_files[:-1]
    
    dmap_prev = -1.0
    
    # Read the first sample    
    state_file, label_file = int_state_files[0], int_shifted_label_files[0]
    state_prev = np.loadtxt(os.path.join(state_path, str(state_file)+".txt"))
    label_prev = np.loadtxt(os.path.join(label_path, str(label_file)+".txt"))
    
    state_ext = np.full((state_prev.shape[0], state_prev.shape[1]+1), -1.0) 
    state_ext[0, :-1] =  state_prev[0, :]
    state_ext[0, -1] = dmap_prev

    
    for i, (state_file, label_file) in enumerate(zip(int_state_files[1:], int_shifted_label_files[1:])):
        
        condition = False #str(state_file)+".txt" == "44.txt"
        print(str(state_file)+".txt")
            
        state = np.loadtxt(os.path.join(state_path, str(state_file)+".txt"))
        label = np.loadtxt(os.path.join(label_path, str(label_file)+".txt"))       
        
        # Find the lastly pruned layer index
        if np.where((state == -1.0).all(axis=1))[0].size != 0:
            if condition:  print("In np.where")
            #print(np.where((state == -1.0).all(axis=1)))
            index_notpruned = np.where((state == -1.0).all(axis=1))[0][0] 
        else:  
            if condition:     print("In np.where ELSE")
            index_notpruned = 44
        layer_index = index_notpruned - 1  
        if condition:  print(f"{layer_index = }")

        #print(f"{index_notpruned = }")           
        
        # Check if state files follow each other continuously
        if differ_in_one_row(state, state_prev):
            if condition:   
                print("Differ in one row")
                print(state_ext)
            
            state_ext[layer_index , :-1] = state[layer_index , :]
            if state[layer_index, -1] != state[layer_index-1, -1]: # Check if there is a new alpha
                
                if condition:  print("State noteq")
                dmap = label[1] 
                state_ext[layer_index , -1] = dmap     
                dmap_prev = dmap
            else:
                if condition:   print("State noteq ELSE")
                state_ext[layer_index, -1] = dmap_prev

        else:
            if condition:  print("Differ in one row ELSE")
            eqflag = False
            for j in range(i-1):
                temp_state = np.loadtxt(os.path.join(state_path, str(int_state_files[j])+".txt"))
                #print(str(int_state_files[j]))
                if np.array_equal(state[:layer_index, :], temp_state[:layer_index, :]):
                    if condition:     
                        print(f"Found prev state: {int_state_files[j]} ")
                    eqflag = True
                    state_ext = np.loadtxt(os.path.join(out_state_path, str(int_state_files[j])+".txt"))
                    label_temp = np.loadtxt(os.path.join(label_path, str(int_shifted_label_files[j])+".txt")) 
                    dmap_temp = label_temp[1]    
                    if condition: print(state_ext)

                    state_ext[layer_index , :-1] = state[layer_index , :]
                    state_ext[layer_index, -1] = dmap_temp
                    dmap_prev = dmap_temp
                    if condition: print(state_ext)

                    break
                                
        
        state_prev = state            
        np.savetxt(os.path.join(out_state_path, str(state_file)+".txt"), state_ext)
        if condition: break
        

          
    

if __name__ == "__main__":
    
    path = "/nas/blanka_phd/DATASETS/SPN/COCO/all"
    add_dmap(path)
