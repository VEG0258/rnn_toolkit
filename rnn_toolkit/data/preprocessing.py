import numpy as np
from scipy.io import savemat
from scipy.io import loadmat
from copy import copy

def preprocessing(name_mat, states, numFlies, numstates, minlength):
    #controls how long is the period of the time series data is used for training by data_fraction
    #this loops sets the size of the numpy array that will contain the data
    minlength=500000
    mat_file = loadmat(name_mat) #each sub-array within 'reduced_states' represents the sequence of states for a single fly over time.
    for i,x in enumerate(mat_file[states]): #x represents each sub-array (or data sequence) in the reduced_states array from your MATLAB file
        if np.size(x[0])<minlength:
            minlength=np.size(x[0])

    #putting the data into a numpy array
    dt = np.zeros((numFlies,minlength)) #59, 50000 
    for i,x in enumerate(mat_file[states]):
        dt[i,:]=x[0][0:minlength,0]-1 #becuase dtype="uint8", so doesnot allow negative, so 0 become the largest number - 1 which is 255
        #remove 0, all 0 become 255 after last line 
        indices = np.where(dt[i,:]==255)[0]
        indices2 = np.where(np.diff(indices)>1)[0]
        for idx in indices:
            if idx==0:
                dt[i,0]=np.random.randint(0,high=numstates) # if 0 is the start of the series, replace with random state from 0 to 112
            else:
                dt[i,idx]=copy(dt[i,idx-1]) #if 0 is not the start of the series, repalce with the last index's value
    return dt
                
