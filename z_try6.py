import cv2
import numpy as np
import os
import pickle
from matplotlib import pyplot as plt

if __name__=="__main__":
    x = [1,2,3,4,5,6,7,8,9,10]
    y = np.array([100, 100, 100, 100, 100, 100, 75, 18, 2, 0]) / 100
    
    plt.figure()
    plt.plot(x, y, marker='o', linewidth=2)
    plt.xlabel('Time Step Interval between Input and Collision')
    plt.ylabel('Crash Rate')
    plt.grid()
    plt.savefig('z_crash_rate.png')