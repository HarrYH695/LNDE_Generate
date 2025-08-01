import cv2
import numpy as np
import os
import pickle
from matplotlib import pyplot as plt

if __name__=="__main__":
    # data = pickle.load(open('zzz_coll_curve.pkl', 'rb'))

    # for key in data:
    #     print([(i[0], i[1]) for i in data[key]])


    data = pickle.load(open('zzz_coll_curve_2_1.pkl', 'rb'))

    for key in data:
        print([(i[0], i[1]) for i in data[key]])