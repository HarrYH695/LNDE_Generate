import numpy as np
import os


if __name__ == '__main__':
    data_pred_len = 1
    
    txt_file_dir = "/nfs/turbo/coe-mcity/hanhy/LNDE_Data/data_txt_files/"
    data_txt_path = txt_file_dir + f"pred_length_{data_pred_len}/"

    if not os.path.exists(data_txt_path):
        os.makedirs(data_txt_path)

    