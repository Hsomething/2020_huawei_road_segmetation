import numpy as np
import cv2,os

def normalize(img):
    r_mean = np.mean(img[:, :, 0]) / 255
    g_mean = np.mean(img[:, :, 1]) / 255
    b_mean = np.mean(img[:, :, 2]) / 255
    r_std = np.std(img[:, :, 0]) / 255
    g_std = np.std(img[:, :, 1]) / 255
    b_std = np.std(img[:, :, 2]) / 255
    mean_ = [r_mean, g_mean, b_mean]
    std_ = [r_std, g_std, b_std]
    return mean_,std_

def get_std_mean(img_path):
    r_means = []
    g_means = []
    b_means = []
    r_std = []
    g_std = []
    b_std = []
    for item in os.listdir(img_path):
        img = cv2.imread(os.path.join(img_path,item))
        mean_,std_ = normalize(img)
        r_means.append(mean_[0])
        g_means.append(mean_[1])
        b_means.append(mean_[2])
        r_std.append(std_[0])
        g_std.append(std_[1])
        b_std.append(std_[2])
    l = len(r_means)
    std_ = (sum(r_std)/l,sum(g_std)/l,sum(b_std)/l)
    mean_ = (sum(r_means)/l,sum(g_means)/l,sum(b_means)/l)
    return {'std':std_,'mean':mean_}

if __name__=='__main__':
    t = get_std_mean(r'E:\Project\Road_split\RSC_Baseline\data\train\images')
    print(t['std'])
    print(t['mean'])