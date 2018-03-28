import proto as p
import numpy as np
from scipy.stats import pearsonr
import os

import matplotlib
import matplotlib.pyplot as plt

output_dir = "results/"

def main():
    samples_dict = load_samples()

    frames_to_mfcc_features = []
    frames_to_mspec_features = []
    for audio_sample in samples_dict:
        
        mfcc_feature_transform = p.mfcc(audio_sample["samples"])
        frames_to_mfcc_features.append(mfcc_feature_transform)

        mspec_feature_transform = p.mspec_only(audio_sample["samples"])
        frames_to_mspec_features.append(mspec_feature_transform)

    mfcc_data = np.vstack(frames_to_mfcc_features)
    mspec_data = np.vstack(frames_to_mspec_features)


    mfcc_r = np.corrcoef(mfcc_data, rowvar = False)
    mspec_r = np.corrcoef(mspec_data, rowvar = False)
    
    fig = plt.figure()
    ax = plt.subplot(121)
    ax.set_title("MFCC Feature correlations")
    plt.pcolormesh(mfcc_r, cmap = "RdBu")
    plt.colorbar()

    ax = plt.subplot(122)
    ax.set_title("MSPEC Feature correlations")
    plt.pcolormesh(mspec_r, cmap = "RdBu")
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, "feature_correlations.png"))
    plt.show()


def load_samples():
    data = np.load('lab1_data.npz')['data']

    return data

if __name__=="__main__":
    main()