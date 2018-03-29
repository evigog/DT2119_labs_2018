import proto as p
import numpy as np
from scipy.spatial.distance import euclidean
import os, tools
from scipy.cluster import hierarchy

import matplotlib
import matplotlib.pyplot as plt

output_dir = "results/"

def main():
    samples_dict = load_samples()

    # ###################### Q5
    if (False):
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


    # ################ Q6
    D = np.zeros((44,44))
    if(True):

        if os.path.exists(os.path.join(output_dir, "global_distances.npy")):
            D = np.load(os.path.join(output_dir, "global_distances.npy"))
        else:
            for i in range(44):
                for j in range(44):
                    speaker1 = samples_dict[i]
                    speaker2 = samples_dict[j]

                    sp1_mfcc = p.mfcc(speaker1["samples"])
                    sp2_mfcc = p.mfcc(speaker2["samples"])

                    N = np.shape(sp1_mfcc)[0]
                    M = np.shape(sp2_mfcc)[0]
                    local_dist = np.zeros((N, M))

                    for n in range(N):
                        for m in range(M):
                            local_dist[n,m] = euclidean(sp1_mfcc[n, :], sp2_mfcc[m, :])

                    print("pair: ", i,' ',j)
                    print("\tlocal dist: ",np.shape(local_dist))
                    print()
                    # fig = plt.figure()
                    # plt.pcolormesh(local_dist, cmap = "RdBu")
                    # plt.colorbar()
                    # plt.show()


                    global_d, acc_d, _ = p.precomputed_dtw(sp1_mfcc, sp2_mfcc, local_dist)
                    
                    D[i, j] = global_d

            np.save(os.path.join(output_dir, "global_distances.npy"), D)


        digits = [d['digit'] for d in samples_dict]

        fig = plt.figure()
        plt.pcolormesh(D, cmap = "jet")
        plt.colorbar()
        plt.title("Global Distances")
        plt.xticks(np.arange(44), digits)
        plt.yticks(np.arange(44), digits)
        plt.savefig(os.path.join(output_dir, "global_distances.png"))
        plt.show()


        labels = tools.tidigit2labels(samples_dict)
        linkage_matrix = hierarchy.linkage(D, method="complete")
        plt.figure()
        dn = hierarchy.dendrogram(linkage_matrix, labels = labels, leaf_rotation=90., leaf_font_size=6)
        plt.savefig(os.path.join(output_dir, "global_distances_dendrogram.png"), dpi = 200)
        plt.show()


def load_samples():
    data = np.load('lab1_data.npz')['data']

    return data

if __name__=="__main__":
    main()