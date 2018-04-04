import proto as p
import numpy as np
from scipy.spatial.distance import euclidean
import os, tools
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

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

#question 7
def compute_mfcc(data):
    coeffs = mfcc(data[0]['samples'])
    for i in range(1, data.shape[0]):
        c = mfcc(data[i]['samples'])
        coeffs = np.append(coeffs, c, axis=0)

    return coeffs

def gmm():
    data = np.load('lab1_data.npz')['data']
    mfcc_coeffs = compute_mfcc(data)

    n_components = [4,8,16,32]

    for comp in n_components:

        #train GMM model
        gmm = GaussianMixture(n_components = comp, covariance_type='diag')
        gmm.fit(mfcc_coeffs)

        select_idx = [16,17,38,39]
        seven = data[select_idx]

        labels = []
        for i in range(4): #for each utterance 'seven'
            test_data = mfcc(seven[i]['samples'])
            prob = gmm.predict_proba(test_data) #compute posterior
            l = gmm.predict(test_data)
            labels.append(l)

            plt.plot(prob)
            plt.title('posterior with %d components for utterance %d' %(comp,i))
            plt.savefig((os.path.join(output_dir,'posteriorDiag_%d_%i'%(comp,i))))
            plt.show()
#
        #one plot for all utterances 'seven'
        seven_mfcc = compute_mfcc(seven)
        all_prob =  gmm.predict_proba(seven_mfcc)
        plt.plot(all_prob)
        plt.title('posterior with %d components for all utterances' %comp)
        plt.savefig((os.path.join(output_dir,'posteriorDiagAll_%d' % comp)))
        plt.show()

        # plot the different labels
        for i in range(2):
             plt.plot(labels[i])
        plt.title('Same utterance for man speaker - %d components' %comp)
        plt.savefig((os.path.join(output_dir,'man_%d' % comp)))
        plt.show()
#
        for i in range(2,4):
             plt.plot(labels[i])
        plt.title('Same utterance for woman speaker  - %d components' %comp)
        plt.savefig((os.path.join(output_dir,'woman_%d' % comp)))
        plt.show()

        plt.plot(labels[0], label='man')
        plt.plot(labels[2], label='woman')
        plt.title('Same utterance for woman and man speaker  - %d components' %comp)
        plt.legend()
        plt.savefig((os.path.join(output_dir,'manwoman_%d' % comp)))
        plt.show()

        #compare 'five' with 'seven' for man
        five_mfcc = mfcc(data[12]['samples']) #five -> 12, one -> 4
        five_label = gmm.predict(five_mfcc)
        plt.plot(labels[0], label='seven')
        plt.plot(five_label, label='five')
        plt.title('One and seven for man speaker  - %d components' % comp)
        plt.legend()
        plt.savefig((os.path.join(output_dir,'oneseven_%d' % comp)))
        plt.show()


def load_samples():
    data = np.load('lab1_data.npz')['data']

    return data

if __name__=="__main__":
    main()