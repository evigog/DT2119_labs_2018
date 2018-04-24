import numpy as np
import proto2
import prondict
import tools2
import matplotlib.pyplot as plt

#score utterances using viterbi
def test_viterbi(data, wordHMMs):
    # Score all utterances using viterbi
    viterbi_res = []
    for u in range(data.size):
        u_hmm = []
        for hmm in wordHMMs.keys():
            # compute log emission probability of utterance u and model hmm
            log_prob = tools2.log_multivariate_normal_density_diag(data[u]['lmfcc'], wordHMMs[hmm]['means'],
                                                            wordHMMs[hmm]['covars'])
            # viterbi
            res = proto2.viterbi(log_prob, wordHMMs[hmm]['startprob'], wordHMMs[hmm]['transmat'])
            res['label'] = hmm
            u_hmm.append(res)

        viterbi_res.append(u_hmm)

    # assign each utterance to hmm with max likelihood and compute error
    count_error = 0
    for i in range(data.size):
        hmm_list = viterbi_res[i]
        hmms_loglik = [hmm['loglik'] for hmm in hmm_list]
        max_idx = hmms_loglik.index(max(hmms_loglik))
        if (data[i]['digit'] != hmm_list[max_idx]['label']):
            count_error += 1

    print('Number of missclassified utterances - Viterbi: ', count_error)



data = np.load('lab2_data.npz')['data']
phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()
example = np.load('lab2_example.npz')['example'].item()


modellist = {}
for digit in prondict.prondict.keys():
    modellist[digit] = ['sil'] + prondict.prondict[digit] + ['sil']

#produce HMM for each model in modelist
wordHMMs = {}
for word in modellist.keys():
    h = proto2.concatHMMs(phoneHMMs, modellist[word])
    wordHMMs[word] = h

obsloglik = example['obsloglik']
startprob = wordHMMs['o']['startprob']
transmat = wordHMMs['o']['transmat']

#forward algorithm for hmm 'o'
forward_prob = proto2.forward(obsloglik, startprob, transmat)

# Sum over all timesteps
# marginal = np.log(np.sum(forward_prob, axis=1))

print('Actual: {}, computed: {}'.format(np.shape(example['logalpha']), np.shape(forward_prob)))

fig = plt.figure()
ax = plt.subplot(121)
ax.set_title('Actual')
plt.pcolormesh(example['logalpha'])

ax = plt.subplot(122)
ax.set_title('Computed')
plt.pcolormesh(forward_prob)
plt.colorbar()

plt.show()


#backward algorithm for hmm 'o'
backward_prob = proto2.backward(example['obsloglik'], startprob, transmat)
diff = np.absolute( backward_prob - example['logbeta'])
print('Backward max difference with example: ', np.max(diff))

#viterbi for hmm 'o'
vit_result = proto2.viterbi(example['obsloglik'], startprob, transmat)
vit_dif = vit_result['loglik'] - example['vloglik'][0]
print('Viterbi diff in probabilities: ', vit_dif)

# plot viterbi path
plt.pcolormesh(forward_prob.T)
plt.plot(vit_result['path'])
plt.colorbar()
plt.savefig('Fig/viterbi_path')
plt.show()

# calculate state posteriors for wordHMMs['o']
gamma_prob = proto2.statePosteriors(forward_prob, backward_prob)



#score all utterances using viterbi
test_viterbi(data, wordHMMs)

