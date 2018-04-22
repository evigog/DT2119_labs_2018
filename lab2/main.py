import numpy as np
import proto2
import prondict

import matplotlib.pyplot as plt

data = np.load('lab2_data.npz')['data']
phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()
example = np.load('lab2_example.npz')['example'].item()


modellist = {}
for digit in prondict.prondict.keys():
    modellist[digit] = ['sil'] + prondict.prondict[digit] + ['sil']

wordHMMs = {}
wordHMMs['o'] = proto2.concatHMMs(phoneHMMs, modellist['o'])

obsloglik = example['obsloglik']
startprob = wordHMMs['o']['startprob']
transmat = wordHMMs['o']['transmat'][:-1, :-1]
forward_prob = proto2.forward(obsloglik, np.log(startprob), np.log(transmat))

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
