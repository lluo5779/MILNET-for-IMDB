import sklearn.manifold as mani
import numpy as np
import collections


## Sample Window Acquisition
window_start = 1
window_size = 200

np.random.seed(13337)
#word_emb_mat = np.random.rand(200000,10)
word_emb_mat = np.load('D:/Osaka_Data_Only/data/weights.npy')
x_train = np.load('D:/Osaka_Data_Only/data/x_test_sort.npy')

flat_list = [word for doc in x_train for sent in doc for word in sent]

counts = collections.Counter(flat_list)
most_w_f = counts.most_common(n = window_size+window_start)[window_start:window_size+1]

most_w_idx = [tuple[0] for tuple in most_w_f]
most_w_mat = [word_emb_mat[idx] for idx in most_w_idx]

### Manifold Learning
## LLE
n_neighbours = 20
n_components = word_emb_mat.shape[1]//2

LLE = mani.LocallyLinearEmbedding(n_neighbors = n_neighbours, n_components = n_components)

new_space = LLE.fit(most_w_mat)
old_vector = word_emb_mat#[1000:1010]
lle_weights = new_space.transform(old_vector)

#np.save('weights_LLE.npy', new_vector)

## ISOMAP
n_neighbours = 20
n_components = word_emb_mat.shape[1]#//2

Isomap = mani.Isomap(n_neighbors = n_neighbours, n_components = n_components)

new_space = Isomap.fit(most_w_mat)
old_vector = word_emb_mat[1000:1010]
new_vector = new_space.transform(old_vector)

##
def replace_matched_items(docs, dictionary):
    new_list = [[[dictionary.get(word, word) for word in sent] for sent in doc] for doc in docs]
    return new_list

list_ = replace_matched_items(x_train.tolist(), collections.Counter())
