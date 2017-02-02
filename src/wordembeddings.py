import numpy as np
import operator
import re
import string
from collections import Counter
from nltk.corpus import brown, stopwords
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

def main():
    base_dir = 'C:/Users/jmzoo/Dropbox/School/CSE/250B/'
    stop_set = set(stopwords.words('english'))
    corpus = standardize_corpus(brown.words(), ignore_words=stop_set)
    word_counts = Counter(corpus)
    top_word_counts = sorted_by_value(word_counts, reverse=True)
    words = [word for word, count in top_word_counts]

    # point-wise mutual information embedding
    v, c, d = 5000, 1000, 100
    word_map = dict((words[i], i) for i in xrange(v))
    adjacencies = get_ngram_adjacencies(corpus, word_map, n=3, dim=(v,c))
    prob_c = np.sum(adjacencies, axis=0).astype('float') / np.sum(adjacencies)
    prob_c_w = adjacencies.astype('float') / np.sum(adjacencies, 1).reshape((v,1))
    phi_c_w = np.maximum(0, np.nan_to_num(np.log(prob_c_w) - np.log(prob_c)))
    U, s, V = np.linalg.svd(phi_c_w, full_matrices=False)
    embedded_vocab = np.dot(s[0:d]*U[:,0:d], V[0:d,:])

    # k-means clustering (k = 100)
    kmeans = KMeans(n_clusters=100).fit(embedded_vocab)
    clusters = get_clusters(kmeans.labels_)
    with open(base_dir + 'word_clusters.txt', 'w') as f:
        f.writelines((ids_to_line(i, c, words) for i, c in clusters.iteritems()))

    # nearest neighbor (n = 4)
    with open(base_dir + 'word_samples.txt', 'r') as f:
        word_samples = f.read().splitlines()
    samples = [word_map[word] for word in word_samples]
    neighs = NearestNeighbors(metric_params={'func': cosine_distance},
                              metric='pyfunc', n_neighbors=4).fit(embedded_vocab)
    neighbors = neighs.kneighbors(embedded_vocab[samples,:], return_distance=False)
    with open(base_dir + 'nearest_neighbors.txt', 'w') as f:
        lines = (ids_to_line(word_samples[i], n, words) for i, n in enumerate(neighbors))
        f.writelines(lines)

def standardize_corpus(corpus, ignore_words):
    words = (standardize_word(word) for word in corpus)
    return [word for word in words if len(word) > 0 and word not in ignore_words]

def standardize_word(word, ignore_chars='[^a-zA-Z]'):
    return re.sub(ignore_chars, '', string.lower(str(word)))

def sorted_by_value(dictionary, reverse=False):
    return sorted(dictionary.iteritems(), key=operator.itemgetter(1), reverse=reverse)

def iterate_ngram_pairs(corpus, n):
    for i, word in enumerate(corpus):
        for j in xrange(max(0, i-n+1), i):
            yield word, corpus[j]

def get_ngram_adjacencies(corpus, word_map, n, dim):
    adjacencies = np.zeros(dim)
    rows, cols = dim

    for word, context_word in iterate_ngram_pairs(corpus, n):
        word_rank = word_map.get(word, float('inf'))
        context_word_rank = word_map.get(context_word, float('inf'))
        if word_rank < rows and context_word_rank < cols:
            adjacencies[word_rank, context_word_rank] += 1
        if word_rank < cols and context_word_rank < rows:
            adjacencies[context_word_rank, word_rank] += 1
    return adjacencies

def get_clusters(labels):
    clusters = {}
    for i, label in enumerate(labels):
        if label in clusters:
            clusters[label].append(i)
        else:
            clusters[label] = [i]
    return clusters

def ids_to_line(line_label, word_ids, vocab):
    return words_to_line(line_label, ids_to_words(word_ids, vocab))

def ids_to_words(word_ids, vocab):
    return [vocab[word_id] for word_id in word_ids]

def words_to_line(line_label, words):
    return '%s: %s\n' % (str(line_label), ' '.join(words))

def cosine_distance(x, y):
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

if __name__ == '__main__':
    main()
