import numpy as np
# from scipy.spatial.distance import pdist, squareform
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag


def centroid_mesure(vectors):
    distances = pdist(vectors, metric='euclidean')
    diversity_measure = np.mean(distances)
    return diversity_measure


def pair_wise_mesure(vectors):
    pairwise_distances = pdist(vectors, metric='euclidean')
    distance_matrix = squareform(pairwise_distances)
    mean_distance = np.mean(pairwise_distances)
    median_distance = np.median(pairwise_distances)
    std_deviation = np.std(pairwise_distances)
    max_distance = np.max(pairwise_distances)
    min_distance = np.min(pairwise_distances)

    return mean_distance, median_distance, std_deviation, max_distance, min_distance


def variance_mesure(vectors):
    '''
        Higher Aggregate Variance: Indicates more diversity across the dimensions of your embeddings.
        Lower Aggregate Variance: Suggests less diversity, meaning the vectors are more similar to each other across their dimensions.
    '''
    variances = np.var(vectors, axis=0)    
    return np.mean(variances)



def mean_distance_from_vector(vectors, center_vector):
    # distances = pdist(vectors, metric='euclidean')
    distances = np.linalg.norm(vectors - center_vector, axis=1)
    diversity_measure = np.mean(distances)
    return diversity_measure



def get_important_words(sentence):
    # Tokenize the sentence
    tokens = word_tokenize(sentence)
    # Part-of-Speech Tagging
    tagged = pos_tag(tokens)
    # Tags of interest, excluding VBZ and VBP
    tags_of_interest = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN']
    # Extracting important words based on the modified tags of interest
    important_words = [word for word, tag in tagged if tag in tags_of_interest]
    return set(important_words)



def word_overlap_mesure(sentence, comments):
    message_set = get_important_words(sentence)
    overlaps = []
    for c in comments:
        c_set = get_important_words(c)
        overlaps.append(len(message_set.intersection(c_set)) / len(message_set))
    return sum(overlaps) / len(overlaps)
