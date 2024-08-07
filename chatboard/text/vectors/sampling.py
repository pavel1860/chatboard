import random

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def under_sample_kmeans_df(df, k=None, label_column='label', embedding_column='vector'):
    if k == None:
        k = min(df[label_column].value_counts())
    sample_df_list = []
    for label in df[label_column].unique():
        label_df = df[df[label_column] == label].copy()
        arr = np.array(label_df[embedding_column].to_list())
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(arr)
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
        label_df['cluster'] = labels
        sample_df_list.append(label_df.groupby('cluster').sample(1))
    return pd.concat(sample_df_list)



def kmeans_undersample(examples, k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit([e.vector['dense'] for e in examples])
    relevant_examples = []
    for label in range(k):
        relevant_examples.append(
            random.choice([
                ex for l, ex in zip(kmeans.labels_, examples) if l == label
            ]))
    return relevant_examples



# def under_sample_kmeans(examples, k=None, label_column='label', embedding_column='vector'):
#     if hasattr(examples, label_column)
#     if k == None:
#         k = min(df[label_column].value_counts())
#     sample_df_list = []
#     for label in df[label_column].unique():
#         label_df = df[df[label_column] == label].copy()
#         arr = np.array(label_df[embedding_column].to_list())
#         kmeans = KMeans(n_clusters=k)
#         kmeans.fit(arr)
#         centroids = kmeans.cluster_centers_
#         labels = kmeans.labels_
#         label_df['cluster'] = labels
#         sample_df_list.append(label_df.groupby('cluster').sample(1))
#     return pd.concat(sample_df_list)

