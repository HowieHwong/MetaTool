import json
import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
import pickle
import sklearn
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


class ClusterTools:
    def __init__(self, filename, savename):
        self.filename = filename
        self.savename = savename

    def read_data(self):
        if not self.filename.endswith('.txt'):
            data = pickle.load(open(self.filename, 'rb'))
            embeddings = [d['embedding'] for d in data]
        else:
            data = open(self.filename, 'r').readlines()
            data = [eval(el.strip('\n')) for el in data]
            embeddings = [d['human_embedding'] for d in data]
        return data, embeddings

    def save_cluster_results(self, data, labels, silhouette_score_samples):
        try:
            model_name = [el['model_name'] for el in data]
        except:
            model_name = [el['name_for_model'] for el in data]
        cluster_label = labels
        pd.DataFrame({'model_name': model_name, 'cluster_label': cluster_label,
                      'silhouette_score': silhouette_score_samples}).to_csv(self.savename, index=False)


class KMeansCluster(ClusterTools):
    def __init__(self, filename, savename, num_clusters):
        super().__init__(filename, savename)
        self.num_clusters = num_clusters

    def cluster_data(self):
        data, embeddings = self.read_data()
        kmeans = KMeans(n_clusters=self.num_clusters)
        kmeans.fit(embeddings)
        labels = kmeans.labels_
        for i, d in enumerate(data):
            d['cluster_label'] = labels[i]

        silhouette_score = sklearn.metrics.silhouette_score(embeddings, labels, metric='euclidean', sample_size=None,
                                                            random_state=None)
        silhouette_score_samples = sklearn.metrics.silhouette_samples(embeddings, labels)
        print(silhouette_score)
        self.save_cluster_results(data, labels, silhouette_score_samples)


class VisualizeCluster:
    def __init__(self, filename, savename, num_clusters, savefig, visual_dim=2):
        self.filename = filename
        self.savename = savename
        self.num_clusters = num_clusters
        self.savefig = savefig
        self.visual_dim = visual_dim

    def cluster_data(self):
        data, embeddings = ClusterTools(self.filename, self.savename).read_data()
        kmeans = KMeans(n_clusters=self.num_clusters)
        kmeans.fit(embeddings)
        labels = kmeans.labels_
        for i, d in enumerate(data):
            d['cluster_label'] = labels[i]

        if self.visual_dim == 2:
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(np.array(embeddings))

            plt.figure(figsize=(6, 5))
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
            plt.colorbar(label='Cluster Label')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.savefig(self.savefig, dpi=200)
            plt.show()
        else:
            tsne = TSNE(n_components=3, random_state=42)
            X_tsne = tsne.fit_transform(embeddings)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=labels, cmap='viridis', alpha=0.7)
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.savefig(self.savefig, dpi=200)
            plt.show()

        silhouette_score = sklearn.metrics.silhouette_score(embeddings, labels, metric='euclidean', sample_size=None,
                                                            random_state=None)
        silhouette_score_samples = sklearn.metrics.silhouette_samples(embeddings, labels)
        print(silhouette_score)
        ClusterTools(self.filename, self.savename).save_cluster_results(data, labels, silhouette_score_samples)


class EnsembleCluster:
    def __init__(self, filename, savename, cluster_times):
        self.filename = filename
        self.savename = savename
        self.cluster_times = cluster_times

    def cluster_data(self):
        data = open(self.filename, 'r').readlines()
        data = [eval(el.strip('\n')) for el in data]
        embeddings = np.array([d['human_embedding'] for d in data])

        num_clusters = 5
        kmeans_results = []
        for _ in range(num_clusters):
            kmeans = KMeans(n_clusters=20)
            kmeans.fit(embeddings)
            kmeans_results.append(kmeans.labels_)

        final_labels = []
        for i in range(len(data)):
            votes = [result[i] for result in kmeans_results]
            final_labels.append(max(set(votes), key=votes.count))

        pd.DataFrame({'model_name': [el['model_name'] for el in data], 'cluster_label': final_labels}).to_csv(self.savename, index=False)


class HierarchyCluster(ClusterTools):
    def __init__(self, filename, savename, threshold=0.5):
        super().__init__(filename, savename)
        self.threshold = threshold

    def cluster_data(self):
        data, embeddings = self.read_data()

        Z = linkage(np.array(embeddings), method='ward')
        print(Z)

        plt.figure(figsize=(20, 5), dpi=200)
        dendrogram(Z)
        plt.title('Dendrogram')
        plt.xlabel('Data Points')
        plt.ylabel('Distance')
        plt.savefig('hierarchy.pdf')
        plt.show()

        labels = fcluster(Z, self.threshold, criterion='distance')
        model_name = [el['model_name'] for el in data]
        df = pd.DataFrame({'Data Point': model_name, 'Cluster': labels})
        df.to_csv(self.savename, index=False)


def get_embedding(text: str, model="text-embedding-ada-002"):
    response = openai.Embedding.create(
        model=model,
        input=[text.replace("\n", " ")]
    )

    embedding = response["data"][0]["embedding"]
    return np.array(embedding)


def visual_overlapped_efficiency():
    with open('cluster_score.json', 'r') as file:
        data = json.load(file)

    nums = [entry['num'] for entry in data]
    new_scores = [entry['new_score'] for entry in data]
    original_scores = [entry['original_score'] for entry in data]

    plt.figure(figsize=(8, 4))
    plt.plot(nums, new_scores, label='New', marker='o', linestyle='-')
    plt.plot(nums, original_scores, label='Original', marker='s', linestyle='--')

    plt.xlabel('Cluster Number')
    plt.ylabel('Score')
    plt.legend()

    plt.grid(True)
    plt.savefig('cluster_score.pdf')
    plt.show()


if __name__ == '__main__':
    pass