import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture


def load_data():
    data_path = './data/Tweets.txt'
    corpus, labels = [], []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            j = json.loads(line.strip())
            corpus.append(j['text'])
            labels.append(j['cluster'])
    return corpus, labels


def my_kmeans(corpus, labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)

    kmeans = KMeans(n_clusters=89, max_iter=50, n_init=10, init='k-means++')
    result_kmeans = kmeans.fit_predict(X.toarray())
    print('K-means accuracy:', normalized_mutual_info_score(result_kmeans, labels))
    # K-means accuracy: 0.790478561984479


def my_AffinityPropagation(corpus, labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)

    affinity_propagation = AffinityPropagation(damping=.55, max_iter=200, convergence_iter=15, copy=False)
    result_affinity_propagation = affinity_propagation.fit_predict(X.toarray())
    print('AffinityPropagation accuracy:', normalized_mutual_info_score(result_affinity_propagation, labels))
    # AffinityPropagation accuracy: 0.7834777200368181


def my_mean_shift(corpus, labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)

    mean_shift = MeanShift(bandwidth=0.65, bin_seeding=True)
    result_mean_shift = mean_shift.fit_predict(X.toarray())
    print('MeanShift accuracy:', normalized_mutual_info_score(result_mean_shift, labels))
    # MeanShift accuracy: -0.7265625
    # MeanShift accuracy: 0.7468492000608157


def my_SpectralClustering(corpus, labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)

    spectral_clustering = SpectralClustering(n_clusters=89, n_init=10)
    result_spectral_clustering = spectral_clustering.fit_predict(X.toarray())
    print('SpectralClustering accuracy:', normalized_mutual_info_score(result_spectral_clustering, labels))
    # SpectralClustering accuracy: 0.6766616506622629


def my_AgglomerativeClustering(corpus, labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)

    agglomerative_clustering = AgglomerativeClustering(n_clusters=89)
    result_agglomerative_clustering = agglomerative_clustering.fit_predict(X.toarray())
    print('AgglomerativeClustering accuracy:', normalized_mutual_info_score(result_agglomerative_clustering, labels))
    # AgglomerativeClustering accuracy: 0.7800394104591923


def my_DBSCAN(corpus, labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)

    dbscan = DBSCAN(eps=0.5, min_samples=1, leaf_size=30)
    result_dbscan = dbscan.fit_predict(X.toarray())
    print('DBSCAN accuracy:', normalized_mutual_info_score(result_dbscan, labels))
    # DBSCAN accuracy: 0.7009526046894612
    # 如果min_samples 设置为默认的5，则accuracy是0.1080121348508573


def my_GaussianMixture(corpus, labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)

    gaussian_mixture = GaussianMixture(n_components=89)
    result_gaussian_mixture = gaussian_mixture.fit_predict(X.toarray())
    print('GaussianMixture accuracy:', normalized_mutual_info_score(result_gaussian_mixture, labels))
    # GaussianMixture accuracy: 0.792451745546511


if __name__ == '__main__':
    corpus, labels = load_data()
    my_kmeans(corpus, labels)
    my_AffinityPropagation(corpus, labels)
    my_mean_shift(corpus, labels)
    my_SpectralClustering(corpus, labels)
    my_AgglomerativeClustering(corpus, labels)
    my_DBSCAN(corpus, labels)
    my_GaussianMixture(corpus, labels)