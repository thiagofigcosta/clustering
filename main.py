import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler

# configs
DATASET_PATH = 'res/IMDB_movie_reviews_details.csv'
REPORT_PATH = 'report.json'
BLOCK_PLOTS = True
PLOT = True
NORMALIZE_FEATURES = True
SILHOUETTE_COMPLEX_PENALTY = 0.07
RUN_KMEANS = True
KMEANS_MAX_CLUSTERS = 10
KMEANS_MAX_ITERS = 500
KMEANS_MAX_RUNS = 11
RUN_DBSCAN = True
DBSCAN_MAX_NEIGHBOURHOOD_DISTANCE = 2.14
DBSCAN_MIN_NEIGHBOURHOOD_SIZE = 10
RUN_GAUSSIAN_MIX = True
GAUSSIAN_MIX_MAX_ITER = 500
GAUSSIAN_MIX_MAX_CLUSTERS = 10


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_results_report(rep):
    try:
        with open(REPORT_PATH, 'w') as f:
            json.dump(rep, f, cls=NumpyEncoder, indent=2)
    except Exception as e:
        print(e)


class Sequentializer():
    def __init__(self):
        self.kv_dict = {}
        self.vk_list = []

    def addId(self, normal_id):
        if normal_id is None:
            raise Exception()
        if normal_id not in self.kv_dict:
            seq_id = len(self.vk_list)
            self.kv_dict[normal_id] = seq_id
            self.vk_list.append(normal_id)
            return seq_id
        return None

    def addBaseTwoId(self, normal_id):
        if normal_id is None:
            raise Exception()
        if normal_id not in self.kv_dict:
            seq_id = 2 ** len(self.vk_list)
            self.kv_dict[normal_id] = seq_id
            self.vk_list.append(normal_id)
            return seq_id
        return None

    def getSeqId(self, normal_id):
        return self.kv_dict.get(normal_id, None)

    def getNormalId(self, seq_id, base_two=False):
        if base_two:
            seq_id = math.log2(seq_id)
        if 0 <= seq_id < len(self.vk_list):
            return self.vk_list[seq_id]
        return None

    def getSeqIdOrAdd(self, normal_id):
        seq_id = self.addId(normal_id)
        if seq_id is None:
            seq_id = self.getSeqId(normal_id)
        return seq_id

    def getSeqBaseTwoIdOrAdd(self, normal_id):
        seq_id = self.addBaseTwoId(normal_id)
        if seq_id is None:
            seq_id = self.getSeqId(normal_id)
        return seq_id


# data loading

df = pd.read_csv(DATASET_PATH)

print(df.head())

amount_of_nan_name = df[df['name'].isnull()].shape[0]
amount_of_nan_year = df[df['year'].isnull()].shape[0]
amount_of_nan_runtime = df[df['runtime'].isnull()].shape[0]
amount_of_nan_genre = df[df['genre'].isnull()].shape[0]
amount_of_nan_rating = df[df['rating'].isnull()].shape[0]
amount_of_nan_metascore = df[df['metascore'].isnull()].shape[0]
amount_of_nan_timeline = df[df['timeline'].isnull()].shape[0]
amount_of_nan_votes = df[df['votes'].isnull()].shape[0]
amount_of_nan_gross = df[df['gross'].isnull()].shape[0]

print(f'Invalid records by name: {amount_of_nan_name}')
print(f'Invalid records by year: {amount_of_nan_year}')
print(f'Invalid records by runtime: {amount_of_nan_runtime}')
print(f'Invalid records by genre: {amount_of_nan_genre}')
print(f'Invalid records by rating: {amount_of_nan_rating}')
print(f'Invalid records by metascore: {amount_of_nan_metascore}')
print(f'Invalid records by timeline: {amount_of_nan_timeline}')
print(f'Invalid records by votes: {amount_of_nan_votes}')
print(f'Invalid records by gross: {amount_of_nan_gross}')

# filter invalid data
df.dropna(subset=['metascore'], inplace=True)
df.dropna(subset=['gross'], inplace=True)

print(df.head())

# data preparation
df['year'] = df['year'].apply(lambda x: int(''.join(filter(str.isdigit, x))))
df['runtime'] = df['runtime'].apply(int)
df['rating'] = df['rating'].apply(float)
df['metascore'] = df['metascore'].apply(int)
df['votes'] = df['votes'].apply(lambda x: int(''.join(filter(str.isdigit, x))))
df['genre'] = df['genre'].apply(lambda x: set([el.strip() for el in x.split(',')]))

amount_of_gross_not_in_million = df.shape[0] - df[df['gross'].str.contains('M')].shape[0]
print(f'Amount of gross not in millions scale {amount_of_gross_not_in_million}')

df['gross'] = df['gross'].apply(lambda x: float(''.join(filter(lambda y: y.isdigit() or y == '.', x))))

print(df.head())

# data understanding
print('Statistical analysis')
amount_of_movies = df.shape[0]
first_year = df['year'].min()
last_year = df['year'].max()
distinct_years = df['year'].nunique()

print(f'First year: {first_year}, Last year: {last_year}, Distinct years: {distinct_years}')
if PLOT:
    grouped_year = df.groupby('year')
    ax = grouped_year['year'].count().plot(kind='bar', title='Movies per year', figsize=(14, 4.8))
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Year')
    plt.tight_layout()
    plt.show(block=BLOCK_PLOTS)

distinct_genres = df['genre'].explode('genre').nunique()
print(f'Distinct genre: {distinct_genres}')
if PLOT:
    grouped_genre = df.explode('genre').groupby('genre')
    ax = grouped_genre['genre'].count().plot(kind='bar', title='Genre Occurrence', figsize=(14, 4.8))
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Genre')
    plt.tight_layout()
    plt.show(block=BLOCK_PLOTS)

min_rate = df['rating'].min()
max_rate = df['rating'].max()
print(f'Minimum rate: {min_rate}, Maximum rate: {max_rate}')
if PLOT:
    bin_width = 0.2
    bins = np.arange(math.floor(df['rating'].min()), math.ceil(df['rating'].max()) + bin_width, bin_width)
    ax = df['rating'].plot.hist(bins=bins, title='Ratings histogram', rwidth=0.5)
    ax.set_xlabel('Rating')
    plt.tight_layout()
    plt.xticks(bins)
    plt.show(block=BLOCK_PLOTS)

min_runtime = df['runtime'].min()
max_runtime = df['runtime'].max()
print(f'Minimum runtime: {min_runtime}, Maximum runtime: {max_runtime}')
if PLOT:
    bin_width = 10
    bins = np.arange(math.floor(min_runtime), math.ceil(max_runtime) + bin_width, bin_width)
    ax = df['runtime'].plot.hist(bins=bins, title='Runtime histogram', rwidth=0.5, figsize=(12, 4.8))
    ax.set_xlabel('Runtime (minutes)')
    plt.tight_layout()
    plt.xticks(bins)
    plt.show(block=BLOCK_PLOTS)

min_metascore = df['metascore'].min()
max_metascore = df['metascore'].max()
print(f'Minimum metascore: {min_metascore}, Maximum metascore: {max_metascore}')
if PLOT:
    bin_width = 5
    bins = np.arange(math.floor(min_metascore), math.ceil(max_metascore) + bin_width, bin_width)
    ax = df['metascore'].plot.hist(bins=bins, title='Metascore histogram', rwidth=0.5)
    ax.set_xlabel('Metascore')
    plt.tight_layout()
    plt.xticks(bins)
    plt.show(block=BLOCK_PLOTS)

min_votes = df['votes'].min()
max_votes = df['votes'].max()
print(f'Minimum votes: {min_votes}, Maximum votes: {max_votes}')
if PLOT:
    amount_bins = 15
    bin_width = int((math.ceil(max_votes) - math.floor(min_votes)) / amount_bins)
    bins = np.arange(math.floor(min_votes), math.ceil(max_votes) + bin_width, bin_width)
    ax = df['votes'].plot.hist(bins=bins, title='Votes histogram', rwidth=0.5, figsize=(12, 4.8))
    ax.set_xlabel('Votes')
    plt.tight_layout()
    plt.xticks(bins)
    plt.show(block=BLOCK_PLOTS)

min_gross = df['gross'].min()
max_gross = df['gross'].max()
print(f'Minimum gross: {min_gross}, Maximum gross: {max_gross}')
if PLOT:
    amount_bins = 15
    bin_width = int((math.ceil(max_gross) - math.floor(min_gross)) / amount_bins)
    bins = np.arange(math.floor(min_gross), math.ceil(max_gross) + bin_width, bin_width)
    ax = df['gross'].plot.hist(bins=bins, title='Gross earnings histogram', rwidth=0.5)
    ax.set_xlabel('Gross earnings (millions of USD)')
    plt.tight_layout()
    plt.xticks(bins)
    plt.show(block=BLOCK_PLOTS)

# last plot
if not BLOCK_PLOTS:
    plt.show()

print()

# data preparation - creating features
features = []
movies = []
genres_sequentializer = Sequentializer()
for i, (idx, row) in enumerate(df.iterrows()):
    movies.append(row['name'].strip())
    row_features = []
    row_features.append(row['year'])  # 0
    row_features.append(row['runtime'])  # 1
    row_features.append(row['rating'])  # 2
    row_features.append(row['metascore'])  # 3
    row_features.append(row['votes'])  # 4
    row_features.append(row['gross'])  # 5
    # row_features.append(row['timeline']) use RAKE to create features from text (lemmatization -> rake -> stop word)
    genres_power_two = [genres_sequentializer.getSeqBaseTwoIdOrAdd(el) for el in row['genre']]
    genres = sum(genres_power_two)
    genres_multi_hot = list(bin(genres).split('b')[1])  # I could use sklearn.preprocessing.MultiLabelBinarizer instead
    genres_multi_hot.reverse()
    genres_multi_hot = [int(el) for el in genres_multi_hot]
    if len(genres_multi_hot) < distinct_genres:
        genres_multi_hot += [0] * (distinct_genres - len(genres_multi_hot))
    row_features += genres_multi_hot
    features.append(row_features)
print()

# sklearn.pipeline.Pipeline can be used to run ColumnTransformer + models ...
preprocessor = ColumnTransformer(
    transformers=[
        # if I don't do the dummy-passthrough, order won't be kept, and index 0 will be after the last transform
        ('dummy', 'passthrough', [0]),  # 0
        ('min_max', MinMaxScaler(), slice(1, 6))  # from 1 to 5 inclusively
    ],
    remainder='passthrough')

if NORMALIZE_FEATURES:
    features = preprocessor.fit_transform(features)  # normalize
    # preprocessor['The transformer name'].inverse_transform(features) # to revert, need to do one by one

# if true labels are known we can use the metrics:
#       Rand index
#       Mutual Information based score
#       Homogeneity
#       Completeness
#       V-measure
#       Fowlkes-Mallows score
# if true labels are unknown we can use the metrics:
#       Silhouette Score
#       Calinski Harabasz Score
#       Davies Bouldin Score

clusters = []
# run kmeans
if RUN_KMEANS:
    graph_x = []
    elbow_graph = []
    silhouette_graph = []
    silhouette_auto_analyser = []
    for c in range(2, 2 + KMEANS_MAX_CLUSTERS):
        kmeans = KMeans(n_clusters=c, init='k-means++', max_iter=KMEANS_MAX_ITERS, n_init=KMEANS_MAX_RUNS, tol=1e-4)
        preds = kmeans.fit_predict(features)
        cluster_centroids = kmeans.cluster_centers_

        cluster_members = [[] for _ in range(c)]
        for m, movie in enumerate(movies):
            cluster_members[preds[m]].append(movie)
        for cm in cluster_members:
            cm.sort()

        inertia = kmeans.inertia_
        silhouette_score = metrics.silhouette_score(features, preds, metric='euclidean')
        silhouette_samples = metrics.silhouette_samples(features, preds, metric='euclidean')
        calinski_harabasz_score = metrics.calinski_harabasz_score(features, preds)
        davies_bouldin_score = metrics.davies_bouldin_score(features, preds)
        graph_x.append(c)
        elbow_graph.append(inertia)
        silhouette_graph.append(silhouette_score)

        cluster_result = {
            'method': 'kmeans',
            'amount_clusters': c,
            'predictions': str(preds.tolist()),
            'centroids': str(cluster_centroids.tolist()),
            'cluster_members': [str(el) for el in cluster_members],
            'metrics': {
                'wcss': inertia,
                # inertia or Within Cluster Sum of Squares (WCSS), measures coherence, the lower, the better
                'silhouette': silhouette_score,  # measures density and separation intra-cluster, the higher, the better
                'silhouette_samples': str(silhouette_samples.tolist()),
                'calinski_harabasz': calinski_harabasz_score,  # the higher, the better defined
                'davies_bouldin': davies_bouldin_score,  # the lower, the better separation
            }
        }
        print(json.dumps(cluster_result, indent=2, cls=NumpyEncoder))
        print()
        clusters.append(cluster_result)
        y_lower = y_upper = 0
        local_silhouette_auto_analyser = [0 for _ in range(c)]  # the most points should be above average line
        for label in range(c):
            label_silhouette_samples = [el for p, el in enumerate(silhouette_samples) if preds[p] == label]
            label_silhouette_samples.sort()
            for el in reversed(label_silhouette_samples):
                dt = el - silhouette_score  # discrete integration
                if dt > 0:
                    local_silhouette_auto_analyser[label] += dt
                else:
                    break
            if PLOT:
                y_upper += len(label_silhouette_samples)
                plt.fill_betweenx(np.arange(y_lower, y_upper), 0, label_silhouette_samples, alpha=.8,
                                  label='_nolegend_')
                plt.text(-0.05, (y_lower + y_upper) / 2 - 11, f'C:{label}')  # 10 is the default font size
                y_lower += len(label_silhouette_samples)
        silhouette_auto_analyser.append(local_silhouette_auto_analyser)
        if PLOT:
            plt.axvline(x=silhouette_score, linestyle='--', color='red', label='Mean silhouette score')
            ax = plt.gca()
            ax.set_xlabel('Silhouette score')
            ax.set_ylabel('Cluster classe')
            plt.title(f'Silhouette method - KMeans size {c}')
            plt.yticks([])
            plt.xlim([min(silhouette_samples) - 0.13, 1])
            plt.tight_layout()
            plt.legend()
            plt.show(block=BLOCK_PLOTS)

    optimum_cluster_size_bic_idx = KneeLocator(graph_x, elbow_graph, curve='convex', direction='decreasing')
    print(f'Optimum size for kmeans using elbow method: {optimum_cluster_size_bic_idx.elbow}')

    # computes the graph area over the average line, also it prioritizes the simpler configs
    silhouette_auto_analyser = [sum(els) - SILHOUETTE_COMPLEX_PENALTY * idx for idx, els in
                                enumerate(silhouette_auto_analyser)]

    best_size_according_to_silu = graph_x[silhouette_auto_analyser.index(max(silhouette_auto_analyser))]
    print(f'Optimum size for kmeans using silhouette method: {best_size_according_to_silu}')
    if PLOT:
        plt.plot(graph_x, elbow_graph, '-o', label='_nolegend_', zorder=1)
        plt.scatter([optimum_cluster_size_bic_idx.elbow],
                    [optimum_cluster_size_bic_idx.elbow_y], color='red', label='optimum size',
                    zorder=2)
        ax = plt.gca()
        ax.set_xlabel('Cluster size')
        ax.set_ylabel('Inertia')
        plt.title('Elbow Method - KMeans')
        plt.xticks(graph_x)
        plt.tight_layout()
        plt.legend()
        plt.show(block=BLOCK_PLOTS)

        plt.plot(graph_x, silhouette_graph, '-o')
        ax = plt.gca()
        ax.set_xlabel('Cluster size')
        ax.set_ylabel('Silhouette')
        plt.title('Silhouettes - KMeans')
        plt.xticks(graph_x)
        plt.tight_layout()
        plt.show(block=BLOCK_PLOTS)
print()
print()
# run dbscan
if RUN_DBSCAN:  # I could loop through some arrays of eps and min_samples, but I believe I got good metrics this way
    dbscan = DBSCAN(eps=DBSCAN_MAX_NEIGHBOURHOOD_DISTANCE, min_samples=DBSCAN_MIN_NEIGHBOURHOOD_SIZE,
                    metric='euclidean')
    preds = dbscan.fit_predict(features)
    amount_clusters = len(set(preds)) - (1 if -1 in preds else 0)
    noisy_points = list(preds).count(-1)
    cluster_members = [[] for _ in range(amount_clusters)]
    noisy_members = []
    cluster_points = [[] for _ in range(amount_clusters)]
    for m, movie in enumerate(movies):
        c_idx = preds[m]
        if c_idx >= 0:
            cluster_members[c_idx].append(movie)
            cluster_points[c_idx].append(features[m])
        else:
            noisy_members.append(movie)
    for cm in cluster_members:
        cm.sort()
    cluster_centroids = []
    for c_points in cluster_points:
        cluster_centroids.append(np.mean(c_points, axis=0))

    sum_of_squared_error = [0 for _ in range(c)]
    for m, point in enumerate(features):
        label = preds[m]
        sum_of_squared_error[label] += np.square(point - cluster_centroids[label]).sum()
    wcss = sum(sum_of_squared_error)

    silhouette_score = metrics.silhouette_score(features, preds, metric='euclidean')
    silhouette_samples = metrics.silhouette_samples(features, preds, metric='euclidean')
    calinski_harabasz_score = metrics.calinski_harabasz_score(features, preds)
    davies_bouldin_score = metrics.davies_bouldin_score(features, preds)

    cluster_result = {
        'method': 'dbscan',
        'amount_clusters': amount_clusters,
        'predictions': str(preds.tolist()),
        'centroids': str([arr.tolist() for arr in cluster_centroids]),
        'cluster_members': [str(el) for el in cluster_members],
        'noisy_members': [str(el) for el in noisy_members],
        'metrics': {
            'wcss': wcss,
            'noise_ratio': noisy_points / len(preds),  # not necessarily a low value is a good thing
            'silhouette': silhouette_score,  # measures density and separation intra-cluster, the higher, the better
            'silhouette_samples': str(silhouette_samples.tolist()),
            'calinski_harabasz': calinski_harabasz_score,  # the higher, the better defined
            'davies_bouldin': davies_bouldin_score,  # the lower, the better separation
        }
    }
    print(json.dumps(cluster_result, indent=2, cls=NumpyEncoder))
    clusters.append(cluster_result)
print()
print()

if RUN_GAUSSIAN_MIX:
    graph_x = []
    bic_graph = []
    silhouette_graph = []
    silhouette_auto_analyser = []
    for c in range(2, 2 + GAUSSIAN_MIX_MAX_CLUSTERS):
        gm = GaussianMixture(n_components=c, max_iter=GAUSSIAN_MIX_MAX_ITER, tol=1e-4)
        preds = gm.fit_predict(features)
        cluster_points = [[] for _ in range(c)]
        cluster_members = [[] for _ in range(c)]
        for m, movie in enumerate(movies):
            c_idx = preds[m]
            cluster_members[c_idx].append(movie)
            cluster_points[c_idx].append(features[m])
        for cm in cluster_members:
            cm.sort()
        cluster_centroids = []
        for c_points in cluster_points:
            cluster_centroids.append(np.mean(c_points, axis=0))

        cluster_members = [[] for _ in range(c)]
        for m, movie in enumerate(movies):
            cluster_members[preds[m]].append(movie)
        for cm in cluster_members:
            cm.sort()

        sum_of_squared_error = [0 for _ in range(c)]
        for m, point in enumerate(features):
            label = preds[m]
            sum_of_squared_error[label] += np.square(point - cluster_centroids[label]).sum()
        wcss = sum(sum_of_squared_error)

        converged = gm.converged_
        aic_score = gm.aic(features)
        bic_score = gm.bic(features)
        silhouette_score = metrics.silhouette_score(features, preds, metric='euclidean')
        silhouette_samples = metrics.silhouette_samples(features, preds, metric='euclidean')
        calinski_harabasz_score = metrics.calinski_harabasz_score(features, preds)
        davies_bouldin_score = metrics.davies_bouldin_score(features, preds)
        graph_x.append(c)
        bic_graph.append(bic_score)
        silhouette_graph.append(silhouette_score)

        cluster_result = {
            'method': 'gaussian mixture',
            'amount_clusters': c,
            'predictions': str(preds.tolist()),
            'centroids': str([arr.tolist() for arr in cluster_centroids]),
            'cluster_members': [str(el) for el in cluster_members],
            'metrics': {
                'wcss': wcss,  # inertia or Within Cluster Sum of Squares, measures coherence, the lower, the better
                'converged': converged,
                'silhouette': silhouette_score,  # measures density and separation intra-cluster, the higher, the better
                'silhouette_samples': str(silhouette_samples.tolist()),
                'calinski_harabasz': calinski_harabasz_score,  # the higher, the better defined
                'davies_bouldin': davies_bouldin_score,  # the lower, the better separation
                'aic_score': aic_score,  # the lower, the better, measures how well a model fits data
                'bic_score': bic_score,  # the lower, the better, measures how well a model describes the unknown
            }
        }
        print(json.dumps(cluster_result, indent=2, cls=NumpyEncoder))
        print()
        clusters.append(cluster_result)
        y_lower = y_upper = 0
        local_silhouette_auto_analyser = [0 for _ in range(c)]  # the most points should be above average line
        for label in range(c):
            label_silhouette_samples = [el for p, el in enumerate(silhouette_samples) if preds[p] == label]
            label_silhouette_samples.sort()
            for el in reversed(label_silhouette_samples):
                dt = el - silhouette_score  # discrete integration
                if dt > 0:
                    local_silhouette_auto_analyser[label] += dt
                else:
                    break
            if PLOT:
                y_upper += len(label_silhouette_samples)
                plt.fill_betweenx(np.arange(y_lower, y_upper), 0, label_silhouette_samples, alpha=.8,
                                  label='_nolegend_')
                plt.text(-0.05, (y_lower + y_upper) / 2 - 11, f'C:{label}')  # 10 is the default font size
                y_lower += len(label_silhouette_samples)
        silhouette_auto_analyser.append(local_silhouette_auto_analyser)
        if PLOT:
            plt.axvline(x=silhouette_score, linestyle='--', color='red', label='Mean silhouette score')
            ax = plt.gca()
            ax.set_xlabel('Silhouette score')
            ax.set_ylabel('Cluster classe')
            plt.title(f'Silhouette method - Gaussian Mixture size {c}')
            plt.yticks([])
            plt.xlim([min(silhouette_samples) - 0.13, 1])
            plt.tight_layout()
            plt.legend()
            plt.show(block=BLOCK_PLOTS)

    optimum_cluster_size_bic_idx = bic_graph.index(min(bic_graph))
    print(f'Optimum size for gaussian mixture using bic method: {graph_x[optimum_cluster_size_bic_idx]}')

    # computes the graph area over the average line, also it prioritizes the simpler configs
    silhouette_auto_analyser = [sum(els) - SILHOUETTE_COMPLEX_PENALTY * idx for idx, els in
                                enumerate(silhouette_auto_analyser)]

    best_size_according_to_silu = graph_x[silhouette_auto_analyser.index(max(silhouette_auto_analyser))]
    print(f'Optimum size for gaussian mixture using silhouette method: {best_size_according_to_silu}')

    if PLOT:
        plt.plot(graph_x, bic_graph, '-o', label='_nolegend_', zorder=1)
        plt.scatter([graph_x[optimum_cluster_size_bic_idx]],
                    [bic_graph[optimum_cluster_size_bic_idx]], color='red', label='optimum size',
                    zorder=2)
        ax = plt.gca()
        ax.set_xlabel('Cluster size')
        ax.set_ylabel('BIC')
        plt.title('BIC Method - Gaussian Mixture')
        plt.xticks(graph_x)
        plt.tight_layout()
        plt.legend()
        plt.show(block=BLOCK_PLOTS)

        plt.plot(graph_x, silhouette_graph, '-o')
        ax = plt.gca()
        ax.set_xlabel('Cluster size')
        ax.set_ylabel('Silhouette')
        plt.title('Silhouettes - Gaussian Mixture')
        plt.xticks(graph_x)
        plt.tight_layout()
        plt.show(block=BLOCK_PLOTS)
print()

save_results_report(clusters)
