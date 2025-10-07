import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import time
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import warnings
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, accuracy_score, make_scorer
from sklearn.decomposition import PCA
import umap.umap_ as umap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.manifold import trustworthiness
import os

warnings.filterwarnings("ignore")

script_dir = os.path.dirname(os.path.realpath(__file__))

df = pd.read_csv(os.path.join(script_dir, "customer.csv"))

df['Ever_Married'].fillna(df['Ever_Married'].mode()[0], inplace=True)
df['Work_Experience'].fillna(df['Work_Experience'].mode()[0], inplace=True)
df['Family_Size'].fillna(df['Family_Size'].mode()[0], inplace=True)
df['Graduated'].fillna(df['Graduated'].mode()[0], inplace=True)
df['Profession'].fillna(df['Profession'].mode()[0], inplace=True)

df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
df['Ever_Married'] = df['Ever_Married'].apply(lambda x: 1 if x == 'Yes' else 0)
df['Graduated'] = df['Graduated'].apply(lambda x: 1 if x == 'Yes' else 0)

le = LabelEncoder()
df['Spending_Score'] = le.fit_transform(df['Spending_Score'])
df['Profession'] = le.fit_transform(df['Profession'])

df.drop('ID', inplace=True, axis=1)
df.drop('Var_1', inplace=True, axis=1)

plt.figure(figsize=(8, 6))
sns.countplot(x='Segmentation', data=df, palette='Set2')
plt.title('Distribution of Segmentation')
plt.xlabel('Segmentation')
plt.ylabel('Count')
plt.show()

X = df.drop('Segmentation', axis=1)
y = df['Segmentation']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

RS_train = RobustScaler()
X_train_scaled = RS_train.fit_transform(X_train)

RS_test = RobustScaler()
X_test_scaled = RS_test.fit_transform(X_test)

pca = PCA()
pca.fit(X_train_scaled)

plt.figure(figsize=(12, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, linestyle='-', color='b')
plt.title('Scree Plot')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1))
plt.xticks(rotation=90)
plt.grid(True)
plt.show()

threshold = 0.95
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
num_components_threshold = np.argmax(cumulative_variance >= threshold) + 1

print(f"Number of components explaining {threshold:.0%} of variance:", num_components_threshold)

plt.figure(figsize=(12, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-', color='r')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained')
plt.xticks(range(1, len(cumulative_variance) + 1))
plt.xticks(rotation=90)
plt.axvline(x=num_components_threshold, color='k', linestyle='--', linewidth=1)
plt.axhline(y=threshold, color='b', linestyle='--', linewidth=1)
plt.text(num_components_threshold + 0.5, threshold - 0.1, f'{threshold:.0%} Variance', color='b')
plt.grid(True)
plt.show()

num_components_retained = 7
pca_retained = PCA(n_components=num_components_retained)
pca_data_train = pca_retained.fit_transform(X_train_scaled)
pca_data_test = pca_retained.transform(X_test_scaled)

ump = umap.UMAP(n_neighbors=5, min_dist=0.25)
ump.fit(X_train_scaled)

umap_data_train = ump.transform(X_train_scaled)
current_trustworthiness = trustworthiness(X_train_scaled, umap_data_train)

best_trustworthiness = 0
best_umap = None
best_umap_data = None
best_num_components = None

# Define parameter grid for manual tuning
umap_param_grid = {'n_neighbors': [5, 10, 15],
              'min_dist': [0.25, 0.5]}
for n_neighbors in umap_param_grid['n_neighbors']:
    for min_dist in umap_param_grid['min_dist']:
        ump = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
        ump.fit(X_train_scaled)
        umap_data_train = ump.transform(X_train_scaled)
        current_trustworthiness = trustworthiness(X_train_scaled, umap_data_train)
        if current_trustworthiness > best_trustworthiness:
            best_trustworthiness = current_trustworthiness
            best_umap = ump
            best_umap_data = umap_data_train
            best_num_components = umap_data_train.shape[1]

print("Best UMAP parameters:", best_umap.get_params())
print("Best trustworthiness:", best_trustworthiness)
print("Best number of components:", best_num_components)

umap_data_train_best = best_umap.transform(X_train_scaled)
umap_data_test_best = best_umap.transform(X_test_scaled)

kmeans_param_grid = {'n_clusters': [3, 4, 5]}
agglo_param_grid = {'n_clusters': [3, 4, 5],
                    'linkage': ['ward', 'single']}

silhouette_scores = pd.DataFrame(index=['Original', 'PCA Reduced', 'UMAP Reduced'],
                                 columns=['KMeans', 'Agglomerative'])

best_parameters = pd.DataFrame(index=['KMeans', 'Agglomerative'], columns=['Method', 'n_clusters', 'linkage'])

best_num_clusters_kmeans = {}
best_num_clusters_agglo = {}

for idx, (method, method_name) in enumerate(zip([X_train_scaled, pca_data_train, umap_data_train_best],
                                                ['Original', 'PCA Reduced', 'UMAP Reduced'])):
    best_kmeans_score = -1
    for n_clusters in kmeans_param_grid['n_clusters']:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(method)
        kmeans_score = silhouette_score(method, kmeans.labels_)
        if kmeans_score > best_kmeans_score:
            best_kmeans_score = kmeans_score
            best_num_clusters_kmeans[method_name] = n_clusters
            best_parameters.loc['KMeans', 'Method'] = method_name
            best_parameters.loc['KMeans', 'n_clusters'] = n_clusters

    silhouette_scores.loc[method_name, 'KMeans'] = best_kmeans_score

    best_agglo_score = -1
    for n_clusters in agglo_param_grid['n_clusters']:
        for linkage in agglo_param_grid['linkage']:
            agglo = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            agglo.fit(method)
            agglo_score = silhouette_score(method, agglo.labels_)
            if agglo_score > best_agglo_score:
                best_agglo_score = agglo_score
                best_num_clusters_agglo[method_name] = (n_clusters, linkage)
                best_parameters.loc['Agglomerative', 'Method'] = method_name
                best_parameters.loc['Agglomerative', 'n_clusters'] = n_clusters
                best_parameters.loc['Agglomerative', 'linkage'] = linkage

    silhouette_scores.loc[method_name, 'Agglomerative'] = best_agglo_score

print("Silhouette Scores:")
print(silhouette_scores)

print("Best number of clusters for KMeans:")
for method_name, num_clusters in best_num_clusters_kmeans.items():
    print(f"{method_name}: {num_clusters}")

print("Best number of clusters for Agglomerative Clustering:")
for method_name, (num_clusters, linkage) in best_num_clusters_agglo.items():
    print(f"{method_name}: {num_clusters} clusters with linkage {linkage}")

fig, axes = plt.subplots(2, figsize=(18, 12))
method = KMeans(n_clusters=3, random_state=42)

method.fit(X_train_scaled)
axes[0].scatter(X_train_scaled[:, 4], X_train_scaled[:, 5], c=method.labels_, cmap='viridis')
axes[0].set_title('KMeans Clustering (Original Data)')

method.fit(pca_data_train)
axes[1].scatter(pca_data_train[:, 4], pca_data_train[:, 5], c=method.labels_, cmap='viridis')
axes[1].set_title('KMeans Clustering (PCA Reduced Data)')

plt.tight_layout()
plt.show()

adaboost = AdaBoostClassifier(random_state=10)
rf = RandomForestClassifier(random_state=10)

adaboost_params = {'n_estimators': [50, 100, 150], 'learning_rate': [0.5, 1.0]}
rf_params = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20]}

data_transformations = {'Original': (X_train_scaled, X_test_scaled),
                        'PCA Reduced': (pca_data_train, pca_data_test),
                        'UMAP Reduced': (best_umap_data, best_umap.transform(X_test_scaled))}

best_models = {}
accuracies = {}

for method_name, (X_train_method, X_test_method) in data_transformations.items():
    adaboost_grid = GridSearchCV(adaboost, adaboost_params, cv=3, scoring='accuracy')
    adaboost_grid.fit(X_train_method, y_train)
    best_adaboost = adaboost_grid.best_estimator_
    best_models[method_name + ' AdaBoost'] = best_adaboost

    rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='accuracy')
    rf_grid.fit(X_train_method, y_train)
    best_rf = rf_grid.best_estimator_
    best_models[method_name + ' Random Forest'] = best_rf

    adaboost_pred = best_adaboost.predict(X_test_method)
    rf_pred = best_rf.predict(X_test_method)

    adaboost_accuracy = accuracy_score(y_test, adaboost_pred)
    rf_accuracy = accuracy_score(y_test, rf_pred)

    accuracies[method_name + ' AdaBoost'] = adaboost_accuracy
    accuracies[method_name + ' Random Forest'] = rf_accuracy

for model_name, model in best_models.items():
    print("Best parameters for", model_name, ":", model.get_params())

for model_name, accuracy in accuracies.items():
    print("Accuracy for", model_name, ":", accuracy)
