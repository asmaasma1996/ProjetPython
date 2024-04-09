import numpy as np

def kmeans(X, k, max_iters=100):
    # Initialisation aléatoire des centres
    centroids = X[np.random.choice(range(len(X)), size=k, replace=False)]

    for _ in range(max_iters):
        # Calcul des distances entre chaque point et chaque centre
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

        # Attribution de chaque point au centre le plus proche
        labels = np.argmin(distances, axis=1)

        # Mise à jour des centres
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # Vérification de la convergence
        if np.allclose(new_centroids, centroids):
            break

        centroids = new_centroids

    return centroids, labels

# Exemple d'utilisation

np.random.seed(0)
X = np.random.randn(100, 2)

# Nombre de clusters
k = 3

# Application de l'algorithme K-means
centroids, labels = kmeans(X, k)

# Affichage des résultats
print("Centres finaux:")
print(centroids)
print("Labels des points:")
print(labels)
