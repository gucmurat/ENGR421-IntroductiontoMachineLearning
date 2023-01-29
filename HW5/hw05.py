import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa
import scipy.stats as stats

dataset = np.genfromtxt("hw05_data_set.csv", delimiter = ",")

N = 1000
K = 5

def update_centroids(memberships, X):
    if memberships is None:
        centroids = X[np.random.choice(range(N), K, False),:]
    else:
        centroids = np.vstack([np.mean(X[memberships == k,:], axis = 0) for k in range(K)])
    return(centroids)

def update_memberships(centroids, X):
    D = spa.distance_matrix(centroids, X)
    memberships = np.argmin(D, axis = 0)
    return(memberships)

def plot_current_state(centroids, memberships, X, initial_means, initial_covs, final_means, final_covs, colors):
    x, y = np.mgrid[-8:+8:0.05, -8:+8:0.05]
    loc = np.dstack((x, y))
    plt.figure(figsize = (10, 10))
    for c in range(K):
        initial = stats.multivariate_normal(initial_means[c], np.array(initial_covs[c])*2).pdf(loc)
        result= stats.multivariate_normal(final_means[c], final_covs[c]*2).pdf(loc)
        plt.plot(dataset[memberships == c, 0], dataset[memberships == c, 1], ".", markersize = 10,
                 color = cluster_colors[c])
        plt.contour(x, y, initial, levels=1, linestyles="dashed", colors="k")
        plt.contour(x, y, result, levels=1, colors = colors[c])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()
    
centroids = np.genfromtxt("hw05_initial_centroids.csv", delimiter = ",")
memberships = update_memberships(centroids, dataset)

covs = [np.cov(dataset[memberships == k].T) for k in range(K)]
print(covs)
priors = [np.sum(memberships == k) / N for k in range(K)]
print(priors)
def update(h, means, X):
    centroids = (np.vstack([np.matmul(h[k], X)/np.sum(h[k], axis = 0) for k in range(K)]))
    covs = list()
    for i in range(K):
        matrix_temp = [[0.0, 0.0], [0.0, 0.0]]
        for j in range(N):
            matrix_temp += np.matmul((X[j] - means[i])[:, None], (X[j] - means[i])[None, :])*h[i][j] / np.sum(h[i], axis = 0)
        covs.append(matrix_temp)
    priors = (np.vstack([np.sum(h[k], axis = 0)/N for k in range(K)]))
    return centroids, covs, priors

total_steps = 100
for i in range(total_steps):
    h = np.array([priors[k] * stats.multivariate_normal(centroids[k], covs[k]).pdf(dataset) for k in range(K)])
    h /= np.sum(h, axis=0)
    centroids, covs, priors = update(h, centroids, dataset)
    memberships = np.argmax(h, axis=0)

print('Means')
print(centroids)

memberships = np.argmax(h, axis = 0)
covariances = [np.cov(dataset[memberships == i].T) for i in range(K)]
class_means = np.array([[+0.0, +5.5],
                        [-5.5, +0.0],
                        [+0.0, +0.0],
                        [+5.5, +0.0],
                        [+0.0, -5.5]])  

class_deviations = np.array([[[+4.8, +0.0], [+0.0, +0.4]],
                             [[+0.4, +0.0], [+0.0, +2.8]],
                             [[+2.4, +0.0], [+0.0, +2.4]],
                             [[+0.4, +0.0], [+0.0, +2.8]],
                             [[+4.8, +0.0], [+0.0, +0.4]]])

cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a"])
    
plot_current_state(centroids, memberships, dataset, class_means, class_deviations, centroids, covs, cluster_colors)