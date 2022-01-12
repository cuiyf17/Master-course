import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt
from scipy.stats import dirichlet, norm, poisson
from theano.tensor import _shared


## ----- 1.read data -----
exprData = pd.read_csv('Brain-expr_matrix-smallData.txt', index_col = 0, sep = '\t')
genes = exprData.index.tolist()


## ----- 2.convert to spot-gene (document-word) index list -----
gene_idx = dict(zip(genes, range(len(genes))))
exprDoc_idx = [[i for i in range(exprData.shape[0]) for k in range(exprData.iloc[i,j])] for j in range(exprData.shape[1])]


## ----- 3.initialize the hyper-parameters -----
batch_size = exprData.shape[1]
n_clusters = 2
n_gene = len(genes)
length_docs = [len(doc) for doc in exprDoc_idx]
alpha = np.ones((batch_size, n_clusters))
beta = np.ones((n_clusters, n_gene))


## ----- 4.construct the LDA model and perform reference and learning
with pm.Model() as model:
    
    thetas = pm.distributions.Dirichlet('thetas', a=alpha, shape=(batch_size, n_clusters))    # Spot-cluster (document-topic) distribution (Dirichlet)
    betas = pm.distributions.Dirichlet('betas', a=beta, shape=(n_clusters, n_gene))     # Cluster-gene (topic-word) distribution (Dirichlet)
    zs = [pm.Categorical("z_d{}".format(d), p=thetas[d], shape=length_docs[d]) for d in range(batch_size)]         # Gene-cluster (word-topic) assignment (Categorical)
    ws = [pm.Categorical("w_{}_{}".format(d,i), p=betas[zs[d][i]], observed=exprDoc_idx[d][i]) for d in range(batch_size) for i in range(length_docs[d])]         # Observed gene (word) (Categorical)
    
    trace = pm.sample(2000)
    # inference and learning
    #~

    
## ----- 5.assign the cluster label to each spot -----
preds = exprData.copy()
preds = preds.drop(preds.index[1:])
preds.index = ["clusters"]
preds.iloc[0] = trace["thetas"].sum(axis=0).argmax(axis=1)
    


## ----- 6.visualize the cluster label on spatial position ------
spotPosition = pd.read_csv('Brain-spot_position-smallData.txt', sep = '\t')
plt.figure(figsize=(4,6), dpi=120)
colors = ["blue", "orange"]
classes = np.unique(preds)
for i, cluster in enumerate(classes):
    idxs = np.where(preds == cluster)[1]
    plt.plot(spotPosition.iloc[idxs, 1], spotPosition.iloc[idxs, 2], "o", color = colors[i], label = "%d"%(i))
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Visualization of Clustering")
plt.legend()
plt.show()

