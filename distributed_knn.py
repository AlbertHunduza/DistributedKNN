import pandas as pd
from time import time
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
import logging

comm = MPI.COMM_WORLD
myRank = comm.Get_rank()
totalRanks = comm.Get_size()
logging.basicConfig(format='%(name)s: %(message)s')
logger = logging.getLogger(str(f"Rank {myRank}"))
splitedFrames = centroids = list()
k = 4  # Set the desired number of clusters
m = 0

if myRank == 0:
    start_time = time()
    df = pd.read_csv("cluster_data.csv", header=0, index_col=0)
    df = df.sample(frac=1)
    n = df.shape[0]
    m = df.shape[1]
    df['cluster'] = np.zeros(n)
    splitedFrames = np.array_split(df, totalRanks)
    centroids = np.zeros((k, m))
    selected_indices = np.random.choice(range(n), size=k, replace=False)
    
    for i, idx in enumerate(selected_indices):
        centroids[i] = df.iloc[idx, :m].values

myFrame = comm.scatter(splitedFrames, root=0)
centroids, k, m, clusterChangeHappened = comm.bcast((centroids, k, m, True), root=0)
distance = np.zeros((myFrame.shape[0], k))
myCentroids = np.zeros((k, m))
counter = 0

while clusterChangeHappened:
    # Cluster assignment - along with distance calculation
    changedCluster = False

    for i in range(myFrame.shape[0]):
        minDist = float('inf')
        clust = 0

        for j in range(k):
            sum = 0

            for l in range(m):
                sum += (centroids[j, l] - myFrame.iloc[i, l]) ** 2

            res = np.sqrt(sum)
            res = round(res, 6)
            distance[i, j] = res

            if res < minDist:
                minDist = res
                clust = j

        if myFrame.iloc[i, m] != clust:
            changedCluster = True
            myFrame.iloc[i, m] = clust

    # Recalculate centroids
    for i in range(k):
        for j in range(m):
            temp = 0
            num = 0

            for l in range(myFrame.shape[0]):
                if myFrame.iloc[l, m] == i:
                    temp += myFrame.iloc[l, j]
                    num += 1

            if num != 0:
                temp /= num

            myCentroids[i, j] = temp

    # Gathering and recalculating centroids
    data = (myCentroids, myFrame.shape[0])
    allDatas = comm.gather(data, root=0)
    temp = np.zeros((k, m))

    if myRank == 0:
        for d in allDatas:
            for i in range(k):
                for j in range(m):
                    temp[i, j] += d[0][i, j]
        
        for i in range(k):
            for j in range(m):
                temp[i, j] /= d[1]  # Divide by the number of data points assigned to each centroid

        centroids = comm.bcast(temp, root=0)


    # Checking for the end
    checks = comm.allgather(changedCluster)
    clusterChangeHappened = False

    for i in checks:
        clusterChangeHappened != i

    counter += 1

result = comm.gather(myFrame, root=0)

if myRank == 0:
    final = pd.concat(result)
    end_time = time()
    print(f"Time taken: {end_time - start_time} seconds")
    X = final['Screen_To_Body_Ratio_(%)']
    Y = final['Weight']
    cluster = final['cluster']
    colors = {0: 'blue', 1: 'red', 2: 'green', 3: 'yellow'}

    # Plotting data points
    plt.scatter(X, Y, c=[colors[c] for c in cluster])

    # Calculating and plotting final centroids as black Xs
    for i in range(k):
        cluster_points = final[final['cluster'] == i]
        if not cluster_points.empty:
            centroid_x = cluster_points['Screen_To_Body_Ratio_(%)'].mean()
            centroid_y = cluster_points['Weight'].mean()
            plt.scatter(centroid_x, centroid_y, marker='x', color='black', s=100)

    plt.show()
