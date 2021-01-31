import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# generate Data randomly
Data = np.random.randn(1000,2)
# random stae=The generator used to initiate centers.
kmeans = KMeans(n_clusters=3,random_state=44).fit(Data)
# get label
labels = kmeans.labels_
x = Data[:,0]
y = Data[:,1]

plt.figure(figsize=(10,12))
plt.subplot(2,1,1)
plt.scatter(x,y)
plt.title("Raw data")
plt.subplot(2,1,2)
plt.scatter(x,y, c = labels)
plt.title("Clustered data")
plt.show()