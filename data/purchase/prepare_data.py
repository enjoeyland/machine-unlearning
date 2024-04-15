import os
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from scipy.sparse import load_npz


data = np.concatenate([load_npz('data1.npz').toarray(), load_npz('data2.npz').toarray()]).astype(int)

num_class = 2

if not os.path.exists(f'{num_class}_kmeans.npy'):
    kmeans = KMeans(n_clusters=num_class, random_state=0).fit(data)
    label = kmeans.labels_
    np.save(f'{num_class}_kmeans.npy', label)
else:
    label = np.load(f'{num_class}_kmeans.npy')

if not os.path.exists(f'purchase{num_class}_train.npy'):
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
    np.save(f'purchase{num_class}_train.npy', {'X': X_train, 'y': y_train})
    np.save(f'purchase{num_class}_test.npy', {'X': X_test, 'y': y_test})
else:
    X_train = np.load(f'purchase{num_class}_train.npy', allow_pickle=True).reshape((1,))[0]['X']
    X_test = np.load(f'purchase{num_class}_test.npy', allow_pickle=True).reshape((1,))[0]['X']

with open('datasetfile', mode='w') as f:
    f.write(json.dumps({
        "nb_train": X_train.shape[0],
        "nb_test": X_test.shape[0],
        "input_shape": list(X_train.shape[1:]),
        "nb_classes": num_class,
        "dataloader": "dataloader"
    }, indent=4))
