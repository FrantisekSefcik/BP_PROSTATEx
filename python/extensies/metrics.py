from scipy.spatial.distance import cdist
from collections import Counter
import operator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def siamese_predict(train_feat, search_feat, dataset,n = 10, distance = 'cosine'):
    dist = cdist(train_feat, [search_feat], distance)
    rank = np.argsort(dist.ravel())
    
    labels = dataset.labels_train[rank[:n]]
    c = Counter(labels) 
    dictionary = dict([(i, c[i] / len(labels) * 100.0) for i in c])
    a = max(dictionary.items(), key=operator.itemgetter(1))[0]
    b = labels[0]
    return a
    

def treshold_predict(train_feat, search_feat, dataset,treshold = 0.5,n = 10, distance = 'cosine'):
    
    dist = cdist(train_feat, [search_feat], distance)
    rank = np.argsort(dist.ravel())
    
    labels = dataset.labels_train[rank[:n]]
    counter = Counter(labels)
    percentage_dict = dict([(i, counter[i] / len(labels)) for i in (0,1)])
    
    if percentage_dict[1] >= treshold:
        return 1
    else:
        return 0
    


def weighted_predict(train_feat, search_feat, dataset,treshold = 0.5,n = 10, distance = 'cosine'):
    dist = cdist(train_feat, [search_feat], distance)
    rank = np.argsort(dist.ravel())
    labels = dataset.labels_train[rank[:n]]
    dictionary = {0:0,1:0}
    for x,y in zip(labels,dist[rank[:n]]):
        dictionary[x] += (1 - y)
    
    dictionary[0] /= dictionary[0] + dictionary[1]
    dictionary[1] /= dictionary[0] + dictionary[1]
    
    if dictionary[1] >= treshold:
        return 1
    else:
        return 0
    



    
def show_image(idxs, data, titles):
    if type(idxs) != np.ndarray:
        idxs = np.array([idxs])
    fig = plt.figure(figsize = (18,10))
    gs = gridspec.GridSpec(1,len(idxs))
    for i in range(len(idxs)):
        ax = fig.add_subplot(gs[0,i])
        ax.set_title(str(titles[idxs[i]]))
        ax.imshow(data[idxs[i],:,:,0],cmap = 'gray')
        ax.axis('off')
    plt.show()
        
def get_separated(y,y_pred):
    dictionary = dict([])
    # ma byt pozitivny a aj je predikovany ako pozitivny
    dictionary['tp'] = np.where([x==y and x == 1 for (x,y) in zip(y,y_pred)])[0]
    # ma byt pozitivny je predikovany ako negativny
    dictionary['fn'] = np.where([x!=y and x == 1 for (x,y) in zip(y,y_pred)])[0]
    # ma byt negativny a aj je predikovany ako negativny
    dictionary['tn'] = np.where([x==y and x == 0 for (x,y) in zip(y,y_pred)])[0]
    # ma byt negativny je predikovany ako pozitivny
    dictionary['fp'] = np.where([x!=y and x == 0 for (x,y) in zip(y,y_pred)])[0]
    return dictionary

def show_most_similar(train_feat, search_feat, dataset,idx = 0,n = 10):
    dist = cdist(train_feat, [search_feat[idx]], 'euclidean')
    rank = np.argsort(dist.ravel())
    labels = dataset.labels_train[rank[:n]]
    plt.title(str(dataset.labels_train[idx]))
    plt.imshow(dataset.images_test[idx,:,:,0],cmap = 'gray')
    show_image(rank[:n],dataset.images_train,dataset.labels_train)