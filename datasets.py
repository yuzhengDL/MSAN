"""
Dataset loading
"""
import numpy as np
from sklearn import preprocessing

def load_dataset(name='f8k', load_train=True):
    """
    Load captions and image features
    Possible options: f8k, f30k, coco
    """
    loc = name + '/'

    # Captions
    train_caps, dev_caps, test_caps = [],[],[]
    if load_train:
        with open(loc+name+'_train_caps.txt', 'rb') as f:
            for line in f:
                train_caps.append(line.strip())
    else:
        train_caps = None
    #with open(loc+name+'_dev_caps.txt', 'rb') as f:
    #    for line in f:
    #        dev_caps.append(line.strip())
    with open(loc+name+'_test_caps.txt', 'rb') as f:
        for line in f:
            test_caps.append(line.strip())
    # Image features
    if load_train:
        train_ims_local = np.load(loc+name+'_train_ims_local.npy')
        train_ims_global = np.load(loc+name+'_train_ims_global.npy')
        #train_ims_global = preprocessing.scale(train_ims_global)
        train_ims_NIC = np.load(loc+name+'_train_ims_NIC.npy')
        train_ims_NIC = preprocessing.scale(train_ims_NIC)
        train_ims_global = np.concatenate((train_ims_global, train_ims_NIC), axis=1)
    else:
        train_ims_local = None
        train_ims_global = None


    test_ims_local = np.load(loc+name+'_test_ims_local.npy')
    test_ims_global = np.load(loc+name+'_test_ims_global.npy')
    #test_ims_global = preprocessing.scale(test_ims_global)
    test_ims_NIC = np.load(loc+name+'_test_ims_NIC.npy')
    test_ims_NIC = preprocessing.scale(test_ims_NIC)
    test_ims_global = np.concatenate((test_ims_global, test_ims_NIC), axis=1)

    return (train_caps, train_ims_local, train_ims_global), (test_caps, test_ims_local, test_ims_global)

