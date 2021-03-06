import numpy as np

import torch
import logging
import pickle
from torch.autograd import Variable
eps = 1e-20


def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""
    formatter = logging.Formatter("%(asctime)s %(message)s", "%m-%d %H:%M:")
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger, handler

def log_params(logger, params):
     for keys, values in params.items():
          st = "Key: {}, item: {}".format(keys, values)
          logger.info(st)     
     return True
          
def randomize2d(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels 

def scale(h):
    h_min = torch.min(h).expand_as(h)
    h_max = torch.max(h).expand_as(h)
    h_out = (h-h_min + eps)/(h_max-h_min + 2*eps)
    return h_out

def scale_embed(h, min, max,GPU_num=None):

    h_min = Variable(torch.FloatTensor(h.size()).fill_(min))

    h_max = Variable(torch.FloatTensor(h.size()).fill_(max))

    h_out = (h-h_min + eps)/(h_max-h_min + 2*eps)
    return h_out



def save_model(model, params):
    path = "../saved_models/{}_{}_{}.pkl".format(params['DATASET'],params['MODEL'],params['EPOCH'])
    pickle.dump(model, open(path, "wb"))
    print("A model is saved successfully as {path}!".format(path))


def load_model(params):
    path = "../saved_models/{}_{}_{}.pkl".format(params['DATASET'],params['MODEL'],params['EPOCH'])

    try:
        model = pickle.load(open(path, "rb"))
        print("Model in {} loaded successfully!".format(path))

        return model
    except:
        print("No available model such as {}.".format(path))
        exit()

