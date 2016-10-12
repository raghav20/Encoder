from AutoEncoder import *
import csv
import tensorflow as tf
#import tflearn.datasets.mnist as mnist
import pandas as pd
import glob


X = pd.read_csv("/home/ubuntu/test/round2.csv")
X = X.as_matrix()

autoencoder1 = AutoEncoder(X,10,learning_rate=0.01)
autoencoder1.train()

