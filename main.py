import os
os.environ['CUDA_DEVICE_ORDER']= "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICE'] =  '"'
 
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense,Activation,Lambda
from tensorflow.keras.optimizers import SGD
import Teacher
import Student
 
(train_data,train_labels),(test_data,test_labels) = keras.datasets.cifar10.load_data()
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')
train_data = train_data/255
test_data = test_data/255
train_labels = keras.utils.to_categorical(train_labels.astype('float32'))
test_labels = keras.utils.to_categorical(test_labels.astype('float32'))
 
def swish(x):
   beta = 1.5
   return beta * x * keras.backend.sigmoid(x)
 
def new_softmax(logits, temperature=1):
   logits = logits/temperature
   return np.exp(logits)/np.sum(np.exp(logits))
 
print(train_data.shape)
 
#teacher
teacher = Teacher()
student = Student()

#student
