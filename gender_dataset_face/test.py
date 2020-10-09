# imports
from tensorflow.keras.processing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as k
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import glob

# initial parameters
epochs = 100
lr = 1e-3
batch_size = 64
img_dims = (96,96,3) #  hight is 96 width is 96 and RGB i.e 3 channel

# two lists
# data list
data = []
#lable list
labels = []

#load image files from the dataset

# we are using glob here, glob gives you the path very easily
# here the first / gives the folder name which in men and women in our case, the next / gives us the image , and the star indicates match anyhting that you see.
image_files = [f for f in glob.glob(r'C:\users\deepak\Desktop\Gender-Detection\Gender-Detection-master\gender_dataset_face' + "/**/*", recursive=True) if not os.path.isdir(f)]
# if we dont shufle glob will take all the men images first, model will learn men completely first and the try to learn women, at that times weights will be completely changing for the women  , data sets should be mixed well, to balance the weights, so we use shuffle
random.shuffle(image_files)

# converting images to arrays and labelling the categories
for img in image_files:
	image = cv2.imread(img)
	image = cv2.resize(image, (img_dims[0],img_dims[1])) #img_dims is (96,96,3) RGB
	image = img_to_array(image) #turn images into an array
	data.append(image)

	label = img.split(os.path.sep)[-2] #
	if label == "woman":
		label = 1;
	else :
		label = 0;
	labels.append([label]) # [ [1], [0], [0], .....] 1 for woman and 0 fro man
# pre-processing
data = np.array(data, dtype ="float")/ 255.0 # by dividing with 255 we ensure that values will lie btween 0 and 1, to check the computational limit
labels = np.array(labels)

#split dataset for training and validation
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

trainY = to_categorical(trainY, num_classes=2) # [[1,0],[0,1], [0,1], ... ]



















