import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC

#citire date

f = open("train.txt", "r")
train_id = []
train_labels = []

rl1 = f.readline()
for rl1 in f.readlines():
    rl1 = rl1.split(',')
    train_id.append(rl1[0])
    train_labels.append(int(rl1[1][0]))

f.close()
f = open("validation.txt")
validation_id = []
validation_labels = []

rl2 = f.readline()
for rl2 in f.readlines():
    rl2 = rl2.split(',')
    validation_id.append(rl2[0])
    validation_labels.append(int(rl2[1][0]))


f.close()
f = open("test.txt")
test_id = []
rl3 = f.readline()
for rl3 in f.readlines():
    test_id.append(rl3[0:-1])

f.close()



train_images = []
validation_images = []
test_images = []

for x in test_id:
    f = open('test/'+x)
    img = mpimg.imread('test/'+x)
    test_images.append(np.array(img).flatten())

f.close()
for x in validation_id:
    f = open('train+validation/'+x)
    img = mpimg.imread('train+validation/'+x)
    validation_images.append(np.array(img).flatten())

f.close()
for x in train_id:
    f = open('train+validation/'+x)
    img = mpimg.imread('train+validation/'+x)
    train_images.append(np.array(img).flatten())
f.close()

#modelul


svmx = SVC()
svmx.fit(train_images, train_labels)
prez = svmx.predict(test_images)


#prez = svmx.predict(validation_images)   Estimare precizie svm
#print(np.mean(prez == validation_labels))


#scriere in fisier
f = open('solutie_svm.txt', 'w')
f.write('id,label\n')
for i in range(len(test_id)):
     f.write(test_id[i])
     f.write(',')
     f.write(str(prez[i]))
     f.write('\n')
f.close()











