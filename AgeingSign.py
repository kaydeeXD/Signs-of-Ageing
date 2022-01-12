import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from keras.preprocessing.image import array_to_img

dir = "DATASET file path" 
categories = ['puffy eyes' , 'wrinkles' , 'dark spots']
data = []

for category in categories:
    path = os.path.join(dir,category) #feature_images_path
    label = categories.index(category)
    for img in os.listdir(path):
        imgpath = os.path.join(path,img)
        age_img = cv2.imread(imgpath,0) #reads_images_as_arrays
        try:
            age_img = cv2.resize(age_img,(50,50))
            image = np.array(age_img).flatten() #flatterns_the_50x50_array_into_1D_array
            data.append([image,label])
        except Exception as e: 
            pass

Pickle_Dataset = open('Data.pickle','wb')
pickle.dump(data,Pickle_Dataset)
Pickle_Dataset.close()

Pickle_Dataset = open('Data.pickle','rb')
data = pickle.load(Pickle_Dataset)
Pickle_Dataset.close()

random.shuffle(data)
features = []
labels = []
for feature, label in data:
    features.append(feature)
    labels.append(label)

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = 0.4)
model = SVC(C = 1, kernel = 'poly' , gamma = 'auto')
model.fit(x_train,y_train)

#pickling_the_model
pick_model= open('model.sav','wb')
pickle.dump(model, pick_model)
pick_model.close()

pick_model= open('model.sav','rb')
model = pickle.load(pick_model)
pick_model.close()

prediction = model.predict(x_test)
accuracy = model.score (x_test,y_test)
categories = ['Puffy eyes' , 'Wrinkles on face' , 'Dark spots on face']
print('Accuracy is :' ,accuracy)
print('Prediction is: ', categories[prediction[1]])

IMAGE = x_test[1].reshape(50,50)
plt.imshow(IMAGE, cmap = 'gray')
plt.show()
