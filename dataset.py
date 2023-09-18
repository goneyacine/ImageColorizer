import cv2  
import os
import tensorflow as tf
import random
def process_data(path='data/raw'):
    for img_name in os.listdir(path):
        img = cv2.imread(os.path.join(path,img_name))
        img = cv2.resize(img,(32,32),interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join('data/processed',img_name),img)
        
def load_ds():
    train_ds = ([],[])
    valid_ds = ([],[])
    test_ds = ([],[])
    img_names = os.listdir('data/raw')
    random.shuffle(img_names)
    for i in range(len(img_names)):
        if i/ len(img_names) < 0.9:
            train_ds[0].append(cv2.cvtColor(cv2.imread(os.path.join('data/processed',img_names[i])),cv2.COLOR_BGR2GRAY).astype('float32') / 255)
            train_ds[1].append(cv2.imread(os.path.join('data/processed',img_names[i])).astype('float32')/255)
        elif i/ len(img_names) < 0.98:
            test_ds[0].append(cv2.cvtColor(cv2.imread(os.path.join('data/processed',img_names[i])),cv2.COLOR_BGR2GRAY).astype('float32') / 255)
            test_ds[1].append(cv2.imread(os.path.join('data/processed',img_names[i])).astype('float32')/255)
        else: 
           valid_ds[0].append(cv2.cvtColor(cv2.imread(os.path.join('data/processed',img_names[i])),cv2.COLOR_BGR2GRAY).astype('float32') / 255)
           valid_ds[1].append(cv2.imread(os.path.join('data/processed',img_names[i])).astype('float32')/255)
    train_ds = tf.data.Dataset.from_tensor_slices(train_ds)
    valid_ds = tf.data.Dataset.from_tensor_slices(valid_ds)
    valid_ds = valid_ds.batch(64)
    test_ds = tf.data.Dataset.from_tensor_slices(test_ds)
    test_ds = test_ds.batch(1)

    return(train_ds,valid_ds,test_ds)
        
    