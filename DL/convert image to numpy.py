# -*- coding: utf-8 -*-
"""pull data.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pykgt-RxKwU701Ev0pWFYvf9JUzbL_Ol

**Import**
"""

import sys, os, errno, shutil
import numpy as np
import cv2

"""**Import drive from GOOGLE DRIVE**"""

from google.colab import drive
drive.mount('/content/drive')

"""**Save to save data to .npy**"""

def save_to_numpy(imgs,label,path,filename):
    np.save(os.path.join(path, filename + '.npy'), imgs)
    np.save(os.path.join(path, filename + '_label' + '.npy'), label)
    print('Numpy files have been saved')

"""**Recieve data from Train Folder with All fold and ALL classes**"""

data = [['Files', 'Classes']]
classes = ['Glaucoma', 'Normal', 'Other']
num_folders = ['1','2','3','4','5']
temp_x = []
temp_y = []
x=[]
y=[]
for k in range(len(num_folders)):
  for i in range(len(classes)):
    images_lst = os.listdir('/content/drive/MyDrive/Dataset/Train/' + classes[i] + '/Fold-' + num_folders[k])
    #print(images_lst)
    count = 0

    for j in images_lst:
      count += 1
      if count == 2:
        break
      print('\n' + j + ' ----> ', count, '/', len(images_lst))
      print('class : ' + classes[i])
      print('fold : ' + num_folders[k])
      img = cv2.imread('/content/drive/MyDrive/Dataset/Train/' + classes[i] + '/Fold-' + num_folders[k]+'/'+j)
      img = cv2.resize(img, (256, 256))
      temp_x.append(np.array(img))
      temp_y.append(i)
      
  x.append(temp_x)
  temp_x = []
  y.append(temp_y)
  temp_y = []

#save to .npy

#output_path = '/content/drive/MyDrive/Dataset/np_data/'
#save_to_numpy(x,y,output_path,'Train')   
print('---------------------------------------------------------------')
print('done')
print(len(x))
print(len(y))

"""**Recieve all data from Test**"""

data = [['Files', 'Classes']]
classes = ['Glaucoma', 'Normal', 'Other']
#num_folders = ['1','2','3','4','5']
#temp_x = []
#temp_y = []
x=[]
y=[]

for i in range(len(classes)):
  images_lst = os.listdir('/content/drive/MyDrive/Dataset/Test/' + classes[i])
  #print(images_lst)
  count = 0

  for j in images_lst:
    count += 1
    print('\n' + j + ' ----> ', count, '/', len(images_lst))
    print('class : ' + classes[i])
    #print('fold : ' + num_folders[k])
    img = cv2.imread('/content/drive/MyDrive/Dataset/Test/' + classes[i] +'/'+ j)
    img = cv2.resize(img, (256, 256))
    x.append(np.array(img))
    if classes[i] == 'Glaucoma':
      y.append('0')
      print('Y appended : 0')
    elif classes[i] == 'Normal':
      y.append('1')
      print('Y appended : 1')
    else:
      y.append('2')
      print('Y appended : 2')

      
#save to .npy
      
#output_path = '/content/drive/MyDrive/Dataset/np_data/'
#save_to_numpy(x,y,output_path,'test folder not the training')   
print('---------------------------------------------------------------')
print('done')
print(len(x))
print(len(y))