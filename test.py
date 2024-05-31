from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from mtcnn import MTCNN
from PIL import Image

feature_list=np.array(pickle.load(open('embedding.pkl', 'rb')))
filenames=pickle.load(open('filenames.pkl', 'rb'))
model=ResNet50(weights='imagenet', include_top=False,input_shape=(224,224,3), pooling='avg')
detector=MTCNN()

sample_img= cv2.imread('Sample/Bhavya_photo.jpg')
results=detector.detect_faces(sample_img)
x,y,width,height=results[0]['box']
face=sample_img[y:y+height,x:x+width]

image=Image.fromarray(face)
image=image.resize((224,224))
face_array=np.asarray(image)
face_array=face_array.astype('float32')
expanded_img=np.expand_dims(face_array,axis=0)
result=model.predict(expanded_img).flatten()

similarity=[]
for i in range(len(feature_list)):
    similarity.append(cosine_similarity(result.reshape(1,-1),feature_list[i].reshape(1,-1))[0][0])
index_pos=sorted(list(enumerate(similarity)),reverse=True,key=lambda x:x[1])[0][0]

temp_img=cv2.imread(filenames[index_pos])
cv2.imshow('output',temp_img)
cv2.waitKey(0)