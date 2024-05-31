import os
import pickle

actors=os.listdir('Data')
filenames=[]
for actor in actors:
    for file in os.listdir(os.path.join('Data',actor)):
        filenames.append(os.path.join('Data',actor,file))
print(filenames)
print(len(filenames))
pickle.dump(filenames,open('filenames.pkl','wb'))

import numpy as np
import pickle
from tqdm import tqdm
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
filenames=pickle.load(open('filenames.pkl', 'rb'))

model=ResNet50(weights='imagenet', include_top=False,input_shape=(224,224,3), pooling='avg')
print(model.summary())

def feature_extractor(img_path,model):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    expanded_img = np.expand_dims(img_array, axis=0)
    result = model.predict(expanded_img).flatten()
    return result

features=[]
for file in tqdm(filenames):
    features.append(feature_extractor(file,model))

pickle.dump(features,open('embedding.pkl','wb'))