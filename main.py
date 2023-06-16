import os
import pickle
import numpy as np
from tqdm.notebook import tqdm

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow import keras

import cv2
import numpy as np

features = 'null'

# load features from pickle
with open('features.pkl', 'rb') as f:
    features = pickle.load(f)

with open('captions.txt', 'r') as f:
    next(f)
    captions_doc = f.read()

model = keras.models.load_model('best_model.h5')

# create mapping of image to captions
mapping = {}
# process lines
for line in tqdm(captions_doc.split('\n')):
    # split the line by comma(,)
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    # remove extension from image ID
    image_id = image_id.split('.')[0]
    # convert caption list to string
    caption = " ".join(caption)
    # create list if needed
    if image_id not in mapping:
        mapping[image_id] = []
    # store the caption
    mapping[image_id].append(caption)

def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            # take one caption at a time
            caption = captions[i]
            # preprocessing steps
            # convert to lowercase
            caption = caption.lower()
            # delete digits, special chars, etc., 
            caption = caption.replace('[^A-Za-z]', '')
            # delete additional spaces
            caption = caption.replace('\s+', ' ')
            # add start and end tags to the caption
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
            captions[i] = caption

# preprocess the text
clean(mapping)

all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)

# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)

# get maximum length of the caption available
max_length = max(len(caption.split()) for caption in all_captions)
max_length

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
      
    return in_text


# TEst with real images

vgg_model = VGG16()
# restructure the model
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

cap = cv2.VideoCapture("video3.mp4")

i=10
while True:
    try:
        success, img = cap.read()
        img = cv2.flip(img, 1)

        if i == 10:
            # convert cv2 image to numpy array
            #image = np.asarray(img)     
            image = img
            image = cv2.resize(image, (224, 224))
            # reshape data for model
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            # preprocess image for vgg
            image = preprocess_input(image)
            # extract features
            feature = vgg_model.predict(image, verbose=0)
            # predict from the trained model
            caption = predict_caption(model, feature, tokenizer, max_length)

            i = 0
        
            
        img = cv2.putText(img, caption, (10,10), cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255), 2)
        print(caption)
        i = i+1      

        cv2.imshow("Image", img)
        cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print("Exception", e)
cap.release()
cv2.destroyAllWindows()

