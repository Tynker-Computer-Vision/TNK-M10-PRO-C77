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

# SA1: load features from features.pkl
with open('features.pkl', 'rb') as f:
    features = pickle.load(f)

# SA1: Load the captions from caption.txt
with open('captions.txt', 'r') as f:
    next(f)
    captions_doc = f.read()


model = keras.models.load_model('best_model.h5')

# SA1: Create mapping of image to captions
mapping = {}
# SA1: Loop through every caption
for line in tqdm(captions_doc.split('\n')):
    # SA1: Split the line by comma(,)
    tokens = line.split(',')
    # SA1: Move to next iteration if length of line is less then 2 characters
    if len(line) < 2:
        continue
    # SA1: Take image_id and caption from token[0], [1] respectively
    image_id, caption = tokens[0], tokens[1:]
    # SA1: Remove extension from image ID
    image_id = image_id.split('.')[0]
    # SA1: Convert caption list to string
    caption = " ".join(caption)
    # Sa1: Create list if needed
    if image_id not in mapping:
        mapping[image_id] = []
    # SA1: Store the caption
    mapping[image_id].append(caption)

# SA1: Print the mapping dictionary
print(mapping["1000268201_693b08cb0e"])

# SA2: Loop throught each key and captions in the mapping
for key, captions in mapping.items():
    # SA2: Go throught each caption in captions for a given image
    for i in range(len(captions)):
        # SA2: Take one caption at a time
        caption = captions[i]
        # SA2: Convert to lowercase
        caption = caption.lower()
        # SA2: Delete digits, special chars, etc., 
        caption = caption.replace('[^a-z]', '')
        # SA2: Delete additional spaces
        caption = caption.replace('\s+', ' ')
        # SA2: Add start and end tags to the caption
        caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
        # SA2: Store the cleaned caption back
        captions[i] = caption

# SA2: Print the cleaned mapping
print("::::::::::::::::::::::::Mapping after cleaning::::::::::::::::::::::::")
print(mapping["1000268201_693b08cb0e"])

# SA3: Create an empty list for storing all_captions
all_captions = []
# SA3: Run a loop for each key in the mapping
for key in mapping:
    # SA3: Run loop for each caption in the mapping[key]
    for caption in mapping[key]:
        # SA3: Append caption to all_captions
        all_captions.append(caption)

# SA3: Create a tokenizer
tokenizer = Tokenizer()
# SA3: Updates internal vocabulary of tokenizer based on a list of texts.
tokenizer.fit_on_texts(all_captions)

max_length = max(len(caption.split()) for caption in all_captions)
max_length

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer):
    global max_length
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
      
    return in_text

vgg_model = VGG16()

vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

cap = cv2.VideoCapture("video3.mp4")

i=10
while True:
    try:
        success, img = cap.read()
        img = cv2.flip(img, 1)

        if i == 10:
            image = np.asarray(img)     
            image = img
            image = cv2.resize(image, (224, 224))
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)
            feature = vgg_model.predict(image, verbose=0)
            
            # SA3: Call the predict_caption function with model, feature, tokenizer and save the returned value in variable caption
            caption = predict_caption(model, feature, tokenizer)

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

