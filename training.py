#!/usr/bin/env python
# coding: utf-8

# In[1]:


from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.layers import *
import numpy as np
import pickle
import random
import glob


# ##Preprocessing Data

# In[2]:


notes=[]

for file in glob.glob("midi_songs/*.mid"):
    midi = converter.parse(file) 
    elements_to_parse = midi.flat.notes
    for element in elements_to_parse:
        if isinstance(element,note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element,chord.Chord):
            notes.append("+".join(str(t) for t in element.normalOrder))
            


# ##Storing Notes

# In[ ]:


with open("notes",'wb') as filepath:
    pickle.dump(notes,filepath)


# ##Mappings

# In[3]:


pitchnames = sorted(set(notes))
element_to_integer = dict((element, num) for num,element in enumerate(pitchnames))


# ##Preparing Data for LSTM

# In[4]:


sequence_length = 100
number_of_examples = len(notes)-sequence_length
number_of_classes = len(pitchnames)  #total kind of notes
network_input = []
network_output = []

for i in range(number_of_examples):
    seq_in = notes[i:i+sequence_length] #list of strings
    seq_out = notes[i+sequence_length] #string
    network_input.append([element_to_integer[s] for s in seq_in])
    network_output.append(element_to_integer[seq_out])
    
#LSTM accepts 3D i/p
network_input = np.reshape(network_input,(number_of_examples,sequence_length,1))
#Normalize 
network_input = network_input/float(number_of_classes)
#Creating One-hot vector
network_output = np_utils.to_categorical(network_output)


# ##Creating LSTM Model

# In[5]:


model = Sequential()
model.add(LSTM(units=512, input_shape=(sequence_length,1), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512))
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(number_of_classes,activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam")
model.summary()


# ##Training Model

# In[ ]:


checkpoint = ModelCheckpoint("model.hd5",monitor="loss", verbos = 0, save_best_only = True, mode="min")
model_his = model.fit(network_input, network_output, epochs=100, batch_size=64, callbacks=[checkpoint])

