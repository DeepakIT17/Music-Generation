#!/usr/bin/env python
# coding: utf-8

# In[23]:


from music21 import converter, instrument, note, chord, stream
from keras.models import load_model
import numpy as np
import pickle
import random


# ##Retrieving Notes

# In[24]:


with open("notes",'rb') as filepath:
    notes = pickle.load(filepath)


# ##Mappings

# In[ ]:


pitchnames = sorted(set(notes))
element_to_integer = dict((element, num) for num,element in enumerate(pitchnames))
integer_to_element = dict((num, element) for num,element in enumerate(pitchnames))


# ##Preprocessing notes

# In[28]:


test_input = []
sequence_length = 100
number_of_examples = len(notes)-sequence_length
number_of_classes = len(pitchnames) 

for i in range(number_of_examples):
    seq_in = notes[i:i+sequence_length] #list of strings
    test_input.append([element_to_integer[s] for s in seq_in])


# ##Load Saved Model

# In[26]:


model = load_model("weights.hdf5")


# ##Predictions

# In[29]:


start = np.random.randint(number_of_examples-1)
pattern = test_input[start]
pattern_output = []

for note_index in range(100):
    print(note_index)
    prediction_input = np.reshape(pattern,(1,len(pattern),1))
    prediction_input = prediction_input/float(number_of_classes)
    prediction = model.predict(prediction_input,verbose=0)
    idx = np.argmax(prediction)
    result = integer_to_element[idx]
    pattern_output.append(result)
    pattern.append(idx)
    pattern = pattern[1:]


# ##Creating MIDI FILE from prediction

# In[30]:


offset =0
output_notes = []

for pattern in pattern_output:
    if ('+' in pattern) or pattern.isdigit():
        notes_in_chord = pattern.split('+')
        temp_notes = []
        for current_note in notes_in_chord:
            new_note = note.Note(int(current_note))
            new_note.storedInstrument = instrument.Piano()
            temp_notes.append(new_note)
        new_chord = chord.Chord(temp_notes)
        new_chord.offset = offset
        output_notes.append(new_chord)
    else :
        new_note = note.Note(pattern)
        new_note.storedInstrument = instrument.Piano()
        new_note.offset = offset
        output_notes.append(new_note)
    offset+=random.random();

midi_stream = stream.Stream(output_notes)
midi_stream.write('midi',fp="test_output4.mid")
    


# In[31]:


midi_stream.show('midi')


# In[ ]:




