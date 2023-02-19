#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
print(tf.__version__)


# # Sequence modelling 
# 
# ## Coding tutorials
#  #### [1.  The IMDb dataset](#coding_tutorial_1)
#  #### [2. Padding and masking sequence data](#coding_tutorial_2)
#  #### [3. The Embedding layer](#coding_tutorial_3)
#  #### [4. The Embedding Projector](#coding_tutorial_4)
#  #### [5. Recurrent neural network layers](#coding_tutorial_5)
#  #### [6. Stacked RNNs and the Bidirectional wrapper](#coding_tutorial_6)

# ***
# <a id="coding_tutorial_1"></a>
# ## The IMDb Dataset

# #### Load the IMDB review sentiment dataset

# In[2]:


# Import imdb

from tensorflow.keras.datasets import imdb


# In[3]:


# Download and assign the data set using load_data()

(x_train, y_train), (x_test, y_test) = imdb.load_data()


# #### Inspect the dataset

# In[4]:


# Inspect the type of the data

print(type(x_train), type(y_train))


# In[5]:


# Inspect the shape of the data

print(x_train.shape, y_train.shape)


# In[6]:


# Display the first dataset element input
# Notice encoding

print(x_train[0])


# In[7]:


# Display the first dataset element output

print(y_train[0])


# #### Load dataset with different options

# In[8]:


# Load the dataset with defaults

(x_train, y_train), (x_test, y_test) = imdb.load_data(index_from=3)
# ~/.keras/dataset/


# In[9]:


# Limit the vocabulary to the top 500 words using num_words

(x_train, y_train), (x_test, y_test) = imdb.load_data(path='imdb.npz', num_words=500)


# In[10]:


# Ignore the top 10 most frequent words using skip_top
(x_train, y_train), (x_test, y_test) = imdb.load_data(skip_top=10)


# In[11]:


# Limit the sequence lengths to 500 using maxlen
(x_train, y_train), (x_test, y_test) = imdb.load_data(maxlen=500)


# In[12]:


# Use '1' as the character that indicates the start of a sequence
(x_train, y_train), (x_test, y_test) = imdb.load_data(start_char=1)


# #### Explore the dataset word index

# In[13]:


# Load the imdb word index using get_word_index()

wi = imdb.get_word_index( path='imdb_word_index.json' )


# In[14]:


# View the word index as a dictionary,
# accounting for index_from.
index_from = 6
wi = {key:value + index_from for key, value in wi.items()}
print(wi)


# In[15]:


# Retrieve a specific word's index

print(wi['drill'])


# In[16]:


# View an input sentence

inv_wi = {value: key for key, value in wi.items()}
print(inv_wi)


# In[17]:


# Get the sentiment value

print(y_train[0])


# ---
# <a id="coding_tutorial_2"></a>
# ## Padding and Masking Sequence Data

# In[10]:


# Load the imdb data set

import tensorflow.keras.datasets.imdb as imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data()


# #### Preprocess the data with padding

# In[12]:


# Inspect the input data shape

x_train.shape
x_train[0]


# In[14]:


# Pad the inputs to the maximum length using maxlen

padded_x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=300, padding='post', truncating='pre')


# In[15]:


# Inspect the output data shape

padded_x_train.shape


# #### Create a Masking layer

# In[16]:


# Import numpy 

import numpy as np


# In[17]:


# Masking expects to see (batch, sequence, features)
# Create a dummy feature dimension using expand_dims

padded_x_train = np.expand_dims(padded_x_train, -1)


# In[18]:


# Create a Masking layer 

tf_x_train = tf.convert_to_tensor(padded_x_train, dtype=tf.float32)
mask_layer = tf.keras.layers.Masking(mask_value=0.0)


# In[19]:


# Pass tf_x_train to it

masked_x_train = mask_layer(tf_x_train)


# In[20]:


# Look at the dataset

masked_x_train


# In[21]:


# Look at the ._keras_mask for the dataset

masked_x_train._keras_mask


# ***
# <a id="coding_tutorial_3"></a>
# ## The Embedding layer

# #### Create and apply an `Embedding` layer

# In[2]:


# Create an embedding layer using layers.Embedding
# Specify input_dim, output_dim, input_length

from tensorflow.keras.layers import Embedding

embedding_layer = Embedding(input_dim=501,  output_dim=16 )


# In[3]:


# Inspect an Embedding layer output for a fixed input
# Expects an input of shape (batch, sequence, feature)

sequence_of_indices = tf.constant([[[0],[1],[5],[500]]])
sequence_of_embeddings = embedding_layer(sequence_of_indices)
sequence_of_embeddings.shape


# In[4]:


# Inspect the Embedding layer weights using get_weights()

embedding_layer.get_weights()[0]


# In[5]:


# Get the embedding for the 14th index

embedding_layer.get_weights()[0][14,:]


# #### Create and apply an `Embedding` layer that uses `mask_zero=True`

# In[6]:


# Create a layer that uses the mask_zero kwarg

masking_embedding_layer = tf.keras.layers.Embedding(input_dim=501, output_dim=16, mask_zero=True)


# In[7]:


# Apply this layer to the sequence and see the _keras_mask property

masked_sequence_of_embeddings = masking_embedding_layer(sequence_of_indices)
masked_sequence_of_embeddings._keras_mask


# ---
# <a id="coding_tutorial_4"></a>
# ## The Embedding Projector

# #### Load and preprocess the IMDb data

# In[18]:


# A function to load and preprocess the IMDB dataset

def get_and_pad_imdb_dataset(num_words=10000, maxlen=None, index_from=2):
    from tensorflow.keras.datasets import imdb

    # Load the reviews
    (x_train, y_train), (x_test, y_test) = imdb.load_data(path='imdb.npz',
                                                          num_words=num_words,
                                                          skip_top=0,
                                                          maxlen=maxlen,
                                                          start_char=1,
                                                          oov_char=2,
                                                          index_from=index_from)

    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train,
                                                        maxlen=None,
                                                        padding='pre',
                                                        truncating='pre',
                                                        value=0)
    
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test,
                                                           maxlen=None,
                                                           padding='pre',
                                                           truncating='pre',
                                                           value=0)
    return (x_train, y_train), (x_test, y_test)


# In[19]:


# Load the dataset

dataset = get_and_pad_imdb_dataset(num_words=10000, maxlen=None, index_from=2)


# In[20]:


# A function to get the dataset word index


def get_imdb_word_index(num_words=10000, index_from=2):
    imdb_word_index = tf.keras.datasets.imdb.get_word_index(
                                        path='imdb_word_index.json')
    imdb_word_index = {key: value + index_from for
                       key, value in imdb_word_index.items() if value <= num_words-index_from}
    return imdb_word_index


# In[25]:


# Get the word index

imdb_word_index = get_imdb_word_index(num_words=10000, index_from=2)


# In[26]:


# Swap the keys and values of the word index


#(x_train, y_train), (x_test, y_test) = dataset
#[index for index in x_train[100] if index > 2]

inv_imdb_word_index = {value: key for key, value in imdb_word_index.items()}


# In[27]:


# View the first dataset example sentence

[inv_imdb_word_index[index] for index in x_train[100] if index > 2]


# #### Build an Embedding layer into a model

# In[28]:


# Get the maximum token value

max_index_value = max(imdb_word_index.values())


# In[29]:


# Specify an embedding dimension

embedding_dim =16


# In[ ]:


# Build a model using Sequential:
#     1. Embedding layer
#     2. GlobalAveragePooling1D
#     3. Dense

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_index_value+1, output_dim=embedding_dim, mask_zero=False),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])


# In[30]:


# Functional API refresher: use the Model to build the same model

review_sequence      = tf.keras.Input((None, ))
embedding_sequence   = tf.keras.layers.Embedding(input_dim=max_index_value+1, output_dim=embedding_dim)(review_sequence)
average_embedding    = tf.keras.layers.GlobalAveragePooling1D()(embedding_sequence)
positive_probability = tf.keras.layers.Dense(units=1, activation='sigmoid')(average_embedding)
model = tf.keras.Model(inputs=review_sequence, outputs=positive_probability)


# In[31]:


model.summary()


# #### Compile, train, and evaluate the model

# In[41]:


# Compile the model with a binary cross-entropy loss

model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')


# In[42]:


# Train the model using .fit(), savng its history

history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test), validation_steps=20)


# In[44]:


# Plot the training and validation accuracy





import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

history_dict = history.history

print(history_dict.keys())


acc      = history_dict['accuracy']
val_acc  = history_dict['val_accuracy']
loss     = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(14,5))
plt.plot(epochs, acc, marker='.', label='Training acc')
plt.plot(epochs, val_acc, marker='.', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Classification accuracy')
plt.legend(loc='lower right')
plt.ylim(0, 1);


# #### The TensorFlow embedding projector
# 
# The Tensorflow embedding projector can be found [here](https://projector.tensorflow.org/).

# In[45]:


# Retrieve the embedding layer's weights from the trained model

weights = model.layers[1].get_weights()[0]


# In[61]:


# Save the word Embeddings to tsv files
# Two files: 
#     one contains the embedding labels (meta.tsv),
#     one contains the embeddings (vecs.tsv)

import io
from os import path


out_v = io.open(path.join('data', 'vecs.tsv'), 'w', encoding='utf-8')
out_m = io.open(path.join('data', 'meta.tsv'), 'w', encoding='utf-8')

k = 0

for word, token in word_index.items():
    if k != 0:
        out_m.write('\n')
        out_v.write('\n')
    
    out_v.write('\t'.join([str(x) for x in weights[token]]))
    out_m.write(word)
    k += 1
    
out_v.close()
out_m.close()


# beware large collections of embeddings!


# ---
# <a id="coding_tutorial_5"></a>
# ## Recurrent neural network layers

# #### Initialize and pass an input to a SimpleRNN layer

# In[3]:


# Create a SimpleRNN layer and test it

# (batch, sequence, features)
simplernn_layer = tf.keras.layers.SimpleRNN(units=16)


# In[4]:


# Note that only the final cell output is returned

sequence = tf.constant([[[1.,1.], [2.,2.], [56., -100.]]])
layer_output = simplernn_layer(sequence)
layer_output


# #### Load and transform the IMDB review sentiment dataset

# In[5]:


# A function to load and preprocess the IMDB dataset

def get_and_pad_imdb_dataset(num_words=10000, maxlen=None, index_from=2):
    from tensorflow.keras.datasets import imdb

    # Load the reviews
    (x_train, y_train), (x_test, y_test) = imdb.load_data(path='imdb.npz',
                                                          num_words=num_words,
                                                          skip_top=0,
                                                          maxlen=maxlen,
                                                          start_char=1,
                                                          oov_char=2,
                                                          index_from=index_from)

    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train,
                                                        maxlen=None,
                                                        padding='pre',
                                                        truncating='pre',
                                                        value=0)
    
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test,
                                                           maxlen=None,
                                                           padding='pre',
                                                           truncating='pre',
                                                           value=0)
    return (x_train, y_train), (x_test, y_test)


# In[14]:


# Load the dataset

 (x_train, y_train), (x_test, y_test) = get_and_pad_imdb_dataset(maxlen=250)


# In[7]:


# A function to get the dataset word index

def get_imdb_word_index(num_words=10000, index_from=2):
    imdb_word_index = tf.keras.datasets.imdb.get_word_index(
                                        path='imdb_word_index.json')
    imdb_word_index = {key: value + index_from for
                       key, value in imdb_word_index.items() if value <= num_words-index_from}
    return imdb_word_index


# In[8]:


# Get the word index using get_imdb_word_index()

imdb_word_index = get_imdb_word_index()


# #### Create a recurrent neural network model

# In[9]:


# Get the maximum index value

max_index_value = max(imdb_word_index.values())
embedding_dim = 16


# In[10]:


# Using Sequential, build the model:
# 1. Embedding.
# 2. LSTM.
# 3. Dense.

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_index_value+1, output_dim=embedding_dim, mask_zero=True),
    tf.keras.layers.LSTM(units=16),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])


# #### Compile and fit the model

# In[11]:


# Compile the model with binary cross-entropy loss

model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')


# In[15]:


# Fit the model and save its training history

history = model.fit(x_train, y_train, epochs=3, batch_size=32, validation_data=(x_test, y_test))


# #### Plot learning curves

# In[16]:


# Plot the training and validation accuracy

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

history_dict = history.history

acc      = history_dict['accuracy']
val_acc  = history_dict['val_accuracy']
loss     = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(14,5))
plt.plot(epochs, acc, marker='.', label='Training acc')
plt.plot(epochs, val_acc, marker='.', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Classification accuracy')
plt.legend(loc='lower right')
plt.ylim(0, 1);


# #### Make predictions with the model

# In[17]:


# View the first test data example sentence
# (invert the word index)

inv_imdb_word_index = {value: key for key, value in imdb_word_index.items()}
[inv_imdb_word_index[index] for index in x_test[0] if index > 2]


# In[18]:


# Get the model prediction using model.predict()

model.predict(x_test[None, 0, :])


# In[19]:


# Get the corresponding label

y_test[0]


# ---
# <a id="coding_tutorial_6"></a>
# ## Stacked RNNs and the Bidirectional wrapper

# #### Load and transform the IMDB review sentiment dataset

# In[3]:


# A function to load and preprocess the IMDB dataset

def get_and_pad_imdb_dataset(num_words=10000, maxlen=None, index_from=2):
    from tensorflow.keras.datasets import imdb

    # Load the reviews
    (x_train, y_train), (x_test, y_test) = imdb.load_data(path='imdb.npz',
                                                          num_words=num_words,
                                                          skip_top=0,
                                                          maxlen=maxlen,
                                                          start_char=1,
                                                          oov_char=2,
                                                          index_from=index_from)

    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train,
                                                        maxlen=None,
                                                        padding='pre',
                                                        truncating='pre',
                                                        value=0)
    
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test,
                                                           maxlen=None,
                                                           padding='pre',
                                                           truncating='pre',
                                                           value=0)
    return (x_train, y_train), (x_test, y_test)


# In[4]:


# Load the dataset

(x_train, y_train), (x_test, y_test) = get_and_pad_imdb_dataset(num_words=5000, maxlen=250)


# In[5]:


# A function to get the dataset word index

def get_imdb_word_index(num_words=10000, index_from=2):
    imdb_word_index = tf.keras.datasets.imdb.get_word_index(
                                        path='imdb_word_index.json')
    imdb_word_index = {key: value + index_from for
                       key, value in imdb_word_index.items() if value <= num_words-index_from}
    return imdb_word_index


# In[6]:


# Get the word index using get_imdb_word_index()

imdb_word_index = get_imdb_word_index(num_words=5000)


# #### Build stacked and bidirectional recurrent models

# In[7]:


# Get the maximum index value and specify an embedding dimension

max_index_value = max(imdb_word_index.values())
embedding_dim = 16


# In[8]:


# Using Sequential, build a stacked LSTM model via return_sequences=True

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_index_value+1, output_dim=embedding_dim, mask_zero=True),
    tf.keras.layers.LSTM(units=32, return_sequences=True),
    tf.keras.layers.LSTM(units=32, return_sequences=False),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])


# In[9]:


# Using Sequential, build a bidirectional RNN with merge_mode='sum'

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_index_value+1, output_dim=embedding_dim, mask_zero=True),
    tf.keras.layers.Bidirectional(layer=tf.keras.layers.LSTM(units=8), merge_mode='sum',
                                 backward_layer=tf.keras.layers.GRU(units=8, go_backwards=True)),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])


# In[10]:


# Create a model featuring both stacked recurrent layers and a bidirectional layer

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_index_value+1, output_dim=embedding_dim),
    tf.keras.layers.Bidirectional(layer=tf.keras.layers.LSTM(units=8, return_sequences=True), merge_mode='concat'),
    tf.keras.layers.GRU(units=8, return_sequences=False),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])


# #### Compile and fit the model

# In[12]:


# Compile the model

model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')


# In[13]:


# Train the model, saving its history


history = model.fit(x_train, y_train, epochs=3, batch_size=32, validation_data=(x_test, y_test))


# In[14]:


# Plot the training and validation accuracy

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

history_dict = history.history

acc      = history_dict['accuracy']
val_acc  = history_dict['val_accuracy']
loss     = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(14,5))
plt.plot(epochs, acc, marker='.', label='Training acc')
plt.plot(epochs, val_acc, marker='.', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Classification accuracy')
plt.legend(loc='lower right')
plt.ylim(0, 1);

