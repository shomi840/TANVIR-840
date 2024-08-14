#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install nltk


# In[2]:


import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

# Download necessary datasets (only needed the first time)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')



# In[3]:


# Sample text
text = "Barack Obama was born on August 4, 1961, in Honolulu, Hawaii. He was the 44th President of the United States."


# In[4]:


# Sentence tokenization
sentences = sent_tokenize(text)
print("Sentences:", sentences)


# In[5]:


# Word tokenization
words = word_tokenize(text)
print("Words:", words)


# In[6]:


# Part-of-speech tagging
pos_tags = pos_tag(words)
print("POS Tags:", pos_tags)


# In[7]:


# Named entity recognition
named_entities = ne_chunk(pos_tags)
print("Named Entities:", named_entities)


# In[ ]:


# Visualize named entities
named_entities.draw()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




