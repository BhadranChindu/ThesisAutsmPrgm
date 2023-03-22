# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 13:54:07 2023

@author: Bhadran
"""
import re
# Reads speech  file
#sample = open("NEWNCS.txt", 'r')
sample = open("NEWACS.txt", 'r')
s = sample.read()

# Replaces escape character with space
f = s.replace("\n", " ")


document = [f]


from sklearn.feature_extraction.text import CountVectorizer

# Create a Vectorizer Object
vectorizer = CountVectorizer()
 
vectorizer.fit(document)
 
# Printing the identified Unique words along with their indices

voc = (vectorizer.vocabulary_)
print(voc)
 
# Encode the Document

#vector = vectorizer.transform(document)
#print(vector)