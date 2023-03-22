# Python program to generate word vectors using Word2Vec

# importing all necessary modules
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings

warnings.filterwarnings(action = 'ignore')

import gensim
from gensim.models import Word2Vec

# Reads speech  file
sample = open("NEWNCS.txt", 'r')
#sample = open("NEWACS.txt", 'r')
s = sample.read()

# Replaces escape character with space
f = s.replace("\n", " ")
f1 = f.replace("(", " ")
f2 = f1.replace(")", " ")
f3 = f2.replace("+", " ")
f4 = f3.replace(":", " ")

document = f4


data = []

# iterate through each sentence in the file
for i in sent_tokenize(document):
	temp = []
	
	# tokenize the sentence into words
	for j in word_tokenize(i):
		temp.append(j.lower())

	data.append(temp)

#print (data)

# Create CBOW model
for i in data:
	model = gensim.models.Word2Vec(data, min_count = 1, vector_size = 100, window = 5)

print (model)



