# Import libraries
import gensim
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
from nltk.tokenize import word_tokenize

def generate_sentence_vectors(sentences):
  """ Generate vectors for each sentences using word vector """
  sentence_vectors = []
  for sentence in sentences:
    words = sentence.split()
    word_vectors = [WORD_VECTOR_MODEL.wv[word] for word in words]
    # print(word_vectors)
    # exit(0)
    # average the n-dimensional word vector to generate sentence vector
    sentence_vector = np.mean(word_vectors, axis=0) 
    # print(sentence_vector)
    # exit(0)
    sentence_vectors.append(sentence_vector)
  return sentence_vectors


def test_model():
  """ Test classifier """
  test_sentences = ["I love this song", "He is very annoying", "This pizza is amazing", "She is very kind"]
  #test_sentences = ["This movie is awesome", "I hate this book", "The food was delicious", "She is very rude"]
  test_labels = [1, 0 ,1 ,1]
  test_sentence_vectors = []
  for test_sentence in test_sentences:
    test_words = test_sentence.split()
    test_word_vectors = [WORD_VECTOR_MODEL.wv[word] for word in test_words if word in WORD_VECTOR_MODEL.wv] 
    test_sentence_vector = np.mean(test_word_vectors, axis=0) # average the word vectors
    test_sentence_vectors.append(test_sentence_vector)
  test_predictions = CLASSIFIER.predict(test_sentence_vectors)
  test_accuracy = accuracy_score(test_labels, test_predictions)
  print("Test accuracy:", test_accuracy)


# Load data
sentences = open("NEWNCS.txt", 'r')
labels = [1, 0, 1, 0] # 1 for positive, 0 for negative

data = []
for i in sentences:
    temp = [] 
    # tokenize the sentence into words
    for j in word_tokenize(i):
        temp.append(j)
        #temp.append(j.lower())
    data.append(temp)

#print(data)
#exit(0)

# Train word embeddings
WORD_VECTOR_MODEL = gensim.models.Word2Vec(data, min_count=1, vector_size=10)

#words = sentences[0].split()
#print(words)
#exit(0)

# Convert sentences to vectors
sentence_vectors = generate_sentence_vectors(sentences)
print(sentence_vectors)
# exit(0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sentence_vectors, labels, test_size=0.2)

# Train classifier
#CLASSIFIER = LogisticRegression()
#CLASSIFIER.fit(sentence_vectors, labels)

# Train classifier
#CLASSIFIER = LogisticRegression()
#CLASSIFIER.fit(X_train, y_train)

# Train classifier
CLASSIFIER = LinearSVC()
CLASSIFIER.fit(X_train, y_train)

# Evaluate classifier on testing data
score = CLASSIFIER.score(X_test, y_test)
print('Accuracy:', score)


test_model()