from JoinAttLayer import Attention
from classifier_capsule import TextClassifier
import classifier_rcnn
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

#embeddings_matrix = np.zeros((len(word_index) + 1, w2_model.vector_size))
#w2_model = KeyedVectors.load_word2vec_format("word2vec/chars.vector", binary=True, encoding='utf8',
#                                             unicode_errors='ignore')
embeddings_matrix = np.zeros((1000, 300))

model1 = TextClassifier().model(embeddings_matrix,1000,2000,4)
model2 = classifier_rcnn.TextClassifier().model(embeddings_matrix,1000,2000,4)

print(model1.summary())
print('-----------------------------')
print(model2.summary())




