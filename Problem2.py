import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras import Sequential
from keras.layers import Embedding, Dense, Dropout, GlobalMaxPool1D, Flatten

ps = PorterStemmer()
tfidVectorizer = TfidfVectorizer()
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
englishStopWords = set(stopwords.words('english'))

myDataFile = pd.read_csv("onion-or-not.csv")
dataTexter = myDataFile.text
label = myDataFile.label

vectorizedText = dataTexter.to_numpy()
vectors_of_words = []


for string in range(len(vectorizedText)):
    for word in word_tokenize(vectorizedText[string]):
        vectorizedText[string] = vectorizedText[string].replace(word, ps.stem(word))
        if word in englishStopWords:
            vectorizedText[string] = vectorizedText[string].replace(word, "")


x = tfidVectorizer.fit(vectorizedText)
x = tfidVectorizer.transform(vectorizedText)

dataframe = pd.DataFrame(x.toarray(), columns=tfidVectorizer.get_feature_names())

dataframe.insert(len(dataframe.columns), "insertedFrame", label, True)
dataframeY = dataframe.insertedFrame
dataframeX = dataframe.drop('insertedFrame', axis=1)


X_train, X_test, y_train, y_test = train_test_split(dataframeX, dataframeY, test_size=0.25)


# Normilizations
X_train = tf.keras.utils.normalize(X_train, axis=1).to_numpy()
X_test = tf.keras.utils.normalize(X_test, axis=1).to_numpy()


y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

model = Sequential()

model.add(Flatten())
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dense(2, activation=tf.nn.softmax))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1, validation_data=(X_test, y_test), callbacks=[es])

val_loss, val_acc = model.evaluate(X_test, y_test)  # Model evaluation.
print("\nLoss:", val_loss, "\nAccuracy: ", val_acc)

predictions = model.predict(X_test)

predictions_list = list()
for i in range(len(predictions)):
    predictions_list.append(np.argmax(predictions[i]))


print(sklearn.metrics.classification_report(y_test, predictions_list))