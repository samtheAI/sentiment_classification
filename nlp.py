pip install -r requirements.txt
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from nltk.stem.porter import PorterStemmer
import joblib
import numpy as np
import streamlit as st



st.title('Model Deployment: Logistic Regression')
   

encoder=joblib.load('tfidf.pkl')
model=joblib.load('modelnlp.pkl')
stemmer=PorterStemmer()

st.subheader('Please enter your input')

 
text = st.text_input("Enter the Text")


def preprocessing(text):

    text=text.lower()
   # print(text)
    text=re.sub('[^a-zA-Z ]' ,'',text)
    #print(text)

    text=text.split()
   # print(text)

    text=[stemmer.stem(word) for word in text if word not in stopwords.words("english") ]
    #print(text)

    text=" ".join(text)
    #print(text)

    return text



if st.button("Predict"):
        
        

        text=preprocessing(text)
        vector=encoder.transform({text}).toarray()
        pred=model.predict(vector)
        st.subheader('Predicted Class')
        if pred[0]==0:
            st.write("Negative")
        else:
                      st.write("Positive")
                     

            
        
       
        

