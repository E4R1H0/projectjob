import streamlit as st
import pandas as pd
import pythainlp
from pythainlp.corpus.common import thai_stopwords
from pythainlp import word_tokenize
from wordcloud import WordCloud, STOPWORDS
#import matplotlib as plt
import matplotlib.pyplot as plt



df = pd.read_csv('./samsungreview.csv')
dftext=pd.DataFrame(df)

html_1 = """
<div style="background-color:#363062;padding:5px;margin-bottom:50px;border-radius:3px;border-bottom: 4px solid #ffffff;border-top: 4px solid #ffffff;">
<center><h3>กราฟเปรียบเทียบความคิดเห็น</h3></center>
</div>
"""
st.markdown(html_1, unsafe_allow_html=True)
st.markdown("")

st.bar_chart(dftext['sentiment'].value_counts())
thai_stopwords = list(thai_stopwords())
#st.write(thai_stopwords)


    
    
#from sklearn.metrics import confusion_matrix,classification_report
#predictions_nb = naive_bayes_model.predict(test_bow)
#test_bow = cvec.transform(X_test['text_tokens'])
##print(classification_report(predictions_nb, y_test))