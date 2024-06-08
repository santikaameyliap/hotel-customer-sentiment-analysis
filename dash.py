import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Dashboard", page_icon="", layout="wide")
st.header("Dashboard Analisis Sentimen Midtown & Lifestyle Hotel")
st.subheader("using Python:snake:")

kpi1, kpi2 = st.columns(2)

st.markdown("###") 

kpi1.metric(label = "JUMLAH DATA HOTEL MIDTOWN",
            value = 508)
kpi2.metric(label = "JUMLAH DATA HOTEL LIFESTYLE",
            value = 1896)


# Dataset
data_midtown= pd.read_csv('data_midtown_baru.csv')
data_lifestyle = pd.read_csv('data_lifestyle_baru.csv')



# Convert rating agar mudah untuk melakukan sentimen analisis
# 1 = postif
# 0 = negatif
data_midtown['rating'] = data_midtown['rating'].apply(lambda x: 1 if x >= 8.5 else 0)
data_lifestyle['rating'] = data_lifestyle['rating'].apply(lambda x: 1 if x >= 8.5 else 0)


sentimen_midtown = data_midtown.rating.value_counts()
sentimen_lifestyle=data_lifestyle.rating.value_counts()

# Plotting Pie Chart berdasarkan rating di Midtown Hotel Surabaya
labels = ['Positif (1)', 'Negatif (0)'] # Labels untuk pie chart
colors = ['green', 'red'] # Warna untuk setiap sektor pie chart
explode = (0.1, 0) # Eksplosi sektor (jika Anda ingin beberapa sektor menonjol)

plt.figure(figsize=(3, 3))  # Ukuran pie chart

# Membuat pie chart
fig1, ax1 = plt.subplots()
ax1.pie(sentimen_midtown, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%', shadow=True)
ax1.set_title('Sentimen Analisis Berdasarkan Rating di Midtown Hotel Surabaya')
ax1.axis('equal')
ax1.legend(labels, loc='upper right')


# Plotting Pie Chart berdasarkan rating di The Life Style Hotel Surabaya 
labels = ['Positif (1)', 'Negatif (0)']
colors = ['green', 'red']
explode = (0.1, 0)

plt.figure(figsize=(6, 6))

# Membuat pie chart
fig2, ax2 = plt.subplots()
ax2.pie(sentimen_lifestyle, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%', shadow=True)
ax2.set_title('Sentimen Analisis Berdasarkan Rating di The Life Style Hotel Surabaya')
ax2.axis('equal')
ax2.legend(labels, loc='upper right')


st.subheader("Visualisasi Data")
tab1, tab2 = st.tabs(["Midtown", "LifeStyle"])
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.text("Grafik")
        st.pyplot(fig1)
    with col2:
        st.text("Data Detail")
        data_midtown
    
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.text("Grafik")
        st.pyplot(fig2)
    with col2:
        st.text("Data Detail")
        data_lifestyle


# Hitung persentase sentimen untuk Dataset 1
total_reviews_dataset1 = len(data_midtown)  # Total jumlah ulasan
positive_reviews_dataset1 = len(data_midtown[data_midtown['rating'] == 1])  # Jumlah ulasan positif
negative_reviews_dataset1 = len(data_midtown[data_midtown['rating'] == 0])  # Jumlah ulasan negatif
percent_positive_dataset1 = (positive_reviews_dataset1 / total_reviews_dataset1) * 100
percent_negative_dataset1 = (negative_reviews_dataset1 / total_reviews_dataset1) * 100

# Hitung persentase sentimen untuk Dataset 2
total_reviews_dataset2 = len(data_lifestyle)
positive_reviews_dataset2 = len(data_lifestyle[data_lifestyle['rating'] == 1])
negative_reviews_dataset2 = len(data_lifestyle[data_lifestyle['rating'] == 0])
percent_positive_dataset2 = (positive_reviews_dataset2 / total_reviews_dataset2) * 100
percent_negative_dataset2 = (negative_reviews_dataset2 / total_reviews_dataset2) * 100


# Visualisasi Perbandingan Sentimen di Kedua Hotel dengan Bar Plot
labels = ['Midtown Hotel', 'The Life Style Hotel'] # Label untuk dataset

# Persentase sentimen dari kedua dataset
percent_positive = [percent_positive_dataset1, percent_positive_dataset2]
percent_negative = [percent_negative_dataset1, percent_negative_dataset2]

bar_width = 0.35 # Mengatur lebar bar
x = range(len(labels)) # Mengatur posisi x untuk dua grup bar


fig3, ax3 = plt.subplots() # Membuat subplot
bar1 = ax3.bar(x, percent_positive, bar_width, label='Positif', color='g') # Membuat bar untuk sentimen positif

# Membuat bar untuk sentimen negatif di sebelah kanan bar positif
bar2 = ax3.bar([i + bar_width for i in x], percent_negative,
              bar_width, label='Negatif', color='r')

# Menambahkan label sumbu x
ax3.set_xlabel('Hotel')
ax3.set_ylabel('Persentase Sentimen (%)')
ax3.set_title('Perbandingan Sentimen Positif dan Negatif antara Dua Hotel')

# Menambahkan label sumbu x pada posisi tengah grup bar
ax3.set_xticks([i + bar_width / 2 for i in x])
ax3.set_xticklabels(labels)

ax3.legend()
#st.pyplot(fig3)




data_midtown['tanggal'] = pd.to_datetime(data_midtown['tanggal'])
data_lifestyle['tanggal'] = pd.to_datetime(data_lifestyle['tanggal'])


# Misalkan Anda memiliki DataFrame data_midtown dan data_lifestyle
# Filter data untuk bulan Januari 2021 hingga Desember 2023 untuk kedua dataset
filtered_df1 = data_midtown[(data_midtown['tanggal'] >= '2021-01-01') &
                            (data_midtown['tanggal'] <= '2023-12-31')]
filtered_df2 = data_lifestyle[(data_lifestyle['tanggal'] >= '2021-01-01') &
                              (data_lifestyle['tanggal'] <= '2023-12-31')]

# Groupby tahun dan bulan, kemudian hitung rata-rata sentimen per bulan untuk kedua dataset
sentimen_per_bulan1 = filtered_df1.groupby(
    [filtered_df1['tanggal'].dt.year, filtered_df1['tanggal'].dt.month])['rating'].mean() * 100
sentimen_per_bulan2 = filtered_df2.groupby(
    [filtered_df2['tanggal'].dt.year, filtered_df2['tanggal'].dt.month])['rating'].mean() * 100

# Ubah indeks menjadi bulan dan tahun
sentimen_per_bulan1.index = [
    f'{year}-{month:02}' for year, month in sentimen_per_bulan1.index]
sentimen_per_bulan2.index = [
    f'{year}-{month:02}' for year, month in sentimen_per_bulan2.index]

# Membuat DataFrame baru untuk Plotly
df_plotly = pd.DataFrame({
    'Bulan': sentimen_per_bulan1.index,
    'Midtown Hotel': sentimen_per_bulan1.values,
    'Life Style Hotel': sentimen_per_bulan2.values
})

# Membuat grafik interaktif dengan Plotly Express
fig = px.line(df_plotly, x='Bulan', y=['Midtown Hotel', 'Life Style Hotel'],
              title='Perbandingan Tren Persentase Sentimen Analisis (Jan 2021 - Des 2023)')
fig.update_xaxes(type='category')  # Menggunakan tipe kategori pada sumbu x

# Menampilkan grafik di Streamlit
#st.plotly_chart(fig)

col1, col2 = st.columns(2)
with col1:
    st.pyplot(fig3)
    
with col2:
    st.plotly_chart(fig)



# Visualisasi HeatMap Calendar pada Midtown Hotel Surabaya
# Set kolom 'Tanggal' sebagai indeks
data_midtown.set_index('tanggal', inplace=True)

# Ekstrak tahun dan bulan ke kolom terpisah
data_midtown['Tahun'] = data_midtown.index.year
data_midtown['Bulan'] = data_midtown.index.month

# Buat matriks berisi sentimen
pivot_data = data_midtown.pivot_table(index='Tahun', columns='Bulan', values='rating', aggfunc='mean')
pivot_data = pivot_data.fillna(0)

# Buat heatmap calendar dengan seaborn
plt.figure(figsize=(12, 6))
plot1 = sns.heatmap(pivot_data, cmap='coolwarm', annot=True, fmt=".2f", cbar=False)
plt.title('Heatmap Calendar Sentimen Analisis pada Midtown Hotel Surabaya (0: Negatif, 1: Positif)')
#st.pyplot(plot1.get_figure())


# Visualisasi HeatMap Calendar pada The Life Style Hotel Surabaya
# Set kolom 'Tanggal' sebagai indeks
data_lifestyle.set_index('tanggal', inplace=True)

# Ekstrak tahun dan bulan ke kolom terpisah
data_lifestyle['Tahun'] = data_lifestyle.index.year
data_lifestyle['Bulan'] = data_lifestyle.index.month

# Buat matriks berisi sentimen
pivot_data = data_lifestyle.pivot_table(
    index='Tahun', columns='Bulan', values='rating', aggfunc='mean')
pivot_data = pivot_data.fillna(0)

# Buat heatmap calendar dengan seaborn
plt.figure(figsize=(12, 6))
plot2 = sns.heatmap(pivot_data, cmap='coolwarm', annot=True, fmt=".2f", cbar=False)
plt.title('Heatmap Calendar Sentimen Analisis pada The Life Style Hotel Surabaya (0: Negatif, 1: Positif)')
#st.pyplot(plot2.get_figure())



col1, col2 = st.columns(2)
with col1:
    st.pyplot(plot1.get_figure())
    
with col2:
    st.pyplot(plot2.get_figure())


#MAPPING

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

#Basic libraries
import pandas as pd 
import numpy as np 


#NLTK libraries
import nltk
import re
import string
from wordcloud import WordCloud, STOPWORDS
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Machine Learning libraries
import sklearn 
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn import svm, datasets
from sklearn import preprocessing 


#Metrics libraries
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

#Visualization libraries
import matplotlib.pyplot as plt 
from matplotlib import rcParams
import seaborn as sns
from textblob import TextBlob
from plotly import tools
import plotly.graph_objs as go
#%matplotlib inline


#Other miscellaneous libraries
from scipy import interp
from itertools import cycle
import cufflinks as cf
from collections import defaultdict
from collections import Counter
from imblearn.over_sampling import SMOTE

import nltk
from nltk.corpus import stopwords

# Download the stopwords dataset if you haven't already
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))


#Dataset
raw_reviews = pd.read_excel('Midtown Hotel Surabaya Review.xlsx')

#Creating a copy
process_reviews=raw_reviews.copy()

process_reviews['ulasan']=process_reviews['ulasan'].fillna('Missing')

def f(row):
    
    '''This function returns sentiment value based on the overall ratings from the user'''
    
    if row['rating'] >= 8.5:
        val = 'Positive'
    elif row['rating'] >= 6.5:
        val = 'Neutral'
    else:
        val = 'Negative'
    return val

process_reviews['sentiment'] = process_reviews.apply(f, axis=1)

#Removing unnecessary columns
process_reviews=process_reviews.drop(['nama'], axis=1)
#Creating a copy 
clean_reviews=process_reviews.copy()

def review_cleaning(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


def casefoldingText(text): # Converting all the characters in a text into lower case
    text = text.lower() 
    return text

def stemmingText(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = [stemmer.stem(word) for word in text]
    text = ' '.join(text)
    return text


def toSentence(list_words): # Convert list of words into sentence
    sentence = ' '.join(word for word in list_words)
    return sentence

from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

process_reviews['ulasan']=process_reviews['ulasan'].apply(lambda x:review_cleaning(x))
process_reviews['ulasan']=process_reviews['ulasan'].apply(lambda x:casefoldingText(x))

#stemming
#Extracting 'reviews' for processing
review_features=process_reviews.copy()
review_features=review_features[['ulasan']].reset_index(drop=True)
review_features.head()

#Performing stemming on the review dataframe
ps = PorterStemmer()

#splitting and adding the stemmed words except stopwords
corpus = []
for i in range(0, len(review_features)):
    review = re.sub('[^a-zA-Z]', ' ', review_features['ulasan'][i])
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stop_words]
    review = ' '.join(review)
    corpus.append(review)  
    

process_reviews=process_reviews.drop(['ulasan'], axis=1)
process_reviews = pd.concat([process_reviews, review_features], axis=1)

import nltk
nltk.download('stopwords')

stop_words = set(stopwords.words('indonesian'))
process_reviews['ulasan'] = process_reviews['ulasan'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))


process_reviews['polarity'] = process_reviews['ulasan'].map(lambda text: TextBlob(text).sentiment.polarity)
process_reviews['review_len'] = process_reviews['ulasan'].astype(str).apply(len)
process_reviews['word_count'] = process_reviews['ulasan'].apply(lambda x: len(str(x).split()))


import plotly.offline as pyo


#Filtering data
review_pos = process_reviews[process_reviews["sentiment"]=='Positive'].dropna()
review_neu = process_reviews[process_reviews["sentiment"]=='Neutral'].dropna()
review_neg = process_reviews[process_reviews["sentiment"]=='Negative'].dropna()

## custom function for ngram generation ##
def generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

## custom function for horizontal bar chart ##
def horizontal_bar_chart(df, color):
    trace = go.Bar(
        y=df["word"].values[::-1],
        x=df["wordcount"].values[::-1],
        showlegend=False,
        orientation = 'h',
        marker=dict(
            color=color,
        ),
    )
    return trace

## Get the bar chart from positive reviews ##
freq_dict = defaultdict(int)
for sent in review_pos["ulasan"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(25), 'green')

## Get the bar chart from neutral reviews ##
freq_dict = defaultdict(int)
for sent in review_neu["ulasan"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace1 = horizontal_bar_chart(fd_sorted.head(25), 'grey')

## Get the bar chart from negative reviews ##
freq_dict = defaultdict(int)
for sent in review_neg["ulasan"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace2 = horizontal_bar_chart(fd_sorted.head(25), 'red')

# Creating two subplots
fig = tools.make_subplots(rows=3, cols=1, vertical_spacing=0.04,
                          subplot_titles=["Frequent words of positive reviews", "Frequent words of neutral reviews",
                                          "Frequent words of negative reviews"])
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 2, 1)
fig.append_trace(trace2, 3, 1)
fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots Hotel Midtown")



text = review_pos["ulasan"]
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(text))
fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

raw_reviews = pd.read_excel('The Life Style Hotel Surabaya Review.xlsx')
#Creating a copy
process_reviews=raw_reviews.copy()

process_reviews['ulasan']=process_reviews['ulasan'].fillna('Missing')

def f(row):
    
    '''This function returns sentiment value based on the overall ratings from the user'''
    
    if row['rating'] >= 8.5:
        val = 'Positive'
    elif row['rating'] >= 6.5:
        val = 'Neutral'
    else:
        val = 'Negative'
    return val

#Applying the function in our new column
process_reviews['sentiment'] = process_reviews.apply(f, axis=1)

#Removing unnecessary columns
process_reviews=process_reviews.drop(['nama'], axis=1)
#Creating a copy 
clean_reviews=process_reviews.copy()

def review_cleaning(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


def casefoldingText(text): # Converting all the characters in a text into lower case
    text = text.lower() 
    return text

def stemmingText(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = [stemmer.stem(word) for word in text]
    text = ' '.join(text)
    return text


def toSentence(list_words): # Convert list of words into sentence
    sentence = ' '.join(word for word in list_words)
    return sentence

from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

process_reviews['ulasan']=process_reviews['ulasan'].apply(lambda x:review_cleaning(x))
process_reviews['ulasan']=process_reviews['ulasan'].apply(lambda x:casefoldingText(x))


#stemming
#Extracting 'reviews' for processing
review_features=process_reviews.copy()
review_features=review_features[['ulasan']].reset_index(drop=True)
review_features.head()

#Performing stemming on the review dataframe
ps = PorterStemmer()

#splitting and adding the stemmed words except stopwords
corpus = []
for i in range(0, len(review_features)):
    review = re.sub('[^a-zA-Z]', ' ', review_features['ulasan'][i])
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stop_words]
    review = ' '.join(review)
    corpus.append(review)  
    

process_reviews=process_reviews.drop(['ulasan'], axis=1)
process_reviews = pd.concat([process_reviews, review_features], axis=1)

stop_words = set(stopwords.words('indonesian'))
process_reviews['ulasan'] = process_reviews['ulasan'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

process_reviews['polarity'] = process_reviews['ulasan'].map(lambda text: TextBlob(text).sentiment.polarity)
process_reviews['review_len'] = process_reviews['ulasan'].astype(str).apply(len)
process_reviews['word_count'] = process_reviews['ulasan'].apply(lambda x: len(str(x).split()))

#Filtering data
review_pos = process_reviews[process_reviews["sentiment"]=='Positive'].dropna()
review_neu = process_reviews[process_reviews["sentiment"]=='Neutral'].dropna()
review_neg = process_reviews[process_reviews["sentiment"]=='Negative'].dropna()

## custom function for ngram generation ##
def generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

## custom function for horizontal bar chart ##
def horizontal_bar_chart(df, color):
    trace = go.Bar(
        y=df["word"].values[::-1],
        x=df["wordcount"].values[::-1],
        showlegend=False,
        orientation = 'h',
        marker=dict(
            color=color,
        ),
    )
    return trace

## Get the bar chart from positive reviews ##
freq_dict = defaultdict(int)
for sent in review_pos["ulasan"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(25), 'green')

## Get the bar chart from neutral reviews ##
freq_dict = defaultdict(int)
for sent in review_neu["ulasan"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace1 = horizontal_bar_chart(fd_sorted.head(25), 'grey')

## Get the bar chart from negative reviews ##
freq_dict = defaultdict(int)
for sent in review_neg["ulasan"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace2 = horizontal_bar_chart(fd_sorted.head(25), 'red')

# Creating two subplots
fig = tools.make_subplots(rows=3, cols=1, vertical_spacing=0.04,
                          subplot_titles=["Frequent words of positive reviews", "Frequent words of neutral reviews",
                                          "Frequent words of negative reviews"])
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 2, 1)
fig.append_trace(trace2, 3, 1)
fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots Hotel Lifestyle")
pyo.plot(fig, filename='word-plots')

