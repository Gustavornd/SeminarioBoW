import numpy as np # linear algebra
import pandas as pd # data processing
pd.options.mode.chained_assignment = None

from wordcloud import WordCloud #Word visualization
import matplotlib.pyplot as plt #Plotting properties
import seaborn as sns #Plotting properties
from sklearn.feature_extraction.text import CountVectorizer #Data transformation
from sklearn.model_selection import train_test_split #Data testing
from sklearn.linear_model import LogisticRegression #Prediction Model
from sklearn.metrics import accuracy_score #Comparison between real and predicted
from sklearn.preprocessing import LabelEncoder #Variable encoding and decoding for XGBoost
import re #Regular expressions
import nltk
from nltk.tokenize import word_tokenize



# Lendo os arquivos de dados e tratando formantando as tabelas.
train = pd.read_csv("twitter_training.csv", header=None)
val = pd.read_csv("twitter_validation.csv",header=None)

train.columns=['id','information','type','text']
train.head()

val.columns = ['id','information','type','text']
val.head()

train_data = train
val_data = val

# Tratando os dados colocando todos em minusculo, convertendo numeros
# e removendo caracteres especiais e emojis
train_data["lower"] = train_data.text.str.lower()
train_data["lower"] = [str(data) for data in train_data.lower]
train_data["lower"] = train_data.lower.apply(
    lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', x))

val_data["lower"]=val_data.text.str.lower() 
val_data["lower"]=[str(data) for data in val_data.lower] 
val_data["lower"]=val_data.lower.apply(
    lambda x: re.sub('[^A-Za-z0-9 ]+', ' ', x)) 

# Divisão do Texto
tokens_text = [word_tokenize(str(word).lower()) for word in train_data['lower']]

# Contandor de palavras
tokens_counter = [item for sublist in tokens_text for item in sublist]
print("Numero de tokens: ", len(set(tokens_counter)))
# Numero de tokens:  30436

# Divisão dos dados principais de treino entre dados de treino e dados de teste
reviews_train, reviews_test = train_test_split(train_data, test_size=0.2, random_state=0)

# n-gramas de 4 palavras
bow_counts = CountVectorizer(
    tokenizer=word_tokenize,
    ngram_range=(1,4)
)

# Transformando os dados de treino, teste e validação em n-gramas
X_train_bow = bow_counts.fit_transform(reviews_train.lower)
X_test_bow = bow_counts.transform(reviews_test.lower)
X_val_bow = bow_counts.transform(val_data.lower)

# Rótulos para codificação de treinamento
y_train_bow = reviews_train['type']
y_test_bow = reviews_test['type']

#Total de registros por sentimento (Type)
distribution = y_test_bow.value_counts() / y_test_bow.shape[0]

# Plotar o gráfico de pizza
plt.figure(figsize=(8, 8))
plt.pie(distribution, labels=distribution.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribuição dos sentimentos nos dados de treino')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# Definindo o modelo de previsão como Regressão Logistica. 
model = LogisticRegression(C=0.9, solver="liblinear",max_iter=1500)

# Treinamento da Regressão logistica 
model.fit(X_train_bow, y_train_bow)
# Teste do treinamento
test_pred = model.predict(X_test_bow)

# Aplicação do modelo de Regressão Logística aos dados de validação
y_val_bow = val_data['type']
Val_pred = model.predict(X_val_bow)
print("Precisão: ", accuracy_score(y_val_bow, Val_pred) * 100)

# Plot do gráfico de erros e acertos da precisão. 
match = pd.DataFrame()
match = y_val_bow == Val_pred
results = match.value_counts()
acertos = results[True] if True in results else 0
erros = results[False] if False in results else 0

labels = ['Acertos', 'Erros']
sizes = [acertos, erros]
colors = ['#66b3ff', '#ff6666']
explode = (0.1, 0)

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title('Comparação de Acertos e Erros nas Previsões')
plt.show()