from sklearn.svm import SVC
import pickle
from nltk.corpus import stopwords
import pandas
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')

path = 'https://raw.githubusercontent.com/matheus-de-araujo/Fake.br-Corpus/master/preprocessed/pre-processed.csv'
data = pandas.read_csv(path)

label_column = data['label'].isnull()
label_column

for i in range(7200):
    if (label_column[i] == True):
        print(i)

data_trial = data
data_trial['preprocessed_news'] = data_trial['preprocessed_news'].str.replace('[^\w\s]', '')
data_trial['preprocessed_news'] = data_trial['preprocessed_news'].str.lower()

y = data_trial.label
data_trial.drop("label", axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(data_trial['preprocessed_news'], y, test_size=0.2, random_state=42)

tf_vectorizer = TfidfVectorizer(stop_words=stopwords.words('portuguese'), analyzer='word', ngram_range=(1, 1), lowercase=True, use_idf=True)

tfidf_train = tf_vectorizer.fit_transform(X_train)

tfidf_test = tf_vectorizer.transform(X_test)

pickle.dump(tf_vectorizer, open('tfidf.pkl', 'wb'))

support_vector_machine = SVC(kernel='linear').fit(tfidf_train, y_train)

pickle.dump(support_vector_machine, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
