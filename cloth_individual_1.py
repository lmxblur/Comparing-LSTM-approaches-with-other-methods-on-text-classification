import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()


from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import classification_report,f1_score
# first predict the score based on greater than 90 or lower than 90

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

data = pd.read_csv('Clothing.csv').iloc[:,1:]

Review_index = data['Review Text'].notnull()

data2 = data[Review_index]

text_length = [len(single.split()) for single in data2['Review Text']]


tag = ['Perfect' if single ==5 else 'With Drawbacks' for single in data2.Rating]
data2['tag'] = tag


Tr_vali, test_data = train_test_split(data2, train_size = 0.7)

train_data, validation_data = train_test_split(Tr_vali, train_size = 0.85)


model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('bayes', BernoulliNB()),
])

parameters = {
    'vectorizer__binary':[True,False],
    'vectorizer__ngram_range':[(1,1),(2,2)],
    'bayes__alpha':[1,0.1,0.5,1.5],
}

GridSearchCV(
    model, 
    param_grid=parameters, 
    refit=True,  
    scoring='accuracy',
    n_jobs=-1,
    cv=5,  
)

grid_search = GridSearchCV(model, parameters, verbose=0)  
grid_search.fit(Tr_vali['Review Text'], Tr_vali['tag'])


print('#'*20)
print('Bayesian cloth')

print(grid_search.best_params_)


tag_pred = grid_search.predict(test_data['Review Text'])

print(classification_report(test_data['tag'],tag_pred))

from sklearn.metrics import confusion_matrix,plot_confusion_matrix

plt.figure(figsize=(8,8));plot_confusion_matrix(grid_search,test_data['Review Text'],test_data.tag);plt.grid(False);plt.show()


ef = classification_report(test_data['tag'],tag_pred,output_dict=True)


print(pd.DataFrame(ef).T.round(2).to_latex())
"""
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('bayes', BernoulliNB()),
])

model.fit(Tr_vali['Review Text'], Tr_vali['tag'])

tag_pred = model.predict(test_data['Review Text'])

print(classification_report(test_data['tag'],tag_pred))



from sklearn.tree import DecisionTreeClassifier

model2 = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('dt', DecisionTreeClassifier()),
])


model2.fit(Tr_vali['Review Text'], Tr_vali['tag'])

tag_pred2 = model2.predict(test_data['Review Text'])

print(classification_report(test_data['tag'],tag_pred2))


from sklearn.neighbors import KNeighborsClassifier


model3 = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('knn', KNeighborsClassifier(n_jobs=-1)),
])


model3.fit(Tr_vali['Review Text'], Tr_vali['tag'])

tag_pred3 = model3.predict(test_data['Review Text'])

print(classification_report(test_data['tag'],tag_pred3))
"""