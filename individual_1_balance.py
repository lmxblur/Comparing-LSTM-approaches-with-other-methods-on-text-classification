import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()
data = pd.read_csv('winemag-data-130k-v2.csv').iloc[:,1:]
data2 = pd.read_csv('winemag-data_first150k.csv').iloc[:,1:]
country_counts = data2.country.value_counts()


text_length = [len(single.split()) for single in data2.description]
# We first decide to predict the wine scole based on the description, as this is the most sensible method
# as the description is the evaluation of a particular wine.

################################

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import classification_report,f1_score
# first predict the score based on greater than 90 or lower than 90

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


score_index = data2.points.notnull()
score = data2[score_index].iloc[:,[1,3]]
tag = ['High End' if single >=90 else 'Middle Range' for single in score.points]
#tag1 = [3 if single >=90  else 2 if single>=85 else 1 for single in score.points]
tag1 = [3 if single >=90  else 2 if single>=87 else 1 for single in score.points]

score['tag'] = tag
score['tag1'] = tag1
plt.figure()
sns.countplot(x= "tag",data=score)
plt.title('Tag counts')
plt.show()
plt.close()


Tr_vali, test_data = train_test_split(score, train_size = 0.7)

train_data, validation_data = train_test_split(Tr_vali, train_size = 0.85)


group = Tr_vali.groupby('tag')

group= pd.DataFrame(group.apply(lambda x: x.sample(group.size().min()).reset_index(drop=True)))
group = group.reset_index(drop=True)
Tr_vali= group


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
grid_search.fit(Tr_vali['description'], Tr_vali['tag'])  
best_parameters = grid_search.best_estimator_.get_params()  

print(grid_search.best_params_)

tag_pred = grid_search.predict(test_data['description'])

print(classification_report(test_data['tag'],tag_pred))

from sklearn.metrics import confusion_matrix,plot_confusion_matrix

plt.figure(figsize=(8,8));plot_confusion_matrix(grid_search,test_data['description'],test_data.tag);plt.grid(False);plt.show()


ef = classification_report(test_data['tag'],tag_pred,output_dict=True)


print(pd.DataFrame(ef).T.round(2).to_latex())
