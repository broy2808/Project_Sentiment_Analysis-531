from flask import Flask
from flask import render_template
from flask import request
import os
import json
import pandas as pd
from pymongo import MongoClient
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import pickle
import numpy as np
import datetime
import re
import string
import nltk
from nltk.corpus import stopwords
import sklearn as sk
import random
from naive_bayes import NB
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score,cross_val_predict,KFold
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from yellowbrick.classifier import ClassificationReport
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve,r2_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import sys
import warnings
from wordcloud import WordCloud
from nltk import pos_tag
from collections import Counter, OrderedDict
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 

      
app = Flask(__name__)
list1=["Starbucks","McDonald's","Taco Bell","Chipotle Mexican Grill","Nacho Daddy","The Cheesecake Factory","Burger King","Wendy's","Subway","Papa John's Pizza"];
def mongodb_read(col): 
     # Connection to the MongoDB Server
     client=MongoClient('mongodb://localhost:27017/')
     # Connection to the database
     db=client.myapp
     #Collection
     if col=="app2":
          collection = db.app2
          details=collection.find({"name":{"$in":list1}})
     elif(col=="app"):
          collection = db.app
          details=collection.find({"date": {"$gte": datetime.datetime(2016, 1, 1)}})
          
     return details

# =============================================================================
# function - text cleaning
# =============================================================================
def remove_punc(text):
     letters_only = re.sub('[^a-zA-Z]', ' ',text)
     letters_only=re.sub('\[[^]]*\]', '', letters_only)
     # remove url
     letters_only = re.sub(r'http\S+', ' ', letters_only);
     # replace special chars with whitespace except '
     #letters_only = re.sub('[^\w^\']', ' ', letters_only)
     #letters_only = re.sub('_', ' ', letters_only)
     # remove number
     #letters_only = re.sub('\d', ' ', letters_only)
     # change any whitespace to one space
     #letters_only = re.sub('\s+', ' ', letters_only)
     # remove start and end whitespace
     letters_only = letters_only.strip()
    
     words = letters_only.lower().split()
     words=remove_stopwords(words)
     return words

# =============================================================================
# stopwords removal
# =============================================================================
def remove_stopwords(words):
     stopwords_eng = set(stopwords.words("english"))
     useful_words = [word for word in words if not word in stopwords_eng]
     return(useful_words)





def top_store(business):
    f,ax = plt.subplots(1,2, figsize=(14,8))
    ax1,ax2, = ax.flatten()
    cnt = business['name'].value_counts()[:20].to_frame()

    #sns.barplot(cnt['name'], cnt.index, palette = 'RdBu', ax =ax1)
    sns.barplot(cnt['name'], cnt.index, ax =ax1)
    ax1.set_xlabel('')
    ax1.set_title('Top name of store in Yelp')
    
    cnt1 = business['city'].value_counts()[:20].to_frame()
    sns.barplot(cnt1['city'], cnt1.index, palette = 'gist_rainbow', ax =ax2)
    ax2.set_xlabel('')
    ax2.set_title('Top city business listed in Yelp')
    plt.show()
    gc.collect()
     
def compare_res(review):
     g = sns.FacetGrid(review, col="name",col_wrap=4,palette = 'gist_rainbow')
     g = g.map(plt.hist, "User_Rating", bins=10,color='blue')
     plt.show()
     gc.collect()
def printdata(url):
   # url="E:/Myproject_am/static/json/yelp_academic_dataset_business.json"
    #df = pd.read_json(url, lines=True)
    df=url
    df2=pd.DataFrame(df,columns=['business_id','name','attributes','city','state'])
    print(df2.head(10))

def graph(reviews):
    g = sns.FacetGrid(data=reviews, col='User_Rating')
    g.map(plt.hist, 'text_length', bins=50)
    plt.show()
    sns.boxplot(x='User_Rating', y='text_length', data=reviews)
    plt.show()

    
# =============================================================================
# Final Function for guessing text emotion. SVM
# =============================================================================
def guess_emo(Text_input):
     app1="app"
     Restaurant_reviews=pd.DataFrame(mongodb_read(app1))
     if not sys.warnoptions:
            warnings.simplefilter("ignore")
     Restaurant_reviews=Restaurant_reviews.assign(text_length=Restaurant_reviews['text'].apply(len))
     #print(Restaurant_reviews['User_Rating'].value_counts())
     Restaurant_reviews['User_Rating'] =Restaurant_reviews['User_Rating'].map({2: 1,1:1,3:3,5:5,4:5})
     #Restaurant_reviews = Restaurant_reviews[(Restaurant_reviews['User_Rating']==5)|(review_res['User_Rating']==1)]
     Restaurant_reviews=Restaurant_reviews.loc[Restaurant_reviews['User_Rating'].isin(['1', '5']),['User_Rating','text','name']]
     print(Restaurant_reviews.head(10))
     data_classes = shuffle(Restaurant_reviews)
     data_classes=data_classes.iloc[ 0:15000, : ]
     
     # Seperate the dataset into X and Y for prediction
     x = data_classes['text']
     y = data_classes['User_Rating']
     #vocab = CountVectorizer(analyzer = remove_punc).fit(x)
     #x=vocab.transform(x)
     
     vectorizer=tf_idf_prog()
     vocab = vectorizer.fit(x)
     x = vocab.transform(x)
    
     x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)
    
     
     svm1 = SVC(random_state=101,probability=True, kernel="linear", class_weight="balanced",C=1)
     svm1.fit(x_train,y_train)
     predsvm = svm1.predict(x_test)
     print("Confusion Matrix for Support Vector Machines for all Fastfood Restaurants:")
     print(confusion_matrix(y_test,predsvm))
     print("Score:",round(accuracy_score(y_test,predsvm)*100,2))
     print("Classification Report for Support Vector Machines for all Fastfood Restaurants:",classification_report(y_test,predsvm))

     #review_transformed = vocab.transform(Text_input)
     
     review_transformed=vocab.transform(Text_input)
     rating=svm1.predict(review_transformed)[0]
     if rating==5:
         emotion="Positive"
     elif rating==1:
         emotion="Negative"
     print("The input sentence rating should be: ",emotion)

################################################
#tf-idf
################################################
def tf_idf_prog():
     # calculate TF-IDF for each word in each document
    # min_df – remove words which have occurred in less than 5 documents
    # max_df – remove words which have occurred in more than 80% of all documents - possibly stopwords or non-significant words like 'movie'
    # sublinear_tf – term frequency (TF)
    # use_idf – inverse document frequency (IDF)
    # stop_words – remove predefined stopwords from the NLTK corpus library
    vectorizer = TfidfVectorizer(min_df = 1, 
                                 max_df = 0.9, 
                                 sublinear_tf = True, 
                                 use_idf = True,
                                 analyzer = remove_punc)
    
    
    return  vectorizer
     
     
def tag_word(text):
    pos=[]
    nltk.download('averaged_perceptron_tagger')
    p=nltk.pos_tag(text,tagset=None, lang='eng')
    print(p)
    pos.append(p)
    return pos
        
def tagged_adjectives(tagged_sentences):
    adjectives = []
    for sentence in tagged_sentences:
        for (word, tag) in sentence:
            if tag.find('JJ') != -1:
                adjectives.append(word)
    return adjectives
    

def convert_tojson(file):
      print(file.head(4))
      pd.DataFrame(file).to_json("E:/Myproject_am/static/json/fastfood_chain_data/test5.JSON",orient='records')
      #x=file.to_json(orient='records')
    

@app.route('/',methods=['GET', 'POST'])
def hello():
      
    if request.method == 'POST':
        item1=request.form["item1"]
        print(item1," ")
        return render_template('Hello.html')
 
    else:
        print("Hello")
        return render_template('Hello.html')

if __name__=="__main__":
     
     app1="app"
     Restaurant_reviews=pd.DataFrame(mongodb_read(app1))
     top_store(Restaurant_reviews)
     if not sys.warnoptions:
            warnings.simplefilter("ignore")

     #new

    # for each in list1:
     each='Starbucks'
     # CLASSIFICATION
     review_res=Restaurant_reviews[(Restaurant_reviews['name']==each)]
     review_res=review_res.assign(text_length=review_res['text'].apply(len))
     #compare_res(review_res)
     #graph(review_res)
     data_classes = review_res[(review_res['User_Rating']==1) | (review_res['User_Rating']==3) | (review_res['User_Rating']==5)]
     
     
     
     print(data_classes.shape)
     
     
     # Seperate the dataset into X and Y for prediction
     x = data_classes['text']
     y = data_classes['User_Rating']
     
     # getting TF-IDF of data
     vectorizer=tf_idf_prog()
     vocab = vectorizer.fit(x)
     x = vocab.transform(x)
     
     

     '''
     #Positive words
     pos_classes = review_res[(review_res['User_Rating']==5)]
     pos_x = pos_classes['text']
     vocab1 = CountVectorizer(analyzer = remove_punc).fit(pos_x)
     pos=tag_word(vocab1.get_feature_names())
     tag=tagged_adjectives(pos)
     
     # Find word count
     counter = Counter()
     for word in tag:
        counter[word] += 1
     print(counter)
     '''
     

################################################
# Word cloud for positive negative words.
##############################################
     #positive, negative words
     nltk.download('vader_lexicon')
     sid = SentimentIntensityAnalyzer()
     pos_word_list=[]
     neu_word_list=[]
     neg_word_list=[]
     #vocab1=CountVectorizer(analyzer = remove_punc).fit(review_res['text'])
     #vectorizer=tf_idf_prog()
     #vocab1 = vectorizer.fit(review_res['text'])
     test_subset=vocab.get_feature_names()
     
     for word in test_subset:
        if (sid.polarity_scores(word)['compound']) >= 0.5:
            pos_word_list.append(word)
        elif (sid.polarity_scores(word)['compound']) <= -0.5:
            neg_word_list.append(word)
        else:
            neu_word_list.append(word)                

     print('Positive :',pos_word_list)        
     print('Neutral :',neu_word_list)    
     print('Negative :',neg_word_list)

     # Wordcloud of top 100 adjectives for positive reviews
     #plt.figure(figsize=(12,12))
     #plt.imshow(wordcloud1)
     wordcloud1 = WordCloud(max_font_size=50, max_words=50).generate(" ".join(pos_word_list))
     wordcloud2 = WordCloud(max_font_size=50, max_words=50).generate(" ".join(neg_word_list))
     wordcloud3 = WordCloud(max_font_size=50, max_words=50).generate(" ".join(neu_word_list))


     f, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2)
     
     review_res['User_Rating'].value_counts().sort_index().plot.bar(ax=ax1, color=('cyan','red','yellow','green','blue'))
     ax1.set_title(each,fontsize=20,color='Red')
     ax1.set(xlabel='Rating', ylabel='Number of Users')

     #plt.subplot(2, 2, 1)
     ax2.imshow(wordcloud1)
     ax2.set_title('Top Positive words',fontsize=20,color='Red')
     ax2.axis("off")
     

     #plt.subplot(2, 2, 2)
     ax3.imshow(wordcloud2)
     ax3.axis("off")
     ax3.set_title('Top Negative words',fontsize=20,color='Red')

     #plt.subplot(2, 2, 3)
     ax4.imshow(wordcloud3)
     ax4.axis("off")
     ax4.set_title('Top Neutral words',fontsize=20,color='Red')

     
     plt.subplots_adjust(wspace = 0.4,hspace = 0.4,top=0.9,bottom=0.1)
     plt.show()
     gc.collect()

 
     # SPLITTING THE DATASET INTO TRAINING SET AND TESTING SET
     x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)

     
     #Naive Bayes
     mnb = MultinomialNB(alpha=0.1)
     mnb.fit(x_train,y_train)
     predmnb = mnb.predict(x_test)
     print("Confusion Matrix for Multinomial Naive Bayes:")
     print(confusion_matrix(y_test,predmnb))
     print("Score:",round(accuracy_score(y_test,predmnb)*100,2))
     print("Classification Report:",classification_report(y_test,predmnb, target_names=['Negative','Neutral','Positive']))
     visualizer = ClassificationReport(MultinomialNB(alpha=0.1),support=True)
     visualizer.fit(x_train, y_train)  # Fit the visualizer and the model
     visualizer.score(x_test, y_test)  # Evaluate the model on the test data
     visualizer.poof()             # Draw/show/poof the data
     fig, ax = plt.subplots(figsize=(7,7))
     labels=[1,3,5]
     cm=confusion_matrix(y_test,predmnb,labels)
     sns.heatmap(cm, annot=True,fmt='d',xticklabels=['Negative','Neutral','Positive'], yticklabels=['Negative','Neutral','Positive'],cmap='Blues')
     plt.ylabel('Actual Emotions')
     plt.xlabel('Predicted Emotions')
     plt.show()
     
     
     #SVM
     # Support Vector Machine
     svm1 = SVC(random_state=101,probability=True, kernel="linear", class_weight="balanced",C=1)
     #scores = cross_val_predict(svm1, x, y, cv=10)
     #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
     svm1.fit(x_train,y_train)
     predsvm = svm1.predict(x_test)
     print("Confusion Matrix for Support Vector Machines:")
     print(confusion_matrix(y_test,predsvm))
     print("Score:",round(accuracy_score(y_test,predsvm)*100,2))
     print("Classification Report:",classification_report(y_test,predsvm,target_names=['Negative','Neutral','Positive']))
     visualizer = ClassificationReport(SVC(random_state=101,probability=True, kernel="linear", class_weight="balanced",C=1),support=True)
     visualizer.fit(x_train, y_train)  # Fit the visualizer and the model
     visualizer.score(x_test, y_test)  # Evaluate the model on the test data
     visualizer.poof()             # Draw/show/poof the data
     fig, ax = plt.subplots(figsize=(7,7))
     labels=[1,3,5]
     cm=confusion_matrix(y_test,predsvm,labels)
     sns.heatmap(cm, annot=True,fmt='d',xticklabels=['Negative','Neutral','Positive'], yticklabels=['Negative','Neutral','Positive'],cmap='Blues')
     plt.ylabel('Actual Emotions')
     plt.xlabel('Predicted Emotions')
     plt.show()
     
      # Support Vector Machine
   
    

     
     #MULTILAYER PERCEPTRON CLASSIFIER(ANN Classifier)
     
     mlp = MLPClassifier()
     mlp.fit(x_train,y_train)
     predmlp = mlp.predict(x_test)
     print("Confusion Matrix for Multilayer Perceptron Classifier:")
     print(confusion_matrix(y_test,predmlp))
     print("Score:",round(accuracy_score(y_test,predmlp)*100,2))
     print("Classification Report:")
     print(classification_report(y_test,predmlp,target_names=['Negative','Neutral','Positive']))
     visualizer = ClassificationReport(MLPClassifier(),support=True)
     visualizer.fit(x_train, y_train)  # Fit the visualizer and the model
     visualizer.score(x_test, y_test)  # Evaluate the model on the test data
     visualizer.poof()             # Draw/show/poof the data
    

     review="Other than the really great happy hour prices, its hit or miss with this place. More often a miss. :(\n\nThe food is less than average, the drinks NOT strong ( at least they are inexpensive) , but the service is truly hit or miss.\n\nI'll pass."
     print("Text sample: ",review)
     guess_emo([review])
      

     review="The Bui vien street has not that many surprises to offer but I do really recommend Babas if you are craving for indian food! Had a quick lunch with my vegan son and we were both pleased with what we ordered. Veggie Samosas as a starter were super and for main course we had two different veggie masalas. It doesn't look that much but we couldn' t eat all! Also a BIG plus is FREE drinking water and we also got a sweet dessert for free. Can highly recommend this place. Super friendly and good service!"
     print("\n\nText sample: ",review)
     guess_emo([review])
    
     
     review="I went by the reviews and ordered food from here via Vietnammm and I have to mention that the food was terrible ; lacked flavour, very oily and the dal was stale. Definitely not worth it."
     print("\n\nText sample: ",review)
     guess_emo([review])
     
     
     
     #new

    
    # app.run()
     
