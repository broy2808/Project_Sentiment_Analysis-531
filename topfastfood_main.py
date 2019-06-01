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
import time
import nltk
from nltk.corpus import stopwords
import sklearn as sk
import random
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
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
from flask import Flask
from flask import render_template,request,url_for
from flask_cache import Cache

      


list1=["Starbucks","McDonald's","Taco Bell","Chipotle Mexican Grill","Nacho Daddy","The Cheesecake Factory","Burger King","Wendy's","Subway","Papa John's Pizza"];
# =============================================================================
# Mongodb connection function
# =============================================================================
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
     # replace special chars 
     letters_only = re.sub('[^a-zA-Z]', ' ',text)
     letters_only=re.sub('\[[^]]*\]', '', letters_only)
     # remove url
     letters_only = re.sub(r'http\S+', ' ', letters_only)     
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
     stopwords_eng.update(['papa','one','called','would','said','location','ordered','johns','get','order','go','drive','thru','place','ever','even','bell','back','got','us','factory','king'])
     useful_words = [word for word in words if not word in stopwords_eng]
     return(useful_words)




# =============================================================================
# List of All store location wise
# =============================================================================
def top_store(business):
    f,ax = plt.subplots(1,2, figsize=(14,8))
    ax1,ax2, = ax.flatten()
    cnt = business['name'].value_counts()[:20].to_frame()

    #sns.barplot(cnt['name'], cnt.index, palette = 'RdBu', ax =ax1)
    sns.barplot(cnt['name'], cnt.index, ax =ax1)
    ax1.set_xlabel('')
    ax1.set_title('Top name of store in Yelp')
    
    cnt1 = business['city'].value_counts()[:20].to_frame()
    sns.barplot(cnt1['city'], cnt1.index, palette = 'gist_rainbow', ax =ax2).figure.savefig("static/images/graph1.jpg")
    ax2.set_xlabel('')
    ax2.set_title('Top city business listed in Yelp')
    #plt.show()
    gc.collect()

# =============================================================================
# User Rating graph// compare avg text_length and User Rating
# =============================================================================     
def compare_res(review):
     g = sns.FacetGrid(review, col="name",col_wrap=4,palette = 'gist_rainbow',margin_titles=True)
     g = g.map(plt.hist, "User_Rating", bins=10,color='blue').savefig("static/images/graph2.jpg")
     gc.collect()
     #plt.show()
     g = sns.FacetGrid(data=review, col='User_Rating')
     g.map(plt.hist, 'text_length', bins=50).savefig("static/images/graph3.jpg")
     #plt.show()
     gc.collect()

# =============================================================================
# User Rating graph// compare avg text_length and User Rating
# =============================================================================     
def compare_loc(review,each):
     Neg_reviews = review[(review['User_Rating']==1)]
     Pos_reviews = review[(review['User_Rating']==5)]


     N1=Neg_reviews.groupby(['city']).count()
     N1=N1.sort_values(by=['review_count'],ascending=False)
     N1=N1['review_count'][:10]
     
     
     P1=Pos_reviews.groupby(['city']).count()
     P1=P1.sort_values(by=['review_count'],ascending=False)
     P1=P1['review_count'][:10]
     

     f,ax = plt.subplots(1,2, figsize=(14,8))
     ax1,ax2, = ax.flatten()
     P1.plot.bar(ax=ax1, color=('cyan','red','yellow','green','blue'))
     ax1.set_title('Positive Reviews',fontsize=20,color='Red')
     ax1.set(xlabel='City Name', ylabel='Number of Positive Reviews')

     N1.plot.bar(ax=ax2, color=('cyan','red','yellow','green','blue'))
     ax2.set_title('Negative Reviews',fontsize=20,color='Red')
     ax2.set(xlabel='City Name', ylabel='Number of Negative Reviews')
     plt.savefig("static/images/graph7.jpg")
     #plt.show()
     gc.collect()
     
    
    
# =============================================================================
# Final Function for guessing text emotion. SVM
# =============================================================================
def guess_emo(Text_input):
     Text_input=[Text_input]
     app1="app"
     Restaurant_reviews=pd.DataFrame(mongodb_read(app1))
     if not sys.warnoptions:
            warnings.simplefilter("ignore")
     Restaurant_reviews=Restaurant_reviews.assign(text_length=Restaurant_reviews['text'].apply(len))
     Restaurant_reviews['User_Rating'] =Restaurant_reviews['User_Rating'].map({2: 1,1:1,3:3,5:5,4:5})
     #Restaurant_reviews = Restaurant_reviews[(Restaurant_reviews['User_Rating']==5)|(review_res['User_Rating']==1)]
     Restaurant_reviews=Restaurant_reviews.loc[Restaurant_reviews['User_Rating'].isin(['1', '5']),['User_Rating','text','name']]
     data_classes = shuffle(Restaurant_reviews)
     data_classes=data_classes.iloc[ 0:13000, : ]
     
     # Seperate the dataset into X and Y for prediction
     x = data_classes['text']
     y = data_classes['User_Rating']
     
     vectorizer=tf_idf_prog()
     vocab = vectorizer.fit(x)
     x = vocab.transform(x)
    
     x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=101)
    
     start = time.time()
     svm1 = SVC(random_state=101,probability=True, kernel="linear", class_weight="balanced",C=1)
     svm1.fit(x_train,y_train)
     predsvm = svm1.predict(x_test)
     print("Confusion Matrix for Support Vector Machines for all Fastfood Restaurants:")
     print(confusion_matrix(y_test,predsvm))
     print("Score:",round(accuracy_score(y_test,predsvm)*100,2))
     print("Classification Report for Support Vector Machines for all Fastfood Restaurants:",classification_report(y_test,predsvm))
     end = time.time()
     print("Total time: ",end - start)
     #review_transformed = vocab.transform(Text_input)
     
     review_transformed=vocab.transform(Text_input)
     rating=svm1.predict(review_transformed)[0]
     if rating==5:
         emotion="Positive"
     elif rating==1:
         emotion="Negative"
     
     print("The input sentence rating should be: ",emotion)
     return emotion
     #Button(master, text=emotion).grid(row=5, column=1, sticky=W, pady=4)
     

################################################
#tf-idf
################################################
def tf_idf_prog():
     # calculate TF-IDF for each word in each document
    # min_df – remove words which have occurred in less than 4 documents
    # max_df – remove words which have occurred in more than 80% of all documents - possibly stopwords or non-significant words like 'movie'
    # sublinear_tf – term frequency (TF)
    # use_idf – inverse document frequency (IDF)
    # stop_words – remove predefined stopwords from the NLTK corpus library
    vectorizer = TfidfVectorizer(min_df = 4, 
                                 max_df = 0.8, 
                                 sublinear_tf = True, 
                                 use_idf = True,
                                 analyzer = remove_punc)
    
    
    return  vectorizer
     
################################################
# Extra preprocessing functionality not used
##############################################
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
      
################################################
# Word cloud for positive negative words.
##############################################
def word_cloud_s1(vocab, review_res,each):
     #positive, negative words
     #nltk.download('vader_lexicon')
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
     plt.savefig("static/images/graph4.jpg")
     #plt.show()
     gc.collect()



     
##############################
#Feature List
##############################
def featurelist(vectorizer,stw_review_list,modelSVM):
    Cvectorizer = CountVectorizer(min_df = 1,
                             max_df = 0.8,
                             tokenizer = nltk.word_tokenize,
                             analyzer = remove_punc)
    #modelSVM = SVC(random_state=101,probability=True, kernel="linear", class_weight="balanced",C=1)
    print("***********************************************************************")

    
    vocab=Cvectorizer.fit(stw_review_list)
    stw_counts = vocab.transform(stw_review_list)
    # number of documents and features
    print("Number of reviews: ", stw_counts.shape[0])

    # apply tfidf from count matrix
    stw_tfidf_transformer = TfidfTransformer()
    stw_counts_tfidf = stw_tfidf_transformer.fit_transform(stw_counts)

    # predict
    stw_reviews_tfidf = vectorizer.transform(stw_review_list)
    stw_predSVM = modelSVM.predict(stw_reviews_tfidf)

    print("Sentiment analysis: \n", stw_predSVM)

    #print(Cvectorizer.get_feature_names())
    stw_terms = Cvectorizer.get_feature_names()

    # finding means of tfidf frequency of each term through documents
    stw_sums = stw_counts_tfidf.sum(axis=0)
    stw_means = stw_sums/stw_counts.shape[0]

    stw_mean_tfidf = []
    for col, term in enumerate(stw_terms):
        stw_mean_tfidf.append((term, stw_means[0, col]))

    #print(stw_mean_tfidf)

    stw_ranking = pd.DataFrame(stw_mean_tfidf, columns=['term', 'mean'])
    #print(stw_ranking)
    stw_ranking = stw_ranking.sort_values(by=['mean'], ascending = False)
    print("Top 10 features: \n", stw_ranking.head(10))

    # export chart figure
    stw_ranking.head(10).plot.bar(x='term', rot=0, figsize=(10, 5)).figure.savefig('static/images/stw_top_fea_fig.jpg')
    gc.collect()
    print("***********************************************************************")
###################################################
#HTML Page generate
###################################################    
PEOPLE_FOLDER = os.path.join('static','images')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

@app.after_request
def add_header(response):
    
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

@app.route('/')
def my_form():
    return render_template('Hello_m.html')



@app.route('/', methods=['POST'])
def my_form_post():
        text=request.form['text']
        processed_text = text.lower()
        val=guess_emo(processed_text)
        print(os.getcwd())
        if val=="Positive":
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'pos.jpg')
            print(full_filename)
        elif val=="Negative":
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'neg.jpg')
        
        return render_template('Hello_m.html',value1=val,val1=text,user_image = full_filename,w1="200",h1="200")
        


@app.route('/form_details', methods=['POST'])
def form_details():
        value=request.form['item1']
        res_revi(value)
        g1= os.path.join(app.config['UPLOAD_FOLDER'], 'graph1.jpg')
        g2= os.path.join(app.config['UPLOAD_FOLDER'], 'graph2.jpg')
        g3= os.path.join(app.config['UPLOAD_FOLDER'], 'graph3.jpg')
        g4= os.path.join(app.config['UPLOAD_FOLDER'], 'graph4.jpg')
        g6= os.path.join(app.config['UPLOAD_FOLDER'], 'graph6.jpg')
        g7= os.path.join(app.config['UPLOAD_FOLDER'], 'stw_top_fea_fig.jpg')
        g8= os.path.join(app.config['UPLOAD_FOLDER'], 'graph7.jpg')
        
        return render_template('Hello_m.html',g1=g1,g2=g2,g3=g3,g6=g6,g7=g7,g8=g8)


       
        
###################################################
#Review validation of each restaurants using SVM
###################################################
def res_revi(value):
     app1="app"
     Restaurant_reviews=pd.DataFrame(mongodb_read(app1))
     top_store(Restaurant_reviews)
     if not sys.warnoptions:
            warnings.simplefilter("ignore")
     ##############################################################################################
     #["Starbucks","McDonald's","Taco Bell","Chipotle Mexican Grill","Nacho Daddy","The Cheesecake Factory","Burger King","Wendy's","Subway","Papa John's Pizza"]
     #for each in list1:
     each=value
     print(each)
     #CLASSIFICATION
     review_res=Restaurant_reviews[(Restaurant_reviews['name']==each)]
     review_res=review_res.assign(text_length=review_res['text'].apply(len))
     compare_res(review_res)
     compare_loc(review_res,each)
     review_res['User_Rating'] =review_res['User_Rating'].map({2: 1,1:1,3:3,5:5,4:5})
     data_classes = review_res[(review_res['User_Rating']==1) |(review_res['User_Rating']==5)]
     data_classes = shuffle(data_classes)
     #Used in Feature selection/more common words graph
     text_feature=data_classes['text']
     
     # Seperate the dataset into X and Y for prediction
     x = data_classes['text']
     y = data_classes['User_Rating']
     # getting TF-IDF of data
     vectorizer=tf_idf_prog()
     vocab = vectorizer.fit(x)
     x = vocab.transform(x)

     # Call word cloud creation
     word_cloud_s1(vocab, review_res,each)
     # SPLITTING THE DATASET INTO TRAINING SET AND TESTING SET
     x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=101)

     #SVM
     # Support Vector Machine
     svm1 = SVC(random_state=101,probability=True, kernel="linear", class_weight="balanced",C=1)
     svm1.fit(x_train,y_train)
     predsvm = svm1.predict(x_test)
     print("Confusion Matrix for Support Vector Machines:")
     print(confusion_matrix(y_test,predsvm))
     print("Score:",round(accuracy_score(y_test,predsvm)*100,2))
     print("Classification Report:",classification_report(y_test,predsvm,target_names=['Negative','Positive']))
     visualizer = ClassificationReport(SVC(random_state=101,probability=True, kernel="linear", class_weight="balanced",C=1),support=True)
     visualizer.fit(x_train, y_train)  # Fit the visualizer and the model
     visualizer.score(x_test, y_test) # Evaluate the model on the test data
     #plt.ylabel('Negative                      Neutral                      Positive')
     #plt.xlabel('Precision                Recall                      f1                     support')
     #plt.savefig("static/images/graph5.jpg")
     #visualizer.poof()          # Draw/show/poof the data
     fig, ax = plt.subplots(figsize=(7,7))
     labels=[1,5]
     cm=confusion_matrix(y_test,predsvm,labels)
     sns.heatmap(cm, annot=True,fmt='d',xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'],cmap='Blues')
     plt.ylabel('Actual Emotions')
     plt.xlabel('Predicted Emotions')
     plt.savefig("static/images/graph6.jpg")
     gc.collect()
     #plt.show()
     
     featurelist(vectorizer,text_feature,svm1);
     
# =============================================================================
# All algorithms on dataset
# =============================================================================
def comp_algo():
     
     app1="app"
     Restaurant_reviews=pd.DataFrame(mongodb_read(app1))
     if not sys.warnoptions:
            warnings.simplefilter("ignore")
     Restaurant_reviews=Restaurant_reviews.assign(text_length=Restaurant_reviews['text'].apply(len))
     Restaurant_reviews['User_Rating'] =Restaurant_reviews['User_Rating'].map({2: 1,1:1,3:3,5:5,4:5})
     Restaurant_reviews=Restaurant_reviews.loc[Restaurant_reviews['User_Rating'].isin(['1', '5']),['User_Rating','text','name']]
     #Uncomment below line for including neutral emotion in analysis
     #Restaurant_reviews=Restaurant_reviews.loc[Restaurant_reviews['User_Rating'].isin(['1','3', '5']),['User_Rating','text','name']]
     data_classes = shuffle(Restaurant_reviews)
     #data_classes=data_classes.iloc[ 0:13000, : ]
     
     # Seperate the dataset into X and Y for prediction
     x = data_classes['text']
     y = data_classes['User_Rating']
     
     vectorizer=tf_idf_prog()
     vocab = vectorizer.fit(x)
     x = vocab.transform(x)
    
     x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=101)


     
     #Naive Bayes
     start = time.time()
     mnb = MultinomialNB(alpha=0.1)
     mnb.fit(x_train,y_train)
     predmnb = mnb.predict(x_test)
     print("Confusion Matrix for Multinomial Naive Bayes:")
     print(confusion_matrix(y_test,predmnb))
     print("Accuracy Score:",round(accuracy_score(y_test,predmnb)*100,2))
     print("Classification Report:")
     #print(classification_report(y_test,predmnb, target_names=['Negative','Neutral','Positive']))
     print(classification_report(y_test,predmnb))
     end = time.time()
     print("Total time: ",end - start)
     
     
     #SVM
     # Support Vector Machine
     start = time.time()
     svm1 = SVC(random_state=101,probability=True, kernel="linear", class_weight="balanced",C=1)
     svm1.fit(x_train,y_train)
     predsvm = svm1.predict(x_test)
     print("Confusion Matrix for Support Vector Machines:")
     print(confusion_matrix(y_test,predsvm))
     print("Accuracy Score:",round(accuracy_score(y_test,predsvm)*100,2))
     print("Classification Report:")
     #print(classification_report(y_test,predsvm,target_names=['Negative','Neutral','Positive']))
     print(classification_report(y_test,predsvm))
     end = time.time()
     print("Total time: ",end - start)
     
     
     #MULTILAYER PERCEPTRON CLASSIFIER(ANN Classifier)
     start = time.time()
     mlp = MLPClassifier(hidden_layer_sizes=(100, 100))
     mlp.fit(x_train,y_train)
     predmlp = mlp.predict(x_test)
     print("Confusion Matrix for Multilayer Perceptron Classifier:")
     print(confusion_matrix(y_test,predmlp))
     print("Accuracy Score:",round(accuracy_score(y_test,predmlp)*100,2))
     print("Classification Report:")
     #print(classification_report(y_test,predmlp,target_names=['Negative','Neutral','Positive']))
     print(classification_report(y_test,predmlp))
     end = time.time()
     print("Total time: ",end - start)


     
##############################################
# Main processing
##############################################

if __name__=="__main__":
     #comp_algo()
     app.run()
     
    
     
     
