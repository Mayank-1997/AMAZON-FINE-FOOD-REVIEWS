   ###Classification And   Clustering  of  Amazon-Food-Reviews- Using Various Machine Learning Models#####

Performed Exploratory Data Analysis, Data Cleaning, Data Visualization and Text Featurization(BOW, tfidf, Word2Vec,tfidf word2vec). I Have Build several classification and clustering ML models with hyperparamter tuning  with each of 4 vectorizers stated above .They are as follows:

Classification Algorithms:
Logistic Regression
Naive Bayes
KNN
Support vector machine
GBDT
XGBOOST
Decision Tree

Clustering Algorithms:

1.K-Means Clustering
2.Agglomerative Hierarchical Clustering
3.DBSCAN clustering


Objective:
Given a text review, determine the sentiment of the review whether its positive or negative.
Data Source: https://www.kaggle.com/snap/amazon-fine-food-reviews
About Dataset
The Amazon Fine Food Reviews dataset consists of reviews of fine foods from Amazon.


Number of reviews: 568,454
Number of users: 256,059
Number of products: 74,258
Timespan: Oct 1999 - Oct 2012
Number of Attributes/Columns in data: 10


Attribute Information:
Id
ProductId - unique identifier for the product
UserId - unqiue identifier for the user
ProfileName
HelpfulnessNumerator - number of users who found the review helpful
HelpfulnessDenominator - number of users who indicated whether they found the review helpful or not
Score - rating between 1 and 5
Time - timestamp for the review
Summary - brief summary of the review
Text - text of the review








                                            NATURAL LANGUAGE PROCESSING
1.Reviews here contain textual data so semantic analysis of reviews is to be done. This is achieved in this project by  various NLP Techniques such as:
1.BAG OF WORDS
2. TERM FREQUENCY -- INVERSE DOCUMENT FREQUENCY (TF_IDF)
3. AVG_ WORD2VEC VECTORIZATION
4.TFIDF WORD2VEC  VECTORIZATION





 
Amazon Food Reviews EDA, NLP, Text Preprocessing and Visualization using TSNE
Defined Problem Statement

Performed Exploratory Data Analysis(EDA) on Amazon Fine Food Reviews Dataset plotted Word Clouds, Histograms, etc.


3.   Performed Data Cleaning & Data Preprocessing by removing unneccesary and duplicates rows and       for text reviews removed html tags, punctuations, Stopwords and Stemmed the words using Porter    Stemmer. Tokenizing and lemmatization of words of words are also done.
4.Documented the concepts clearly
Plotted TSNE plots for Different Featurization of Data viz. BOW(uni-gram), tfidf, Avg-Word2Vec and tf-idf-Word2Vec by taking sample of words and by checking with various values of n_iterations and perplexity value with uni_gram and bi_gram values.





2 KNN CLASSIFICATION ALGORITHM
Considering top 50k points of reviews , data is splitted into train and test data(Since knn is very slow to run  considering only top 50k words) .
Data is splitted into Train data  30k and test data is 20k  with timestamp based splitting
Used both brute & kd-tree implementation of KNN with BOW vectorization.
Hyperparameter tuning is done (for k) for getting the best hyperparameters to prevent model from overfitting and underfitting  
Evaluated the test data on various performance metrics such as f1_score .
Plotted the confusion matrix for test data with the help of heatmap of test data.
  This whole modelling is done  again with other vectorizations  such as tfidf ,avg word2vec,tfidf word2vec also.
     
Conclusions:
KNN is a very slow Algorithm takes very long time to train.
Best Accuracy is achieved by Avg Word2Vec Featurization which is of 89.38%.
Both kd-tree and brute algorithms of KNN gives comparatively similar results.
Overall KNN was not that good for this dataset.

3 Naive Bayes Algorithm
Considering top 100k points of reviews , data is splitted into train and test data with timestamp based spilting .
Train data is 80k and test data is 20k 
Applied Naive Bayes using Multinomial NB with    BOW vectorization .
Hyperparameter tuning is done for parameter ‘alpha’ . Best value of alpha is that which minimizes the cv_error on cross validation_data.
Evaluated the test data on various performance metrics like accuracy, f1-score, precision,   
            recall,etc. Also plotted Confusion matrix using seaborne with heatmap
Gridsearchcv is used to find the best value of hyperparameter  ‘  alpha’.

This whole  process is also  done for tfidf vectorization.


Conclusions:
Naive Bayes is much faster algorithm than KNN
The   multinomial naive bayes worked very well .
Best F1 score is acheived by BOW featurization which is 0.9672

  Logistic Regression
       1   Considering top 100k points of reviews , data is splitted into train and test data with timestamp      based spilting .
Train data is 80k and test data is 20k .Applied logistic regression model with bow vectorization.
 Performing pertubation test to check whether our data features are colliner or not and   plotting    the result
      4     Finding the best hyperparameter using gridsearchcv with train data and cross-validation  
      data by plotting the resluts of varoius train data and cross validation data
       5    Using the appropriate value of   hyper parameter , testing  accuracy on test data using f1-  
       score
       5.	Plotting the confusion matrix to get the precisoin ,recall value with help of heatmap for test data
      6.   For model interpretation , printing the top 20 features for both positive and negative class
Showed How Sparsity increases as we increase lambda or decrease C when L1 Regularizer is used for each featurization.
Did pertubation test to check whether the features are multi-collinear or not. It was found that around 98% of the features are non-collinear
Steps  1 to 8 are repeated for other vectorizations also such as tfidf,avg word2vec,tfidf word2vec
Conclusions:
          1. L2 regularization works fast then L1 reguralizer .
          2. Tfidf _Word2vec Featurization performs best with F1_score of 0.987 and Accuracy of 97.39.
          3.  Logistic Regression is faster algorithm.



5.  SUPPORT VECTOR MACHINE
 1. Applying svm with bow vectorization with two kernels -[LINEAR and RBF].For linear svm , sgdclassifier is used. 
2. Finding the best hyperparameter using gridsearchcv with train data and cross-validation data by  plotting the resluts of cross validation data uisng heatmap.
3.Regularizer type (L1 or L2) and alpha were taken as hyperparamters tuning.
4.To improve model performace  and estimate probabilistic output of classider the,calibrated classifier is used with all  methods such as sigmoid, isotonic.
5.It was found that isotonic calibrated classifier  is performing best .Thus it  is used for test data  
3. Plotting of roc curve to check for the AUC value both for train data and test data.
4. Results of gridsearchcv is stored in a dataframe and heatmap is plotted for various values of  hyperparamters  .Those hyperparameters are chosen which are minimizing  cross validation _error. 
4. Using the apropriate value of hyperparameter ,testing accuracy on test data using f1_score
5. Plotting the confusion matrix to get the precisoin ,recall value with help of heatmap for test data
6. Printing the top 30 most important features.
7. Steps 1 to 6  is repeated for other vectorizations also such as tfidf ,avg word2vec,tfidf word2vec.

CONCLUSIONS:.
Using  SGDClasiifier  with linear kernel  takes very less time to train.
SVM with rbf kernel takes a lot of time to run 
Avg word2vec vectorization is having highest f1 score of all vectorization ie .946 

6 Decision Trees
1. Applying decision tree with bow vectorization
2.Finding the best hyperparameter (hyperparameters used here are max_depth and min_sample_split) using gridsearchcv with train data and cross-validation data by plotting the results of train data and cross validation data
3.Getting the results from gridsearchcv into dataframe and then using heatmap ,chosing those values of hyperparameters which are reducing the cv_error most of all possible combinations of hyperparameter  
3.Using the appropriate values of hyperparameters ,testing accuracy on test data using f1-score.
4.Plotting the confusion matrix to get the precisoin ,recall value with help of heatmap .
5.Decision tree is visualized with the help of graphviz library  for all vectorizations
6 plotting the roc curve for test data to check how our model is performing
7..Printing the top 30 most important features of our model. Using wordcloud library
8.steps 1 to 8  is repeated again for other vectorization  such as tfidf,avg word2vec,tfidf word2vec.
Conclusions:

1.Avg word2vec vectorization(max_depth=5) gave the best results with accuracy of 85.8%   and F1- 
    score of 0.93.
 2.Other vectorizations also gave f1_score above 90%.



6 Ensembles(RF&GBDT)
1. Considering top 100k points of reviews , data is splitted into train and test data with timestamp based spilting .
2.Train data is 80k and test data is 20k 
3. Finding the best hyper-parameters i.e max_deth and min_sample split  using gridsearchcv with train data and cross-validation data by plotting  the heatmap
5.plotting the roc_curve for checking how our model is behaving both for train and test data
6. Using the appropriate value of these  hyperparameters ,testing accuracy on test data using f1-score
7. Plotting the confusion matrix to get the precisoin ,recall value with help of heatmap on test data.
8. Printing the top 30 most important features using wordclpud library.
Conclusions:
TFIDF Featurization in Random Forest (BASE-LEARNERS=10) with random search gave the best results with F1-score of 0.857.
TFIDF Featurization in GBDT (BASE-LEARNERS=275, DEPTH=10) gave the best results with F1-score of 0.8708.





APPLYING KMEANS,AGLOMERATIVE,DBSCAN  CLUSTERING

1. Used a sample size of 50k datapoints and applied K-Means Clustering on all the 4 vectorizers(BOW, TFIDF, AVG-W2V, TFIDF-AVG_W2V).
2. Find the best ‘k’ using the elbow-knee method (plot k vs inertia_)
3. Once after I found the k clusters, plot the word cloud per each cluster so that at a single go we can analyze the words in a cluster.
4. Used a sample size of 5k-10k datapoints and apply Agglomerative Clustering on AVG-W2V, TFIDF-AVG_W2V vectorizers.
5. Applied agglomerative algorithm and try a different number of clusters like 2,5 etc.
6.  Plotted word clouds for each cluster and summarize in your own words what that cluster is representing.
7. Used a sample size of 5k-10k datapoints and apply Agglomerative Clustering on AVG-W2V, TFIDF-AVG_W2V vectorizers.
8. Foun the best ‘Eps’ using the elbow-knee method.
9.  plotted word clouds for each cluster and summarize in your own words what that cluster is representing.

 
