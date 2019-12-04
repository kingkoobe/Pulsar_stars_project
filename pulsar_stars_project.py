#!/usr/bin/env python
# coding: utf-8

# # Import packages

# In[1]:


# import packages in dataset analysis and processing part,if you do not install you can install in advance.
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


# import packages in machine learning part,if you do not install you can install in advance.
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# for Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# for Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
import os

# for Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

# for KNN
from sklearn.neighbors import KNeighborsClassifier

# for SVM
from sklearn.svm import SVC


# # Let's load the data!

# In[3]:


df = pd.read_csv('pulsar_stars.csv') 


# In[4]:


# check the first 10th datasets from the cvs file
df.head(10)


# # Let's see how many data in total

# In[5]:


print("How many data:",len(df.index))


# # We need a pair plot here!

# In[6]:


# pairplot
sns.pairplot(data=df,
             palette="husl",
             hue="target_class",
             vars=[df.columns[0],
                   df.columns[1],
                   df.columns[2],
                   df.columns[3],
                   df.columns[4],
                   df.columns[5],
                   df.columns[6],
                   df.columns[7],])

plt.suptitle("PairPlot of the DataFrame",fontsize=20)


# # The most common one in each feature!

# In[7]:


# most appearance in each column
for i in range(0,len(df.columns)-1):
    print("The # column:",i+1,"\n", df[df.columns[i]].value_counts().head(1))


# # The great Data Visualization!

# In[8]:


# heatmap, correlation plot
correlation = df.corr()
plt.figure(figsize=(12,12))
sns.heatmap(correlation,cmap=sns.color_palette("plasma"),
            annot=True,linewidth=5)
plt.title("Correlation Plot")
plt.show()
# from the heatmap, we can find "the excess kurtosis of the integrated profile" and "the skewness of the integrated profile" 
# have more relastions with the target class.


# # The most important joint plot of the most important features!

# In[9]:


# joint plot of mean and std
# since excess Kurtosis of the integrated profile and skewness of the isntegrated profile has
# the best positive correlation with the target class, we want to see the jointplot of these two
sns.jointplot(df[df.columns[2]],df[df.columns[3]],kind="reg")
plt.show()


# # Box plot time~

# In[10]:


df.drop('target_class', axis=1).plot(kind='box', subplots=True, layout=(4,2), 
                                     figsize=(9,18), title='We need a box plot!')
plt.show()
# From the box plot, we can find this dataset has many mild outliers and extreme outliers, it is not a very good dataset.
# But, I will still use this dataset to do machine learning to prediet.
# We can find the data difference between these values is relatively large, and the extreme value is very different from the center. 


# # Pie plot of type 0 and type 1

# In[11]:


# see how what is the proportion of is a star and not a star

plt.figure(figsize=(8,8))
df[df.columns[-1]].value_counts().plot.pie(labels = ["is star","not a star"], 
                                           autopct = "%1.2f%%",shadow = True,explode=[0.2,0.2])
plt.title("Is star vs Not a star")
plt.show()


# # Let's go to the Machine Learning Part!

# In[12]:


# get the labels and features
labels = df[df.columns[-1]].values
df2 = df.drop(["target_class"],axis=1,inplace=False)
features = df2.values


# In[13]:


# do a min max scaler here
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
features_scaled = scaler.fit_transform(features)


# In[14]:


# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features_scaled,labels,random_state = 101,test_size=0.3)
print(" x_train len:",len(x_train),"\n","y_train len:",len(y_train),"\n","x_test len:",len(x_test),"\n","y_test len:",len(y_test))


# # Logistic Regression!

# In[15]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR_model = LogisticRegression(random_state=101)
LR_model.fit(x_train,y_train)
y_head_LR = LR_model.predict(x_test)
LR_score = LR_model.score(x_test,y_test)


# In[16]:


print("Confusion matrix for Logistic Regression:\n",confusion_matrix(y_test,y_head_LR))
con_LR = confusion_matrix(y_test,y_head_LR)

# show the confusion matrix plot
plt.matshow(con_LR)
plt.suptitle("confusion matrix for Logistic Regression")
plt.colorbar()
plt.show()
print("Accuracy for Logistic Regression type 0:",con_LR[0][0]/(con_LR[0][0]+con_LR[0][1]))
print("Accuracy for Logistic Regression type 1:",con_LR[1][1]/(con_LR[1][0]+con_LR[1][1]))
print("Total accuracy for Logistic Regression:",LR_score)


# # Do you understand decision tree?

# In[17]:


from sklearn.tree import DecisionTreeClassifier
DT_model = DecisionTreeClassifier()
DT_model.fit(x_train,y_train)
y_head_DT = DT_model.predict(x_test)
DT_score = DT_model.score(x_test,y_test)


# In[18]:


print("Confusion matrix for DT:\n",confusion_matrix(y_test,y_head_DT))
con_DT = confusion_matrix(y_test,y_head_DT)

# show the confusion matrix plot
plt.matshow(con_DT)
plt.suptitle("confusion matrix for Decision Tree")
plt.colorbar()
plt.show()
print("Accuracy for Decision Tree type 0:",con_DT[0][0]/(con_DT[0][0]+con_DT[0][1]))
print("Accuracy for Decision Tree type 1:",con_DT[1][1]/(con_DT[1][0]+con_DT[1][1]))
print("Total accuracy for Decision Tree:",DT_score)


# # This is from the example in class!!!

# In[19]:


from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
import os

feature_cols = df.columns[0:8]
os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"
dot_data = StringIO()
export_graphviz(DT_model, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())


# # Powerful Gaussian Naive Bayes :）

# In[20]:


from sklearn.naive_bayes import GaussianNB
NB_model = GaussianNB()
NB_model.fit(x_train,y_train)
y_head_NB = NB_model.predict(x_test)
NB_score = NB_model.score(x_test,y_test)


# In[21]:


print("Confusion matrix for NB:\n",confusion_matrix(y_test,y_head_NB))
con_NB = confusion_matrix(y_test,y_head_NB)

# show the confusion matrix plot
plt.matshow(con_NB)
plt.suptitle("confusion matrix for Naive Bayes")
plt.colorbar()
plt.show()
print("Accuracy for Naive Bayes type 0:",con_NB[0][0]/(con_NB[0][0]+con_NB[0][1]))
print("Accuracy for Naive Bayes type 1:",con_NB[1][1]/(con_NB[1][0]+con_NB[1][1]))
print("Total accuracy for Naive Bayes:",NB_score)


# # K.N.N !

# In[22]:


score = []
N = 20
for i in range(1,N):
    from sklearn.neighbors import KNeighborsClassifier
    KNN_model = KNeighborsClassifier(n_neighbors=i,weights="distance")
    KNN_model.fit(x_train,y_train)
    y_head_KNN = KNN_model.predict(x_test)
    KNN_score = KNN_model.score(x_test,y_test)
    score.append(KNN_score)


# In[23]:


# find out the socres for each KNN model by using different K values
# choose k is from 1 - 19
print("Your score is:\n",np.array(score).round(3))
plt.figure(figsize = (12,8))
plt.xlabel("Number K")
plt.ylabel("Accuracy")
plt.xticks(range(1,21),range(1,21))
plt.suptitle("Analysis on number of K")
plt.scatter(range(1,N),score,edgecolors = "r",marker = "^",linewidths = 5)
plt.show()


# In[24]:


print("Confusion matrix for KNN:\n",confusion_matrix(y_test,y_head_KNN))
con_KNN = confusion_matrix(y_test,y_head_KNN)

# show the confusion matrix plot
plt.matshow(con_KNN)
plt.suptitle("confusion matrix for KNN")
plt.colorbar()
plt.show()
print("Accuracy for KNN type 0:",con_KNN[0][0]/(con_KNN[0][0]+con_KNN[0][1]))
print("Accuracy for KNN type 1:",con_KNN[1][1]/(con_KNN[1][0]+con_KNN[1][1]))
print("Total accuracy for KNN:",KNN_score)


# # The last one: SVM!

# In[25]:


from sklearn.svm import SVC
SVM_model = SVC()
SVM_model.fit(x_train, y_train)
y_head_SVM = SVM_model.predict(x_test)
SVM_score = SVM_model.score(x_test,y_test)


# In[26]:


print("Confusion matrix for SVM:\n",confusion_matrix(y_test,y_head_SVM))
con_SVM = confusion_matrix(y_test,y_head_SVM)

# show the confusion matrix plot
plt.matshow(con_SVM)
plt.suptitle("confusion matrix for SVM")
plt.colorbar()
plt.show()
print("Accuracy for SVM type 0:",con_SVM[0][0]/(con_SVM[0][0]+con_SVM[0][1]))
print("Accuracy for SVM type 1:",con_SVM[1][1]/(con_SVM[1][0]+con_SVM[1][1]))
print("Total accuracy for SVM:",SVM_score)


# # Accuracy comparison

# In[27]:


# Use a histogram to compare
scores = (LR_score,DT_score,NB_score,KNN_score,SVM_score)
name = ("Logistic Regression","Decision Tree","Naive Bayes","KNN","SVM")

plt.figure(figsize=(14,8))
plt.xticks(range(1,6),name,fontsize=15)
plt.ylim(0.92,1.00)
plt.bar(range(1,6),scores)
plt.suptitle("Accuracy Comparsion of different methods!",fontsize=22)
plt.show()


# # Conclusion
# ## From the figure, we can see all of them perform well, escept the Naiver Bayes classifier. even though the Naiver Bayes is the worst, the gap of accuracy is just 3.5%.
# 
# ### Logistic Regression: It mainly solves the two-category problem and is used to indicate the possibility of something happening. Thus, it is suitable for this case.
# ### Decision tree: It is an algorithm for solving classification problems. It is easy to implement, highly interpretable, and fully in line with human intuitive thinking.
# ### Naive Bayes classifier: Naive Bayes is a simple but surprisingly powerful predictive modeling algorithm. The model consists of two types of probabilities that can be calculated directly from your training data: the probability of each class and the conditional probability of each class for each x value. Once calculated, the probabilistic model can be used to predict new data using Bayes' theorem. When the data is real, you usually assume a Gaussian distribution (bell curve) so that you can easily estimate these probabilities.
# ### K-Nearest Neighbors: he KNN algorithm is very simple and very efficient. The model representation of KNN is the entire training data set. New data points are predicted by searching the entire training set of the K most similar instances (neighbors) and summarizing the output variables of those K instances.
# ### Support Vector Machine: A hyperplane is a line that divides the input variable space. In the SVM, the hyperplane is selected to optimally separate the points in the input variable space from their classes (yes or no).
# 
# ## Wolpert and Macready (1995) said "All algorithms that search for an extremum of a cost function perform exactly the same, when averaged over all possible cost functions."  We call it "no free lunch", regardless of the machine learning algorithm, there are no better distinctions in terms of some criteria. In oher words, no matter how complicated the neural network is, it is the same as the simplest knn method.

# # Additional trying to use Neural Network by numpy

# In[28]:


# hand coding neural network
# inputs are the fist four columns
training_inputs = np.array([df[df.columns[0]].to_numpy(),
                            df[df.columns[1]].to_numpy(),
                            df[df.columns[2]].to_numpy(),
                            df[df.columns[3]].to_numpy(),
                            df[df.columns[4]].to_numpy(),
                            df[df.columns[5]].to_numpy(),
                            df[df.columns[6]].to_numpy(),
                            df[df.columns[7]].to_numpy()])
training_inputs = training_inputs.T
training_inputs = training_inputs[0:12529][:]
training_outputs = np.array([df[df.columns[-1]][0:12529]]).T
# check inputs and outputs
print("length of training_inputs are:",len(training_inputs[1]))
print("length of training_outputs are:",len(training_outputs[1]))


# In[29]:


## sigmoid function and sigmoid derivertive function

def sigmoid(x):
    x= 1/(1+np.exp(-x))
    return x

def sigmoid_de(x):
    x = x*(1-x)
    return x


# In[30]:


# initializing weights
weights_1 = 0.5*np.random.random((len(training_inputs.T),1))-1
weights_2 = 0.5*np.random.random((len(training_inputs.T),1))-1
weights_3 = 0.5*np.random.random((len(training_inputs.T),1))-1
weights_4 = 0.5*np.random.random((len(training_inputs.T),1))-1

print("Initial weights1 are:\n", weights_1)
print("Initial weights2 are:\n", weights_2)
print("Initial weights3 are:\n", weights_3)


# In[31]:


# training neurons
for i in range(0,10000):
    Inputs_layer = training_inputs
    
    outputs_layer = sigmoid(np.dot(training_inputs,weights_1)+np.dot(training_inputs,weights_2)
                           +np.dot(training_inputs,weights_3)+np.dot(training_inputs,weights_4))
    
#     calculating the error adjustment, weights for each neural in the hidden layer
#     adjustments are defined as the error times the sigmoid derivertive function 
#     we have three nuerals in the hidden layer
    error_1 = training_outputs - outputs_layer 
    adjustments_1 = error_1 * sigmoid_de(outputs_layer)
    weights_1 += np.dot(Inputs_layer.T, adjustments_1)
                            
    error_2 = training_outputs - outputs_layer 
    adjustments_2 = error_2 * sigmoid_de(outputs_layer)
    weights_2 += np.dot(Inputs_layer.T, adjustments_2)
    
    error_3 = training_outputs - outputs_layer 
    adjustments_3 = error_3 * sigmoid_de(outputs_layer)
    weights_3 += np.dot(Inputs_layer.T, adjustments_3)
    
    error_4 = training_outputs - outputs_layer 
    adjustments_4 = error_4 * sigmoid_de(outputs_layer)
    weights_4 += np.dot(Inputs_layer.T, adjustments_4)

print('weights: ')
print(weights_1)

print("Output After Training:")
print(outputs_layer)


# In[32]:


num=0
num_zero = 0
num_one = 0
a=0
b=0
c=0


# find out how many we predected for type 0 and 1 are right
for k in range(len(outputs_layer)):
    if outputs_layer[k]==training_outputs[k]:
        num +=1
    if training_outputs[k]==0:
        a+=1
        if outputs_layer[k]==training_outputs[k]:
            num_zero+=1
    if training_outputs[k]==1:
        b+=1
        if outputs_layer[k]==training_outputs[k]:
            num_one+=1

print("Total zeros:",a)
print("Total ones:",b)
print("Predicted zeros:",num_zero)
print("Predicted ones:",num_one)


# ## I do not have enough time for the last part which is a neural network code for this project. As far as my TA and I  concerned, this is the most fun and interesting part in this project! For now, I can only go this far as you can see. The neural network just can predicted zeros, but cannot predict ones. I think what I can do in the next period is that I can play around with parameters like the learning rate, how I generate the random weights and the numbers of the neurals in the hidden layer and so forth. 
# ## However, there is a proverb: failure is the mother of success, this part of the code is meant to use the powerful neural network built from scratch.  At this point, I think I have learnt something from doing this code and most importantly, I will finish the rest of this part of the code in the future! Thank you so much! Please kindly grade it!
