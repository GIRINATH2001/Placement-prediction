#!/usr/bin/env python
# coding: utf-8

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import sweetviz as sv
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn import metrics
from sklearn import tree

import streamlit as st

def main():
    st.title("Predict Yourself")
    st.write("Rate Yourself in scale of 1-10")
    n1 = st.number_input("Rate Yourself on Programming language: ",0,10)/10
    n2 = st.number_input("Rate Yourself on Frontend dev: ",0,10)/10
    n3 = st.number_input("Rate Yourself on DBMS: ",0,10)/10
    n4 = st.number_input("Rate Yourself on Backend dev: ",0,10)/10
    n5 = st.number_input("Rate Yourself on Operating System: ",0,10)/10
    n6 = st.number_input("Rate Yourself on Communication Skills: ",0,10)/10
    list = [n1,n2,n3,n4,n5,n6]
    if st.button("Submit"):
        st.write("On the Basis of these: ")
        main_prog(list)



def main_prog(list):
    warnings.filterwarnings('ignore')
    df  = pd.read_excel("C:\\Users\\hp\\devaminiproject\\Newowndataset.xlsx")
    #df
    # # Histogram plots
    #report = sv.analyze(df)
    #report.show_html("./report.html")
    # # Data cleaning and processing
    df2=df.drop(['Roll No','Name','Placed','Gender','CTC','Diploma %','Backlogs'],axis='columns')
    df2['Placed_category'].unique()
    df2['Placed_category'].replace({'L1General':0,'L2General':1,'Dream':2,'SuperDream':3},inplace=True)
    df2.rename(columns = {'Placed_category':'categoryplaced'}, inplace = True)
    df2['ProgrammingLanguage'] = df2['ProgrammingLanguage'].astype(float)
    df2['Frontend dev']=df2['Frontend dev'].astype(float)
    df2['DBMS']=df2['DBMS'].astype(float)
    df2['Backend dev ']=df2['Backend dev '].astype(float)
    df2['Operating System ']=df2['Operating System '].astype(float)
    df2['Communication_skills']=df2['Communication_skills'].astype(float)
    #df2.describe().T
    #df2.sample(10)
    #df2.isnull().sum()
    #df2.info()
    # # Spliting of Test and Training Data
    y = df2["categoryplaced"]
    x = df2.drop(["Class 10 %","Class 12 %","UG percentage","Current course %","categoryplaced"],axis='columns')
    x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0) 
    # determine the mutual information
    mutual_info = mutual_info_classif(x_train, y_train)
    #mutual_info
    mutual_info = pd.Series(mutual_info)
    mutual_info.index = x_train.columns
    mutual_info.sort_values(ascending=False)
    mutual_info.sort_values(ascending=False).plot.bar(figsize=(20, 8))
    sel_five_cols = SelectKBest(mutual_info_classif, k=5)
    sel_five_cols.fit(x_train, y_train)
    #x_train.columns[sel_five_cols.get_support()]
    # # Creating Decision Tree classifer object
    classifier = DecisionTreeClassifier(criterion="entropy", max_depth=4)# Train Decision Tree Classifer
    classifier = classifier.fit(x_train,y_train)#Predict the response for test dataset
    # # Model Accuracy
    y_pred = classifier.predict(x_test)
    #print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)
    # # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classifier.classes_)
    disp.plot()
    #plt.show()
    # # Prediction of Input Data 
    def Prediction(features):
        predict = classifier.predict(features)
        if predict == 0:
            print("You will get place in L1 General category" )
            st.title("You will be place in L1 General category")
        if predict == 1:
            print("You will get place in L2 General category" )
            st.title("You will be place in L2 General category")
        if predict == 2:
            print("You will get place in Dream Category")
            st.title("You will be place in Dream Category")
        if predict == 3:
            print("You will get place in SuperDream Category")
            st.title("You will be place in SuperDream Category")

    #[Programming language,Frontend dev,DBMS,Backend dev,Operating System,Communication_skills]  
    features = np.array([list])
    #features = arrayinput
    #features.reshape(1, -1)

    Prediction(features)

    fn=['Programming language','Frontend dev l','DBMS','Backend dev','Operating System','Communication_skills']
    cn=['L1', 'L2', 'Dream','SuperDream']
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (15,10), dpi=300)
    tree.plot_tree(classifier,
                   feature_names = fn, 
                   class_names=cn,
                   filled = True);
    #fig.savefig('imagename.png')





if __name__ == '__main__':
    main()
