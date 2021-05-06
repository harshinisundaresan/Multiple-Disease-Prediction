from django.http import HttpResponse
from django.shortcuts import render
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def home(request):
    return render(request,"home.html")

def model1(request):
    return render(request,"model1.html")

def result1(request):
    df = pd.read_csv("diabetes.csv")

    #except from pregnancies and outcome no other value can have a minimum of zero. So we replce it by their medians
    df['Glucose']=df.Glucose.mask(df.Glucose == 0,df['Glucose'].median())
    df['BloodPressure']=df.BloodPressure.mask(df.BloodPressure == 0,df['BloodPressure'].median())
    df['SkinThickness']=df.SkinThickness.mask(df.SkinThickness == 0,df['SkinThickness'].median())
    df['Insulin']=df.Insulin.mask(df.Insulin == 0,df['Insulin'].median())
    df['BMI']=df.BMI.mask(df.BMI == 0,df['BMI'].median())
    pd.set_option("max_rows", None)

    X=df.drop('Outcome',axis=1)
    y=df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=None)

    std=StandardScaler()
    X_train_std=std.fit_transform(X_train)
    X_test_std=std.transform(X_test)

    mlp=MLPClassifier(hidden_layer_sizes=(9,9))
    mlp.fit(X_train_std,y_train)
    
    Pregnancies=float(request.GET['Pregnancies'])
    Glucose=float(request.GET['Glucose'])
    BloodPressure=float(request.GET['BloodPressure'])
    SkinThickness=float(request.GET['SkinThickness'])
    Insulin=float(request.GET['Insulin'])
    BMI=float(request.GET['BMI'])
    DiabetesPedigreeFunction=float(request.GET['DiabetesPedigreeFunction'])
    Age=float(request.GET['Age'])

    pred=mlp.predict(np.array([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]).reshape(1,-1))

    ans=''
    if(pred==[1]):
        ans="The person have diabetes"
    else:
        ans="The person does not have diabetes"
    
    
    return render(request,"result1.html",{'ans':ans})

def model2(request):
    return render(request,"model2.html")
def result2(request):
    df = pd.read_csv("heart.csv")
    X=df.drop('target',axis=1)
    y=df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=None)
    std=StandardScaler()
    X_train_std=std.fit_transform(X_train)
    X_test_std=std.transform(X_test)
    rft=RandomForestClassifier()
    rft.fit(X_train_std,y_train)

    age = float(request.GET['age'])
    sex = float(request.GET['sex'])
    cp = float(request.GET['cp'])
    trestbps = float(request.GET['trestbps'])
    chol = float(request.GET['chol'])
    fbs = float(request.GET['fbs'])
    restecg = float(request.GET['restecg'])
    thalach = float(request.GET['thalach'])
    exang = float(request.GET['exang'])
    oldpeak = float(request.GET['oldpeak'])
    slope = float(request.GET['slope'])
    ca = float(request.GET['ca'])
    thal = float(request.GET['thal'])

    pred=rft.predict(np.array([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]).reshape(1,-1))

    ans=''
    if(pred==[1]):
        ans="The person have heart disease"
    else:
        ans="The person does not have heart disease"
    
    return render(request,"result2.html",{'ans':ans})
    
def model3(request):
    return render(request,"model3.html")
def result3(request):
    df = pd.read_csv("indian_liver_patient.csv")
    df['Albumin_and_Globulin_Ratio']=df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].mean())

    X=df.drop('Dataset',axis=1)
    y=df['Dataset']
    X.Gender=X.Gender.map({'Male':0,'Female':1})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=None)
    std=StandardScaler()
    X_train_std=std.fit_transform(X_train)
    X_test_std=std.transform(X_test)
    lr=LogisticRegression()
    lr.fit(X_train_std,y_train)

    Age = float(request.GET['Age'])
    Gender = float(request.GET['Gender'])
    Total_Bilirubin = float(request.GET['Total_Bilirubin'])
    Direct_Bilirubin = float(request.GET['Direct_Bilirubin'])
    Alkaline_Phosphotase = float(request.GET['Alkaline_Phosphotase'])
    Alamine_Aminotransferase = float(request.GET['Alamine_Aminotransferase'])
    Aspartate_Aminotransferase = float(request.GET['Aspartate_Aminotransferase'])
    Total_Protiens = float(request.GET['Total_Protiens'])
    Albumin = float(request.GET['Albumin'])
    Albumin_and_Globulin_Ratio = float(request.GET['Albumin_and_Globulin_Ratio'])

    pred=lr.predict(np.array([Age,Gender,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio]).reshape(1,-1))

    ans=''
    if(pred==[2]):
        ans="The person have liver disease"
    else:
        ans="The person does not have liver disease"

    return render(request,"result3.html",{'ans':ans})