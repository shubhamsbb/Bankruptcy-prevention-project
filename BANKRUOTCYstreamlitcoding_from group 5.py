# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 21:08:26 2023

@author: MAHESH
"""
import numpy as np
import pickle
import pandas as pd
import streamlit as st


df = pd.read_csv("bankruptcy-prevention.csv", sep=";")
df

df["class_as"] = 0

# df.loc[df['class'] == 'bankruptcy', 'class_as'] = 0

df.loc[df[" class"] == 'non-bankruptcy', 'class_as'] = 1
df.sample(10)

df.drop(columns = ' class',axis =1,inplace = True)
df

x = df.drop(columns = 'class_as',axis =1)
y = df['class_as']

from imblearn.over_sampling import RandomOverSampler
ros=RandomOverSampler(random_state=0)

x,y=ros.fit_resample(x,y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=2)

from sklearn.svm import SVC
clf_poly = SVC(kernel='poly')
clf_poly.fit(x_train , y_train)


pickle_in = open("clf_poly.pkl","rb")
classifier=pickle.load(pickle_in)


def welcome():
    return "Welcome ALL"
  
st.title('Bankruptcy Detector')

st.write("""
### Let your serene mind be more productive
and far away from someone spoiling it for you,
BANKRUPTCY = 0 ,NON-BANKRUPTCY = 1
""")

model_name = st.sidebar.selectbox(
    'Select Model',
    ('polySVC','DecisionTreeClassifier','AdaBoostClassifier','GradientBoostingClassifier','XGBClassifier')
)
      
def predict_bankruptcy(industrial_risk,management_risk,financial_flexibility,credibility,competitiveness,operating_risk):
     prediction=classifier.predict([[industrial_risk,management_risk,financial_flexibility,credibility,competitiveness,operating_risk]])
     print(prediction)
     return prediction

def main():
    st.title("Bankruptcy Detector")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bankruptcy Detector ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    industrial_risk = st.text_input("industrial_risk","Type Here")
    management_risk = st.text_input("management_risk","Type Here")
    financial_flexibility = st.text_input("financial_flexibility","Type Here")
    credibility = st.text_input("credibility","Type Here")
    competitiveness = st.text_input("competitiveness","Type Here")
    operating_risk = st.text_input("operating_risk","Type Here")
    result=""
    if st.button("Predict"):

        result = predict_bankruptcy(industrial_risk,management_risk,financial_flexibility,credibility,competitiveness,operating_risk)
    st.success('The output is{}'.format(result))
    if st.button("About"):
        st.text("Lets Learn")
        st.text("Buildt with Streamlit")
if __name__=='__main__':
    main()

    
