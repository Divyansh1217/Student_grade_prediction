import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st
from sklearn.metrics import mean_absolute_error as mae,mean_squared_error as mse 
data=pd.read_csv("student-mat.csv")
st.write(data.head(10))
data=data[["G1","G2","G3","studytime","failures","absences"]]
predict="G3"
x=np.array(data.drop([predict], axis=1))
y=np.array(data[predict])
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.2,random_state=7)
linear_reg=LinearRegression()
linear_reg.fit(x_train,y_train)
accuracy = linear_reg.score(x_test, y_test)
st.write("Accuracy:",accuracy)
predictions = linear_reg.predict(x_test)
st.write("Mean Square Error",mse(y_test,predictions))
st.write("Mean Absolute Error",mae(y_test,predictions))
