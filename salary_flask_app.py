#!/usr/bin/env python
# coding: utf-8

# # Creating flask app for deployment of salary model

# In[1]:


import numpy as np
from flask import Flask, request, render_template
import pickle


# In[ ]:


app = Flask(__name__) #instantiate the flask app

model = pickle.load(open('salary_model.pkl','rb')) #load the ML model which is ready with us for deployment

@app.route('/') # route to home page.Home directiory is given by '/'
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0],2)
    return render_template('index.html',prediction_text = "Employee Salary should be $ {}".format(output))

if __name__ == '__main__':
    app.run(debug = True)


# In[ ]:




