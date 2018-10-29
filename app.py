from flask import Flask, request, json, Response, redirect, url_for, render_template
from wtforms import Form, BooleanField, StringField, PasswordField, IntegerField, FloatField, validators
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

diabetes = pd.read_csv('diabetes.csv')
del diabetes['DiabetesPedigreeFunction']
print(diabetes.info())

knn = None
X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'], diabetes['Outcome'], stratify=diabetes['Outcome'], random_state=66)

neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    # build the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

app = Flask(__name__)

class DiabetesForm(Form):
    age = IntegerField('Age')
    bmi = FloatField('BMI')
    bp = IntegerField('Diastolic blood pressure (mm Hg)')
    glucose = IntegerField('Glucose (plasma glucose concentration over 2 hours)')
    insulin = IntegerField('2 hour serum insulin (mu U/mL)')
    skin = IntegerField('Skin thickness (triceps skin fold thickness in mm)')
    pregnancies = IntegerField('Number of pregnancies')
    accept_tos = BooleanField('I will not substitute this for a doctor\'s assessment', [validators.DataRequired()])

@app.route('/', methods=['GET', 'POST'])
def main():
    global knn
    form = DiabetesForm(request.form)
    if request.method == 'POST' and form.validate():
        res = knn.predict_proba(
            [[
                float(form.pregnancies.data),
                float(form.glucose.data),
                float(form.bp.data),
                float(form.skin.data),
                float(form.insulin.data),
                float(form.bmi.data),
                float(form.age.data)
            ]]
        )
        for i in res.tolist():
            print(i)
        return render_template("result.html", res=int(res[0][0] * 100))
        
    return render_template("index.html", form=form)