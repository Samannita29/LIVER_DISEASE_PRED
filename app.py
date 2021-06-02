import numpy as np 
import pandas as pd
from sklearn import metrics
from flask import Flask,jsonify, request, render_template
from flask_cors import CORS
import re
import math
import pickle

app = Flask("__name__",template_folder='templates')
CORS(app)

q = ""

@app.route("/")
def loadPage():
	#return render_template('home.html', query="")
    return jsonify({"msg":"page loaded successfully"})



@app.route("/predict", methods=['POST'])
def LiverPrediction():
    df = pd.read_csv('indian_liver_patient.csv')

    df.info()
    print(request.form)

    inputQuery1 = request.form['Age']
    inputQuery3 = request.form['Total_Bilirubin']
    inputQuery4 = request.form['Direct_Bilirubin']
    inputQuery5 = request.form['Alkaline']
    inputQuery6 = request.form['Alamine']
    inputQuery7 = request.form['Aspartate']
    inputQuery8 = request.form['Total_Protiens']
    inputQuery9 = request.form['Albumin']
    inputQuery10 = request.form['Albumin_Globulin_Ratio']

    model=pickle.load(open('model12.sav','rb'))
    
    data = np.array([inputQuery1, inputQuery3, inputQuery4, inputQuery5,inputQuery6,inputQuery7,inputQuery8,inputQuery9,inputQuery10]).reshape(1,-1).astype('float64')
    
   # new_df = pd.DataFrame(data,p columns = ['Preg', 'Plas', 'Pres', 'Skin', 'Test','mass','Pedi','age'])
    #print(new_df)
    single = model.predict(data)
    print(single)
    
    #probability = model.predict_proba(new_df)[:,0:8]
    #probability=model.score(data[:][0:8],data[:][8])
    #print(probability)
    if single==1:
        output = "The patient is diagnosed with Liver Disease"
        #output1 = "Confidence: {}".format(probability*100)
    if single==2:
        output = "The patient is not diagnosed with Liver  Disease"
        #output1 = ""
    
    #return render_template('home.html', output1=output, query1 = request.form['query1'], query2 = request.form['query2'],query3 = request.form['query3'],query4 = request.form['query4'],query5 = request.form['query5'],query6 = request.form['query6'],query7 = request.form['query7'],query8 = request.form['query8'])
    return jsonify({"message":"Data processed",
    "output":output})
if __name__ == "__main__":
    app.run(debug=True)
