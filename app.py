
from flask import Flask, render_template, request
# libraries for making count matrix and similarity matrix

import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from sklearn.externals import joblib
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
nutrition_df = pd.read_csv('Test.csv', header=0)

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
   # print(request.get_data('Fat'))
    print("hello")
    #print(request.form.to_dict())
    print(request.json)
    
    print(request.json['Fat'])
    #print(request.get_json()["Fat"])
    #print(request.args.get('Fat'))
    lis=[]
    t=[]
    lis.append(int(request.json['Weight']))
    lis.append(int(request.json['Calories']))
    lis.append(int(request.json['Fat']))
    lis.append(int(request.json['Sodium']))
    lis.append(int(request.json['Carbo']))
    lis.append(int(request.json['Cholesterol']))
    lis.append(int(request.json['Protein']))
    lis.append(int(request.json['Calcium']))
    lis.append(int(request.json['Potassium']))
    lis.append(int(request.json['Iron']))
    lis.append(int(request.json['VitaminA']))
    lis.append(int(request.json['VitaminC']))
    
    lis.append(int(request.json['A']))
    lis.append(int(request.json['B']))
    lis.append(int(request.json['C']))
    lis.append(int(request.json['D']))
    lis.append(int(request.json['E']))
    lis.append(int(request.json['F']))
    lis.append(int(request.json['G']))
    lis.append(int(request.json['H']))
    lis.append(int(request.json['I']))
    lis.append(int(request.json['J']))
    lis.append(int(request.json['K']))
    lis.append(int(request.json['L']))
    lis.append(int(request.json['M']))
    lis.append(int(request.json['N']))
    lis.append(int(request.json['O']))
    lis.append(int(request.json['P']))
   


    t.append(lis)
    
    
    
    #t=[[48.2, 180, 4.5, 150, 28, 0,10,0.22,0.26,0.009,3,0.036]]
    

    #int_features = [int(x) for x in request.form.values()]
    #int_features.reshape(1,-1)
    #print(int_features)
    #final_features = np.array(int_features)
    distances, indices = model.kneighbors(t, n_neighbors=5)
    
    temp=[]
    for i in indices[0]:
        dic={}
        #print (nutrition_df.loc[i]['Brand'])
        dic['Brand']=nutrition_df.loc[i]['Brand']
        dic['FoodName']=nutrition_df.loc[i]['FoodName']
        dic['Category']=nutrition_df.loc[i]['Category']

        temp.append(dic)

    #recommended_products = [nutrition_df.loc[i]['FoodName'] for i in indices[0]]
   
    return str(temp)

if __name__ == "__main__":
    app.run(debug=True)
