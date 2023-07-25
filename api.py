from flask import Flask, request,jsonify
import joblib
# import json
import pandas as pd
import numpy as np

api_app=Flask(__name__)
model = joblib.load('models/BHP.pkl')


@api_app.route('/predict', methods=['POST'])
def predict(): 
    req=request.json
    inp=pd.DataFrame(req.items()).T
    inp.columns=inp.iloc[0]
    inp=inp.iloc[1:,:]
    # print(inp)
    
    # # Make the prediction using the model
    predicted_price = model.predict(inp)
    # print(round(predicted_price[0],2))
    # # Return the predicted price as the response
    return jsonify({'ans':str(np.round(predicted_price[0]*100000,2))}),201


if __name__=='__main__':
    api_app.run(debug=True)

