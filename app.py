from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('models/BHP.pkl')
df=pd.read_csv('cleaned_data.csv')



@app.route('/')
def index():
    locations=sorted(df['location'].unique()) 
    return render_template('index.html',locations=locations)

@app.route('/predict', methods=['POST'])
def predict(): 
    # Get the input values from the form
    location = request.form.get('location')
    bhk = int(request.form.get('bhk'))
    bath = int(request.form.get('bath'))
    total_sqft = float(request.form.get('total_sqft'))

    # Create a DataFrame with the input data
    input_data = pd.DataFrame([[location, bhk, bath, total_sqft]],
                              columns=['location', 'bhk', 'bath', 'total_sqft'])

    # Make the prediction using the model
    predicted_price = model.predict(input_data)
    # print(round(predicted_price[0],2))
    # Return the predicted price as the response
    return str(np.round(predicted_price[0]*100000,2))

if __name__ == '__main__':
    app.run(debug=True)
