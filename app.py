from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)


Housing = pd.read_csv("./Housing_recommendation.csv")
Housing = Housing.dropna(axis=1, how='all')  
Housing = Housing.dropna(axis=0, how='all') 
Housing['rent'] = Housing['rent'].str.extract(r'(\d+)').astype(float)


Label_encoders = {}
categorical_column = ['Laundry', 'type_of_room', 'Gender', 'Food', 'Type_of_accomodation']
for col in categorical_column:
    le = LabelEncoder()
    Housing[col] = le.fit_transform(Housing[col])
    Label_encoders[col] = le



scaler = MinMaxScaler()
numerical_columns = ['Bedrooms', 'Bath', 'Dist.from_univ ', 'rent']
Housing_new = scaler.fit_transform(Housing[numerical_columns])


feature_columns = numerical_columns + categorical_column
feature_matrix = Housing[feature_columns].to_numpy()


knn_model = NearestNeighbors(n_neighbors=6, metric='cosine')
knn_model.fit(feature_matrix)


def recomendation(**kwargs):
    filtered_df = Housing.copy()
    for col in Housing.select_dtypes(include=['object']).columns:
        Housing[col] = Housing[col].str.strip().str.lower()
    kwargs = {k: (v.strip().lower() if isinstance(v, str) else v) for k, v in kwargs.items()}
    print(kwargs)


    for key, value in kwargs.items():
        if value is not None:
            if key not in Housing.columns:
                return f"Error: Column '{key}' not found in dataset."
            value = int(value)
            filtered_df = filtered_df[filtered_df[key] == value]

    if filtered_df.empty:
        return "No recommendations available for the given criteria."

 
    index = filtered_df.index[0]
    distances, indices = knn_model.kneighbors(feature_matrix[index].reshape(1, -1))

    
    recommendations = [Housing.iloc[i] for i in indices[0][1:7]]

    
    recommendation_df = pd.DataFrame(recommendations)[['Address', 'Zip Code', 'Transportation', 'Amenities']]
    result = []
    for idx, row in recommendation_df.iterrows():
        res = {
            'Address': row['Address'],
            'Zip Code': row['Zip Code'],
            'Transportation': row['Transportation'],
            'Amenities': row['Amenities']
        }
        result.append(res)

    return result


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        
        form_data = {
            'Bedrooms': request.form['Bedrooms'],
            'Bath': request.form['Bath'],
            'Food': request.form['Food']
        }
        return redirect(url_for('recommend', **form_data))
    return render_template('index.html')


@app.route('/recommend')
def recommend():
    kwargs = {key: value for key, value in request.args.items()}
    recommendations = recomendation(**kwargs)
    if recommendations is None:
        message = "No recommendations available for the given criteria."
    else:
        message = None
    return render_template('result.html', recommendations=recommendations, message=message)



if __name__ == '__main__':
    app.run(debug=True)
