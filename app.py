from flask import Flask, request, jsonify, send_from_directory
import pickle
import os

app = Flask(__name__)

# Load the medicine data and similarity matrix
medicine_dict = pickle.load(open('medicine_dict.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Convert medicine_dict back to DataFrame-like structure for lookup
# medicines = medicine_dict['Drug_Name']
# tags = medicine_dict['tags']

import pandas as pd

# Convert medicine_dict to DataFrame for easier lookup
df_medicines = pd.DataFrame(medicine_dict)

@app.route('/')
def serve_frontend():
    return send_from_directory(os.getcwd(), 'index.html')

@app.route('/recommend', methods=['GET'])
def recommend():
    medicine_name = request.args.get('medicine')
    if not medicine_name:
        return jsonify({'error': 'Please provide a medicine name using the "medicine" query parameter.'}), 400

    try:
        medicine_index = df_medicines[df_medicines['Drug_Name'] == medicine_name].index[0]
    except IndexError:
        return jsonify({'error': f'Medicine "{medicine_name}" not found in database.'}), 404

    distances = similarity[medicine_index]
    medicines_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_medicines = [df_medicines.iloc[i[0]]['Drug_Name'] for i in medicines_list]

    return jsonify({'recommended_medicines': recommended_medicines})

from Personalized_Medicine_Recommending_System import recommend_by_symptoms

@app.route('/recommend_by_symptoms', methods=['POST'])
def recommend_by_symptoms_api():
    data = request.get_json()
    if not data or 'symptoms' not in data:
        return jsonify({'error': 'Please provide symptoms as a JSON array in the request body.'}), 400

    symptoms = data['symptoms']
    if not isinstance(symptoms, list) or not all(isinstance(s, str) for s in symptoms):
        return jsonify({'error': 'Symptoms should be a list of strings.'}), 400

    recommended_medicines = recommend_by_symptoms(symptoms)
    return jsonify({'recommended_medicines': recommended_medicines})

if __name__ == '__main__':
    app.run(debug=True)
