import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle
# import firebase_admin
# from firebase_admin import credentials, firestore

CropRecommendationApplication = Flask(__name__)
croprecommendationmodel = pickle.load(
    open('Croprecommendationmodel.pkl', 'rb'))


# @CropRecommendationApplication.route('/database', methods=['GET'])
# def showDatabase():
#     ref = firebase_admin.firestore.client(
#         app=None).collection(u"data").stream()

#     for doc in ref:
#         print(f'{doc.id} => {doc.to_dict()}')

#     return jsonify(message="Success")


@CropRecommendationApplication.route('/predict_mobile', methods=['GET','POST'])
def predictMobile():

    features = [float(x) for x in request.form.values()]
    f_features = [np.array(features)]

    print(features)

    if(len(features) < 7):
        return jsonify(
            message="Invalid parameters"
        )

    else:
        prediction = croprecommendationmodel.predict(f_features)
        out = int(prediction)
        decoded_form = {0: 'apple', 1: 'banana', 2: 'blackgram', 3: 'chickpea', 4: 'coconut', 5: 'coffee', 6: 'cotton', 7: 'grapes', 8: 'jute', 9: 'kidneybeans',
                        10: 'lentil', 11: 'maize', 12: 'mango', 13: 'mothbeans', 14: 'mungbean', 15: 'muskmelon', 16: 'orange', 17: 'papaya', 18: 'pigeonpeas',
                        19: 'pomegranate', 20: 'rice', 21: 'watermelon'}

        keyMap = {"N": features[0], "P": features[1], "K": features[2], "temperature": features[3],
                  "humidity": features[4], "pH": features[5], "moisture": features[6]}
        return jsonify(
            message="OK",
            predicted_crop=decoded_form[out],
            attributes=keyMap)


if __name__ == "__main__":

    # cred = credentials.Certificate("serviceAccountKey.json")
    # firebase_admin.initialize_app(cred)

    CropRecommendationApplication.run(debug=True,host='192.168.10.71', port=5000)
