from flask import Flask, request, jsonify
from models import spacyTrain
from flask_cors import CORS
import spacy
from spacy.lang.en.examples import sentences
from spacy.tokens import DocBin

app = Flask(__name__)
CORS(app)

spacyTrain = spacyTrain.SpacyTrain()

@app.route('/trainData', methods=['GET'])
def trainData():
    return jsonify(spacyTrain.ner()), 200

@app.route('/trainData2', methods=['GET'])
def trainData2():
    return jsonify(spacyTrain.ner2(True)), 200

@app.route('/trainFlair', methods=['GET'])
def trainData3():
    return jsonify(spacyTrain.nerFlair()), 200

@app.route('/tweetData', methods=['GET'])
def tweetData():
    return jsonify(spacyTrain.tweetData()), 200

@app.route('/findData', methods=['POST'])
def findData():
    if request.method == "POST":
        text = request.form['text']
        response = spacyTrain.findData({'text': text})
        return response, 201

if __name__ == '__main__':
    app.run(debug=True)
