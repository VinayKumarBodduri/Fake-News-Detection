from flask import Flask,render_template,request
import pickle
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))
vector = pickle.load(open('vector.pkl','rb'))



@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        text = str(request.form['text'])
        fpred = model.predict(vector.transform([text]))
        return render_template('index.html', pred = fpred)
    else:
        return render_template('index.html', pred = "Contact admin")

if __name__ == '__main__':
    app.run(debug=True)