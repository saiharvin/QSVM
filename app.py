import numpy as np
import pickle
from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def entry_page():
    if request.method == 'POST':
        val1 = request.form['val1']
        val2 = request.form['val2']
        val3 = request.form['val3']
        val4 = request.form['val4']
        temp = [float(val1), float(val2),float(val3), float(val4)]
        loaded_model = pickle.load(open("model.pkl", "rb"))
        result = loaded_model.predict(temp)
        if result == 0:
            value = "Bad"
        else:
            value = "Good"
        return render_template('result.html', prediction=value)
        # return f"<h1>{result}</h1>"
    else:
        return render_template('file.html')


if __name__ == '__main__':
    app.run(debug=True)
