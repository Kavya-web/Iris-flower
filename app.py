

from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__,template_folder='templates')


model_pk = pickle.load(open('iri.pkl','rb'))

@app.route('/')
def get_info():
    return render_template('info.html')

@app.route('/api_predict', methods = ['POST','GET'])
def api_predict():
    
        sepal_length = request.form["sepal_length"]
        sepal_width = request.form["sepal_width"]
        petal_length = request.form["petal_length"]
        petal_width = request.form["petal_width"]
    
        data = np.array([[sepal_length, sepal_width, 
                          petal_length, petal_width]])
           
        prediction = model_pk.predict(data)
        return str(prediction)

if __name__ == "__main__":
    app.run(debug=True)
    
    
