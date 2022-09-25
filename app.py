from flask import Flask, request, render_template
from datetime import date


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/form')
def forms():
    return render_template('form.html')


@app.route('/result', methods=('GET', 'POST'))
def result():
    if request.method == 'POST':
        age = request.form.get("age")
        age_days = age.split('-')
        f_date = date(int(age_days[0]), int(age_days[1]), int(age_days[2]))
        l_date = date.today()
        delta = l_date - f_date
        age_days = delta.days

        height = request.form.get("height")

        weight = request.form.get("weight")

        gender = request.form.get("gender")
        if gender == "Male":
            gender = 0
        else:
            gender = 1

        sbp = int(request.form.get("sbp"))

        dbp = int(request.form.get("dbp"))

        cholestrol = request.form.get("cholestrol")
        if cholestrol == "Normal":
            cholestrol = 0
        elif cholestrol == "Above Normal":
            cholestrol = 1
        else:
            cholestrol = 2

        glucose = request.form.get("glucose")
        if glucose == "Normal":
            glucose = 0
        elif glucose == "Above Normal":
            glucose = 1
        else:
            glucose = 2

        smoke = request.form.get("smoking")
        if smoke == "Yes":
            smoke = 0
        else :
            smoke = 1

        alcohol = request.form.get("alcohol")
        if alcohol == "Yes":
            alcohol = 0
        else:
            alcohol = 1

        activity = request.form.get("activity")
        if activity == "Yes":
            activity = 0
        else:
            activity = 1
        import numpy as np
        import pickle
        import warnings
        warnings.filterwarnings("ignore")

        model = pickle.load(open('check_health.pkl', 'rb'))
        input_data = (23,age_days,height,weight,gender,sbp,dbp,cholestrol,glucose,smoke,alcohol,activity)

        # change the input data to a numpy array
        input_data_as_numpy_array= np.array(input_data)

        # reshape the numpy array as we are predicting for only on instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        prediction = model.predict(input_data_reshaped)
        a = ""
        if (prediction[0]== 0):
            a = "+result.html"
        else:
            a = "-result.html"
    return render_template(a)


@app.route('/about-us')
def about_us():
    return render_template('about_us.html')



app.run()
