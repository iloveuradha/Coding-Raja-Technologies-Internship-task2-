from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load your model
# Move model loading outside of the route
model = joblib.load('trained_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract features from the request (adjust as needed based on your input form)
        features = [float(request.form['feature1']),
                    float(request.form['feature2']),
                    float(request.form['feature3']),
                    float(request.form['feature4']),
                    float(request.form['feature5']),
                    float(request.form['feature6']),
                    float(request.form['feature7']),
                    float(request.form['feature8']),
                    float(request.form['feature9']),
                    float(request.form['feature10'])

                    ]

        # Make prediction
        prediction = model.predict([features])[0]

        # Display the result on a new page
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

