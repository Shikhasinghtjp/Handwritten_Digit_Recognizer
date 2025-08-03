from flask import Flask, render_template, request
from predict import predict_digit
from PIL import Image
import base64
import io
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None

    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            img = Image.open(file.stream)
            prediction, confidence = predict_digit(img)

    return render_template('index.html', prediction=prediction, confidence=confidence)


@app.route('/canvas', methods=['POST'])
def canvas_predict():
    prediction = None
    confidence = None

    try:
        canvas_data = request.form['canvas_data']
        if canvas_data:
            # Strip off the base64 header
            canvas_data = canvas_data.split(',')[1]
            image_bytes = base64.b64decode(canvas_data)
            image = Image.open(io.BytesIO(image_bytes))

            prediction, confidence = predict_digit(image)
    except Exception as e:
        print("Canvas prediction error:", e)

    return render_template('index.html', prediction=prediction, confidence=confidence)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)