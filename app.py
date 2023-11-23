from flask import Flask, render_template, request

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

app = Flask(__name__)
model = VGG16()

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)
    label = decode_predictions(yhat)
    label = label[0][0]

    # Determine the classification and select the appropriate emoji
    if 'banana' in label[1].lower():
        classification = 'This is a banana! (%.2f%%)' % (label[2]*100)
        emoji = '😄'  # Happy emoji
    else:
        classification = 'This is not a banana.'
        emoji = '😢'  # Sad emoji

    return render_template('index.html', prediction=classification, emoji=emoji)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
