from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from io import BytesIO
import base64
from flask import send_file
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the fashion dataset
fashion = pd.read_csv("fashion-mnist_train.csv")
new_df = fashion.drop(['label'], axis=1)
reshaped_data = new_df.values.reshape(-1, 28, 28, 1)

# Load the generator model
new_model = tf.keras.models.load_model('genetor.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_real_images', methods=['GET'])
def get_real_images():
    imgs = reshaped_data[394:400]
    img_list = []
    for i in range(6):
        buffer = BytesIO()
        plt.imsave(buffer, imgs[i][:, :, 0], cmap='gist_yarg')
        img_str = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode('utf-8')
        img_list.append(img_str)

    return jsonify(img_list)

@app.route('/get_real_images_file/<int:index>', methods=['GET'])
def get_real_images_file(index):
    img = reshaped_data[index][:, :, 0]
    buffer = BytesIO()
    plt.imsave(buffer, img, cmap='gist_yarg')
    buffer.seek(0)

    return send_file(buffer, mimetype='image/png', as_attachment=True, download_name=f'real_image_{index}.png')

@app.route('/show_images')
def show_images():
    # Render the images.html template
    return render_template('images.html')

@app.route('/get_more_real_images', methods=['GET'])
def get_more_real_images():
    more_imgs = reshaped_data[:5000]
    img_list = []

    for i in range(5000):
        buffer = BytesIO()
        plt.imsave(buffer, more_imgs[i][:, :, 0], cmap='gist_yarg')
        img_str = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode('utf-8')
        img_list.append(img_str)

    # Render the images.html template and pass img_list as a variable
    return render_template('images.html', images=img_list)

@app.route('/get_more_real_images_file/<int:index>', methods=['GET'])
def get_more_real_images_file(index):
    img = reshaped_data[index][:, :, 0]
    buffer = BytesIO()
    plt.imsave(buffer, img, cmap='gist_yarg')
    buffer.seek(0)

    return send_file(buffer, mimetype='image/png', as_attachment=True, download_name=f'real_image_{index}.png')

@app.route('/generate_images', methods=['GET'])
def generate_images():
    imgs = new_model.predict(tf.random.normal((6, 128)))
    img_list = []

    for i in range(6):
        buffer = BytesIO()
        plt.imsave(buffer, imgs[i][:, :, 0], cmap='bone')
        img_str = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode('utf-8')
        img_list.append(img_str)

    return jsonify(img_list)

if __name__ == '__main__':
    app.run(debug=True)
