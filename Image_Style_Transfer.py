# Import necessary libraries
import cv2
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, Response
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

# Load pre-trained style transfer model
model = tf.keras.models.load_model('your_model_path')

# Function to apply style transfer
def apply_style(input_image, style_index):
    # Preprocess the image for the model
    input_image = cv2.resize(input_image, (256, 256))
    input_image = input_image / 255.0
    input_image = np.expand_dims(input_image, axis=0)

    # Apply style transfer using the loaded model
    stylized_image = model.predict([input_image, style_index])[0]

    # Post-process the stylized image
    stylized_image = (stylized_image * 255).astype(np.uint8)

    return stylized_image

# Function to capture webcam feed
def generate_frames(style_index):
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()

        if not success:
            break

        stylized_frame = apply_style(frame, style_index)

        # Combine original and stylized frames
        output_frame = np.hstack((frame, stylized_frame))

        _, buffer = cv2.imencode('.jpg', output_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index_complex.html')

@socketio.on('style_change')
def handle_style_change(style_index):
    # Start a new thread to generate frames with the selected style
    socketio.start_background_task(target=send_frames, style_index=style_index)

def send_frames(style_index):
    for frame in generate_frames(style_index):
        socketio.emit('video_feed', {'frame': frame})

if __name__ == '__main__':
    socketio.run(app, debug=True)
