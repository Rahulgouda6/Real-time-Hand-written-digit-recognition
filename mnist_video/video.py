import tensorflow as tf
import numpy as np
import cv2
import pyttsx3
import gradio as gr

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Set speaking speed
engine.setProperty('volume', 1.0)  # Set volume (0.0 to 1.0)

# Function to announce the prediction
def announce_number(number):
    engine.say(f"The detected number is {number}")
    engine.runAndWait()

# Function to predict digit using the model
def predict(model, img):
    imgs = np.array([img])
    res = model.predict(imgs)
    index = np.argmax(res)
    return str(index)

# Main webcam processing function
def webcam_digit_detector(video_frame, threshold=150):
    global model

    # Convert Gradio frame to OpenCV-compatible format
    frame = cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply threshold
    _, thr = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY_INV)

    # Get central region for processing
    center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
    cropped_frame = thr[center_y-75:center_y+75, center_x-75:center_x+75]

    # Resize to 28x28 for model input
    if cropped_frame.shape[:2] == (150, 150):
        resized_frame = cv2.resize(cropped_frame, (28, 28))

        # Predict using the trained model
        detected_digit = predict(model, resized_frame)

        # Announce the detected digit
        announce_number(detected_digit)

        # Draw bounding box and prediction on the original frame
        cv2.rectangle(frame, (center_x-80, center_y-80), (center_x+80, center_y+80), (0, 255, 0), 2)
        cv2.putText(frame, f"Detected: {detected_digit}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return frame[:, :, ::-1], detected_digit  # Convert to RGB for Gradio display
    else:
        return frame[:, :, ::-1], "No Prediction"  # No digit predicted

# Load or train the model
def load_or_train_model():
    try:
        model = tf.keras.models.load_model('model.keras')
        print('Loaded saved model.')
    except:
        from tensorflow.keras.datasets import mnist

        print("Getting MNIST data...")
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        print("Training model...")
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=10)

        print("Saving model...")
        model.save('model.keras')

    return model

# Load the model
model = load_or_train_model()

# Gradio interface
import gradio as gr

import cv2
import numpy as np
import tensorflow as tf

# Load your pre-trained TensorFlow model
model = tf.keras.models.load_model("model.keras")  # Replace with your model's path

def process_video(video_frame):
    """
    Process the video frame to detect and classify handwritten digits.
    Steps:
    1. Convert the frame to grayscale.
    2. Apply thresholding to isolate the digit.
    3. Resize the region of interest to 28x28 pixels.
    4. Use the trained model to predict the digit.
    5. Display the prediction on the frame.
    """
    # Convert the video frame (BGR format) to grayscale
    gray = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary image
    _, thresholded = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Detect the largest contour (assuming it's the handwritten digit)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Ensure the bounding box is reasonably sized
        if w > 20 and h > 20:
            # Extract the region of interest (ROI)
            roi = thresholded[y:y+h, x:x+w]

            # Resize the ROI to 28x28 pixels
            roi_resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

            # Normalize the pixel values
            roi_normalized = roi_resized / 255.0

            # Expand dimensions to match the input shape of the model
            roi_input = np.expand_dims(roi_normalized, axis=(0, -1))

            # Predict the digit
            prediction = model.predict(roi_input)
            digit = np.argmax(prediction)

            # Display the predicted digit on the video frame
            cv2.putText(video_frame, f"Digit: {digit}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Return the annotated video frame
    return video_frame


video_input = gr.Video(label="Webcam Input")
import gradio as gr
import cv2
import numpy as np

def process_video(video_frame):
    # Process the video frame here (e.g., detect digits or any other task)
    # Returning the processed frame
    return video_frame

# Create the Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        webcam = gr.Video(label="Webcam", streaming=True)
        output = gr.Image(label="Processed Frame")

    webcam.stream(process_video, outputs=output)

demo.launch()


# Launch the app
if __name__ == "__main__":
    app.launch()
