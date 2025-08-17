import tensorflow as tf
import numpy as np
import cv2
import pyttsx3  # Text-to-speech library

# Helper functions and global variables
startInference = False
threshold = 50
last_prediction = None
prediction_start_time = None
DISPLAY_THRESHOLD = 1 # seconds
DISPLAY_DURATION = 5  # seconds
is_displaying = False
clear_display_time = None

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Set speaking speed
engine.setProperty('volume', 1.0)  # Set volume (0.0 to 1.0)

# Left mouse click handler
def ifClicked(event, x, y, flags, params):
    global startInference
    if event == cv2.EVENT_LBUTTONDOWN:
        startInference = not startInference

# Threshold slider handler
def on_threshold(x):
    global threshold
    threshold = x

# Function to predict digit using the model
def predict(model, img):
    imgs = np.array([img])
    res = model.predict(imgs)
    index = np.argmax(res)
    return str(index)

# Function to announce the prediction
def announce_number(number):
    engine.say(f"The detected number is {number}")
    engine.runAndWait()

# OpenCV display loop
def start_cv(model):
    global threshold, last_prediction, prediction_start_time, is_displaying, clear_display_time
    cap = cv2.VideoCapture(0)

    # Increase the frame size to 1280x960
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    cv2.namedWindow('background')
    cv2.setMouseCallback('background', ifClicked)
    cv2.createTrackbar('threshold', 'background', 150, 255, on_threshold)

    background = np.zeros((960, 1280), np.uint8)

    while True:
        ret, frame = cap.read()

        if startInference:
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply threshold
            _, thr = cv2.threshold(grayFrame, threshold, 255, cv2.THRESH_BINARY_INV)

            # Get central image
            resizedFrame = thr[480-75:480+75, 640-75:640+75]
            background[480-75:480+75, 640-75:640+75] = resizedFrame

            # Resize for inference
            iconImg = cv2.resize(resizedFrame, (28, 28))

            # Get the current time
            current_time = cv2.getTickCount() / cv2.getTickFrequency()

            if is_displaying:
                # Clear display after showing for DISPLAY_DURATION seconds
                if current_time - clear_display_time >= DISPLAY_DURATION:
                    is_displaying = False
                    last_prediction = None
                    prediction_start_time = None
                    background.fill(0)  # Clear the background
            else:
                # Predict using the model
                current_prediction = predict(model, iconImg)

                # Check if prediction is stable
                if current_prediction == last_prediction:
                    if prediction_start_time is None:
                        prediction_start_time = current_time

                    # Check duration
                    if current_time - prediction_start_time >= DISPLAY_THRESHOLD:
                        # Display the prediction
                        cv2.putText(background, current_prediction, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                        is_displaying = True
                        clear_display_time = current_time

                        # Announce the number
                        announce_number(current_prediction)
                else:
                    # Reset tracking if prediction changes
                    last_prediction = current_prediction
                    prediction_start_time = None

            # Draw bounding box
            cv2.rectangle(background, (640-80, 480-80), (640+80, 480+80), (255, 255, 255), thickness=3)

            # Display frame
            cv2.imshow('background', background)
        else:
            # Display normal video
            cv2.imshow('background', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    model = None
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

    print("Starting OpenCV...")
    start_cv(model)

if __name__ == '__main__':
    main()