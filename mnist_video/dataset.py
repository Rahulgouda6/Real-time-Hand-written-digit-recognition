import tensorflow as tf
import numpy as np
import cv2
from tkinter import Tk, Button, Label, StringVar
from PIL import Image, ImageTk
 
# Load MNIST data
def get_mnist_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return x_train, y_train, x_test, y_test

# Train model with MNIST data
def train_model(x_train, y_train, x_test, y_test):
    class MyCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs.get('accuracy') > 0.99:
                print("\nReached 99% accuracy, stopping training!")
                self.model.stop_training = True

    callbacks = MyCallback()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
    return model

# Predict digits in a batch of images
def predict_digits(model, images):
    images = np.array(images) / 255.0
    predictions = model.predict(images)
    return [np.argmax(pred) for pred in predictions]

# Capture image and predict digits
def capture_and_predict(model, result_text, result_label):
    cap = cv2.VideoCapture(0)

    # Start webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        cap.release()
        return

    # Display the captured frame
    cv2.imshow('Captured Image', frame)

    # Wait for the user to press space to capture the image
    print("Press 'Space' to capture the image")
    key = cv2.waitKey(0) & 0xFF
    if key == ord(' '):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digit_images = []
        bounding_boxes = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if 20 < w < 100 and 20 < h < 100:  # Filter for valid digit sizes
                digit_roi = thresh[y:y + h, x:x + w]
                digit_resized = cv2.resize(digit_roi, (28, 28))
                digit_images.append(digit_resized)
                bounding_boxes.append((x, y, w, h))

        if digit_images:
            predictions = predict_digits(model, digit_images)

            # Create a copy of the image to display results
            result_image = frame.copy()
            for (x, y, w, h), digit in zip(bounding_boxes, predictions):
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(result_image, str(digit), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Convert the image to PIL format for display in Tkinter
            result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(result_image)
            tk_image = ImageTk.PhotoImage(pil_image)

            # Display the image in the GUI
            result_label.configure(image=tk_image)
            result_label.image = tk_image
            result_text.set(f"Predicted digits: {', '.join(map(str, predictions))}")
        
        cap.release()
        cv2.destroyAllWindows()

# GUI for capturing and recognizing digits
def create_gui(model):
    def on_capture_button_click():
        capture_and_predict(model, result_text, result_label)

    root = Tk()
    root.title("Digit Recognition")
    root.geometry("600x400")

    label = Label(root, text="Capture an image to recognize digits.", font=("Arial", 14))
    label.pack(pady=10)

    capture_button = Button(root, text="Capture Image", command=on_capture_button_click, font=("Arial", 12))
    capture_button.pack(pady=10)

    result_text = StringVar()
    result_text.set("Predicted digits will appear here.")
    result_label_text = Label(root, textvariable=result_text, font=("Arial", 12))
    result_label_text.pack(pady=10)

    result_label = Label(root)
    result_label.pack(pady=10)

    root.mainloop()

# Main function
def main():
    try:
        model = tf.keras.models.load_model('model.keras')
        print('Loaded saved model.')
    except:
        print("Training new model...")
        x_train, y_train, x_test, y_test = get_mnist_data()
        model = train_model(x_train, y_train, x_test, y_test)
        model.save('model.keras')

    create_gui(model)

if __name__ == '__main__':
    main()
