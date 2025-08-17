# 📚 Numeracy Tutor for Children  

## 📌 Overview  
The *Numeracy Tutor for Children* is an interactive learning project designed to help children practice numeracy skills with the assistance of digit recognition and multimedia support. By integrating the **MNIST handwritten digit dataset**, visualization, and voice features, this project makes learning numbers and basic arithmetic engaging and effective.  

## 🚀 Features  
- 🧮 **Handwritten Digit Recognition** using MNIST dataset  
- 🎥 **Video generation** to display digits visually  
- 🔊 **Voice assistance** to pronounce numbers for children  
- 🎨 Child-friendly design for easy interaction  
- 📊 Practical exercises for number identification and basic math  

## 🛠️ Technologies Used  
- **Python** (Core Programming)  
- **TensorFlow / Keras** (for digit recognition)  
- **OpenCV** (for video handling)  
- **gTTS / pyttsx3** (for voice synthesis)  
- **Matplotlib / NumPy** (for visualization & dataset handling)  
- **Jupyter Notebook** for experimentation   

## Project Structure
mini-project-main/
│── mnist_video/
│   ├── dataset.py          # Dataset preparation  
│   ├── mnist_intro.ipynb   # Notebook for model & demo  
│   ├── mnist_video.py      # Video generation script  
│   ├── video.py            # Video handling utilities  
│   ├── voice.py            # Voice assistant module  
│   ├── 0.png - 9.png       # Sample digit images  
│── mini project report new.pdf   # Project report  
│── README.md               # Project documentation  


## ⚙️ Installation & Setup  
1. Clone or download this repository.  
2. Install the required dependencies:  
   ```bash
   pip install tensorflow keras opencv-python matplotlib numpy gTTS pyttsx3
3. Run the Jupyter Notebook (mnist_intro.ipynb) to explore training and testing.
4. Use mnist_video.py and voice.py to generate digit-based videos with voice output.

## 🎯 Skills Applied  
- Machine Learning (MNIST digit recognition)  
- Python programming and modular coding  
- Data visualization and preprocessing  
- Integration of multimedia (video + audio) for education  
- Problem-solving and project design  

## 📌 Future Enhancements  
- Expand to arithmetic problem-solving with digit recognition  
- Add gamification with rewards and levels  
- Build a web/mobile app version for wider accessibility  
