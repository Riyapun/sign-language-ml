#  Sign Language to Text and Voice Using Machine Learning

A machine learning-based system that translates hand sign gestures into readable text and audio output. This project is aimed at helping individuals with hearing or speech disabilities communicate more effectively through gesture recognition using a webcam.

2. ##Short Description:
   
A machine learning-based system that captures hand gestures using a webcam, recognizes sign language through a trained model, and translates it into text or speech to aid communication for individuals with hearing or speech difficulties.

## Project Summary

This is a final-year BCA Data Science project that focuses on real-time sign language detection using self-collected hand gesture data. The system captures hand gestures, 
identifies them using a trained ML model, and converts the recognized signs into both **text** and **spoken audio**.

 Objectives

- Detect hand gestures using computer vision.
- Recognize and classify signs from custom training data.
- Convert detected gestures into readable text.
- Use text-to-speech to output the translated gesture as audio.

##  Technologies Used

| Tool/Library       | Purpose                        |
|--------------------|--------------------------------|
| **Python**         | Core programming language      |
| **OpenCV**         | Hand gesture image capture     |
| **TensorFlow/Keras** | Model training & prediction |
| **Pyttsx3**        | Text-to-speech conversion      |
| **NumPy, Pandas**  | Data handling & preprocessing  |
| **Matplotlib**     | Data visualization             |

Folder Structure :

bash
Copy
Edit

 📁 project 1.0/
 
├── 📁 Data/               # Images collected from webcam

├── 📁 venv/               # Virtual environment (not uploaded)

├── 📄 datacollection.py   # Script to collect training data

├── 📄 test.py             # Script to test trained model

├── 📄 testing.py          # Additional functionality

├── 📄 README.md           # Project documentation



 How to Run the Project

1. **Clone this repository**
   ```bash
   git clone https://github.com/yourusername/sign-language-ml
   cd sign-language-ml

2-##Install required libraries

bash
Copy
Edit
pip install -r requirements.txt

3-##To Collect the data
  
    python datacollection.py

4-##To run real-time prediction
   
      python test.py

##Dataset
Data was collected manually using a webcam.

Each gesture class has multiple sample images.
You can improve accuracy by collecting more samples and retraining the model.

##Author
 
 Reshma Pun
🎓 Final Year BCA (with Data Science specialization)
🧠 Passionate about AI, ML, and Human-Computer Interaction
📍 Project created using 100% self-collected data.

 ##License
    
This project is for academic and learning purposes. You are free to use, modify, and build upon this work with proper credit.
