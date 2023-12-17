# Speech Recognition and Prediction App

This is a simple speech recognition and prediction application written in Python. The application listens to your speech, recognizes the spoken words, and predicts the next word based on the patterns learned from previous input.

## Requirements

- Python 3.x
- Required Python packages (`pip install -r requirements.txt`):
  - `speech_recognition`
  - `numpy`
  - `scikit-learn`
  - `joblib`
  - `pyttsx3`
  - `matplotlib`

## Usage

1. Run the application:

   ```bash
   python main.py
   ```

2. Speak something when prompted. The application will recognize your speech, predict the next word, and read it back to you.

3. To exit the application, say "exit."

## Features

- The application continuously learns from your input and updates its prediction model.
- Predicts the next word in real-time based on the learned patterns.
- Uses a RandomForestClassifier for word prediction.
- Provides a simple command ("exit") to exit the application.

## Additional Notes

- The dataset is stored in a CSV file (`dataset.csv`), and recognized words are stored in another CSV file (`recognized_words.csv`).
- The model is saved as `model.pkl` and is updated as the program runs.
- Adjust the "exit" keyword in the code to your desired keyword for exiting the application.

Feel free to explore and customize the application for your specific use case!
