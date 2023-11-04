import speech_recognition as sr
import os
import csv
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
import pyttsx3
import sys
import matplotlib.pyplot as plt

MODEL_FILENAME = "model.pkl"
DATASET_FILE = 'dataset.csv'  # Define the name of the dataset CSV file
RESET_FILES = False

# Initialize arrays with 20 zeros
model = None

def initialize_dataset(file_name):
    print("Initialzing dataset...")
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([0] * 20)  # Write the first row with 20 columns having zero values

def train_model_with_dataset(dataset_file):
    global model

    X, y = [], []

    with open(dataset_file, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            row = list(map(int, row))
            X.append(row[:-1])
            y.append(row[-1])

    # Train the RandomForestClassifier with the dataset's data
    if model is None:
        model = RandomForestClassifier()
    
    model.fit(X, y)

def append_recent_words_to_dataset(recent_words, dataset_file):
    with open(dataset_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(recent_words)

def create_csv_if_not_exist(file_name):
    if not os.path.exists(file_name):
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            pass  # No header needed

def append_to_csv(file_name, data):
    with open(file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([data])

def check_existing_words(file_name):
    existing_words = set()
    if os.path.exists(file_name):
        with open(file_name, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                existing_words.add(row[0])
    return existing_words

def find_word_index_in_csv(file_name, word):
    index = None
    if os.path.exists(file_name):
        with open(file_name, 'r', newline='') as file:
            reader = csv.reader(file)
            for i, row in enumerate(reader):
                if row[0] == word:
                    index = i
                    break
    return index + 1

# Function to retrieve the word from the CSV based on the index
def retrieve_word_from_index(file_name, index):
    word = ""
    with open(file_name, 'r', newline='') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i == index - 1:
                word = row[0]
                break
    return word

def count_rows_in_csv(file_name):
    count = 0
    if os.path.exists(file_name):
        with open(file_name, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                count += 1
    return count

def speech_recognition_loop(dataset_file):
    global model

    csv_file = 'recognized_words.csv'
    create_csv_if_not_exist(csv_file)
    existing_words = check_existing_words(csv_file)
    recognizer = sr.Recognizer()

    # Start an infinite loop for continuous speech recognition
    while True:
        recent_words = np.zeros(20, dtype=int)  # Array to store the 20 most recent recognized words
        predicted_text = ""
        predicted_words = []
        prediction = 1
        total_rows = count_rows_in_csv(csv_file)

        with sr.Microphone() as source:
            print("Speak something...")
            recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
            audio_data = recognizer.listen(source)

            try:
                # Recognize speech using Google Web Speech API
                text = recognizer.recognize_google(audio_data)
                print(f"You said: {text}")

                if "exit" in text.lower():  # Change "exit" to your desired keyword
                    print("Exiting...")
                    break  # Exit the loop

                words = text.lower().split()  # Split recognized text into individual words

                for word in words:
                    if word not in existing_words:
                        append_to_csv(csv_file, word)
                        existing_words.add(word)
                        # print(f"Added '{word}' to the CSV")

                for i, word in enumerate(words):
                    index = find_word_index_in_csv(csv_file, word)
                    recent_words = np.roll(recent_words, -1)  # Shift elements to the left
                    recent_words[-1] = index  # Place the new index at the last position
                    # print(recent_words)
                    
                    # Append the recent_words to the dataset CSV file
                    append_recent_words_to_dataset(recent_words, DATASET_FILE)

                    if i == len(words) - 1:
                        index = 0
                        recent_words = np.roll(recent_words, -1)  # Shift elements to the left
                        recent_words[-1] = index  # Place the new index at the last position
                        
                        # Append the recent_words to the dataset CSV file
                        append_recent_words_to_dataset(recent_words, DATASET_FILE)

                    # Train the model using the data from the dataset
                    train_model_with_dataset(DATASET_FILE)

                # Remove the first element
                recent_words = recent_words[1:]

                # print(prediction)

                # Predict the 20th element and retrieve words from the CSV
                while prediction > 0:
                    # Make the prediction for the 20th element
                    recent_words_array = np.array(recent_words).reshape(1, -1)
                    prediction = model.predict(recent_words_array)[0]
                    recent_words = np.roll(recent_words, -1)  # Shift elements to the left
                    recent_words[-1] = prediction  # Place the new index at the last position
                    # print(recent_words)
                    predicted_words.append(prediction)

                # print(predicted_words)

                for index in predicted_words:
                    word = retrieve_word_from_index(csv_file, index)
                    predicted_text += f"{word} "
                
                print(f"Predicted Text: {predicted_text}")

                # Text-to-speech using pyttsx3 library
                engine = pyttsx3.init()
                engine.say(predicted_text)
                engine.runAndWait()
                    
            except sr.UnknownValueError:
                print("Sorry, I couldn't understand the audio.")
            except sr.RequestError as e:
                print(f"Error: {e}")

def save_model(model):
    global MODEL_FILENAME

    if model is not None:
        joblib.dump(model, MODEL_FILENAME)
        print(f"Model saved to {MODEL_FILENAME}")

if __name__ == '__main__':
    if RESET_FILES:
        initialize_dataset(DATASET_FILE)
    try:
        speech_recognition_loop(DATASET_FILE)
    except (KeyboardInterrupt, SystemExit):
        print("Exiting due to keyboard interrupt or system exit.")
        save_model(model)
    except Exception as e:
        print(f"Error occurred: {e}")
        save_model(model)
    finally:
        save_model(model)