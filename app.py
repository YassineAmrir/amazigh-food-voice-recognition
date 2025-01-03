from flask import Flask, render_template, request, jsonify
import os
import speech_recognition as sr
import tensorflow as tf  # Or use your specific model type
import requests
import joblib

app = Flask(__name__)

# Folder to save uploaded audio files
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your pre-trained audio recognition model

model = joblib.load('best_random_forest_model.joblib')  # Replace with your model's path

# Simple food-to-ingredients mapping (you can replace this with an API call)
food_to_ingredients = {
    "amlou": ["  بالطريقة التقليدية، نأخذ اللوز محمر ثم يتم طحنه في أزرݣ مع زيت أرݣان"],
    "ourimken": ["الحمص - الفول - الشعرية - الروز - اللوبية - الدشيشة - الݣديديوضع في الطنجرة مع المياه حتى يطهى ثم نضيف أخيرا القليل من الطحين"],
    "tagala": ["مياه مغلية الدشيشة توضع في المياه  نضيف الملح حسب الذوق تحرك حتى تطهى تزين بأملو و أرݣان"],
}

def process_audio_with_model(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)
    food_name = recognizer.recognize_google(audio)  # Recognize the food name from audio
    food_name = post_process_prediction(food_name)
    return food_name
def post_process_prediction(prediction):
    corrections = {"amlo": "amlou", "Oregon": "ourimken", "Douglas": "tagala"}
    return corrections.get(prediction, prediction)


def get_ingredients(food_name):
    # Lookup ingredients for the recognized food name
    return food_to_ingredients.get(food_name, [])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'audio' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['audio']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file:
        # Save the uploaded audio file
        audio_file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(audio_file_path)
        
        # Process the audio file with your model to recognize food name
        food_name = process_audio_with_model(audio_file_path)
        
        # Get the ingredients for the recognized food
        ingredients = get_ingredients(food_name)

        # Load images based on ingredients (if available in the static/images folder)
        ingredient_images = []
        for ingredient in ingredients:
            image_path = os.path.join('static/images', f"{food_name.lower()}.jpg")
            if os.path.exists(image_path):
                ingredient_images.append(image_path)

        return render_template('index.html', food_name=food_name, ingredients=ingredients, images=ingredient_images)

if __name__ == '__main__':
    app.run(debug=True)
