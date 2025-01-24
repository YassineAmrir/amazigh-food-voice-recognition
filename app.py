from flask import Flask, render_template, request, jsonify
import os
import librosa
import joblib
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Dossier pour les fichiers téléchargés
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


model = load_model("audio_classification_model.h5")
label_encoder = joblib.load("label_encoder.pkl")

#model = load_model("audio_classification_model.h5")
# Charger le modèle et le scaler
mode = joblib.load('best_random_forest_model.joblib')
scaler = joblib.load('scaler.joblib')
encoder = joblib.load('encoder.joblib')  # Assurez-vous d'avoir un encodeur pour inverser la transformation

# Map des aliments vers leurs ingrédients
food_to_ingredients = {
    "amlou": ["بالطريقة التقليدية، نأخذ اللوز محمر ثم يتم طحنه في أزرݣ مع زيت أرݣان"],
    "ourimken": ["الحمص - الفول - الشعرية - الروز - اللوبية - الدشيشة - الݣديديوضع في الطنجرة مع المياه حتى يطهى ثم نضيف أخيرا القليل من الطحين"],
    "tagala": ["مياه مغلية الدشيشة توضع في المياه نضيف الملح حسب الذوق تحرك حتى تطهى تزين بأملو وأرݣان"],
}

def extract_features(file_path, n_mfcc=13, max_pad_len=100):
    try:
        audio, sample_rate = librosa.load(file_path, sr=16000)  # Load audio
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)  # Extract MFCC
        mfcc = librosa.util.normalize(mfcc)  # Normalize MFCC features
        pad_width = max(0, max_pad_len - mfcc.shape[1])  # Pad or truncate to fixed length
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        mfcc = mfcc[:, :max_pad_len]  # Ensure fixed size
        return mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def process_audio_with_model(audio_file_path):
    features = extract_features(audio_file_path)
    if features is None:
        return "unknown"

    # Reshape features to match the input shape of the model
    features = np.expand_dims(features, axis=-1)  # Add channel dimension
    features = np.expand_dims(features, axis=0)   # Add batch dimension

    # Predict with the TensorFlow model
    predictions = model.predict(features)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Map the prediction index to the corresponding class name
    label_mapping = {0: "amlou", 1: "ourimken", 2: "tagala"}
    return label_mapping.get(predicted_class, "unknown")


def post_process_prediction(prediction):
    corrections = {"Amlou": "amlou", "Ourkimen": "ourimken", "Tagala": "tagala"}
    return corrections.get(prediction, prediction)

def get_ingredients(food_name):
    return food_to_ingredients.get(food_name, [])

@app.route('/')
def index():
    """
    Afficher la page principale.
    """
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """
    Gérer l'upload du fichier audio et la prédiction.
    """
    if 'audio' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['audio']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file:
        # Sauvegarder le fichier audio téléchargé
        audio_file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(audio_file_path)
    
        # Processus de prédiction sur l'audio
        food_name = process_audio_with_model(audio_file_path)
        
        if food_name == "unknown" or food_name not in food_to_ingredients:
            return render_template('index.html', error_message="Je ne reconnais pas ce plat. Veuillez réessayer.")
        
        # Obtenir les ingrédients pour le nom de l'aliment prédit
        ingredients = get_ingredients(food_name)
        
        # Trouver les images associées aux ingrédients
        ingredient_images = []
        for ingredient in ingredients:
            image_path = os.path.join('static/images', f"{food_name.lower()}.jpg")
            if os.path.exists(image_path):
                ingredient_images.append(image_path)

        return render_template('index.html', food_name=food_name, ingredients=ingredients, images=ingredient_images)

if __name__ == '__main__':
    app.run(debug=True)
