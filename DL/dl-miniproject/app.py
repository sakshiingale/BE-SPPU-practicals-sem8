import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import seaborn as sns
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Face Emotion Recognition",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define CSS for styling
st.markdown("""
<style>
    .title {
        font-size: 42px;
        font-weight: bold;
        color: #4169E1;
        text-align: center;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 24px;
        color: #6495ED;
        text-align: center;
        margin-bottom: 30px;
    }
    .emotion-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        text-align: center;
        font-weight: bold;
    }
    .stButton > button {
        width: 100%;
        background-color: #4169E1;
        color: white;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #1E90FF;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='title'>Face Emotion Recognition</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Deep Learning Mini Project</div>", unsafe_allow_html=True)

# Create sidebar
st.sidebar.title("Settings")

# Function to load emotion dict
@st.cache_data
def load_emotion_dict():
    # Default emotion mapping
    default_emotion_dict = {
        0: 'Angry',
        1: 'Disgust',
        2: 'Fear',
        3: 'Happy',
        4: 'Sad',
        5: 'Surprise',
        6: 'Neutral'
    }
    
    # Try to load custom emotion mapping if it exists
    try:
        if os.path.exists("emotion_map.txt"):
            custom_emotion_dict = {}
            with open("emotion_map.txt", "r") as f:
                for line in f:
                    if ":" in line:
                        idx, emotion = line.strip().split(":", 1)
                        custom_emotion_dict[int(idx.strip())] = emotion.strip()
            
            if custom_emotion_dict:  # If we successfully loaded custom mappings
                return custom_emotion_dict
    except Exception as e:
        st.warning(f"Could not load custom emotion mapping: {e}. Using default mapping.")
    
    return default_emotion_dict

# Load emotion dictionary
emotion_dict = load_emotion_dict()

# Define emotion colors
emotion_colors = {
    'Angry': '#FF5252',      # Red
    'Disgust': '#9C27B0',    # Purple
    'Fear': '#FFC107',       # Amber
    'Happy': '#4CAF50',      # Green
    'Sad': '#2196F3',        # Blue
    'Surprise': '#FF9800',   # Orange
    'Neutral': '#9E9E9E',    # Grey
    # Add mappings for any directory-based emotion names
    'angry': '#FF5252',
    'disgust': '#9C27B0',
    'fear': '#FFC107',
    'happy': '#4CAF50',
    'sad': '#2196F3',
    'surprise': '#FF9800',
    'neutral': '#9E9E9E'
}

# Function to get color for any emotion
def get_emotion_color(emotion):
    return emotion_colors.get(emotion, '#9E9E9E')  # Default to grey if not found

# Function to load model
@st.cache_resource
@st.cache_resource
def load_model():
    """
    Load the emotion recognition model from JSON and weights files.
    Implements multiple fallback strategies for loading weights.
    """
    try:
        # Check if model files exist in the current directory
        if not os.path.exists("model.json"):
            st.error("Model JSON file not found. Please make sure 'model.json' is in the current directory.")
            return None
            
        # Try to load model architecture from JSON
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        
        # Try different approaches to load weights
        weight_loaded = False
        
        # List of possible weight file names to try
        weight_files = ["model.h5", "model.weights.h5", "weights.h5"]
        
        # Check if any of the standard weight files exist
        for weight_file in weight_files:
            if os.path.exists(weight_file):
                try:
                    # Try standard loading
                    model.load_weights(weight_file)
                    st.info(f"Successfully loaded weights from {weight_file}")
                    weight_loaded = True
                    break
                except ValueError:
                    try:
                        # Try with by_name parameter
                        model.load_weights(weight_file, by_name=True)
                        st.info(f"Successfully loaded weights from {weight_file} using by_name=True")
                        weight_loaded = True
                        break
                    except Exception as e:
                        st.warning(f"Failed to load {weight_file}: {e}")
                        continue
        
        # If none of the standard files worked, look for any .h5 file
        if not weight_loaded:
            h5_files = [f for f in os.listdir('.') if f.endswith('.h5')]
            if h5_files:
                try:
                    model.load_weights(h5_files[0])
                    st.info(f"Successfully loaded weights from {h5_files[0]}")
                    weight_loaded = True
                except Exception as e:
                    st.error(f"Error loading weights from {h5_files[0]}: {e}")
            
        # Check if weights were loaded successfully
        if not weight_loaded:
            # As a last resort, try loading a full model if it exists
            if os.path.exists("full_model.h5"):
                try:
                    from tensorflow.keras.models import load_model as keras_load_model
                    model = keras_load_model("full_model.h5")
                    st.info("Successfully loaded full model from full_model.h5")
                    weight_loaded = True
                except Exception as e:
                    st.error(f"Error loading full model: {e}")
            
        if not weight_loaded:
            st.error("Failed to load model weights. Please ensure weight files are correctly formatted and accessible.")
            return None
        
        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to load face cascade classifier
@st.cache_resource
def load_face_cascade():
    # Load haarcascade classifier
    try:
        # Use OpenCV's included Haar cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return face_cascade
    except Exception as e:
        st.error(f"Error loading face cascade: {e}")
        return None

# Function to preprocess the face for prediction
def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (48, 48))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = np.reshape(face_img, [1, 48, 48, 1]) / 255.0
    return face_img

# Function to predict emotion
def predict_emotion(face_img, model):
    processed_img = preprocess_face(face_img)
    emotion_prediction = model.predict(processed_img)
    return emotion_prediction

# Function to draw face rectangle with emotion label
def draw_face_info(frame, x, y, w, h, emotion_label, confidence):
    # Calculate color based on emotion
    color = get_emotion_color(emotion_label)
    # Convert hex color to BGR (reverse order)
    b = int(color[5:7], 16)
    g = int(color[3:5], 16)
    r = int(color[1:3], 16)
    color_bgr = (b, g, r)
    
    # Draw rectangle around face
    cv2.rectangle(frame, (x, y), (x+w, y+h), color_bgr, 2)
    
    # Draw filled rectangle for text background
    cv2.rectangle(frame, (x, y-40), (x+w, y), color_bgr, -1)
    
    # Add emotion text with confidence
    text = f"{emotion_label} ({confidence:.1%})"
    cv2.putText(frame, text, (x+5, y-15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

# Function to generate emotion distribution chart
def generate_emotion_chart(emotion_counts):
    if not emotion_counts:
        return None
    
    df = pd.DataFrame(list(emotion_counts.items()), columns=['Emotion', 'Count'])
    df = df.sort_values('Count', ascending=False)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Create a list of colors for each bar
    colors = [get_emotion_color(emotion) for emotion in df['Emotion']]
    
    bars = sns.barplot(x='Emotion', y='Count', data=df, palette=colors, ax=ax)
    
    # Add count numbers on top of bars
    for i, p in enumerate(bars.patches):
        bars.annotate(f'{p.get_height():.0f}', 
                      (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha='center', va='center', 
                      xytext=(0, 9), 
                      textcoords='offset points')
    
    plt.title('Emotion Distribution', fontsize=16)
    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    
    return img

# Function to get emoji for emotion
def get_emotion_emoji(emotion):
    emoji_dict = {
        'Angry': '😠',
        'Disgust': '🤢',
        'Fear': '😨',
        'Happy': '😊',
        'Sad': '😢',
        'Surprise': '😲',
        'Neutral': '😐',
        # Add lowercase versions for directory-based emotions
        'angry': '😠',
        'disgust': '🤢',
        'fear': '😨',
        'happy': '😊',
        'sad': '😢',
        'surprise': '😲',
        'neutral': '😐'
    }
    return emoji_dict.get(emotion, '')

# Function to display webcam and process video
def process_webcam(model, face_cascade):
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Cannot open webcam. Please check your camera connection.")
        return
    
    # Create placeholder for video frame
    frame_placeholder = st.empty()
    
    # Create placeholder for emotion text
    emotion_text = st.empty()
    
    # Create placeholder for emotion chart
    chart_placeholder = st.empty()
    
    # Initialize emotion tracking
    emotion_counts = {emotion: 0 for emotion in emotion_dict.values()}
    dominant_emotion = list(emotion_dict.values())[0]  # Default to first emotion
    
    # Initialize time tracking for chart update
    last_chart_update = time.time()
    chart_update_interval = 1.0  # Update chart every 1 second
    
    stop_button_pressed = st.button("Stop Camera")
    
    while not stop_button_pressed:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            st.error("Failed to capture image from camera.")
            break
        
        # Convert to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Process each face
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            
            # Predict emotion
            if face_roi.size > 0:
                emotion_probabilities = predict_emotion(face_roi, model)[0]
                emotion_idx = np.argmax(emotion_probabilities)
                emotion_label = emotion_dict[emotion_idx]
                confidence = emotion_probabilities[emotion_idx]
                
                # Update emotion counts
                emotion_counts[emotion_label] += 1
                
                # Determine dominant emotion
                dominant_emotion = max(emotion_counts, key=emotion_counts.get)
                
                # Draw face information on frame
                frame = draw_face_info(frame, x, y, w, h, emotion_label, confidence)
        
        # Convert frame to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display frame
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        
        # Display current dominant emotion
        emoji = get_emotion_emoji(dominant_emotion)
        emotion_color = get_emotion_color(dominant_emotion)
        emotion_text.markdown(f"<div class='emotion-box' style='background-color: {emotion_color}; color: white;'>Dominant Emotion: {dominant_emotion} {emoji}</div>", unsafe_allow_html=True)
        
        # Update chart periodically
        current_time = time.time()
        if current_time - last_chart_update > chart_update_interval:
            chart_img = generate_emotion_chart(emotion_counts)
            if chart_img:
                chart_placeholder.image(chart_img, use_column_width=True)
            last_chart_update = current_time
        
        # Check if stop button was pressed
        if stop_button_pressed:
            break
            
        # Add small delay
        time.sleep(0.05)
    
    # Release resources
    cap.release()
    
    # Final emotion distribution chart
    final_chart = generate_emotion_chart(emotion_counts)
    if final_chart:
        st.subheader("Final Emotion Distribution")
        st.image(final_chart, use_column_width=True)
    
    st.success("Camera stopped")

# Function to process uploaded image
def process_image(uploaded_file, model, face_cascade):
    # Read image
    image = Image.open(uploaded_file)
    image_array = np.array(image.convert('RGB'))
    
    # Convert to OpenCV format
    frame = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        st.warning("No faces detected in the image.")
        st.image(image, caption="Original Image", use_column_width=True)
        return
    
    # Process each face
    results = []
    for (x, y, w, h) in faces:
        # Extract face region
        face_roi = frame[y:y+h, x:x+w]
        
        # Predict emotion
        emotion_probabilities = predict_emotion(face_roi, model)[0]
        emotion_idx = np.argmax(emotion_probabilities)
        emotion_label = emotion_dict[emotion_idx]
        confidence = emotion_probabilities[emotion_idx]
        
        # Draw face information on frame
        frame = draw_face_info(frame, x, y, w, h, emotion_label, confidence)
        
        # Store result
        results.append({
            'emotion': emotion_label,
            'confidence': confidence,
            'probabilities': emotion_probabilities
        })
    
    # Convert frame to RGB for display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Display result
    st.image(frame_rgb, caption="Detected Emotions", use_column_width=True)
    
    # Display detailed results
    st.subheader("Detailed Analysis")
    
    for i, result in enumerate(results):
        with st.expander(f"Face {i+1} - {result['emotion']} {get_emotion_emoji(result['emotion'])}"):
            # Create DataFrame for probabilities
            prob_df = pd.DataFrame({
                'Emotion': list(emotion_dict.values()),
                'Probability': result['probabilities'] * 100
            })
            
            # Sort by probability
            prob_df = prob_df.sort_values('Probability', ascending=False)
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = [get_emotion_color(emotion) for emotion in prob_df['Emotion']]
            bars = sns.barplot(x='Emotion', y='Probability', data=prob_df, palette=colors, ax=ax)
            
            # Add percentage labels
            for i, p in enumerate(bars.patches):
                bars.annotate(f'{p.get_height():.1f}%', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='center', 
                            xytext=(0, 5), 
                            textcoords='offset points')
            
            plt.title('Emotion Probabilities', fontsize=16)
            plt.xlabel('Emotion', fontsize=12)
            plt.ylabel('Probability (%)', fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig)

# About section
def show_about():
    st.subheader("About this application")
    st.write("""
    This face emotion recognition application uses a Convolutional Neural Network (CNN) to detect emotions in real-time. 
    The model was trained to recognize seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.
    
    ### How it works:
    1. The application captures video from your webcam or processes uploaded images
    2. Detects faces using the Haar Cascade classifier
    3. For each detected face, the CNN model predicts the emotion
    4. Results are displayed in real-time with visual indicators
    
    ### Model Architecture:
    The CNN model consists of multiple convolutional layers, batch normalization, max pooling, and dropout layers, 
    followed by fully connected layers for classification.
    """)
    
    # Display model architecture
    st.subheader("Model Architecture")
    architecture = """
    Sequential model with:
    - Conv2D layers (starting with 64 filters)
    - BatchNormalization
    - MaxPooling2D
    - Dropout (0.25)
    - Multiple convolutional blocks with increasing filters
    - Fully connected layers (1024 units)
    - Output layer with 7 units (for 7 emotions)
    """
    st.code(architecture)

# Main function
def main():
    # Load model
    model = load_model()
    
    # Load face cascade
    face_cascade = load_face_cascade()
    
    if model is None or face_cascade is None:
        st.error("Failed to load required models. Cannot continue.")
        return
    
    # Display emotion mapping
    with st.sidebar.expander("Emotion Classes"):
        for idx, emotion in emotion_dict.items():
            emoji = get_emotion_emoji(emotion)
            st.write(f"{idx}: {emotion} {emoji}")
    
    # Sidebar navigation
    app_mode = st.sidebar.selectbox("Choose the app mode", 
                                    ["About", "Image Detection", "Webcam Detection"])
    
    # App mode selection
    if app_mode == "About":
        show_about()
    
    elif app_mode == "Webcam Detection":
        st.subheader("Webcam Face Emotion Detection")
        st.write("Click the button below to start the webcam and detect emotions in real-time.")
        
        start_button = st.button("Start Camera")
        
        if start_button:
            process_webcam(model, face_cascade)
    
    elif app_mode == "Image Detection":
        st.subheader("Image Face Emotion Detection")
        st.write("Upload an image to detect emotions.")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            process_image(uploaded_file, model, face_cascade)

if __name__ == "__main__":
    main()