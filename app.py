import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import time
import speech_recognition as sr
import mediapipe as mp
from textblob import TextBlob
import matplotlib.pyplot as plt
import openai
from ultralytics import YOLO
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

class AdvancedInterviewAnalyzer:
    def __init__(self, openai_key):
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh()
        self.recognizer = sr.Recognizer()
        self.yolo_model = YOLO('yolov8n.pt')
        openai.api_key = openai_key
        
    def analyze_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.1, 4)
        emotions = []
        
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            emotion = self.analyze_emotion(face_roi)
            emotions.append({
                'emotion': emotion,
                'bbox': (x, y, w, h)
            })
        return emotions
    
    def analyze_emotion(self, face_img):
        # Simplified emotion detection
        emotions = ['happy', 'sad', 'neutral', 'angry', 'surprised']
        return np.random.choice(emotions)
        
    def analyze_objects(self, frame):
        results = self.yolo_model(frame)
        detected_objects = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = box.cls[0]
                name = self.yolo_model.names[int(cls)]
                detected_objects.append({
                    'name': name,
                    'confidence': float(conf),
                    'box': (int(x1), int(y1), int(x2), int(y2))
                })
        return detected_objects

    def get_interview_feedback(self, response_text, question):
        try:
            completion = openai.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "system",
                    "content": """You are an expert interview coach. Analyze the candidate's response and provide feedback on:
                    1. Content relevance
                    2. Clarity of communication
                    3. Specific improvements
                    Keep feedback constructive and actionable."""
                }, {
                    "role": "user",
                    "content": f"Question: {question}\nResponse: {response_text}\nProvide feedback:"
                }]
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error getting feedback: {str(e)}"

    def check_interview_environment(self, objects):
        warnings = []
        prohibited_objects = ['cell phone', 'laptop', 'book', 'tv', 'remote']
        distracting_objects = ['person', 'dog', 'cat']
        
        for obj in objects:
            if obj['name'] in prohibited_objects:
                warnings.append(f"⚠️ Warning: {obj['name']} detected - please remove from interview space")
            elif obj['name'] in distracting_objects:
                warnings.append(f"⚠️ Notice: {obj['name']} detected - potential distraction")
        return warnings

    def save_session_data(self, responses, warnings):
        session_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            'responses': responses,
            'warnings': warnings
        }
        
        os.makedirs('session_data', exist_ok=True)
        filename = f"session_data/interview_{session_data['timestamp']}.json"
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, default=str)
        return filename

def initialize_session_state():
    if 'running' not in st.session_state:
        st.session_state.running = True
    if 'responses' not in st.session_state:
        st.session_state.responses = []
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'warnings' not in st.session_state:
        st.session_state.warnings = []
    if 'feedback_history' not in st.session_state:
        st.session_state.feedback_history = []

def main():
    st.set_page_config(page_title="AI Interview Analyzer", layout="wide")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar configuration
    with st.sidebar:
        st.title("Interview Settings")
        openai_key = st.text_input("OpenAI API Key", type="password", 
                                 value=os.getenv('OPENAI_API_KEY', ''))
        
        if not openai_key:
            st.error("Please enter OpenAI API key")
            return
            
        st.divider()
        st.subheader("Detection Settings")
        object_detection = st.toggle("Enable Object Detection", value=True)
        face_detection = st.toggle("Enable Face Detection", value=True)
        
        if 'analyzer' not in st.session_state:
            st.session_state.analyzer = AdvancedInterviewAnalyzer(openai_key)
    
    # Main content
    st.title("AI-Powered Interview Analysis System")
    
    questions = [
        "Tell me about yourself and your background.",
        "What are your greatest strengths?",
        "Where do you see yourself in 5 years?",
        "Describe a challenging situation you've faced at work.",
        "Why are you interested in this position?"
    ]
    
    # Layout
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("Interview Progress")
        progress = st.progress(int((st.session_state.current_question / len(questions)) * 100))
        
        # Question and response section
        if st.session_state.current_question < len(questions):
            current_q = questions[st.session_state.current_question]
            st.write(f"Q{st.session_state.current_question + 1}: {current_q}")
            
            # Voice recording
            if st.button("Record Response", key="record"):
                with st.spinner("Recording... Speak now"):
                    with sr.Microphone() as source:
                        try:
                            audio = st.session_state.analyzer.recognizer.listen(source, timeout=10)
                            response_text = st.session_state.analyzer.recognizer.recognize_google(audio)
                            st.success("Response recorded!")
                            
                            with st.spinner("Analyzing response..."):
                                feedback = st.session_state.analyzer.get_interview_feedback(
                                    response_text, current_q)
                            
                            st.write("AI Feedback:", feedback)
                            
                            st.session_state.responses.append({
                                'question': current_q,
                                'response': response_text,
                                'feedback': feedback,
                                'timestamp': datetime.now()
                            })
                        except sr.WaitTimeoutError:
                            st.error("No speech detected. Please try again.")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            
            if st.button("Next Question", key="next"):
                st.session_state.current_question += 1
                
        else:
            st.success("Interview Complete!")
            
            if st.button("Generate Report", key="report"):
                if st.session_state.responses:
                    report_data = pd.DataFrame(st.session_state.responses)
                    st.write("Interview Analysis Report")
                    st.dataframe(report_data)
                    
                    # Visualizations
                    fig = plt.figure(figsize=(12, 8))
                    gs = fig.add_gridspec(2, 2)
                    
                    # Response length analysis
                    ax1 = fig.add_subplot(gs[0, :])
                    response_lengths = [len(r['response']) for r in st.session_state.responses]
                    ax1.bar(range(len(response_lengths)), response_lengths)
                    ax1.set_title("Response Length by Question")
                    ax1.set_xlabel("Question Number")
                    ax1.set_ylabel("Response Length (characters)")
                    
                    # Warning distribution
                    if st.session_state.warnings:
                        ax2 = fig.add_subplot(gs[1, 0])
                        warning_counts = pd.Series(st.session_state.warnings).value_counts()
                        warning_counts.plot(kind='pie', ax=ax2, title="Warning Distribution")
                    
                    # Save session data
                    filename = st.session_state.analyzer.save_session_data(
                        st.session_state.responses, 
                        st.session_state.warnings
                    )
                    st.success(f"Session data saved to {filename}")
                    
                    st.pyplot(fig)
                else:
                    st.warning("No responses recorded yet!")
    
    with col2:
        st.subheader("Live Analysis")
        video_feed = st.empty()
        warnings_placeholder = st.empty()
        metrics_placeholder = st.empty()
        
        if st.button("End Session", key="end"):
            st.session_state.running = False
            st.experimental_rerun()
        
        video = cv2.VideoCapture(0)
        
        try:
            while st.session_state.running:
                ret, frame = video.read()
                if ret:
                    # Object detection
                    if object_detection:
                        objects = st.session_state.analyzer.analyze_objects(frame)
                        
                        for obj in objects:
                            x1, y1, x2, y2 = obj['box']
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, 
                                      f"{obj['name']} {obj['confidence']:.2f}", 
                                      (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.6, 
                                      (0, 255, 0), 
                                      2)
                    
                    # Face detection
                    if face_detection:
                        emotions = st.session_state.analyzer.analyze_face(frame)
                        for emotion_data in emotions:
                            x, y, w, h = emotion_data['bbox']
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                            cv2.putText(frame, 
                                      emotion_data['emotion'], 
                                      (x, y-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.6, 
                                      (255, 0, 0), 
                                      2)
                    
                    # Check environment
                    if object_detection:
                        warnings = st.session_state.analyzer.check_interview_environment(objects)
                        if warnings:
                            st.session_state.warnings.extend(warnings)
                            warnings_placeholder.warning("\n".join(warnings))
                    
                    # Display frame
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_feed.image(frame)
                    
                    time.sleep(0.1)
        
        finally:
            video.release()

if __name__ == "__main__":
    main()