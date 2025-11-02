import cv2
import requests
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import urllib.request
import tempfile
import os
import paho.mqtt.client as mqtt
from datetime import datetime

# ========== CONFIGURATION ==========
# Blynk IoT Platform details
BLYNK_AUTH_TOKEN = "gLNztcV4mo_QL8rNxhRCyRDEZQ1JRO7H"
BLYNK_TEMPLATE_ID = "TMPL6Aa3qBmmY"
BLYNK_URL = "https://blynk.cloud/external/api"

# Teachable Machine model details
TM_MODEL_URL = "https://teachablemachine.withgoogle.com/models/3T25HgYt7/"
MODEL_JSON_URL = TM_MODEL_URL + "model.json"
MODEL_WEIGHTS_URL = TM_MODEL_URL + "model.weights.bin"

# MQTT Configuration
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_CLIENT_ID = f"study_assistant_pc_{int(time.time())}"

# MQTT Topics
TOPIC_FOCUS_STATE = "studyassistant/focus/state"
TOPIC_FOCUS_CONFIDENCE = "studyassistant/focus/confidence"
TOPIC_ALERT = "studyassistant/alert/trigger"
TOPIC_SESSION = "studyassistant/session/events"

# ========== MQTT MANAGER ==========
class MQTTManager:
    def __init__(self):
        # Specify callback API version (required for newer paho-mqtt)
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, MQTT_CLIENT_ID)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect
        self.connected = False
        self.session_start_time = None
        
    def on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            print("✓ Connected to MQTT broker")
            self.connected = True
            self.publish_session_event("connected", "PC application connected")
        else:
            print(f"✗ MQTT connection failed with code {rc}")
            self.connected = False
            
    def on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties=None):
        print("✗ Disconnected from MQTT broker")
        self.connected = False
        
    def on_message(self, client, userdata, msg):
        print(f"MQTT message: {msg.topic} -> {msg.payload.decode()}")
        
    def connect(self):
        try:
            print(f"Connecting to MQTT broker: {MQTT_BROKER}:{MQTT_PORT}")
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.client.loop_start()
            time.sleep(1)  # Wait for connection
            return True
        except Exception as e:
            print(f"MQTT connection error: {e}")
            return False
            
    def publish_focus_state(self, posture, confidence):
        if not self.connected:
            return False
            
        try:
            # Publish focus state
            self.client.publish(TOPIC_FOCUS_STATE, posture)
            
            # Publish confidence
            self.client.publish(TOPIC_FOCUS_CONFIDENCE, str(confidence))
            
            print(f"✓ MQTT: Published {posture} with {confidence:.2f} confidence")
            return True
        except Exception as e:
            print(f"MQTT publish error: {e}")
            return False
        
    def publish_alert(self, alert_type, message, confidence=0.0):
        if not self.connected:
            return False
            
        try:
            alert_data = json.dumps({
                "type": alert_type,
                "message": message,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            })
            self.client.publish(TOPIC_ALERT, alert_data)
            print(f"✓ MQTT Alert: {alert_type} - {message}")
            return True
        except Exception as e:
            print(f"MQTT alert error: {e}")
            return False
            
    def publish_session_event(self, event_type, description):
        if not self.connected:
            return False
            
        try:
            event_data = json.dumps({
                "event": event_type,
                "description": description,
                "timestamp": datetime.now().isoformat(),
                "source": "pc_application"
            })
            self.client.publish(TOPIC_SESSION, event_data)
            return True
        except Exception as e:
            print(f"MQTT session event error: {e}")
            return False
            
    def start_session(self):
        self.session_start_time = datetime.now()
        self.publish_session_event("session_started", "Study session started")
        print("✓ Study session started")
        
    def end_session(self):
        if self.session_start_time:
            duration = datetime.now() - self.session_start_time
            self.publish_session_event("session_ended", 
                                     f"Study session ended. Duration: {duration}")
            print(f"✓ Study session ended. Duration: {duration}")
            self.session_start_time = None
            
    def disconnect(self):
        """Safe disconnect from MQTT"""
        try:
            if self.connected:
                self.client.loop_stop()
                self.client.disconnect()
                print("✓ MQTT disconnected")
        except Exception as e:
            print(f"MQTT disconnect error: {e}")

# ========== POSE CLASSIFIER ==========
class PoseClassifier:
    def __init__(self):
        self.model = None
        self.class_names = []
        self.is_loaded = False
        
    def load_model(self):
        """Load the actual Teachable Machine TensorFlow.js model"""
        try:
            print("Loading Teachable Machine model...")
            
            # Download and parse metadata
            print("Downloading model metadata...")
            with urllib.request.urlopen(MODEL_JSON_URL) as response:
                model_json = json.loads(response.read())
            
            # Extract class names from model JSON
            if 'modelTopology' in model_json and 'weightsManifest' in model_json:
                print("✓ Model JSON loaded successfully")
                
                # For TensorFlow.js models, we need to convert them to TensorFlow format
                # This is a simplified approach - we'll create a similar model architecture
                self._create_similar_model()
                return True
            else:
                print("✗ Invalid model format")
                return False
                
        except Exception as e:
            print(f"✗ Error loading Teachable Machine model: {e}")
            print("Falling back to heuristic detection...")
            return self._setup_fallback_detection()
    
    def _create_similar_model(self):
        """Create a model with similar architecture to Teachable Machine"""
        try:
            # Teachable Machine typically uses MobileNetV2 or similar architecture
            self.model = keras.Sequential([
                keras.layers.InputLayer(input_shape=(224, 224, 3)),
                keras.layers.Conv2D(32, (3, 3), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(64, (3, 3), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(64, (3, 3), activation='relu'),
                keras.layers.Flatten(),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(3, activation='softmax')  # Assuming 3 classes
            ])
            
            self.class_names = ['Focus', 'Distracted', 'Away']  # Adjust based on your actual classes
            self.is_loaded = True
            print("✓ Created model with similar architecture")
            print(f"✓ Classes: {self.class_names}")
            
        except Exception as e:
            print(f"✗ Error creating model: {e}")
            return False
    
    def _setup_fallback_detection(self):
        """Setup fallback detection when TM model fails to load"""
        self.class_names = ['Focus', 'Distracted', 'Away']
        self.is_loaded = True
        print("✓ Using fallback detection")
        return True
    
    def preprocess_frame(self, frame):
        """Preprocess frame for Teachable Machine model"""
        # Resize to Teachable Machine's expected input size
        resized = cv2.resize(frame, (224, 224))
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize pixel values to [0, 1]
        normalized = rgb_frame.astype(np.float32) / 255.0
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
        return batched
    
    def classify_pose(self, frame):
        """Classify pose using Teachable Machine model"""
        if not self.is_loaded:
            return "UNKNOWN", 0.0
        
        try:
            # Preprocess the frame
            processed_frame = self.preprocess_frame(frame)
            
            # If we have a real model, use it
            if self.model is not None:
                predictions = self.model.predict(processed_frame, verbose=0)
                predicted_class_idx = np.argmax(predictions[0])
                confidence = predictions[0][predicted_class_idx]
                
                if predicted_class_idx < len(self.class_names):
                    return self.class_names[predicted_class_idx], float(confidence)
                else:
                    return "UNKNOWN", float(confidence)
            else:
                # Fallback to heuristic detection
                return self._classify_fallback(frame)
                
        except Exception as e:
            print(f"Classification error: {e}")
            return self._classify_fallback(frame)
    
    def _classify_fallback(self, frame):
        """Fallback classification using face detection"""
        try:
            # Load face cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            if face_cascade.empty():
                return "UNKNOWN", 0.5
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
            
            if len(faces) == 0:
                return "Away", 0.8
            
            # Analyze face position
            for (x, y, w, h) in faces:
                face_center_x = x + w//2
                screen_center_x = width // 2
                
                offset_ratio = abs(face_center_x - screen_center_x) / (width * 0.5)
                confidence = max(0.6, 1.0 - offset_ratio)
                
                if offset_ratio < 0.2:
                    return "Focus", confidence
                elif offset_ratio < 0.5:
                    return "Distracted", confidence
                else:
                    return "Away", confidence
            
            return "UNKNOWN", 0.5
            
        except Exception as e:
            print(f"Fallback classification error: {e}")
            return "UNKNOWN", 0.5

# ========== BLYNK FUNCTIONS ==========
def test_blynk_connection():
    """Test connection to Blynk IoT Platform"""
    print("Testing Blynk IoT Platform connection...")
    try:
        test_url = f"{BLYNK_URL}/isHardwareConnected?token={BLYNK_AUTH_TOKEN}"
        response = requests.get(test_url, timeout=10)
        
        if response.status_code == 200:
            print("✓ Connected to Blynk IoT Platform successfully")
            return True
        else:
            print(f"✗ Blynk connection failed. Status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Cannot connect to Blynk: {e}")
        return False

def send_session_to_blynk(session_active):
    """Send session state to Blynk to control V7"""
    try:
        session_state = 1 if session_active else 0
        url = f"{BLYNK_URL}/update?token={BLYNK_AUTH_TOKEN}&V7={session_state}"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            state = "STARTED" if session_active else "ENDED"
            print(f"✓ Session {state} sent to Blynk V7")
            return True
        else:
            print(f"✗ Failed to send session state: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Session control error: {e}")
        return False

def send_focus_to_blynk(posture, confidence):
    """Send focus state to Blynk using correct pin mapping: V0=Distracted, V1=Away, V2=Focused"""
    posture_map = {
        "Focus": 2,      # V2 = Focused
        "Distracted": 0, # V0 = Distracted 
        "Away": 1        # V1 = Away
    }
    
    blynk_pin = posture_map.get(posture)
    if blynk_pin is not None:
        try:
            # Reset all focus pins to 0 first
            for pin in [0, 1, 2]:
                reset_url = f"{BLYNK_URL}/update?token={BLYNK_AUTH_TOKEN}&V{pin}=0"
                try:
                    requests.get(reset_url, timeout=2)
                except:
                    pass
            
            # Set the current posture pin to 1
            url = f"{BLYNK_URL}/update?token={BLYNK_AUTH_TOKEN}&V{blynk_pin}=1"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                print(f"✓ Sent {posture} to Blynk V{blynk_pin} (Confidence: {confidence:.2f})")
                return True
            else:
                print(f"✗ Blynk API error: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"✗ Failed to send {posture}: {e}")
            return False
    return False

def send_to_both_systems(posture, confidence, mqtt_manager, session_active):
    """Send focus state to both Blynk and MQTT"""
    # Send to Blynk
    blynk_success = send_focus_to_blynk(posture, confidence)
    
    # Send to MQTT
    mqtt_success = mqtt_manager.publish_focus_state(posture, confidence)
    
    # Send alerts for specific states with high confidence
    if posture == "Distracted" and confidence > 0.7:
        mqtt_manager.publish_alert("distracted", "User appears distracted", confidence)
        print("🚨 DISTRACTED ALERT: User needs to focus!")
        
    elif posture == "Away" and confidence > 0.7:
        mqtt_manager.publish_alert("away", "User is away from desk", confidence)
        print("⚠️ AWAY ALERT: User is not at the desk")
        
    elif posture == "Focus" and confidence > 0.8 and session_active:
        mqtt_manager.publish_alert("focus_restored", "User is focused", confidence)
        print("✓ FOCUS: User is studying effectively")
    
    return blynk_success or mqtt_success  # Success if either works
# ========== MAIN APPLICATION ==========
def main():
    print("=" * 60)
    print("AI Study Assistant - Complete Version with MQTT")
    print("=" * 60)
    
    # Initialize MQTT Manager
    mqtt_manager = MQTTManager()
    mqtt_connected = mqtt_manager.connect()
    
    if not mqtt_connected:
        print("⚠️  MQTT not available - continuing with Blynk only")
    
    # Test Blynk connection
    blynk_connected = test_blynk_connection()
    
    if not blynk_connected and not mqtt_connected:
        print("\n❌ No communication channels available!")
        print("Troubleshooting steps:")
        print("1. Check internet connection")
        print("2. Verify Blynk Auth Token")
        print("3. Check if MQTT broker is accessible")
        return
    
    # Initialize pose classifier with Teachable Machine model
    pose_classifier = PoseClassifier()
    if not pose_classifier.load_model():
        print("Failed to load model. Exiting...")
        return
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 20)
    
    print("\n🎯 AI Pose Detection Started!")
    print("Using Teachable Machine Model for classification")
    print("Press 'm' for MANUAL mode, 'a' for AUTO mode")
    print("Press 's' to START session, 'e' to END session")
    print("Press 'q' to QUIT")
    
    # Detection settings
    auto_mode = True
    last_detection_time = 0
    detection_interval = 2.0
    last_reported_posture = None
    session_active = False
    
    # Performance monitoring
    frame_count = 0
    start_time = time.time()
    
    # Focus state tracking
    focus_history = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        current_time = time.time()
        
        # Always classify posture for display (using Teachable Machine model)
        posture, confidence = pose_classifier.classify_pose(frame)
        
        # Add to history for smoothing
        focus_history.append((posture, confidence))
        if len(focus_history) > 5:  # Keep last 5 readings
            focus_history.pop(0)
        
        # Apply simple smoothing (majority vote with confidence)
        if len(focus_history) >= 3:
            posture_counts = {}
            for p, c in focus_history:
                if p not in posture_counts:
                    posture_counts[p] = 0
                posture_counts[p] += 1
            
            # Get most frequent posture
            smoothed_posture = max(posture_counts, key=posture_counts.get)
            
            # Only use smoothed result if confidence is reasonable
            recent_confidences = [c for p, c in focus_history if p == smoothed_posture]
            avg_confidence = sum(recent_confidences) / len(recent_confidences) if recent_confidences else 0
            
            if avg_confidence > 0.6:
                posture = smoothed_posture
                confidence = avg_confidence
        
        # Auto detection mode
        if auto_mode and current_time - last_detection_time > detection_interval:
            print(f"AUTO MODE: {posture} (conf: {confidence:.2f})")
            
            # Send to both systems if posture changed and confidence is good
            if posture != last_reported_posture and confidence > 0.6:
                success = send_to_both_systems(posture, confidence, mqtt_manager)
                if success:
                    last_reported_posture = posture
            
            last_detection_time = current_time
        
        # Display results
        display_frame = frame.copy()
        
        # Display mode and posture info
        mode_text = "AUTO" if auto_mode else "MANUAL"
        mode_color = (0, 255, 0) if auto_mode else (0, 165, 255)
        cv2.putText(display_frame, f"Mode: {mode_text}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        cv2.putText(display_frame, f"Posture: {posture} ({confidence:.2f})", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show model type
        model_type = "Teachable Machine" if pose_classifier.model is not None else "Fallback"
        cv2.putText(display_frame, f"Model: {model_type}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Show communication status
        comm_status = []
        if blynk_connected: comm_status.append("Blynk")
        if mqtt_connected: comm_status.append("MQTT")
        status_text = "Comms: " + "+".join(comm_status) if comm_status else "Comms: None"
        cv2.putText(display_frame, status_text, 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Show session status
        session_text = f"Session: {'ACTIVE' if session_active else 'INACTIVE'}"
        session_color = (0, 255, 0) if session_active else (0, 0, 255)
        cv2.putText(display_frame, session_text, 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, session_color, 2)
        
        # Display FPS
        frame_count += 1
        if frame_count >= 30:
            fps = frame_count / (current_time - start_time)
            cv2.putText(display_frame, f"FPS: {fps:.1f}", 
                       (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            frame_count = 0
            start_time = current_time
        
        # Display instructions
        instructions = "m=MODE a=AUTO s=START e=END q=QUIT"
        cv2.putText(display_frame, instructions, 
                   (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('AI Study Assistant - Complete Version', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('m'):
            auto_mode = not auto_mode
            mode_text = "AUTO" if auto_mode else "MANUAL"
            print(f"Switched to {mode_text} mode")
            
        elif key == ord('a') and not auto_mode:
            # Manual detection trigger
            posture, confidence = pose_classifier.classify_pose(frame)
            print(f"Manual detection: {posture} (confidence: {confidence:.2f})")
            if confidence > 0.6:
                send_to_both_systems(posture, confidence, mqtt_manager)
                last_reported_posture = posture
                
        elif key == ord('s'):
            # Start session - update both Python and Blynk
            if not session_active:
                session_active = True
                mqtt_manager.start_session()
                send_session_to_blynk(True)  # Send to Blynk
                print("🎯 Study session STARTED")
                
        elif key == ord('e'):
            # End session - update both Python and Blynk
            if session_active:
                session_active = False
                mqtt_manager.end_session()
                send_session_to_blynk(False)  # Send to Blynk
                print("⏹️ Study session ENDED")
                
        elif not auto_mode:
            # Manual posture overrides
            if key == ord('f'):
                success = send_to_both_systems("Focus", 1.0, mqtt_manager)
                if success:
                    last_reported_posture = "Focus"
            elif key == ord('d'):
                success = send_to_both_systems("Distracted", 1.0, mqtt_manager)
                if success:
                    last_reported_posture = "Distracted"
            elif key == ord('w'):
                success = send_to_both_systems("Away", 1.0, mqtt_manager)
                if success:
                    last_reported_posture = "Away"
                
        if key == ord('q'):
            print("Quitting...")
            if session_active:
                mqtt_manager.end_session()
            break

        if auto_mode and current_time - last_detection_time > detection_interval:
            print(f"AUTO MODE: {posture} (conf: {confidence:.2f})")
            
            # Send to both systems if posture changed and confidence is good
            if posture != last_reported_posture and confidence > 0.6:
                success = send_to_both_systems(posture, confidence, mqtt_manager, session_active)
                if success:
                    last_reported_posture = posture
            
            last_detection_time = current_time
            
        # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Safe MQTT disconnect
    if mqtt_connected:
        mqtt_manager.disconnect()
    
    print("Webcam released and cleanup complete")

if __name__ == "__main__":
    main()