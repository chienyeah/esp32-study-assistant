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
import threading

# ========== CONFIGURATION ==========
BLYNK_AUTH_TOKEN = "gLNztcV4mo_QL8rNxhRCyRDEZQ1JRO7H"
BLYNK_TEMPLATE_ID = "TMPL6Aa3qBmmY"
BLYNK_URL = "https://blynk.cloud/external/api"

# Model path configuration
MODEL_BASE_NAME = os.path.join(os.path.dirname(__file__), "EIE3127_StudyAssistant")

# MQTT Configuration
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_CLIENT_ID = f"study_assistant_pc_{int(time.time())}"
TOPIC_FOCUS_STATE = "studyassistant/focus/state"
TOPIC_FOCUS_CONFIDENCE = "studyassistant/focus/confidence"
TOPIC_ALERT = "studyassistant/alert/trigger"
TOPIC_SESSION = "studyassistant/session/events"
TOPIC_TEMPERATURE = "studyassistant/env/temperature" 
TOPIC_HUMIDITY = "studyassistant/env/humidity"    
TOPIC_LIGHT_LEVEL = "studyassistant/env/light"    
ENV_DATA_POLL_INTERVAL = 5  # Poll every 5 seconds

# ========== MQTT MANAGER ==========
class MQTTManager:
    def __init__(self):
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, MQTT_CLIENT_ID)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect
        self.connected = False
        self.session_start_time = None
        self.session_active = False
        
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
            time.sleep(1)
            return True
        except Exception as e:
            print(f"MQTT connection error: {e}")
            return False
            
    def publish_focus_state(self, posture, confidence):
        if not self.connected:
            return False
        try:
            # Existing MQTT publish logic
            self.client.publish(TOPIC_FOCUS_STATE, posture)
            self.client.publish(TOPIC_FOCUS_CONFIDENCE, str(confidence))
            print(f"✓ MQTT: Published {posture} with {confidence:.2f} confidence")

            # New: Update Blynk pins
            self.update_blynk_pins(posture)

            return True
        except Exception as e:
            print(f"MQTT publish error: {e}")
            return False
        
    def get_blynk_environmental_data(self):
        """Retrieve V4 (temperature), V5 (humidity), V6 (light) values from Blynk"""
        try:
            # Construct URLs for each virtual pin
            temp_url = f"{BLYNK_URL}/get?token={BLYNK_AUTH_TOKEN}&v4"
            humidity_url = f"{BLYNK_URL}/get?token={BLYNK_AUTH_TOKEN}&v5"
            light_url = f"{BLYNK_URL}/get?token={BLYNK_AUTH_TOKEN}&v6"
            
            # Make HTTP GET requests to Blynk API
            temperature = float(requests.get(temp_url, timeout=5).text)
            humidity = float(requests.get(humidity_url, timeout=5).text)
            light_level = float(requests.get(light_url, timeout=5).text)
            
            print(f"✓ Blynk data retrieved - Temp: {temperature}°C, Humi: {humidity}%, Light: {light_level}%")
            return temperature, humidity, light_level
            
        except Exception as e:
            print(f"✗ Blynk environmental data retrieval error: {e}")
            return None, None, None

    def get_blynk_session_control(self):
        """Retrieve V7 (session control) value from Blynk - returns True for start, False for stop"""
        try:
            # Construct URL for V7 pin
            session_url = f"{BLYNK_URL}/get?token={BLYNK_AUTH_TOKEN}&v7"
            
            # Make HTTP GET request to Blynk API
            session_value = int(requests.get(session_url, timeout=5).text)
            
            # V7 is a switch: 1 = start session, 0 = stop session
            session_state = bool(session_value)
            print(f"✓ Blynk session control retrieved - V7: {session_value} ({'START' if session_state else 'STOP'})")
            return session_state
            
        except Exception as e:
            print(f"✗ Blynk session control retrieval error: {e}")
            return None
        
    def publish_environmental_data(self, temperature, humidity, light_level):
        if not self.connected:
            return False
        try:
            # Publish each environmental parameter to its topic
            self.client.publish(TOPIC_TEMPERATURE, f"{temperature:.1f}")
            self.client.publish(TOPIC_HUMIDITY, f"{humidity:.0f}")
            self.client.publish(TOPIC_LIGHT_LEVEL, f"{light_level:.0f}")
            
            print(f"✓ MQTT: Published env data - Temp: {temperature:.1f}°C, Humi: {humidity}%, Light: {light_level}%")
            return True
        except Exception as e:
            print(f"MQTT environmental publish error: {e}")
            return False

    def handle_session_control(self, session_state):
        """Handle session control based on V7 value from Blynk"""
        if session_state is None:
            return
            
        if session_state and not self.session_active:
            # Start session
            self.start_session()
            self.session_active = True
            print("✓ Session STARTED via Blynk V7")
            self.publish_session_event("session_started", "Study session started via Blynk V7")
            
        elif not session_state and self.session_active:
            # Stop session
            self.end_session()
            self.session_active = False
            print("✓ Session STOPPED via Blynk V7")
            self.publish_session_event("session_ended", "Study session ended via Blynk V7")
        
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
        self.publish_session_event("session_started", "Study session started via Blynk V7")
        print("✓ Study session started")
        
    def end_session(self):
        if self.session_start_time:
            duration = datetime.now() - self.session_start_time
            self.publish_session_event("session_ended", 
                                     f"Study session ended via Blynk V7. Duration: {duration}")
            print(f"✓ Study session ended. Duration: {duration}")
            self.session_start_time = None
            
    def disconnect(self):
        try:
            if self.connected:
                self.client.loop_stop()
                self.client.disconnect()
                print("✓ MQTT disconnected")
        except Exception as e:
            print(f"MQTT disconnect error: {e}")

    def update_blynk_pins(self, state):
        """Update Blynk virtual pins based on detected state (only V0, V1, V2)"""
        if not BLYNK_AUTH_TOKEN:
            print("✗ Blynk auth token missing - skipping update")
            return False

        try:
            # Reset all state pins first (ensure only one state is active)
            requests.get(f"{BLYNK_URL}/update?token={BLYNK_AUTH_TOKEN}&v0=0&v1=0&v2=0")

            # Map detected state to corresponding virtual pin
            if state == "Distracted" or state == "Distracted - Phone":
                # Set V0 (Distracted) to 1
                requests.get(f"{BLYNK_URL}/update?token={BLYNK_AUTH_TOKEN}&v0=1")
                requests.get(f"{BLYNK_URL}/update?token={BLYNK_AUTH_TOKEN}&v1=0")
                requests.get(f"{BLYNK_URL}/update?token={BLYNK_AUTH_TOKEN}&v2=0")
            elif state == "Away":
                # Set V1 (Away) to 1
                requests.get(f"{BLYNK_URL}/update?token={BLYNK_AUTH_TOKEN}&v0=0")
                requests.get(f"{BLYNK_URL}/update?token={BLYNK_AUTH_TOKEN}&v1=1")
                requests.get(f"{BLYNK_URL}/update?token={BLYNK_AUTH_TOKEN}&v2=0")
            elif state == "Focus":
                # Set V2 (Focused) to 1
                requests.get(f"{BLYNK_URL}/update?token={BLYNK_AUTH_TOKEN}&v0=0")
                requests.get(f"{BLYNK_URL}/update?token={BLYNK_AUTH_TOKEN}&v1=0")
                requests.get(f"{BLYNK_URL}/update?token={BLYNK_AUTH_TOKEN}&v2=1")

            print(f"✓ Blynk updated: {state} (V0/V1/V2)")
            return True

        except Exception as e:
            print(f"✗ Blynk update error: {e}")
            return False

# ========== POSE CLASSIFIER ==========
class PoseClassifier:
    def __init__(self):
        self.model = None
        self.class_names = []
        self.is_loaded = False
        self.input_size = (224, 224)
        self.is_multi_input = False
        
    def load_model(self):
        """Load Teachable Machine model with error handling"""
        try:
            print("Loading Teachable Machine Pose Model...")

            model_dir = MODEL_BASE_NAME
            h5_path = os.path.join(model_dir, "keras_model.h5")
            labels_path = os.path.join(model_dir, "labels.txt")
            
            print(f"Looking for model in: {model_dir}")
            print(f"h5 path: {h5_path}")
            print(f"labels path: {labels_path}")

            if not os.path.exists(h5_path):
                print(f"✗ Model file not found at: {h5_path}")
                return False
                
            if not os.path.exists(labels_path):
                print(f"✗ Labels file not found at: {labels_path}")
                return False
            
            # Load labels
            self.class_names = self._load_labels(labels_path)
            print(f"✓ Loaded labels: {self.class_names}")
            
            # Load the h5 model using method 1
            if self._load_h5_model(h5_path):
                self.is_loaded = True
                print("✓ Model loaded successfully")
                return True
            else:
                print("✗ Failed to load h5 model")
                return False
                
        except Exception as e:
            print(f"✗ Model load error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_h5_model(self, h5_path):
        """Load h5 model with DepthwiseConv2D compatibility fix"""
        from tensorflow.keras.layers import DepthwiseConv2D
        import tensorflow.keras.utils as keras_utils

        # Custom DepthwiseConv2D that ignores unsupported 'groups' parameter
        class CustomDepthwiseConv2D(DepthwiseConv2D):
            def __init__(self, *args, **kwargs):
                # Remove 'groups' if present (not supported in all TF versions)
                kwargs.pop('groups', None)
                super().__init__(*args, **kwargs)

        custom_objects = {
            'DepthwiseConv2D': CustomDepthwiseConv2D,
            'DepthwiseConv2D': keras_utils.get_custom_objects().get('DepthwiseConv2D', CustomDepthwiseConv2D)
        }

        self.model = tf.keras.models.load_model(
            h5_path,
            compile=False,
            custom_objects=custom_objects
        )
        self._analyze_model()
        return True
    
    def _analyze_model(self):
        """Analyze the loaded model structure"""
        if self.model is None:
            return
            
        input_shape = self.model.input_shape
        
        if isinstance(input_shape, list):
            self.is_multi_input = True
            if len(input_shape) > 0 and input_shape[0] is not None:
                self.input_size = (input_shape[0][1], input_shape[0][2])
            print(f"✓ Model analysis: Multi-input ({len(input_shape)} inputs), size: {self.input_size}")
        else:
            self.is_multi_input = False
            if input_shape is not None:
                self.input_size = (input_shape[1], input_shape[2])
            print(f"✓ Model analysis: Single input, size: {self.input_size}")
    
    def _load_labels(self, labels_path):
        """Load labels from text file"""
        with open(labels_path, 'r') as f:
            return [line.strip().split(' ', 1)[1] for line in f if line.strip()]
    
    def preprocess_image(self, frame):
        """Preprocess image for model prediction"""
        # Resize to model input size
        img = cv2.resize(frame, self.input_size)
        # Convert to RGB (Teachable Machine uses RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize pixel values to [0, 1]
        img = img / 255.0
        # Add batch dimension
        return np.expand_dims(img, axis=0)
    
    def predict(self, frame):
        """Make prediction on frame"""
        if not self.is_loaded or self.model is None:
            return None, 0.0
            
        try:
            processed = self.preprocess_image(frame)
            
            # Handle multi-input models
            if self.is_multi_input:
                predictions = self.model.predict([processed])[0]
            else:
                predictions = self.model.predict(processed)[0]
                
            max_index = np.argmax(predictions)
            confidence = float(predictions[max_index])
            class_name = self.class_names[max_index] if max_index < len(self.class_names) else "Unknown"
            
            return class_name, confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0

# ========== MAIN APPLICATION ==========
class StudyAssistantApp:
    def __init__(self):
        self.classifier = PoseClassifier()
        self.mqtt = MQTTManager()
        self.running = False
        self.cap = None
        self.last_env_poll_time = 0
        self.last_session_poll_time = 0
        self.blynk_thread = None
        self.stop_blynk_thread = False
        
    def start_blynk_polling(self):
        """Run Blynk polling in a separate thread to avoid lag"""
        while self.running and not self.stop_blynk_thread:
            try:
                # Poll environmental data
                temperature, humidity, light_level = self.mqtt.get_blynk_environmental_data()
                if temperature is not None and humidity is not None and light_level is not None:
                    self.mqtt.publish_environmental_data(temperature, humidity, light_level)
                
                # Poll session control
                session_state = self.mqtt.get_blynk_session_control()
                if session_state is not None:
                    self.mqtt.handle_session_control(session_state)
                    
            except Exception as e:
                print(f"Blynk polling error: {e}")
            
            # Sleep for polling interval
            time.sleep(ENV_DATA_POLL_INTERVAL)

    def setup(self):
        """Setup all components"""
        # Load model
        if not self.classifier.load_model():
            print("✗ Critical error: Could not load model. Exiting.")
            return False
            
        # Connect MQTT
        if not self.mqtt.connect():
            print("⚠️ Warning: Could not connect to MQTT. Continuing with local mode.")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)  # Use 0 for default camera
        if self.cap.isOpened():
            # Reduce resolution to lower CPU load
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # instead of 1280
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # instead of 720
            print("✓ Camera initialized with reduced resolution")
        
        if not self.cap.isOpened():
            print("✗ Could not open camera. Trying other indices...")
            # Try other common camera indices
            for i in range(1, 4):
                self.cap = cv2.VideoCapture(i)
                if self.cap.isOpened():
                    print(f"✓ Camera found at index {i}")
                    break
            if not self.cap.isOpened():
                print("✗ Critical error: No camera available. Exiting.")
                return False
                
        self.running = True
        return True
        
    def poll_environmental_data(self):
        """Poll Blynk for environmental data at regular intervals"""
        current_time = time.time()
        if current_time - self.last_env_poll_time >= ENV_DATA_POLL_INTERVAL:
            temperature, humidity, light_level = self.mqtt.get_blynk_environmental_data()
            if temperature is not None and humidity is not None and light_level is not None:
                self.mqtt.publish_environmental_data(temperature, humidity, light_level)
            self.last_env_poll_time = current_time

    def poll_session_control(self):
        """Poll Blynk for session control (V7) at regular intervals"""
        current_time = time.time()
        if current_time - self.last_session_poll_time >= ENV_DATA_POLL_INTERVAL:
            session_state = self.mqtt.get_blynk_session_control()
            if session_state is not None:
                self.mqtt.handle_session_control(session_state)
            self.last_session_poll_time = current_time
        
    def run(self):
        """Main loop"""
        if not self.running:
            if not self.setup():
                return
                
        print("\nStarting main loop. Press 'q' to quit.")
        
        # START BLYNK POLLING THREAD (ADD THIS)
        self.stop_blynk_thread = False
        self.blynk_thread = threading.Thread(target=self.start_blynk_polling)
        self.blynk_thread.daemon = True
        self.blynk_thread.start()
        
        prediction_counter = 0
        PREDICTION_INTERVAL = 3  # Only predict every 3 frames
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("✗ Could not read frame from camera")
                    time.sleep(1)
                    continue
                
                # Get prediction
                prediction_counter += 1
                if prediction_counter % PREDICTION_INTERVAL == 0:
                    class_name, confidence = self.classifier.predict(frame)
                    prediction_counter = 0

                    if class_name:
                        # Store last state for display
                        self.last_class_name = class_name
                        self.last_confidence = confidence
                        
                        # Display on frame
                        cv2.putText(frame, 
                                    f"{class_name}: {confidence:.2f}", 
                                    (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    1, (0, 255, 0), 2)
                        
                        # Publish to MQTT if connected
                        self.mqtt.publish_focus_state(class_name, confidence)
                        
                        # Trigger alerts for low focus states (only if session is active)
                        if self.mqtt.session_active and ("Distracted" in class_name or class_name == "Away"):
                            if confidence > 0.7:
                                self.mqtt.publish_alert("focus_loss", f"Detected {class_name}", confidence)
                else:
                    if hasattr(self, 'last_class_name'):
                        cv2.putText(frame, f"{self.last_class_name}: {self.last_confidence:.2f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                try:
                    cv2.imshow('Study Assistant - Press Q to Exit', frame)
                except Exception as e:
                    print(f"OpenCV display error: {e}")
                    break

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Q or ESC key
                    self.running = False
                
                time.sleep(0.1)  # ~10 FPS
                
        except KeyboardInterrupt:
            print("\nReceived keyboard interrupt")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        self.running = False
        self.stop_blynk_thread = True  # Stop the thread
        
        # Wait for thread to finish (ADD THIS)
        if self.blynk_thread and self.blynk_thread.is_alive():
            self.blynk_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        if self.mqtt.session_active:
            self.mqtt.end_session()
        self.mqtt.disconnect()
        print("✓ Cleanup complete")

# ========== RUN APPLICATION ==========
if __name__ == "__main__":
    app = StudyAssistantApp()
    app.run()