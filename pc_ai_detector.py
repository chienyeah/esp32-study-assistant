import cv2
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import os
import paho.mqtt.client as mqtt
from datetime import datetime
import threading

# ========== CONFIGURATION ==========
MODEL_BASE_NAME = os.path.join(os.path.dirname(__file__), "models")

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_CLIENT_ID = f"study_assistant_pc_{int(time.time())}"
TOPIC_FOCUS_STATE = "studyassistant/focus/state"
TOPIC_FOCUS_CONFIDENCE = "studyassistant/focus/confidence"
TOPIC_ALERT = "studyassistant/alert/trigger"
TOPIC_SESSION = "studyassistant/session/events"

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
            # Publish focus state and confidence to MQTT
            self.client.publish(TOPIC_FOCUS_STATE, posture)
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
        self.input_size = (224, 224)
        self.is_multi_input = False
        
    def load_model(self):
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
        # Resize to model input size
        img = cv2.resize(frame, self.input_size)
        # Convert to RGB (Teachable Machine uses RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize pixel values to [0, 1]
        img = img / 255.0
        # Add batch dimension
        return np.expand_dims(img, axis=0)
    
    def predict(self, frame):
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

# ========== AI DETECTION THREAD ==========
class AIDetectionThread(threading.Thread):
    def __init__(self, classifier, mqtt_manager):
        super().__init__()
        self.classifier = classifier
        self.mqtt = mqtt_manager
        self.running = True
        self.cap = None
        self.last_class_name = "Unknown"
        self.last_confidence = 0.0
        
    def setup_camera(self):
        """Initialize camera for AI detection"""
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print("✓ AI Detection: Camera initialized")
            return True
        else:
            print("✗ AI Detection: Could not open camera")
            return False
    
    def run(self):
        """Main AI detection loop"""
        print("Starting AI Detection Thread")
        if not self.setup_camera():
            return
        prediction_counter = 0
        PREDICTION_INTERVAL = 3
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("✗ AI Detection: Could not read frame from camera")
                    time.sleep(1)
                    continue
                # Get prediction
                prediction_counter += 1
                if prediction_counter % PREDICTION_INTERVAL == 0:
                    class_name, confidence = self.classifier.predict(frame)
                    prediction_counter = 0

                    if class_name:
                        # Store last state
                        self.last_class_name = class_name
                        self.last_confidence = confidence
                        
                        # Display on frame
                        cv2.putText(frame, 
                                    f"{class_name}: {confidence:.2f}", 
                                    (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    1, (0, 255, 0), 2)
                        
                        # Send detection data to MQTT publisher via queue
                        detection_data = {
                            'type': 'detection',
                            'posture': class_name,
                            'confidence': confidence,
                            'timestamp': time.time()
                        }
                        
                        # Store for MQTT publisher thread to pick up
                        self.last_detection_data = detection_data
                        
                        # Trigger alerts for low focus states (only if session is active)
                        if self.mqtt.session_active and ("Distracted" in class_name or class_name == "Away"):
                            if confidence > 0.7:
                                self.mqtt.publish_alert("focus_loss", f"Detected {class_name}", confidence)
                else:
                    # Display last known state
                    cv2.putText(frame, 
                               f"{self.last_class_name}: {self.last_confidence:.2f}", 
                               (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                try:
                    cv2.imshow('Study Assistant - AI Detection (Press Q to Exit)', frame)
                except Exception as e:
                    print(f"AI Detection: OpenCV display error: {e}")
                    break

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    self.running = False
                
                time.sleep(0.1)
                
        except Exception as e:
            print(f"AI Detection Thread error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up AI detection resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("✓ AI Detection Thread cleaned up")
    
    def stop(self):
        """Stop the AI detection thread"""
        self.running = False

# ========== MQTT PUBLISHER THREAD ==========
class MQTTPublisherThread(threading.Thread):
    def __init__(self, mqtt_manager, ai_detection_thread):
        super().__init__()
        self.mqtt = mqtt_manager
        self.ai_thread = ai_detection_thread
        self.running = True
        self.detection_poll_interval = 1  # Check for new detections every 1 second
        self.last_detection_poll_time = 0
        
    def run(self):
        """Main MQTT publishing loop"""
        print("Starting MQTT Publisher Thread")
        
        try:
            while self.running:
                current_time = time.time()
                
                # Check for new detection data from AI thread
                if current_time - self.last_detection_poll_time >= self.detection_poll_interval:
                    if hasattr(self.ai_thread, 'last_detection_data'):
                        detection_data = self.ai_thread.last_detection_data
                        if detection_data['type'] == 'detection':
                            # Publish detection data to MQTT
                            self.mqtt.publish_focus_state(
                                detection_data['posture'], 
                                detection_data['confidence']
                            )
                    self.last_detection_poll_time = current_time
                
                time.sleep(0.1) 
                
        except Exception as e:
            print(f"MQTT Publisher Thread error: {e}")
        finally:
            print("✓ MQTT Publisher Thread stopped")
    
    def stop(self):
        """Stop the MQTT publisher thread"""
        self.running = False

# ========== MAIN APPLICATION ==========
class StudyAssistantApp:
    def __init__(self):
        self.classifier = PoseClassifier()
        self.mqtt = MQTTManager()
        self.ai_detection_thread = None
        self.mqtt_publisher_thread = None
        self.running = False
        
    def setup(self):
        """Setup all components"""
        if not self.classifier.load_model():
            print("✗ Critical error: Could not load model. Exiting.")
            return False
            
        # Connect MQTT
        if not self.mqtt.connect():
            print("Could not connect to MQTT. Continuing with local mode.")
        
        self.running = True
        return True
        
    def run(self):
        """Start both threads and manage application"""
        if not self.setup():
            return
            
        print("\nStarting Study Assistant with Double Threads")
        print("   - Thread 1: AI Detection (Camera processing)")
        print("   - Thread 2: MQTT Publisher (Data publishing)")
        print("   Press 'q' in the camera window to exit\n")
        
        try:
            # Create and start threads
            self.ai_detection_thread = AIDetectionThread(self.classifier, self.mqtt)
            self.mqtt_publisher_thread = MQTTPublisherThread(self.mqtt, self.ai_detection_thread)
            
            # Start threads
            self.ai_detection_thread.start()
            self.mqtt_publisher_thread.start()
            
            self.ai_detection_thread.join()
            
        except KeyboardInterrupt:
            print("\nReceived keyboard interrupt")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up all resources"""
        print("\nCleaning up...")
        self.running = False
        
        # Stop threads
        if self.ai_detection_thread:
            self.ai_detection_thread.stop()
            self.ai_detection_thread.join(timeout=2.0)
            
        if self.mqtt_publisher_thread:
            self.mqtt_publisher_thread.stop()
            self.mqtt_publisher_thread.join(timeout=2.0)

        if self.mqtt.session_active:
            self.mqtt.end_session()

        self.mqtt.disconnect()
        print("✓ Cleanup complete")

# ========== RUN APPLICATION ==========
if __name__ == "__main__":
    app = StudyAssistantApp()
    app.run()