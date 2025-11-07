/* 
  Study Assistant - ESP32 Environment Monitor
  Updated for correct Blynk pin mapping:
  V0=Distracted, V1=Away, V2=Focused
*/

#include <Arduino.h>
#include "secrets.h"
#include <BlynkSimpleEsp32.h>
#include <WiFi.h>
#include "env_monitor.h"

// --------- Pin Definitions ----------
#define PIN_DHT       17
#define PIN_LDR_AO    7
#define PIN_I2C_SDA   4
#define PIN_I2C_SCL   5

// RGB LEDs
#define LEDA_R        13  // Environment status
#define LEDA_Y        14
#define LEDA_G        21
#define LEDB_R        10  // Focus status
#define LEDB_Y        11
#define LEDB_G        12
#define PIN_BUZZER    18

// Blynk Virtual Pins
#define V0 0  // Distracted
#define V1 1  // Away  
#define V2 2  // Focused
#define V4 4  // Temperature
#define V5 5  // Humidity
#define V6 6  // Light
#define V7 7  // Session control
#define V8 8  // Focus ratio

// Create the monitor
EnvMonitor env(PIN_DHT, PIN_LDR_AO,
               LEDA_R, LEDA_Y, LEDA_G,
               LEDB_R, LEDB_Y, LEDB_G, PIN_BUZZER,
               PIN_I2C_SDA, PIN_I2C_SCL);

// Session tracking
bool sessionActive = false;
unsigned long sessionStartTime = 0;
unsigned long focusTime = 0;
bool blynkConnected = false;

// Focus state tracking
String currentFocusState = "UNKNOWN";
unsigned long lastFocusUpdate = 0;
const unsigned long FOCUS_TIMEOUT = 10000;

void setup() {
  Serial.begin(115200);
  delay(2000);
  
  Serial.println("\n=== Study Environment Monitor Starting ===");
  Serial.println("Blynk Pins: V0=Distracted, V1=Away, V2=Focused");

  // Init environment monitor first
  env.begin();
  Serial.println("Environment monitor initialized");

  // Initialize WiFi and Blynk
  setupWiFi();
  setupBlynk();
  
  Serial.println("=== Setup Complete ===");
  Serial.println("Commands: STATUS, START, STOP, FOCUSED, DISTRACTED, AWAY, SYNC, RESET");
}

void setupWiFi() {
  Serial.println("Setting up WiFi...");
  Serial.printf("SSID: %s\n", WIFI_SSID);
  
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.persistent(true);
  
  Serial.print("Connecting to WiFi");
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  
  unsigned long wifiStart = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - wifiStart < 15000) {
    delay(500);
    Serial.print(".");
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✓ WiFi Connected!");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\n✗ WiFi Failed - Continuing offline");
  }
}

void setupBlynk() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Skipping Blynk - No WiFi");
    return;
  }
  
  Serial.print("Connecting to Blynk...");
  
  Blynk.config(BLYNK_AUTH_TOKEN);
  
  unsigned long blynkStart = millis();
  while (!Blynk.connect() && millis() - blynkStart < 10000) {
    delay(500);
    Serial.print(".");
  }
  
  if (Blynk.connected()) {
    blynkConnected = true;
    Serial.println("✓ Blynk Connected!");
    
    // Sync virtual pins
    Blynk.syncVirtual(V0, V1, V2, V7);
    
    // Reset all focus state switches to OFF
    Blynk.virtualWrite(V0, 0); // Distracted OFF
    Blynk.virtualWrite(V1, 0); // Away OFF
    Blynk.virtualWrite(V2, 0); // Focused OFF
    Blynk.virtualWrite(V7, 0); // Session inactive
    Blynk.virtualWrite(V8, 0); // Focus ratio 0%
    
  } else {
    blynkConnected = false;
    Serial.println("✗ Blynk Failed - Continuing offline");
  }
}

void loop() {
  static unsigned long lastReconnectAttempt = 0;
  static unsigned long lastStatusCheck = 0;
  static unsigned long lastBlynkUpdate = 0;
  
  // Main environment monitoring
  env.tick();
  
  // Handle Blynk connection
  handleBlynkConnection();
  
  // Update session metrics
  updateSessionMetrics();

  updateFocusTimeTracking();
  
  // Send environment data to Blynk every 10 seconds
  if (millis() - lastBlynkUpdate > 10000 && blynkConnected) {
    sendEnvironmentToBlynk();
    lastBlynkUpdate = millis();
  }
  
  // Check for focus state timeout
  checkFocusTimeout();
  
  // Print status every 30 seconds
  if (millis() - lastStatusCheck > 30000) {
    lastStatusCheck = millis();
    printQuickStatus();
  }
  
  // Handle serial commands
  if (Serial.available()) {
    handleSerialCommand();
  }
  
  delay(100);
}

void handleBlynkConnection() {
  static unsigned long lastReconnectAttempt = 0;
  
  if (blynkConnected && !Blynk.connected()) {
    if (millis() - lastReconnectAttempt > 30000) {
      Serial.println("Blynk disconnected - attempting reconnect...");
      blynkConnected = Blynk.connect();
      if (blynkConnected) {
        Serial.println("✓ Blynk Reconnected!");
        Blynk.syncVirtual(V0, V1, V2, V7);
      }
      lastReconnectAttempt = millis();
    }
  }
  
  if (blynkConnected) {
    Blynk.run();
  }
}

void updateSessionMetrics() {
  if (!sessionActive) return;
  
  static unsigned long lastFocusUpdate = 0;
  if (millis() - lastFocusUpdate > 30000 && blynkConnected) {
    unsigned long totalTime = (millis() - sessionStartTime) / 1000;
    if (totalTime > 0) {
      float focusRatio = (float)focusTime / totalTime;
      Blynk.virtualWrite(V8, focusRatio * 100);
    }
    lastFocusUpdate = millis();
  }
}

void sendEnvironmentToBlynk() {
  if (!blynkConnected) return;
  
  auto readings = env.get();
  
  if (!isnan(readings.tempC)) {
    Blynk.virtualWrite(V4, readings.tempC);
  }
  if (readings.humi >= 0) {
    Blynk.virtualWrite(V5, readings.humi);
  }
  Blynk.virtualWrite(V6, readings.lightPct);
}

void updateBlynkFocusState(String state) {
  if (!blynkConnected) return;
  
  // Turn OFF all focus state switches first
  Blynk.virtualWrite(V0, 0); // Distracted OFF
  Blynk.virtualWrite(V1, 0); // Away OFF
  Blynk.virtualWrite(V2, 0); // Focused OFF
  
  // Turn ON the correct switch based on state
  if (state == "DISTRACTED") {
    Blynk.virtualWrite(V0, 1); // Distracted ON
    Serial.println("Blynk: DISTRACTED (V0=1)");
  } else if (state == "AWAY") {
    Blynk.virtualWrite(V1, 1); // Away ON
    Serial.println("Blynk: AWAY (V1=1)");
  } else if (state == "FOCUSED") {
    Blynk.virtualWrite(V2, 1); // Focused ON
    Serial.println("Blynk: FOCUSED (V2=1)");
  }
}

void checkFocusTimeout() {
  if (millis() - lastFocusUpdate > FOCUS_TIMEOUT) {
    if (currentFocusState != "NO_DATA") {
      currentFocusState = "NO_DATA";
      Serial.println("Focus state: NO DATA (timeout)");
      if (blynkConnected) {
        // Turn off all states when no data
        Blynk.virtualWrite(V0, 0);
        Blynk.virtualWrite(V1, 0);
        Blynk.virtualWrite(V2, 0);
      }
    }
  }
}

// Blynk handler for V0 - Distracted Switch
BLYNK_WRITE(V0) {
  int state = param.asInt();
  if (state == 1) {
    env.setDetectionMode(EnvMonitor::DET_DISTRACTED);
    currentFocusState = "DISTRACTED";
    lastFocusUpdate = millis();
    Serial.println("Blynk V0: DISTRACTED");
  }
}

// Blynk handler for V1 - Away Switch
BLYNK_WRITE(V1) {
  int state = param.asInt();
  if (state == 1) {
    env.setDetectionMode(EnvMonitor::DET_AWAY);
    currentFocusState = "AWAY";
    lastFocusUpdate = millis();
    Serial.println("Blynk V1: AWAY");
  }
}

// Blynk handler for V2 - Focused Switch
BLYNK_WRITE(V2) {
  int state = param.asInt();
  if (state == 1) {
    env.setDetectionMode(EnvMonitor::DET_FOCUSED);
    currentFocusState = "FOCUSED";
    lastFocusUpdate = millis();
    Serial.println("Blynk V2: FOCUSED");
    // REMOVED the focusTime += 30 line to prevent double-counting
  }
}

void updateFocusTimeTracking() {
  static unsigned long lastFocusCheck = 0;
  static String previousFocusState = "UNKNOWN";
  
  if (millis() - lastFocusCheck > 1000) { // Check every second
    if (sessionActive) {
      // Only count time when actually in FOCUSED state
      if (currentFocusState == "FOCUSED") {
        focusTime++;
        Serial.printf("Focus time: %lu seconds\n", focusTime);
      }
    }
    lastFocusCheck = millis();
    previousFocusState = currentFocusState;
  }
}

// Blynk handler for V7 - Session Control
BLYNK_WRITE(V7) {
  int sessionState = param.asInt();
  
  if (sessionState == 1 && !sessionActive) {
    sessionStartTime = millis();
    focusTime = 0;
    sessionActive = true;
    Blynk.virtualWrite(V8, 0);
    Serial.println("🎯 Session STARTED from Blynk");
  } 
  else if (sessionState == 0 && sessionActive) {
    sessionActive = false;
    unsigned long totalTime = (millis() - sessionStartTime) / 1000;
    float focusRatio = totalTime > 0 ? (float)focusTime / totalTime : 0;
    Blynk.virtualWrite(V8, focusRatio * 100);
    Serial.printf("⏹️ Session ENDED - Focus Ratio: %.1f%%\n", focusRatio * 100);
  }
}

void handleSerialCommand() {
  String command = Serial.readStringUntil('\n');
  command.trim();
  
  using DM = EnvMonitor::DetMode;
  
  if (command == "STATUS") {
    printStatus();
  } else if (command == "START") {
    if (blynkConnected) {
      Blynk.virtualWrite(V7, 1);
    } else {
      sessionStartTime = millis();
      focusTime = 0;
      sessionActive = true;
      Serial.println("Session STARTED (manual)");
    }
  } else if (command == "STOP") {
    if (blynkConnected) {
      Blynk.virtualWrite(V7, 0);
    } else {
      unsigned long totalTime = (millis() - sessionStartTime) / 1000;
      float focusRatio = totalTime > 0 ? (float)focusTime / totalTime : 0;
      Serial.printf("⏹️ Session ENDED - Focus: %lus/%lus (%.1f%%)\n", 
                   focusTime, totalTime, focusRatio * 100);
      sessionActive = false;
    }
  } else if (command == "FOCUSED") {
    env.setDetectionMode(DM::DET_FOCUSED);
    currentFocusState = "FOCUSED";
    lastFocusUpdate = millis();
    Serial.println("Manual: FOCUSED");
    if (blynkConnected) updateBlynkFocusState("FOCUSED");
  } else if (command == "DISTRACTED") {
    env.setDetectionMode(DM::DET_DISTRACTED);
    currentFocusState = "DISTRACTED";
    lastFocusUpdate = millis();
    Serial.println("Manual: DISTRACTED");
    if (blynkConnected) updateBlynkFocusState("DISTRACTED");
  } else if (command == "AWAY") {
    env.setDetectionMode(DM::DET_AWAY);
    currentFocusState = "AWAY";
    lastFocusUpdate = millis();
    Serial.println("Manual: AWAY");
    if (blynkConnected) updateBlynkFocusState("AWAY");
  } else if (command == "SYNC") {
    if (blynkConnected) {
      Blynk.syncVirtual(V0, V1, V2, V7);
      sendEnvironmentToBlynk();
      updateBlynkFocusState(currentFocusState);
      Serial.println("Blynk sync completed");
    }
  } else if (command == "RESET") {
    Serial.println("Resetting...");
    ESP.restart();
  }
}

void printQuickStatus() {
  auto readings = env.get();
  
  Serial.printf("[Status] WiFi:%s Blynk:%s Session:%s Focus:%s\n", 
                WiFi.status() == WL_CONNECTED ? "ON" : "OFF",
                blynkConnected ? "ON" : "OFF",
                sessionActive ? "ACTIVE" : "INACTIVE",
                currentFocusState.c_str());
}

void printStatus() {
  auto readings = env.get();
  
  Serial.println("\n=== System Status ===");
  Serial.printf("WiFi: %s\n", WiFi.status() == WL_CONNECTED ? "CONNECTED" : "DISCONNECTED");
  Serial.printf("Blynk: %s\n", blynkConnected ? "CONNECTED" : "DISCONNECTED");
  Serial.printf("Session: %s\n", sessionActive ? "ACTIVE" : "INACTIVE");
  Serial.printf("Focus State: %s\n", currentFocusState.c_str());
  Serial.printf("Environment: %.1f°C, %d%%, %d%% light\n", 
                readings.tempC, readings.humi, readings.lightPct);
  
  if (sessionActive) {
    unsigned long totalTime = (millis() - sessionStartTime) / 1000;
    float focusRatio = totalTime > 0 ? (float)focusTime / totalTime : 0;
    Serial.printf("Session Time: %lu:%02lu\n", totalTime / 60, totalTime % 60);
    Serial.printf("Focus Time: %lu:%02lu\n", focusTime / 60, focusTime % 60);
    Serial.printf("Focus Ratio: %.1f%%\n", focusRatio * 100);
  }
}