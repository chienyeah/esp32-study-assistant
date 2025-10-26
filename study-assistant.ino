/*  EIE3127 – Study Environment Monitor + Blynk (ESP32-S3 UNO)
    Wiring (unchanged):
      OLED SSD1306 I2C: SDA=IO4, SCL=IO5 (3V3)
      DHT11: DATA=IO17 (3V3)
      LDR AO: IO7 (3V3)
      RGB-A (env): R=IO13, Y=IO14, G=IO21 (ACTIVE-HIGH)
      RGB-B (det): R=IO9,  Y=IO11, G=IO12 (ACTIVE-HIGH)
      Buzzer: IO18 (KY-006 passive buzzer)
*/

#include <Arduino.h>
// #include <WiFi.h>
#include "secrets.h"     // define WIFI_SSID, WIFI_PASS, BLYNK_AUTH
#include <BlynkSimpleEsp32.h>

#include "env_monitor.h" // environment screen + detection overlay

// --------- Pin map ----------
#define PIN_I2C_SDA   4
#define PIN_I2C_SCL   5
#define PIN_DHT       17
#define PIN_LDR_AO     7

// RGB-A (environment judgement) — ACTIVE-HIGH
#define LEDA_R        13
#define LEDA_Y        14
#define LEDA_G        21

// RGB-B (detection/result) — ACTIVE-HIGH
#define LEDB_R        10
#define LEDB_Y        11
#define LEDB_G        12

// Buzzer
#define PIN_BUZZER    18

// Create the monitor (pins + I2C pins)
EnvMonitor env(PIN_DHT, PIN_LDR_AO,
               LEDA_R, LEDA_Y, LEDA_G,
               LEDB_R, LEDB_Y, LEDB_G, PIN_BUZZER,
               PIN_I2C_SDA, PIN_I2C_SCL);

// Track which Blynk state is currently active (-1 = none)
// 0 -> V0 Distracted, 1 -> V1 Away, 2 -> V2 Focused
int g_activeStateIndex = -1;

// ----- Helpers -----
static inline void setActiveState(int idx) {
  g_activeStateIndex = idx;
  using DM = EnvMonitor::DetMode;
  switch (idx) {
    case 0: env.setDetectionMode(DM::DET_DISTRACTED); break; // V0
    case 1: env.setDetectionMode(DM::DET_AWAY);        break; // V1
    case 2: env.setDetectionMode(DM::DET_FOCUSED);     break; // V2
    default: env.setDetectionMode(DM::DET_NONE);       break;
  }
}

static inline void bounceBackIfBusy(uint8_t vpin) {
  // If some state is already active, reject new ON command and reset that widget to 0
  if (g_activeStateIndex != -1) {
    Blynk.virtualWrite(vpin, 0);
  }
}

// ----- Blynk handlers -----
// V0: Distracted (blink red + buzzer + big blinking text)
BLYNK_WRITE(V0) {
  int val = param.asInt();
  if (val == 1) {
    if (g_activeStateIndex == -1) setActiveState(0);
    else                          bounceBackIfBusy(V0);
  } else {
    if (g_activeStateIndex == 0) setActiveState(-1);
  }
}

// V1: Away (yellow solid + big text)
BLYNK_WRITE(V1) {
  int val = param.asInt();
  if (val == 1) {
    if (g_activeStateIndex == -1) setActiveState(1);
    else                          bounceBackIfBusy(V1);
  } else {
    if (g_activeStateIndex == 1) setActiveState(-1);
  }
}

// V2: Focused (green solid)
BLYNK_WRITE(V2) {
  int val = param.asInt();
  if (val == 1) {
    if (g_activeStateIndex == -1) setActiveState(2);
    else                          bounceBackIfBusy(V2);
  } else {
    if (g_activeStateIndex == 2) setActiveState(-1);
  }
}

void setup() {
  Serial.begin(115200);
  delay(50);

  // Connect Blynk
  Blynk.begin(BLYNK_AUTH_TOKEN, WIFI_SSID, WIFI_PASS);
  while (!Blynk.connected()) { Serial.println("Connecting to Blynk..."); }

  // Init environment monitor
  env.begin();

  // Ensure dashboard sync (optional)
  Blynk.syncVirtual(V0, V1, V2);
}

void loop() {
  env.tick();   // sensor read + env LEDs + OLED or overlay + detection outputs
  Blynk.run();
  delay(20);
}
