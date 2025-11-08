#include "env_monitor.h"

EnvMonitor::EnvMonitor(uint8_t pinDHT, uint8_t pinLdrAO,
                       uint8_t ledA_R, uint8_t ledA_Y, uint8_t ledA_G,
                       uint8_t ledB_R, uint8_t ledB_Y, uint8_t ledB_G,
                       uint8_t buzzerPin,
                       uint8_t pinSDA, uint8_t pinSCL)
  : m_pinDHT(pinDHT), m_pinLDR(pinLdrAO),
    m_ledA_R(ledA_R), m_ledA_Y(ledA_Y), m_ledA_G(ledA_G),
    m_ledB_R(ledB_R), m_ledB_Y(ledB_Y), m_ledB_G(ledB_G),
    m_buzzer(buzzerPin),
    m_pinSDA(pinSDA), m_pinSCL(pinSCL),
    dht(pinDHT, DHT11),
    oled(U8G2_R0, /*reset=*/U8X8_PIN_NONE, /*clock=*/pinSCL, /*data=*/pinSDA)
{}

void EnvMonitor::begin() {
  // LEDs (ACTIVE-HIGH)
  pinMode(m_ledA_R, OUTPUT); digitalWrite(m_ledA_R, LOW);
  pinMode(m_ledA_Y, OUTPUT); digitalWrite(m_ledA_Y, LOW);
  pinMode(m_ledA_G, OUTPUT); digitalWrite(m_ledA_G, LOW);

  pinMode(m_ledB_R, OUTPUT); digitalWrite(m_ledB_R, LOW);
  pinMode(m_ledB_Y, OUTPUT); digitalWrite(m_ledB_Y, LOW);
  pinMode(m_ledB_G, OUTPUT); digitalWrite(m_ledB_G, LOW);

  // Buzzer
  pinMode(m_buzzer, OUTPUT); digitalWrite(m_buzzer, LOW);
  ledcAttach(m_buzzer, BUZZ_TONE_HZ, 8);   // set up PWM on this pin (auto channel), 8-bit res
  ledcWriteTone(m_buzzer, 0);  

  // Sensors
  dht.begin();

  // ADC config for ESP32-S3
  analogReadResolution(12);
  analogSetPinAttenuation(m_pinLDR, ADC_11db);

  // OLED
  oled.begin();
  oled.clearBuffer();
  oled.setFont(u8g2_font_6x13_tf);
  oled.drawStr(0, 24, "Initializing...");
  oled.sendBuffer();

  // Initial readings
  m_rd.lightPct = readLightPercent();
  m_lastDhtMs = millis() - DHT_PERIOD_MS; // force immediate DHT read
}

void EnvMonitor::tick() {
  uint32_t now = millis();

  // Periodic DHT
  if (now - m_lastDhtMs >= DHT_PERIOD_MS) {
    m_lastDhtMs = now;
    float h = dht.readHumidity();
    float t = dht.readTemperature();
    if (isnan(h) || isnan(t)) {
      Serial.println(F("DHT read failed"));
      m_rd.humi  = -1;
      m_rd.tempC = NAN;
    } else {
      m_rd.humi  = (int)roundf(h);
      m_rd.tempC = t;
    }
  }

  // Fast LDR
  m_rd.lightPct = readLightPercent();

  // Scoring (for env LEDs)
  m_rd.scoreT = scoreTemp(m_rd.tempC);
  m_rd.scoreH = scoreHumi(m_rd.humi);
  m_rd.scoreL = scoreLight(m_rd.lightPct);

  // Overall env
  if (m_rd.scoreT == 0 || m_rd.scoreH == 0 || m_rd.scoreL == 0) m_rd.overall = COMF_RED;
  else if (m_rd.scoreT == 1 || m_rd.scoreH == 1 || m_rd.scoreL == 1) m_rd.overall = COMF_YELLOW;
  else m_rd.overall = COMF_GREEN;

  // Drive environment LEDs
  setEnvLED(m_rd.overall);

  // Detection outputs (LEDB/Buzzer + blink timing)
  driveDetectionOutputs();

  // Display (overlay if Away/Distracted, else normal env screen)
  drawOverlayIfNeeded();
}

// ---------------- Helpers ----------------

int EnvMonitor::readLightPercent() {
  uint32_t acc = 0;
  for (uint8_t i = 0; i < NUM_LDR_SAMPLES; ++i) {
    acc += analogRead(m_pinLDR);       // 0..4095
    delay(2);
  }
  uint16_t raw = acc / NUM_LDR_SAMPLES;

  // Inverted mapping: brighter -> larger %
  int pct = map(raw, 4095, 0, 0, 100);
  return constrain(pct, 0, 100);
}

int EnvMonitor::scoreTemp(float t) const {
  if (isnan(t)) return 0;
  if (t >= m_TEMP_GOOD_MIN && t <= m_TEMP_GOOD_MAX) return 2;
  if ((t >= m_TEMP_WARN_MIN && t < m_TEMP_GOOD_MIN) ||
      (t >  m_TEMP_GOOD_MAX && t <= m_TEMP_WARN_MAX)) return 1;
  return 0;
}

int EnvMonitor::scoreHumi(int h) const {
  if (h < 0 || h > 100) return 0; // invalid -> bad
  if (h >= m_HUMI_GOOD_MIN && h <= m_HUMI_GOOD_MAX) return 2;
  if ((h >= m_HUMI_WARN_MIN && h < m_HUMI_GOOD_MIN) ||
      (h >  m_HUMI_GOOD_MAX && h <= m_HUMI_WARN_MAX)) return 1;
  return 0;
}

int EnvMonitor::scoreLight(int p) const {
  if (p >= m_LIGHT_GOOD_PCT) return 2;
  if (p >= m_LIGHT_LOW_PCT)  return 1;
  return 0;
}

void EnvMonitor::setEnvLED(Comfort z) {
  digitalWrite(m_ledA_R, (z == COMF_RED)    ? HIGH : LOW);
  digitalWrite(m_ledA_Y, (z == COMF_YELLOW) ? HIGH : LOW);
  digitalWrite(m_ledA_G, (z == COMF_GREEN)  ? HIGH : LOW);
}

void EnvMonitor::setDetectionMode(DetMode m) {
  m_detMode = m;
  m_lastBlinkMs = millis();

  // Reset detection outputs immediately
  digitalWrite(m_ledB_R, LOW);
  digitalWrite(m_ledB_Y, LOW);
  digitalWrite(m_ledB_G, LOW);
  ledcWriteTone(m_buzzer, 0);   // ensure silent by default

  switch (m_detMode) {
    case DET_AWAY:
      m_blinkOn = false;
      digitalWrite(m_ledB_Y, HIGH); // solid yellow
      break;

    case DET_DISTRACTED:
      // Start with ON state immediately; blink handled in driveDetectionOutputs()
      m_blinkOn = true;
      digitalWrite(m_ledB_R, HIGH);
      ledcWriteTone(m_buzzer, BUZZ_TONE_HZ);
      break;

    case DET_FOCUSED:
      m_blinkOn = false;
      digitalWrite(m_ledB_G, HIGH); // solid green
      break;

    case DET_NONE:
    default:
      m_blinkOn = false;
      break;
  }
}

void EnvMonitor::driveDetectionOutputs() {
  // Handle 1 Hz blink for DET_DISTRACTED (buzzer toggles with LED)
  if (m_detMode == DET_DISTRACTED) {
    uint32_t now = millis();
    if (now - m_lastBlinkMs >= BLINK_HALF_PERIOD_MS) {
      m_lastBlinkMs = now;
      m_blinkOn = !m_blinkOn;

      digitalWrite(m_ledB_R, m_blinkOn ? HIGH : LOW);
      ledcWriteTone(m_buzzer, m_blinkOn ? BUZZ_TONE_HZ : 0); // play/stop buzzer
    }
  } else {
    // Other modes: make sure buzzer is off
    ledcWriteTone(m_buzzer, 0);
  }
}

static inline const char* _statusText(int s) {
  return (s==2) ? "[GOOD]" : (s==1) ? "[WARN]" : "[BAD]";
}

void EnvMonitor::drawRow(uint8_t y, const char* name, const char* value, int score) {
  oled.drawStr(0,  y, name);
  oled.drawStr(36, y, value);
  uint8_t w = oled.getStrWidth(_statusText(score));
  oled.drawStr(128 - w, y, _statusText(score));
}

void EnvMonitor::drawEnvScreen() {
  oled.clearBuffer();
  oled.setFont(u8g2_font_6x13_tf);
  oled.drawStr(0, 12, "Study Assistant");

  char v1[20], v2[20], v3[20];

  if (!isnan(m_rd.tempC)) snprintf(v1, sizeof(v1), "%5.1fC", m_rd.tempC);
  else                    snprintf(v1, sizeof(v1), "  --- ");

  if (m_rd.humi >= 0 && m_rd.humi <= 100) snprintf(v2, sizeof(v2), "%3d %%", m_rd.humi);
  else                                    snprintf(v2, sizeof(v2), "  --- ");

  snprintf(v3, sizeof(v3), "%3d %%", m_rd.lightPct);

  drawRow(28, "Temp:",  v1, m_rd.scoreT);
  drawRow(44, "Humi:",  v2, m_rd.scoreH);
  drawRow(60, "Light:", v3, m_rd.scoreL);

  // Overall tag (small, top-right)
  const char* ov = (m_rd.overall==COMF_GREEN) ? "[GOOD]" :
                   (m_rd.overall==COMF_YELLOW)? "[WARN]" : "[BAD]";
  uint8_t ww = oled.getStrWidth(ov);
  oled.drawStr(128 - ww, 12, ov);

  oled.sendBuffer();
}

void EnvMonitor::drawOverlayIfNeeded() {
  if (m_detMode == DET_AWAY || m_detMode == DET_DISTRACTED) {
    oled.clearBuffer();
    oled.setFont(u8g2_font_logisoso20_tf);

    const char* msg = (m_detMode == DET_AWAY) ? "Away" : "Distracted";

    // For distracted: blink the text in sync with LED/buzzer
    if (m_detMode == DET_DISTRACTED && !m_blinkOn) {
      oled.sendBuffer();
      return;
    }

    uint8_t w = oled.getStrWidth(msg);
    uint8_t x = (128 > w) ? (128 - w) / 2 : 0;
    uint8_t y = 40;
    oled.drawStr(x, y, msg);
    oled.sendBuffer();
  } else {
    drawEnvScreen();
  }
}

// ---------- Optional setters ----------
void EnvMonitor::setTempRanges(float gMin, float gMax, float wMin, float wMax) {
  m_TEMP_GOOD_MIN = gMin; m_TEMP_GOOD_MAX = gMax;
  m_TEMP_WARN_MIN = wMin; m_TEMP_WARN_MAX = wMax;
}
void EnvMonitor::setHumiRanges(int gMin, int gMax, int wMin, int wMax) {
  m_HUMI_GOOD_MIN = gMin; m_HUMI_GOOD_MAX = gMax;
  m_HUMI_WARN_MIN = wMin; m_HUMI_WARN_MAX = wMax;
}
void EnvMonitor::setLightThresholds(int lowPct, int goodPct) {
  m_LIGHT_LOW_PCT = lowPct; m_LIGHT_GOOD_PCT = goodPct;
}