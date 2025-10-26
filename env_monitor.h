#pragma once
#include <Arduino.h>
#include <U8g2lib.h>
#include <DHT.h>

class EnvMonitor {
public:
  enum Comfort  { COMF_RED, COMF_YELLOW, COMF_GREEN };
  enum DetMode  { DET_NONE, DET_AWAY, DET_DISTRACTED, DET_FOCUSED };

  struct Readings {
    float   tempC = NAN;
    int     humi  = -1;        // -1 = invalid
    int     lightPct = -1;     // 0..100
    int     scoreT = 0;        // 0/1/2 (BAD/WARN/GOOD)
    int     scoreH = 0;
    int     scoreL = 0;
    Comfort overall = COMF_RED;
  };

  EnvMonitor(uint8_t pinDHT, uint8_t pinLdrAO,
             uint8_t ledA_R, uint8_t ledA_Y, uint8_t ledA_G,
             uint8_t ledB_R, uint8_t ledB_Y, uint8_t ledB_G,
             uint8_t buzzerPin,
             uint8_t pinSDA, uint8_t pinSCL);

  void begin();   // init IO, sensors, OLED, ADC
  void tick();    // periodic DHT read, fast LDR read, judge, LEDs, OLED/overlay

  const Readings& get() const { return m_rd; }

  // Detection mode (Blynk-controlled)
  void setDetectionMode(DetMode m);

  // Optional threshold setters
  void setTempRanges(float goodMin, float goodMax, float warnMin, float warnMax);
  void setHumiRanges(int goodMin, int goodMax, int warnMin, int warnMax);
  void setLightThresholds(int lowPct, int goodPct);

private:
  // Pins
  uint8_t m_pinDHT, m_pinLDR, m_ledA_R, m_ledA_Y, m_ledA_G;
  uint8_t m_ledB_R, m_ledB_Y, m_ledB_G, m_buzzer;
  uint8_t m_pinSDA, m_pinSCL;

  // Peripherals
  DHT dht;
  U8G2_SSD1306_128X64_NONAME_F_HW_I2C oled;

  // Timing / config
  const uint32_t DHT_PERIOD_MS = 2000;
  const uint8_t  NUM_LDR_SAMPLES = 8;
  uint32_t m_lastDhtMs = 0;

  // Thresholds (defaults match your sketch)
  float m_TEMP_GOOD_MIN = 21.0f, m_TEMP_GOOD_MAX = 26.0f;
  float m_TEMP_WARN_MIN = 16.0f, m_TEMP_WARN_MAX = 30.0f;

  int   m_HUMI_GOOD_MIN = 40,    m_HUMI_GOOD_MAX = 60;
  int   m_HUMI_WARN_MIN = 30,    m_HUMI_WARN_MAX = 70;

  int   m_LIGHT_LOW_PCT = 30;    // <30% bad
  int   m_LIGHT_GOOD_PCT = 50;   // >=50% good

  // Buzzer tone (passive buzzer needs PWM tone). 2.5 kHz is a crisp “high” tone.
  const uint32_t BUZZ_TONE_HZ = 2500;

  // State
  Readings m_rd;

  // Detection overlay state
  DetMode  m_detMode = DET_NONE;
  bool     m_blinkOn  = false;
  uint32_t m_lastBlinkMs = 0;
  const uint32_t BLINK_HALF_PERIOD_MS = 500; // 1 Hz blink (on/off every 500 ms)

  // Helpers
  int  readLightPercent();
  int  scoreTemp(float t) const;
  int  scoreHumi(int h) const;
  int  scoreLight(int p) const;

  void setEnvLED(Comfort z);

  // Detection actuators
  void driveDetectionOutputs();   // LEDB / Buzzer (handles blink)
  void drawOverlayIfNeeded();     // Draws center text for AWAY/DISTRACTED; else normal screen
  void drawRow(uint8_t y, const char* name, const char* value, int score);
  void drawEnvScreen();           // Normal environment screen
};
