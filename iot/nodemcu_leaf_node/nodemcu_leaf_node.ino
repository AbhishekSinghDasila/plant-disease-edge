// NodeMCU (ESP8266) field node for the plant-disease Pi.
//  - Press the button  -> publishes to "plant/capture"  -> the Pi takes a photo and predicts.
//  - Listens on "plant/diagnosis" and lights an LED:  GREEN = healthy, RED = disease,
//    YELLOW = not sure (retake).  LED turns off after 10 s.
//  - Then listens on "plant/advice" (final verdict after the cloud check) and updates the LED:
//    e.g. the edge said disease but the cloud disagreed -> YELLOW.
//  - Watches "plant/status" so the on-board LED shows whether the Pi app is online.
//  - Soil probe (optional, SOIL_ENABLED): capacitive soil-moisture sensor on A0, publishes "plant/soil"
//    every 30 s. The cloud's soil simulator pauses by itself while real readings arrive.
//
// Needs library: "PubSubClient" by Nick O'Leary (Library Manager).
// Board: NodeMCU 1.0 (ESP-12E Module). Serial Monitor: 9600 baud.
// MQTT payloads from the Pi are JSON but are > 256 bytes, so the buffer is enlarged below.
//
// Wiring (each LED through a 220-330 ohm resistor to GND):
//   D5 (GPIO14) -> GREEN LED      D6 (GPIO12) -> RED LED      D7 (GPIO13) -> YELLOW LED
//   Button: the on-board FLASH button (GPIO0 / D3) works with no extra wiring;
//           an external push button goes between D3 and GND.
//   Capacitive soil-moisture sensor v1.2/v2.0 (3 pins): VCC -> 3V3, GND -> G, AOUT -> A0.
//           Use 3.3 V. Push the probe into the soil up to the line, not deeper (electronics on top).
//           Calibrate once: read "raw" in the Serial Monitor with the probe in dry air (SOIL_DRY_RAW)
//           and in a glass of water (SOIL_WET_RAW), and put the two numbers below.

#include <ESP8266WiFi.h>
#include <PubSubClient.h>

// ---------- CHANGE THESE ----------
const char* WIFI_SSID = "Abhishek";       // the same Wi-Fi as the Pi and the laptop
const char* WIFI_PASS = "";               // <- put the Wi-Fi password here before uploading
const char* BROKER    = "10.93.8.209";    // the Pi (find it with: hostname -I); Pi bridges to the cloud
const bool  SOIL_ENABLED = true;          // false if no soil probe is connected
const int   SOIL_DRY_RAW = 820;           // A0 reading in dry air   (calibrate, see above)
const int   SOIL_WET_RAW = 380;           // A0 reading in water     (calibrate, see above)
const unsigned long SOIL_EVERY_MS = 30000;
// ----------------------------------

const char* TOPIC_CAPTURE   = "plant/capture";
const char* TOPIC_DIAGNOSIS = "plant/diagnosis";
const char* TOPIC_STATUS    = "plant/status";
const char* TOPIC_ADVICE    = "plant/advice";
const char* TOPIC_SOIL      = "plant/soil";

const int PIN_GREEN  = D5;
const int PIN_RED    = D6;
const int PIN_YELLOW = D7;
const int PIN_BUTTON = D3;            // FLASH button, active LOW
const int PIN_ONBOARD = LED_BUILTIN;  // active LOW
const unsigned long LED_ON_MS = 10000;

WiFiClient net;
PubSubClient mqtt(net);
unsigned long ledOffAt = 0, lastPress = 0, waitingSince = 0, lastSoil = 0;
bool piOnline = false;
String nodeId;

void allLedsOff() {
  digitalWrite(PIN_GREEN, LOW); digitalWrite(PIN_RED, LOW); digitalWrite(PIN_YELLOW, LOW);
}

void showResult(int pin) {
  allLedsOff();
  digitalWrite(pin, HIGH);
  ledOffAt = millis() + LED_ON_MS;
  waitingSince = 0;
}

// Tiny JSON field reader: finds "key": "value" or "key": true/false. Enough for our payload.
String jsonField(const String& s, const char* key) {
  String k = String("\"") + key + "\"";
  int i = s.indexOf(k);
  if (i < 0) return "";
  i = s.indexOf(':', i) + 1;
  while (i < (int)s.length() && s[i] == ' ') i++;
  if (s[i] == '"') { int e = s.indexOf('"', i + 1); return s.substring(i + 1, e); }
  int e = i;
  while (e < (int)s.length() && s[e] != ',' && s[e] != '}') e++;
  return s.substring(i, e);
}

void onMessage(char* topic, byte* payload, unsigned int len) {
  String msg; msg.reserve(len);
  for (unsigned int i = 0; i < len; i++) msg += (char)payload[i];

  if (strcmp(topic, TOPIC_STATUS) == 0) {
    piOnline = jsonField(msg, "state") == "online";
    digitalWrite(PIN_ONBOARD, piOnline ? LOW : HIGH);
    Serial.printf("Pi app is %s\n", piOnline ? "ONLINE" : "OFFLINE");
    return;
  }
  if (strcmp(topic, TOPIC_ADVICE) == 0) {          // final verdict (edge + cloud check)
    if (jsonField(msg, "pending_cloud") == "true") return;
    String status = jsonField(msg, "status");
    Serial.printf("Final verdict: %s (%s)\n", jsonField(msg, "title").c_str(), status.c_str());
    if (status == "healthy")      showResult(PIN_GREEN);
    else if (status == "disease") showResult(PIN_RED);
    else                          showResult(PIN_YELLOW);
    return;
  }
  if (strcmp(topic, TOPIC_DIAGNOSIS) == 0) {
    String label = jsonField(msg, "label");
    String conf  = jsonField(msg, "confidence");
    Serial.printf("Diagnosis: %s (confidence %s)\n", label.c_str(), conf.c_str());
    String status = jsonField(msg, "status");   // healthy | disease | not_sure (newer Pi app)
    if (label == "not_sure" || status == "not_sure")                           showResult(PIN_YELLOW);
    else if (status == "healthy" || label == "healthy" || label.endsWith(".healthy")) showResult(PIN_GREEN);
    else                          showResult(PIN_RED);
  }
}

// Soil moisture: average of 10 A0 readings -> 0-100 % between the dry and wet calibration values.
void publishSoil() {
  long sum = 0;
  for (int i = 0; i < 10; i++) { sum += analogRead(A0); delay(10); }
  int raw = sum / 10;
  if (raw > 1000 || raw < 50) {   // nothing connected (floating pin) or a short
    Serial.printf("Soil probe: raw %d looks unplugged, not sent\n", raw);
    return;
  }
  float pct = 100.0 * (SOIL_DRY_RAW - raw) / (SOIL_DRY_RAW - SOIL_WET_RAW);
  pct = constrain(pct, 0.0, 100.0);
  String payload = "{\"device\":\"soil-" + String(ESP.getChipId(), HEX) + "\",\"moisture_pct\":" + String(pct, 1) +
                   ",\"raw\":" + String(raw) + ",\"simulated\":false}";
  mqtt.publish(TOPIC_SOIL, payload.c_str());
  Serial.printf("Soil: raw %d -> %.1f %%\n", raw, pct);
}

void connectWiFi() {
  Serial.printf("Connecting to Wi-Fi '%s'", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.printf("\nWi-Fi OK, my IP: %s\n", WiFi.localIP().toString().c_str());
}

void connectMQTT() {
  while (!mqtt.connected()) {
    Serial.printf("Connecting to MQTT broker %s ... ", BROKER);
    if (mqtt.connect(nodeId.c_str())) {
      Serial.println("OK");
      mqtt.subscribe(TOPIC_DIAGNOSIS);
      mqtt.subscribe(TOPIC_STATUS);
      mqtt.subscribe(TOPIC_ADVICE);
    } else {
      Serial.printf("failed (state %d), retry in 3 s\n", mqtt.state());
      delay(3000);
    }
  }
}

void setup() {
  Serial.begin(9600);
  delay(300);
  pinMode(PIN_GREEN, OUTPUT); pinMode(PIN_RED, OUTPUT); pinMode(PIN_YELLOW, OUTPUT);
  pinMode(PIN_ONBOARD, OUTPUT); digitalWrite(PIN_ONBOARD, HIGH);
  pinMode(PIN_BUTTON, INPUT_PULLUP);
  allLedsOff();
  nodeId = "leaf-node-" + String(ESP.getChipId(), HEX);

  // power-on self test: blink each LED once
  for (int pin : {PIN_GREEN, PIN_RED, PIN_YELLOW}) { digitalWrite(pin, HIGH); delay(200); digitalWrite(pin, LOW); }

  connectWiFi();
  mqtt.setServer(BROKER, 1883);
  mqtt.setBufferSize(4096);   // advice messages carry the measures lists
  mqtt.setCallback(onMessage);
  connectMQTT();
  Serial.println("Ready. Press the button to diagnose a leaf.");
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) connectWiFi();
  if (!mqtt.connected()) connectMQTT();
  mqtt.loop();

  // button with 1 s debounce / lock-out
  if (digitalRead(PIN_BUTTON) == LOW && millis() - lastPress > 1000) {
    lastPress = millis();
    String payload = "{\"device\":\"" + nodeId + "\",\"time_ms\":" + String(millis()) + "}";
    mqtt.publish(TOPIC_CAPTURE, payload.c_str());
    Serial.println("Button -> capture request sent");
    waitingSince = millis();
  }

  // blink yellow while waiting for the Pi's answer (max 15 s)
  if (waitingSince) {
    if (millis() - waitingSince > 15000) { waitingSince = 0; allLedsOff(); Serial.println("No answer from Pi"); }
    else digitalWrite(PIN_YELLOW, (millis() / 200) % 2);
  }

  if (ledOffAt && millis() > ledOffAt) { allLedsOff(); ledOffAt = 0; }

  if (SOIL_ENABLED && millis() - lastSoil > SOIL_EVERY_MS) { lastSoil = millis(); publishSoil(); }
}
