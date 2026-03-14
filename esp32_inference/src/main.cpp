/*
 * main.cpp - ESP32 TinyMobileNet CIFAR-10 Inference
 *
 * Two modes:
 *   1. STANDALONE: runs inference on embedded test image at boot
 *   2. SERIAL: waits for images sent from PC via Serial, runs inference,
 * returns results
 *
 * Serial protocol:
 *   PC sends: "IMG\n" header + 3072 raw bytes (uint8, CHW format, already
 * quantized) ESP32 replies: "RES\n" + 10 lines of "class_name probability\n" +
 * "END\n"
 */

#include "esp_heap_caps.h"
#include <Arduino.h>

extern "C" {
#include "model.h"
}
#include "input_image.h"

static float output[NUM_CLASSES];

static const char *class_names[NUM_CLASSES] = {
    "airplane", "automobile", "bird",  "cat",  "deer",
    "dog",      "frog",       "horse", "ship", "truck"};

/* ---- Print inference results ---- */
void print_results(float *probs) {
  int best = 0;
  for (int i = 0; i < NUM_CLASSES; i++) {
    Serial.print("  [");
    Serial.print(i);
    Serial.print("] ");
    Serial.print(class_names[i]);
    Serial.print(": ");
    Serial.print(probs[i] * 100.0f, 2);
    Serial.println("%");
    if (probs[i] > probs[best])
      best = i;
  }
  Serial.println();
  Serial.print(">> Predicted: ");
  Serial.print(class_names[best]);
  Serial.print(" (");
  Serial.print(probs[best] * 100.0f, 2);
  Serial.println("%)");
}

/* ---- Serial inference: receive image from PC ---- */
static uint8_t serial_img[INPUT_SIZE];

bool receive_image_serial() {
  /* Wait for "IMG\n" header */
  Serial.println("READY");

  /* Read 3072 bytes with timeout */
  unsigned long start = millis();
  int received = 0;
  while (received < INPUT_SIZE) {
    if (Serial.available()) {
      serial_img[received++] = Serial.read();
      start = millis(); /* reset timeout on each byte */
    }
    if (millis() - start > 5000) {
      Serial.println("ERR:TIMEOUT");
      return false;
    }
  }
  return true;
}

void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println("\nTinyMobileNet CIFAR-10 INT8 - ESP32 Booting...");
  Serial.println("Running embedded test image:");
  Serial.print("Expected: ");
  Serial.println(EXPECTED_CLASS_NAME);

  unsigned long t0 = millis();
  predict(input_image, output);
  unsigned long t1 = millis();

  Serial.print("Inference time: ");
  Serial.print(t1 - t0);
  Serial.println(" ms");
  print_results(output);

  /* --- Enter serial inference loop --- */
  Serial.println("\nEntering Serial Inference Loop...");
  Serial.println("Ready to receive 'IMG' + 3072 bytes.");
}

void loop() {
  /* Check for incoming serial data */
  if (Serial.available() >= 3) {
    char header[4] = {0};
    header[0] = Serial.read();
    header[1] = Serial.read();
    header[2] = Serial.read();

    /* Consume newline if present */
    if (Serial.available() && Serial.peek() == '\n')
      Serial.read();

    if (header[0] == 'I' && header[1] == 'M' && header[2] == 'G') {
      if (receive_image_serial()) {
        /* Run inference */
        unsigned long t0 = millis();
        predict(serial_img, output);
        unsigned long t1 = millis();

        /* Send results back */
        Serial.println("RES");
        Serial.print("TIME:");
        Serial.println(t1 - t0);

        /* Calculate memory */
        uint32_t free_ram = heap_caps_get_free_size(MALLOC_CAP_8BIT);
        uint32_t min_free_ram =
            heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT);
        uint32_t max_alloc = heap_caps_get_largest_free_block(MALLOC_CAP_8BIT);
        uint32_t static_ram = (MAX_BUF_SIZE * 2) + RESIDUAL_SIZE;
        uint32_t cpu_freq = ESP.getCpuFreqMHz();

        Serial.print("FREE_HEAP:");
        Serial.println(free_ram);
        Serial.print("MIN_HEAP:");
        Serial.println(min_free_ram);
        Serial.print("MAX_ALLOC:");
        Serial.println(max_alloc);
        Serial.print("STATIC_RAM:");
        Serial.println(static_ram);
        Serial.print("CPU_FREQ:");
        Serial.println(cpu_freq);

        for (int i = 0; i < NUM_CLASSES; i++) {
          Serial.print("CLASS:");
          Serial.print(class_names[i]);
          Serial.print(":");
          Serial.println(output[i], 6);
        }
        Serial.println("END");
      }
    }
  }
  delay(10);
}
