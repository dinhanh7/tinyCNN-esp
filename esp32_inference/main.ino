extern "C" {
#include "model.h"
// #include "model.c"
}
#include "input_image.h"

static float output[NUM_CLASSES];

/* CIFAR-10 class names */
static const char *class_names[NUM_CLASSES] = {
    "airplane", "automobile", "bird",  "cat",  "deer",
    "dog",      "frog",       "horse", "ship", "truck"};

void setup() {
  Serial.begin(115200);
  delay(2000); // đợi Serial Monitor mở

  Serial.println("==================================");
  Serial.println("TinyMobileNet CIFAR-10 INT8 Inference");
  Serial.println("==================================");

  Serial.print("Expected class: ");
  Serial.println(EXPECTED_CLASS_NAME);

  predict(input_image, output);

  Serial.println("Prediction:");

  for (int i = 0; i < NUM_CLASSES; i++) {
    Serial.print(class_names[i]);
    Serial.print(": ");
    Serial.println(output[i], 6);
  }
}

void loop() {}
