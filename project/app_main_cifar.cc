#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_log.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model.h"
#include "test_inputs.h"

static const char *TAG = "tflite_cifar";

// tensor arena — scratch memory for TFLite Micro
constexpr int kTensorArenaSize = 200 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

extern "C" void app_main(void) {
    tflite::InitializeTarget();

    // load model
    const tflite::Model* tfl_model = tflite::GetModel(model_data);
    if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema mismatch!");
        return;
    }

    // register ops used by the model
    static tflite::MicroMutableOpResolver<14> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddMaxPool2D();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddShape();
    resolver.AddStridedSlice();
    resolver.AddPack();
    resolver.AddAdd();
    resolver.AddMul();
    resolver.AddMean();

    // build interpreter
    static tflite::MicroInterpreter interpreter(
        tfl_model, resolver, tensor_arena, kTensorArenaSize);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed!");
        return;
    }

    TfLiteTensor* input  = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    ESP_LOGI(TAG, "Input shape:  [%d, %d, %d, %d]",
        input->dims->data[0], input->dims->data[1],
        input->dims->data[2], input->dims->data[3]);
    ESP_LOGI(TAG, "Output shape: [%d, %d]",
        output->dims->data[0], output->dims->data[1]);
    ESP_LOGI(TAG, "TFLite Micro ready. Starting inference loop...\n");

    int iteration = 0;
    while (true) {
        // alternate between input1 and input2
        const int8_t* current_input = (iteration % 2 == 0) ? input1 : input2;
        int true_label              = (iteration % 2 == 0) ? 1 : 7;
        const char* true_name       = (iteration % 2 == 0) ? "automobile" : "horse";

        // copy input into tensor
        memcpy(input->data.int8, current_input, 3072);

        // run inference with timer
        int64_t t1 = esp_timer_get_time();
        if (interpreter.Invoke() != kTfLiteOk) {
            ESP_LOGE(TAG, "Invoke failed!");
            break;
        }
        int64_t t2 = esp_timer_get_time();
        int64_t inference_us = t2 - t1;

        // find argmax of output
        int8_t max_val = output->data.int8[0];
        int predicted  = 0;
        for (int i = 1; i < 10; i++) {
            if (output->data.int8[i] > max_val) {
                max_val   = output->data.int8[i];
                predicted = i;
            }
        }

        // print results
        printf("\n── Iteration %d ──\n", iteration + 1);
        printf("Input:       %s (true label=%d)\n", true_name, true_label);
        printf("Predicted:   %s (%d)\n", class_names[predicted], predicted);
        printf("Correct:     %s\n", (predicted == true_label) ? "YES" : "NO");
        printf("Inference time: %lld us (%.2f ms)\n",
               inference_us, inference_us / 1000.0f);

        iteration++;
        vTaskDelay(pdMS_TO_TICKS(2000));
    }
}
