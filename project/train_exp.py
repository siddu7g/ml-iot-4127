import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, layers

print(f"TensorFlow version: {tf.__version__}")

# ── config ────────────────────────────────────────────────────────────────────
EPOCHS = 10
os.makedirs('saved_models', exist_ok=True)
saved_model_path = 'saved_models/cifar_cnn_model'
tfl_file_name    = saved_model_path + '.tflite'

# ── load CIFAR-10 ─────────────────────────────────────────────────────────────
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
train_labels = train_labels.squeeze()
test_labels  = test_labels.squeeze()
input_shape  = train_images.shape[1:]   # (32, 32, 3)
train_images = train_images / 255.0
test_images  = test_images  / 255.0
print(f"Train images: {train_images.shape}, range [{train_images.min():.2f}, {train_images.max():.2f}]")

# ── build model ───────────────────────────────────────────────────────────────
def conv_block(num_channels=32, kernel_size=(3,3), pool_size=(1,1),
               padding='same', drop_rate=None, use_batchnorm=False):
    def fn(inputs):
        x = layers.Conv2D(num_channels, kernel_size=kernel_size, padding=padding)(inputs)
        if use_batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        if pool_size not in [(1,1), 1]:
            x = layers.MaxPooling2D(pool_size=pool_size)(x)
        if drop_rate is not None:
            x = layers.Dropout(drop_rate)(x)
        return x
    return fn

inputs = Input(shape=input_shape)
x = conv_block(32,  pool_size=2, drop_rate=.25, use_batchnorm=True)(inputs)
x = conv_block(64,  pool_size=2, drop_rate=.25, use_batchnorm=True)(x)
x = conv_block(128, pool_size=2, drop_rate=.25, use_batchnorm=True)(x)
x = layers.Flatten()(x)
x = layers.Dense(10)(x)
model = tf.keras.models.Model(inputs=inputs, outputs=x)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

# ── train ─────────────────────────────────────────────────────────────────────
print(f"\nTraining for {EPOCHS} epochs...")
model.fit(train_images, train_labels,
          validation_data=(test_images, test_labels),
          epochs=EPOCHS)

# ── quantize to TFLite INT8 ───────────────────────────────────────────────────
print("\nQuantizing to INT8 TFLite...")

def representative_dataset():
    for i in range(500):
        yield [train_images[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
with open(tfl_file_name, 'wb') as f:
    f.write(tflite_model)
print(f"TFLite model saved: {tfl_file_name} ({len(tflite_model)} bytes)")

# ── verify TFLite in Python ───────────────────────────────────────────────────
interpreter = tf.lite.Interpreter(model_path=tfl_file_name)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_scale      = input_details[0]['quantization'][0]
input_zero_point = input_details[0]['quantization'][1]
print(f"\nTFLite input scale={input_scale:.6f}, zero_point={input_zero_point}")

# accuracy on 100 samples
output_data, labels = [], []
for i in range(100):
    dat_q = np.array(test_images[i:i+1] / input_scale + input_zero_point, dtype=np.int8)
    interpreter.set_tensor(input_details[0]['index'], dat_q)
    interpreter.invoke()
    output_data.append(np.argmax(interpreter.get_tensor(output_details[0]['index'])))
    labels.append(test_labels[i])
num_correct = np.sum(np.array(labels) == output_data)
print(f"TFLite Python accuracy = {num_correct/100:.3f} ({num_correct}/100)")

# ── pick automobile (1) and horse (7) ────────────────────────────────────────
idx1 = int(np.where(test_labels == 1)[0][0])   # automobile
idx2 = int(np.where(test_labels == 7)[0][0])   # horse
print(f"\nImage 1 → idx={idx1}, true label: {class_names[test_labels[idx1]]}")
print(f"Image 2 → idx={idx2}, true label: {class_names[test_labels[idx2]]}")

# Keras predictions
keras_pred1 = class_names[np.argmax(model.predict(test_images[idx1:idx1+1], verbose=0))]
keras_pred2 = class_names[np.argmax(model.predict(test_images[idx2:idx2+1], verbose=0))]
print(f"Keras prediction 1: {keras_pred1}")
print(f"Keras prediction 2: {keras_pred2}")

# quantize inputs
img1_q = np.array(test_images[idx1:idx1+1] / input_scale + input_zero_point, dtype=np.int8)
img2_q = np.array(test_images[idx2:idx2+1] / input_scale + input_zero_point, dtype=np.int8)

# TFLite Python predictions
interpreter.set_tensor(input_details[0]['index'], img1_q)
interpreter.invoke()
tflite_pred1 = class_names[np.argmax(interpreter.get_tensor(output_details[0]['index']))]

interpreter.set_tensor(input_details[0]['index'], img2_q)
interpreter.invoke()
tflite_pred2 = class_names[np.argmax(interpreter.get_tensor(output_details[0]['index']))]
print(f"TFLite Python prediction 1: {tflite_pred1}")
print(f"TFLite Python prediction 2: {tflite_pred2}")

# ── export model.h ────────────────────────────────────────────────────────────
with open("model.h", "w") as f:
    f.write("#pragma once\n")
    f.write(f"const unsigned int model_len = {len(tflite_model)};\n")
    f.write("alignas(8) const unsigned char model_data[] = {\n")
    f.write(", ".join(str(b) for b in tflite_model))
    f.write("\n};\n")
print(f"\nmodel.h written ({len(tflite_model)} bytes)")

# ── export test_inputs.h ──────────────────────────────────────────────────────
with open("test_inputs.h", "w") as f:
    f.write("#pragma once\n")
    f.write("#include <stdint.h>\n\n")
    f.write(f"// Input 1 — true label: {class_names[test_labels[idx1]]} ({test_labels[idx1]})\n")
    f.write(f"const int8_t input1[3072] = {{\n")
    f.write(", ".join(str(x) for x in img1_q.flatten()))
    f.write("\n};\n\n")
    f.write(f"// Input 2 — true label: {class_names[test_labels[idx2]]} ({test_labels[idx2]})\n")
    f.write(f"const int8_t input2[3072] = {{\n")
    f.write(", ".join(str(x) for x in img2_q.flatten()))
    f.write("\n};\n\n")
    f.write('const char* class_names[10] = {"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};\n')
print("test_inputs.h written")

print(f"\nAll files saved to: {os.path.abspath('.')}")
print("\n── Summary table (ESP32 column to be filled after flashing) ──")
print(f"{'Input':<10} {'True label':<12} {'Keras':<12} {'TFLite Python':<15} {'TFLite ESP32'}")
print(f"{'Image 1':<10} {class_names[test_labels[idx1]]:<12} {keras_pred1:<12} {tflite_pred1:<15} ???")
print(f"{'Image 2':<10} {class_names[test_labels[idx2]]:<12} {keras_pred2:<12} {tflite_pred2:<15} ???")