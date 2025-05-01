import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Hiển thị tất cả log
import tensorflow as tf
print(tf.__version__)
print("CUDA Available:", tf.test.is_built_with_cuda())
print("GPU Devices:", tf.config.list_physical_devices('GPU'))
tf.debugging.set_log_device_placement(True)  # Hiển thị thiết bị thực thi