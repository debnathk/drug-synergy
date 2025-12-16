import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU: {gpus.name}")
else:
    print("Tensorflow did not detect GPU.")