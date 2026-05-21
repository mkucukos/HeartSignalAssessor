import tensorflow as tf

from utils import generate_ecg_data, create_animation

_MODEL = tf.saved_model.load("./model/")

if __name__ == "__main__":
    df, valid_features = generate_ecg_data(_MODEL)
    create_animation(df, valid_features)
