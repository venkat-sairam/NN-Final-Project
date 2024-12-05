
from src.utils import os, warnings, np, random, tf
from absl import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'  # Disable XLA devices

warnings.filterwarnings('ignore')  # Suppress Python warnings

logging.set_verbosity(logging.ERROR)  # Suppress Abseil logs


seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPUs.")
    except RuntimeError as e:
        print(e)