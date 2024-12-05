
from src.utils import np, os, tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input

from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import tensorflow_datasets as tfds
import re
import string
from src.utils.logger import logging as logger
from tensorflow.keras.models import Model

# logger = logging.get_logger(__name__)

def preprocess_image(example):
    logger.info("Preprocessing the dataset to extract the images and ids only.")
    img = example['image']
    img = tf.image.resize(img, (299, 299))
    img = preprocess_input(img)
    # Return only 'image' and 'image/id' to avoid batching issues
    logger.info("Extracted image and imade-id from the COCO dataset successfully.")
    return {'image': img, 'image/id': example['image/id']}

def load_inception_v3_model():
    logger.info("Initializing the InceptionV3 Model")
    base_model = InceptionV3(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    logger.info(f"InceptionV3 Model summary: {model.summary()}")
    return model

from tqdm import tqdm

def extract_features(dataset, features_dir, model):
    try:    
        batch_size = 50  
        dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        for batch in tqdm(dataset, desc="Extracting features", unit="batch"):
            images = batch['image']
            image_ids = batch['image/id'].numpy()

            # Extract features for the batch
            features_batch = model.predict(images, verbose=0)

            for idx, image_id in enumerate(image_ids):
                image_id_str = str(image_id)
                feature = features_batch[idx]
                # Save the feature to disk
                feature_path = os.path.join(features_dir, f"{image_id_str}.npy")
                np.save(feature_path, feature)
    except Exception as e:
        logger.info(f"failed to extract the features from images...{e}")
        raise


def extract_captions(dataset):
    try:
        captions_dict = {}
        for example in tqdm(tfds.as_numpy(dataset), desc="Extracting captions"):
            image_id = str(example['image/id'])
            
            captions = [caption.decode('utf-8') for caption in example['captions']['text']]
            captions_dict[image_id] = captions
        logger.info("Captions extracted successfully.")
        return captions_dict
    except Exception as e:
        logger.error(f"Failed to extract captions: {e}")
        raise



def clean_captions(captions_dict):
    table = str.maketrans('', '', string.punctuation)
    for key, captions_list in captions_dict.items():
        cleaned_captions = []
        for caption in captions_list:
            # Convert to lowercase
            caption = caption.lower()
            # Remove punctuation
            caption = caption.translate(table)
            # Remove digits and special characters
            caption = re.sub(r'[^a-z\s]', '', caption)
            # Remove extra whitespace
            caption = re.sub(r'\s+', ' ', caption).strip()
            # Add start and end tokens
            caption = 'startseq ' + caption + ' endseq'
            cleaned_captions.append(caption)
        captions_dict[key] = cleaned_captions
    return captions_dict

