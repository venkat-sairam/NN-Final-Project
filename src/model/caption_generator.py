from src.utils import random, os, np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from src.utils.logger import logging

def data_generator(captions, features_dir, tokenizer, max_length, vocab_size, batch_size):
    logging.info("Executing the data generator function. ")
    image_ids = list(captions.keys())
    while True:
        random.shuffle(image_ids)
        X1, X2, y = [], [], []
        for image_id in image_ids:
            # Load feature
            feature_path = os.path.join(features_dir, f"{image_id}.npy")
            try:
                photo_feature = np.load(feature_path).astype('float32')
                photo_feature = np.squeeze(photo_feature)  # Ensure shape is (2048,)
            except FileNotFoundError:
                logging.info(f"Warning: Feature file not found for image ID {image_id}. Skipping this image.")
                continue  # Skip to the next image_id
            
            # Get captions
            captions_list = captions[image_id]
            for caption in captions_list:
                # Encode the sequence
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    # Split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # Pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post')[0]
                    # Store
                    X1.append(photo_feature)
                    X2.append(in_seq)
                    y.append(out_seq)
                    if len(X1) == batch_size:
                        # Convert y to categorical and ensure dtype is float32
                        y_cat = to_categorical(y, num_classes=vocab_size).astype('float32')
                        yield ((np.stack(X1), np.array(X2)), y_cat)
                        X1, X2, y = [], [], []
        if len(X1) > 0:
            y_cat = to_categorical(y, num_classes=vocab_size).astype('float32')
            yield ((np.stack(X1), np.array(X2)), y_cat)
            X1, X2, y = [], [], []

def generate_caption(model, tokenizer, photo_feature, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        # Convert text to integer sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # Pad the sequence
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        # Predict the next word
        yhat = model.predict([photo_feature.reshape(1, 2048), sequence], verbose=0)
        # Get the index with highest probability
        yhat = np.argmax(yhat)
        # Convert index to word
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        # Append word to input text
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text


def filter_captions_with_features(captions_dict, features_dir):
    logging.info(f"Filtering the captions with no image id in {features_dir}")
    available_image_ids = []
    missing_image_ids = []
    for image_id in captions_dict.keys():
        feature_path = os.path.join(features_dir, f"{image_id}.npy")
        if os.path.exists(feature_path):
            available_image_ids.append(image_id)
        else:
            missing_image_ids.append(image_id)
            logging.warning(f"Warning: Feature file not found for image ID {image_id}. Excluding this image from training.")

    # Create a new captions dictionary with available image IDs
    filtered_captions = {image_id: captions_dict[image_id] for image_id in available_image_ids}
    logging.info("returning the filtered captions to the caller. ")
    return filtered_captions, missing_image_ids

