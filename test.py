from src.utils import os, warnings
from src.data.preprocessing import *
from src.data.data_loader import load_dataset
from src.data.preprocessing import load_inception_v3_model, extract_captions
from src.data.tokenizer import tokenize_captions
from sklearn.model_selection import train_test_split
from src.model.model_builder import get_image_feature_extractor_model
from src.model.caption_generator import data_generator, generate_caption
from src.model.model_trainer import model_trainer
from src.model.model_evaluator import evaluate_model
from src.model.caption_generator import filter_captions_with_features

dataset, info = load_dataset()
model = load_inception_v3_model()

# Directory for saving preprocessed features
features_dir = "preprocessed_features"
os.makedirs(features_dir, exist_ok=True)

# extract_features(dataset, features_dir, model)
captions = extract_captions(dataset)

# Clean the captions
captions = clean_captions(captions)

vocab_size, max_length, tokenizer = tokenize_captions(captions=captions)




# Get image IDs
image_ids = list(captions.keys())

# Split image IDs into training and validation sets
train_ids, val_ids = train_test_split(image_ids, test_size=0.1, random_state=42)

# Create dictionaries for training and validation captions
train_captions = {img_id: captions[img_id] for img_id in train_ids}
val_captions = {img_id: captions[img_id] for img_id in val_ids}

# Filter training and validation captions
train_captions, missing_train_ids = filter_captions_with_features(train_captions, features_dir)
val_captions, missing_val_ids = filter_captions_with_features(val_captions, features_dir)

image_model = get_image_feature_extractor_model(vocab_size= vocab_size, max_length=max_length)
batch_size = 64
# Create generators
train_generator = data_generator(train_captions, features_dir, tokenizer, max_length, vocab_size, batch_size)
val_generator = data_generator(val_captions, features_dir, tokenizer, max_length, vocab_size, batch_size)


trained_image_model = model_trainer(
    max_length = max_length,
    tokenizer = tokenizer,
    train_captions = train_captions,
    val_captions = val_captions,
    train_generator = train_generator,
    val_generator = val_generator,
    model = image_model,
    features_dir = features_dir
)

evaluate_model(
    model = trained_image_model,
    captions_dict=val_captions,
    features_dir= features_dir,
    tokenizer= tokenizer,
    max_length=  max_length
)
