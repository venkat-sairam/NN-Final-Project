from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm  # Import tqdm
from src.utils import os, np
from src.model.caption_generator import generate_caption
from src.utils.logger import logging
def evaluate_model(model, captions_dict, features_dir, tokenizer, max_length):
    actual, predicted = [], []
    logging.info("evaluating the model using given params.")
    for image_id, captions_list in tqdm(captions_dict.items(), desc="Evaluating", total=len(captions_dict)):
        # Load the photo feature
        
        feature_path = os.path.join(features_dir, f"{image_id}.npy")
        photo_feature = np.load(feature_path).astype('float32')
        photo_feature = np.squeeze(photo_feature)  # Ensure shape is (2048,)
        
        # logging.info("generating captions for images ....pred ones")
        y_pred = generate_caption(model, tokenizer, photo_feature, max_length)
        y_pred = y_pred.split()[1:-1] 
        references = [c.split()[1:-1] for c in captions_list]
        actual.append(references)
        predicted.append(y_pred)

    logging.info("Calculating BLEU scores")
    
    bleu1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    bleu2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))
    logging.info("BLEU Evaluation metrics:")
    
    logging.info(f'BLEU-1:{ bleu1}')
    logging.info(f'BLEU-2:{ bleu2}')
    logging.info(f'BLEU-3:{ bleu3}')
    logging.info(f'BLEU-4:{ bleu4}')
