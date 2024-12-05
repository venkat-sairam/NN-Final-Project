from tensorflow.keras.preprocessing.text import Tokenizer
from src.utils.logger import logging as logger

# logger = logging.get_logger(__name__)

def tokenize_captions(captions):
    try:
        logger.info("Preparing a list of all captions")    
       
        all_captions = []
        for captions_list in captions.values():
            all_captions.extend(captions_list)

        logger.info("Initializing and fit the tokenizer with all captions.")
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(all_captions)
        logger.info("successfully instantiated tokenizer.")
        # Save the vocabulary size
        vocab_size = len(tokenizer.word_index) + 1  # Adding 1 for zero padding
        logger.info(f'Vocabulary Size:{ vocab_size}')

        max_length = max(len(caption.split()) for caption in all_captions)
        logger.info(f'Maximum Caption Length:{ max_length}')

        return vocab_size, max_length, tokenizer
    except Exception as e:
        logger.error(f"failure occured while tokenizing the captions: {e}")