import tensorflow_datasets as tfds
from tqdm import tqdm
from src.utils.logger import logging as logger
# logger = logging.get_logger(__name__)

def load_dataset(split='train[:5000]'):
    try:
        dataset, info = tfds.load('coco_captions', with_info=True)
        logger.info("Dataset loaded successfully.")
        # Concatenate the splits together (train + val + test + restval)
        dataset = dataset['train'].concatenate(dataset['val']).concatenate(dataset['test']).concatenate(dataset['restval'])
        dataset_size = dataset.cardinality().numpy()
        # # Verify dataset size
        # for split, split_dataset in dataset.items():
        #     dataset_size = split_dataset.cardinality().numpy()
        #     logger.info(f"Dataset size for {split}: {dataset_size}")
        logger.info(f"Dataset size: {dataset_size}")
        logger.info(f"Writing the dataset information: {info}")
        return dataset, info
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise
