from src.model.caption_generator import data_generator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from src.utils import tf
from src.utils.logger import logging


def calc_steps(captions_dict, batch_size):
    logging.info(" Function to calculate steps per epoch")
    total_captions = sum(len(captions_list) for captions_list in captions_dict.values())
    total_sequences = sum(len(caption.split()) - 1 for captions_list in captions_dict.values() for caption in captions_list)
    steps = total_sequences // batch_size
    return steps


def calc_total_sequences(captions_dict):
    logging.info("Calculate total sequences")
    total = 0
    for captions_list in captions_dict.values():
        for caption in captions_list:
            total += len(caption.split()) - 1
    return total
   
def model_trainer(
        tokenizer,
        train_captions,
        val_captions,
        train_generator,
        val_generator,
        model,
        max_length,
        features_dir
):

     

    batch_size = 64  # Adjust based on your system's capabilities
    vocab_size = len(tokenizer.word_index) + 1
    
    train_sequences = calc_total_sequences(train_captions)
    val_sequences = calc_total_sequences(val_captions)
    train_steps = train_sequences // batch_size
    val_steps = val_sequences // batch_size
    logging.info(f'Train steps per epoch: {train_steps}, Validation steps per epoch: {val_steps}')


    logging.info("defining callbacks on model")
    checkpoint = ModelCheckpoint('best_model.keras', monitor='loss', verbose=1, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='loss', patience=3)

    logging.info("Defining output Signatures in the model")
    output_signature = (
        (tf.TensorSpec(shape=(None, 2048), dtype=tf.float32),
        tf.TensorSpec(shape=(None, max_length), dtype=tf.int32)),
        tf.TensorSpec(shape=(None, vocab_size), dtype=tf.float32)
    )
    logging.info("creating training datasets using data generator.")
    # Create datasets
    train_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(train_captions, features_dir, tokenizer, max_length, vocab_size, batch_size),
        output_signature=output_signature
    )

    logging.info("creating validation datasets using data generator.")
    val_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(val_captions, features_dir, tokenizer, max_length, vocab_size, batch_size),
        output_signature=output_signature
    )
    logging.info("Model training has started.")
    # Train the model
    model.fit(
        train_dataset,
        steps_per_epoch=train_steps,
        epochs=20,
        validation_data=val_dataset,
        validation_steps=val_steps,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )

    logging.info("Successfully finished training the model.")
    return model