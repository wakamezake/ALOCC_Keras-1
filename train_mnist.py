import logging
from pathlib import Path

import numpy as np
from keras.datasets import mnist

from models import AloccModel


def main():
    # Make log folder if not exist.
    log_path = Path("log")
    if not log_path.exists():
        log_path.mkdir()

    checkpoint_path = Path("checkpoint")
    if not checkpoint_path.exists():
        checkpoint_path.mkdir()

    log_file_name = "ALOCC_loss.log"
    logging.basicConfig(filename=log_path.joinpath(log_file_name),
                        level=logging.INFO)

    # prepare datasets
    attention_label = 1
    (images, labels), (_, _) = mnist.load_data()
    # Make the data range between 0~1.
    images = images / 255
    image_size = images.shape[1:]
    c_dim = 1
    specific_idx = np.where(labels == attention_label)[0]
    data = images[specific_idx].reshape(-1, image_size[0], image_size[1], c_dim)

    # train params
    epochs = 5
    batch_size = 128
    sample_interval = 500

    model = AloccModel(dataset_name='mnist', input_height=image_size[0], input_width=image_size[1])
    model.train(epochs=epochs, batch_size=batch_size, sample_interval=sample_interval)


if __name__ == '__main__':
    main()
