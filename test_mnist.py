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

    sample_path = Path("sample")
    if not sample_path.exists():
        sample_path.mkdir()

    # prepare datasets
    attention_label = 1
    (_, _), (images, labels) = mnist.load_data()
    # Make the data range between 0~1.
    images = images / 255
    image_size = images.shape[1:]
    c_dim = 1
    specific_idx = np.where(labels == attention_label)[0]
    data = images[specific_idx].reshape(-1, image_size[0], image_size[1], c_dim)

    model = AloccModel(data=data, checkpoint_dir=checkpoint_path, sample_dir=sample_path,
                       input_height=image_size[0], input_width=image_size[1])

    weight_path = checkpoint_path.joinpath("ALOCC_Model_4.h5")
    predicts = model.predict(data, weight_path=weight_path)
    print(predicts[1])


if __name__ == '__main__':
    main()
