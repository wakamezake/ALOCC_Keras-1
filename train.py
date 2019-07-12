import logging
from pathlib import Path

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

    model = AloccModel(dataset_name='mnist', input_height=28, input_width=28)
    model.train(epochs=5, batch_size=128, sample_interval=500)


if __name__ == '__main__':
    main()
