from data_loader.data_generator import DataGenerator
from utils.config import process_config
from utils.util import get_args, create_dirs
from models.cnn import CNNModel
from trainers.model_trainer import ModelTrainer

def main():

    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs(([config.summary_dir, config.checkpoint_dir]))

    print("Create the data generator")
    data_generator = DataGenerator(config)

    print("Create the model.")
    model = CNNModel(config, data_generator.get_word_index())

    print("Trainer initiatise")
    trainer =ModelTrainer(model.model, data_generator.get_train_data(), config)

    print("Training Start")
    trainer.train()

    print("Visualization of loss and accuracy")
    trainer.visualize("FastText +CNN")

if __name__ == "__main__":
    main()