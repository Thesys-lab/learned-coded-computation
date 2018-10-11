import argparse
import json

import code_trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs the training process for the configuration"
        " specified by `config_file`.")
    parser.add_argument("config_file", type=str,
                        help="Configuration file that specifies components and"
                             " parameter used in training.")
    args = parser.parse_args()

    with open(args.config_file) as infile:
        config_map = json.load(infile)

    trainer = code_trainer.CodeTrainer(config_map)
    trainer.train()
