import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility script to get the reconstruction- and overall-"
        "accuracies of training a code. This script takes as"
        " input the directory containing stats files produced"
        " during training of a code, and reports the training"
        " and validation reconstruction- and overall-accuracies"
        " that correspond to the maximum reconstruciton-accuracy"
        " attained over the training set")
    parser.add_argument("stats_dir", type=str,
                        help="Directory containing stats files")
    args = parser.parse_args()

    max_idx = None
    to_print = []
    for i, fil in enumerate(["train_reconstruction_accuracy", "val_reconstruction_accuracy",
                             "train_overall_accuracy", "val_overall_accuracy"]):
        filename = os.path.join(args.stats_dir, "{}.txt".format(fil))
        with open(filename, 'r') as infile:
            vals = [float(x) for x in infile.readlines()]
            if i == 0:
                print("Epochs:", len(vals))

            # Get index of maximum training reconstruction accuracy
            if max_idx == None:
                max_idx = vals.index(max(vals))

            print(fil, vals[max_idx])
            to_print.append("{:.4f}".format(vals[max_idx]))
    print('\t'.join(to_print))
