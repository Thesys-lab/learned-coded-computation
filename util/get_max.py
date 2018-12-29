import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility script to get the reconstruction- and overall-"
        "accuracies of training a code. This script takes as"
        " input the directory containing stats files produced"
        " during training of a code, and reports the training,"
        " validation, and test reconstruction- and overall-accuracies"
        " that correspond to the maximum overall-accuracy"
        " attained over the validation set")
    parser.add_argument("stats_dir", type=str,
                        help="Directory containing stats files")
    args = parser.parse_args()

    # Get index to search for by using the validation overall accuracy
    val_file = os.path.join(
            args.stats_dir, "val_overall_accuracy.txt")
    with open(val_file, 'r') as infile:
        vals = [float(x) for x in infile.readlines()]
        max_idx = vals.index(max(vals))
        print("Epochs:", len(vals))

    to_print = []
    for fil in ["test_reconstruction_accuracy", "test_overall_accuracy",
                "val_reconstruction_accuracy", "val_overall_accuracy",
                "train_reconstruction_accuracy", "train_overall_accuracy"]:
        filename = os.path.join(args.stats_dir, "{}.txt".format(fil))
        with open(filename, 'r') as infile:
            vals = [float(x) for x in infile.readlines()]
            print(fil, vals[max_idx])
            to_print.append("{:.4f}".format(vals[max_idx]))
    print('\t'.join(to_print))
