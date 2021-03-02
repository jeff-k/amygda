import argparse
import shutil
from pathlib import Path

predefined_names = "Einstein,Curie,Newton,Darwin,Hawking,Edison,Sanger,Sulston,Kendrew,Tesla,Pasteur,Franklin,Watson,Crick,Lovelace,Planck,Boyle,Jenner,Fleming,Hodgkin,Avery,Waksman,Luria,Petri,Leeuwenhoek,Koch,Ehrlich,Mendel,Linnaeus,Wallace".split(",")

def get_args():
    parser = argparse.ArgumentParser(description='Add training dataset to participants dataset')
    parser.add_argument('--participants', type=str, help='Path to participants dataset', required=True)
    parser.add_argument('--training', type=str, help='Path to training dataset', required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    participants_folder = Path(args.participants)
    training_folder = Path(args.training)
    for name in predefined_names:
        participant_folder = participants_folder / name
        if participant_folder.exists():
            shutil.copytree(training_folder, participant_folder / "training")
            with open(participant_folder / "training.csv", "w") as csv_training:
                print("wells", file=csv_training)
                sorted_files = sorted(list(training_folder.iterdir()))
                for file in sorted_files:
                    print(f"participants_datasets/{name}/training/{file.name}", file=csv_training)