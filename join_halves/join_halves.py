import argparse
import pandas as pd
from pathlib import Path
import os

def concat_and_symlink(csvs_filepaths, output_dir_filepath):
    output_dir = Path(output_dir_filepath)
    output_dir.mkdir(parents=True)
    output_dir_all_split_wells_concatenated = output_dir / "all_split_wells_concatenated"
    output_dir_all_split_wells_concatenated.mkdir(parents=True)
    old_csvs = pd.concat([pd.read_csv(csv_filepath) for csv_filepath in csvs_filepaths])

    new_csv = pd.DataFrame(columns=old_csvs.columns)

    for index, row in enumerate(old_csvs.iterrows()):
        row = row[1]
        original_plate_path_file = row["original_plate_path_file"]
        old_anonymous_plate_path_dir_well_split = row["anonymous_plate_path_dir_well_split"]
        old_anonymous_plate_path_dir_well_split_filtered = row["anonymous_plate_path_dir_well_split_filtered"]

        new_anonymous_plate_path_dir_well_split = output_dir_all_split_wells_concatenated / f"{index}"
        new_anonymous_plate_path_dir_well_split_filtered = output_dir_all_split_wells_concatenated / f"{index}_filtered"

        new_anonymous_plate_path_dir_well_split = new_anonymous_plate_path_dir_well_split.resolve()
        new_anonymous_plate_path_dir_well_split_filtered = new_anonymous_plate_path_dir_well_split_filtered.resolve()

        os.symlink(old_anonymous_plate_path_dir_well_split, new_anonymous_plate_path_dir_well_split, target_is_directory=True)
        os.symlink(old_anonymous_plate_path_dir_well_split_filtered, new_anonymous_plate_path_dir_well_split_filtered, target_is_directory=True)

        new_csv.loc[index] = [original_plate_path_file, new_anonymous_plate_path_dir_well_split, new_anonymous_plate_path_dir_well_split_filtered]

    new_csv.to_csv(output_dir / "all_plates.translation.csv", index=False)

def get_args():
    parser = argparse.ArgumentParser(description='Join two or more files created by create_split_well_script.py using symlinks.')
    parser.add_argument('--csvs', type=str, nargs="+", help='Path to all csvs', required=True)
    parser.add_argument('--output_dir', type=str, help='Path to all csvs', required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    csvs = args.csvs
    output_dir = args.output_dir

    print(f"Processing {csvs}")
    concat_and_symlink(csvs, output_dir)