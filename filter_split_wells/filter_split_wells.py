import re
import argparse
import pandas as pd
import random

def get_args():
    parser = argparse.ArgumentParser(description='Filter all_plates.translation.csv output by the create_split_well_images.py script to the interesting stuff.')
    parser.add_argument('--plates_translation_csvs', type=str, nargs="+", help='Path to all plates_translation_csvs', required=True)
    parser.add_argument('--nb_full_time_series', type=int, help='Number of full time series plates to keep', required=True)
    parser.add_argument('--output_prefix', type=str, help='Output_prefix', required=True)
    args = parser.parse_args()
    return args

def get_concatenated_df(plates_translation_csvs_filenames):
    plates_translation_csvs = [pd.read_csv(plates_translation_csvs_filename) for plates_translation_csvs_filename in
                               plates_translation_csvs_filenames]
    df = pd.concat(plates_translation_csvs)
    return df

def add_info_to_df (df):
    # add sites, subjects, labs, isolates, days and plate_names columns
    sites=[]
    subjects=[]
    labs=[]
    isolates=[]
    days=[]
    plate_names=[]
    for original_plate_path_file in df["original_plate_path_file"]:
        site, subject, lab, isolate, day, plate_name = re.match(".*half/(.+)/(.+)/(.+)/(.+)/(\d+)/(.+)-raw.png", original_plate_path_file).group(*list(range(1, 7)))
        sites.append(site)
        subjects.append(subject)
        labs.append(lab)
        isolates.append(isolate)
        days.append(day)
        plate_names.append(plate_name)
    df["sites"]=sites
    df["subjects"]=subjects
    df["labs"]=labs
    df["isolates"]=isolates
    df["days"]=days
    df["plate_names"]=plate_names



def get_series_plates_csv (df, nb_full_time_series):
    # identify the isolates with all time series
    grouped_df = df.groupby(by=["sites", "subjects", "labs", "isolates"]).count()
    isolates_with_all_time_series_keys = grouped_df[grouped_df.days == 4].index.values

    # shuffle them and get the n first
    random.shuffle(isolates_with_all_time_series_keys)
    isolates_with_all_time_series_keys = isolates_with_all_time_series_keys[:nb_full_time_series]

    # get the isolates with the 4 time series
    indexed_df = df.set_index(["sites", "subjects", "labs", "isolates"], drop=False)
    df_with_selected_time_series = [indexed_df.xs(isolates_with_all_time_series_key) for isolates_with_all_time_series_key in isolates_with_all_time_series_keys]
    df_with_selected_time_series = pd.concat(df_with_selected_time_series, ignore_index=True)

    return df_with_selected_time_series


def remove_rows_in_the_first_df_that_are_in_the_second(df_1, df_2):
    return pd.concat([df, df_with_selected_time_series]).drop_duplicates(keep=False)


def get_df_with_14_day_plates_only(df):
    return df[df.days=="14"]

if __name__ == "__main__":
    args = get_args()
    plates_translation_csvs = args.plates_translation_csvs
    nb_full_time_series = args.nb_full_time_series
    output_prefix = args.output_prefix
    print(f"Processing {plates_translation_csvs}")

    df = get_concatenated_df(plates_translation_csvs)
    add_info_to_df(df)

    df_with_selected_time_series = get_series_plates_csv(df, nb_full_time_series)

    df = remove_rows_in_the_first_df_that_are_in_the_second(df, df_with_selected_time_series)

    df_with_14_day_plates_only = get_df_with_14_day_plates_only(df)

    df_with_selected_time_series.to_csv(f"{output_prefix}.series_plates.csv", index=False)
    df_with_14_day_plates_only.to_csv(f"{output_prefix}.14_days.csv", index=False)