import argparse
import pandas as pd
from pathlib import Path
import random
import os
import re
import shutil

predefined_names = "Einstein,Curie,Newton,Darwin,Hawking,Edison,Sanger,Sulston,Kendrew,Tesla,Pasteur,Franklin,Watson,Crick,Lovelace,Planck,Boyle,Jenner,Fleming,Hodgkin,Avery,Waksman,Luria,Petri,Leeuwenhoek,Koch,Ehrlich,Mendel,Linnaeus,Wallace".split(",")
rows_to_consider = "ABGH"

def get_total_number_of_wells(all_plates_translation_csv):
    global rows_to_consider
    number_of_considered_well_per_plate = len(rows_to_consider) * 12 # 4 because we are just looking at the first and last 2 rows
    total_number_of_wells = len(all_plates_translation_csv) * number_of_considered_well_per_plate
    return total_number_of_wells


def get_all_control_wells(all_plates_translation_csv, use_filtered_images):
    column = "anonymous_plate_path_dir_well_split"
    if use_filtered_images: column+="_filtered"

    all_control_wells = []
    for well_split_dir in all_plates_translation_csv[column]:
        all_control_wells.append(f"{well_split_dir}/H11.png")
        all_control_wells.append(f"{well_split_dir}/H12.png")
    return all_control_wells


def get_all_non_control_wells(all_plates_translation_csv, use_filtered_images):
    global rows_to_consider
    column = "anonymous_plate_path_dir_well_split"
    if use_filtered_images: column += "_filtered"

    all_non_control_wells = []
    for well_split_dir in all_plates_translation_csv[column]:
        for letter in rows_to_consider:
            for number in range(1, 13):
                if (letter, number) != ("H", 11) and (letter, number) != ("H", 12):
                    all_non_control_wells.append(f"{well_split_dir}/{letter}{number}.png")
    return all_non_control_wells

def partition(list_of_values):
    list_of_values = list_of_values.copy()
    partitioned_values = []
    while len(list_of_values) > 3:
        partitioned_values.append([list_of_values.pop(0), list_of_values.pop(0)])
    partitioned_values.append(list_of_values)
    return partitioned_values


def add_shared_control_wells_for_pairs_of_participants(participant_index_to_list_of_its_wells,
                                                       all_control_wells,
                                                       number_of_shared_control_wells_per_part):
    random.shuffle(all_control_wells)
    partitioned_participants = partition(participant_index_to_list_of_its_wells)
    for partitioned_participants in partitioned_participants:
        for i in range(number_of_shared_control_wells_per_part):
            well_to_be_added = all_control_wells.pop()
            for participant_well in partitioned_participants:
                participant_well.append(well_to_be_added)


def add_wells_for_each_participant(participant_index_to_list_of_its_wells,
                                           wells,
                                           amount):
    random.shuffle(wells)
    for participant_well in participant_index_to_list_of_its_wells:
        for i in range(amount):
            participant_well.append(wells.pop())


def add_control_wells_for_each_participant(participant_index_to_list_of_its_wells,
                                           all_control_wells,
                                           number_of_non_shared_control_wells_per_part):
    add_wells_for_each_participant(participant_index_to_list_of_its_wells, all_control_wells, number_of_non_shared_control_wells_per_part)


def add_non_control_wells_for_each_participant(participant_index_to_list_of_its_wells,
                                           all_non_control_wells,
                                           number_of_non_control_wells_per_part):
    add_wells_for_each_participant(participant_index_to_list_of_its_wells, all_non_control_wells,
                                   number_of_non_control_wells_per_part)


def add_extra_control_wells_per_participant(participant_index_to_list_of_its_extra_control_wells,
                                            all_control_wells):
    random.shuffle(all_control_wells)

    i = 0
    while len(all_control_wells) > 0:
        participant_well = participant_index_to_list_of_its_extra_control_wells[ i % len(participant_index_to_list_of_its_extra_control_wells) ]
        participant_well.append(all_control_wells.pop())
        i += 1


def create_list_with_n_empty_lists(n):
    the_list = []
    for i in range(n):
        the_list.append([])
    return the_list


def make_participants_datasets(number_of_participants, number_of_control_wells_per_part,
                               number_of_non_control_wells_per_part, number_of_shared_control_wells_per_part,
                               all_control_wells, all_non_control_wells):
    participant_index_to_list_of_its_wells = create_list_with_n_empty_lists(number_of_participants)

    nb_of_control_wells_before = len(all_control_wells)
    add_shared_control_wells_for_pairs_of_participants(participant_index_to_list_of_its_wells,
                                                       all_control_wells,
                                                       number_of_shared_control_wells_per_part)
    nb_of_control_wells_after = len(all_control_wells)
    nb_of_pairs = int(number_of_participants/2)
    assert nb_of_control_wells_before == nb_of_control_wells_after + (nb_of_pairs * number_of_shared_control_wells_per_part)


    number_of_non_shared_control_wells_per_part = number_of_control_wells_per_part - number_of_shared_control_wells_per_part
    assert number_of_non_shared_control_wells_per_part >= 0
    nb_of_control_wells_before = len(all_control_wells)
    add_control_wells_for_each_participant(participant_index_to_list_of_its_wells,
                                           all_control_wells,
                                           number_of_non_shared_control_wells_per_part)
    nb_of_control_wells_after = len(all_control_wells)
    assert nb_of_control_wells_before == nb_of_control_wells_after + (
                number_of_participants * number_of_non_shared_control_wells_per_part)


    nb_of_non_control_wells_before = len(all_non_control_wells)
    add_non_control_wells_for_each_participant(participant_index_to_list_of_its_wells,
                                           all_non_control_wells,
                                           number_of_non_control_wells_per_part)
    nb_of_non_control_wells_after = len(all_non_control_wells)
    assert nb_of_non_control_wells_before == nb_of_non_control_wells_after + (number_of_participants * number_of_non_control_wells_per_part)

    participant_index_to_list_of_its_extra_control_wells = create_list_with_n_empty_lists(number_of_participants)
    add_extra_control_wells_per_participant(participant_index_to_list_of_its_extra_control_wells,
                                            all_control_wells)
    assert len(all_control_wells) == 0

    return participant_index_to_list_of_its_wells, participant_index_to_list_of_its_extra_control_wells


def put_wells_in_dirs(wells, dir):
    wells_order = []
    random.shuffle(wells)
    for well in wells:
        new_well_name = re.match(".*/(\d+(_filtered)?/.*\.png)", well).group(1)
        new_well_name = new_well_name.replace("/", "_")
        new_well_path = dir/new_well_name
        shutil.copy(well, new_well_path)
        wells_order.append(new_well_path)
    return wells_order


def create_participants_datasets_in_disk(participant_index_to_list_of_its_wells, participant_index_to_list_of_its_extra_control_wells,
                                         output_dir):
    global predefined_names
    output_dir.mkdir(parents=True)
    for participant_name, participant_wells, participant_extra_wells in \
        zip(predefined_names, participant_index_to_list_of_its_wells, participant_index_to_list_of_its_extra_control_wells):
        output_dir_for_this_participant = output_dir / participant_name
        well_output_dir_for_this_participant = output_dir_for_this_participant / "wells"
        well_output_dir_for_this_participant.mkdir(parents=True)
        participant_wells_order = put_wells_in_dirs(participant_wells, well_output_dir_for_this_participant)

        extra_output_dir_for_this_participant = output_dir / participant_name / "extra"
        extra_output_dir_for_this_participant.mkdir(parents=True)
        participant_extra_wells_order = put_wells_in_dirs(participant_extra_wells, extra_output_dir_for_this_participant)

        participant_wells_order_csv = pd.DataFrame.from_dict({"wells": participant_wells_order})
        participant_wells_order_csv.to_csv(output_dir_for_this_participant/"wells.csv", index=False)
        participant_extra_wells_order_csv = pd.DataFrame.from_dict({"wells": participant_extra_wells_order})
        participant_extra_wells_order_csv.to_csv(output_dir_for_this_participant/"extra.csv", index=False)



def force_add_control_wells_to_participants_list(force_add_controls_csv, participant_index_to_list_of_its_wells, use_filtered_images):
    controls_csv = pd.read_csv(force_add_controls_csv)
    all_control_wells = get_all_control_wells(controls_csv, use_filtered_images)
    nb_of_participants = len(participant_index_to_list_of_its_wells)
    nb_of_control_wells_per_part = int(len(all_control_wells) / nb_of_participants)
    add_control_wells_for_each_participant(participant_index_to_list_of_its_wells,
                                           all_control_wells,
                                           nb_of_control_wells_per_part)


def create_participants_dataset(all_plates_translation_csv_filepath, number_of_participants,
                                nb_of_wells_per_part,
                                percentage_of_images_from_control_wells,
                                percentage_of_shared_control_well_between_participants,
                                output_dir, use_filtered_images, force_add_controls_csv):
    all_plates_translation_csv = pd.read_csv(all_plates_translation_csv_filepath)
    total_number_of_wells = get_total_number_of_wells(all_plates_translation_csv)
    max_number_of_wells_per_part = int(total_number_of_wells / number_of_participants)

    nb_of_wells_per_part = min(nb_of_wells_per_part, max_number_of_wells_per_part)

    number_of_control_wells_per_part = int((percentage_of_images_from_control_wells/100)*nb_of_wells_per_part)
    assert 0 <= number_of_control_wells_per_part <= nb_of_wells_per_part

    number_of_non_control_wells_per_part = nb_of_wells_per_part - number_of_control_wells_per_part
    assert 0 <= number_of_non_control_wells_per_part <= nb_of_wells_per_part

    assert number_of_control_wells_per_part + number_of_non_control_wells_per_part == nb_of_wells_per_part

    number_of_shared_control_wells_per_part = int((percentage_of_shared_control_well_between_participants/100)*number_of_control_wells_per_part)

    all_control_wells = get_all_control_wells(all_plates_translation_csv, use_filtered_images)
    all_non_control_wells = get_all_non_control_wells(all_plates_translation_csv, use_filtered_images)
    try:
        participant_index_to_list_of_its_wells, participant_index_to_list_of_its_extra_control_wells =\
            make_participants_datasets(number_of_participants, number_of_control_wells_per_part,
                                   number_of_non_control_wells_per_part, number_of_shared_control_wells_per_part,
                                   all_control_wells, all_non_control_wells)
    except IndexError:
        print("You dont have enough wells for what you have asked, change params")
        os._exit(1)

    if force_add_controls_csv != None:
        force_add_control_wells_to_participants_list(force_add_controls_csv, participant_index_to_list_of_its_wells, use_filtered_images)

    create_participants_datasets_in_disk(participant_index_to_list_of_its_wells, participant_index_to_list_of_its_extra_control_wells,
                                         output_dir)





def get_args():
    parser = argparse.ArgumentParser(description='Create participants datasets.')
    parser.add_argument('--all_plates_translation_csv_filepath', type=str, help='all_plates.translation.csv output by create_split_well_images.py script', required=True)
    parser.add_argument('--nb_part', type=int, help='Number of participants', required=True)
    parser.add_argument('--nb_of_wells_per_part', type=int, help='Number of wells per participant', required=True)
    parser.add_argument('--control_well', type=float, help='Percentage of images that should come from control wells', required=True)
    parser.add_argument('--share_control_well', type=float, help='Percentage of control wells between pair of participants',
                        required=True)
    parser.add_argument('--output_dir', type=str, help='Directory to output the participants dataset', required=True)
    parser.add_argument('--filtered', action="store_true", help='Use the filtered images', default=False)
    parser.add_argument('--seed', type=int, help='Seed for random values', default=-1)
    parser.add_argument('--force_add_controls_csv', type=str, help='The control wells in this file will be forcedly distributed to the participants')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    all_plates_translation_csv_filepath = Path(args.all_plates_translation_csv_filepath)
    number_of_participants = args.nb_part
    percentage_of_images_from_control_wells = args.control_well
    percentage_of_shared_control_well_between_participants = args.share_control_well
    nb_of_wells_per_part = args.nb_of_wells_per_part
    output_dir = Path(args.output_dir)
    use_filtered_images = args.filtered
    seed = args.seed
    force_add_controls_csv = args.force_add_controls_csv
    if seed != -1:
        random.seed(seed)
    create_participants_dataset(all_plates_translation_csv_filepath, number_of_participants,
                                nb_of_wells_per_part,
                                percentage_of_images_from_control_wells,
                                percentage_of_shared_control_well_between_participants,
                                output_dir, use_filtered_images, force_add_controls_csv)

