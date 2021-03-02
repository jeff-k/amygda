#!/usr/bin/env bash

array_contains () {
    local array="$1[@]"
    local seeking=$2
    local in=1
    for element in "${!array}"; do
        if [[ $element == "$seeking" ]]; then
            in=0
            break
        fi
    done
    return $in
}


if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <user_id> <training or wells or extra>"
    echo "Example: $0 Curie wells"
    exit 1
fi

# configs
amygda_home="$HOME/amygda/"
dataset_folder="/media/penelopeprime/bacterial_growth_imaging"
virtualenv_activate_script="$HOME/.local/share/virtualenvs/amygda-CYtbES-g/bin/activate"
user=$1
dataset_type=$2
output_folder="${dataset_folder}/outputs/${user}/"
participants_datasets_folder="${dataset_folder}/participants_datasets"
scientist_folder="${participants_datasets_folder}/${user}"
all_participants=$(ls "$participants_datasets_folder" | tr '\n' ' ')

if [[ ! -d "$scientist_folder" ]]
then
    echo "Error: participant not found. Should be one of these values (the one given to you): ${all_participants}."
    exit 1
fi


dataset_types=(wells extra training)
array_contains dataset_types "$dataset_type" && echo "Running ${dataset_type} for ${user}" || (echo "The dataset type (second argument) must be \"wells\", \"extra\" or \"training\"" && exit 1)

source "$virtualenv_activate_script"
cd "$dataset_folder" || exit
mkdir -p "$output_folder"

python "${amygda_home}"/manual_annotator/manual_annotator.py  \
--wells_csv "${participants_datasets_folder}/${user}/${dataset_type}".csv \
--output_csv "${output_folder}"/output_"${user}"_"${dataset_type}".csv

deactivate
