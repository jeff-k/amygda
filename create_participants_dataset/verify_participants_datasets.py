import pandas as pd


def get_nb_common_wells (csv_1, csv_2):
    wells_1 = [line.split("/")[-1] for line in csv_1["wells"]]
    wells_2 = [line.split("/")[-1] for line in csv_2["wells"]]
    common_elements = set(wells_1).intersection(wells_2)
    return len(common_elements)


predefined_names = "Einstein,Curie,Newton,Darwin,Hawking,Edison,Sanger,Sulston,Kendrew,Tesla,Pasteur,Franklin,Watson,Crick,Lovelace,Planck,Boyle,Jenner,Fleming,Hodgkin,Avery,Waksman,Luria,Petri,Leeuwenhoek,Koch,Ehrlich,Mendel,Linnaeus,Wallace".split(",")
predefined_names = predefined_names[:20]


def print_common_wells(csv):
    print(f"Veryfing {csv}...")
    wells_csvs = {}
    for name in predefined_names:
        wells_csvs[name] = pd.read_csv(f"/home/leandro/git/amygda/participants_dataset/participants_datasets/{name}/{csv}")

    for name_1 in predefined_names:
        for name_2 in predefined_names:
            if name_1 != name_2:
                nb_common_wells = get_nb_common_wells(wells_csvs[name_1], wells_csvs[name_2])
                if nb_common_wells>0:
                    print(f"Common wells between {name_1} and {name_2}: {nb_common_wells}")

print_common_wells("wells.csv")
print_common_wells("extra.csv")