from pathlib import Path

dir_with_images = "examples/sample-images"
get_just_the_first_n_images = 100


all_input_images = list(str(path) for path in Path(dir_with_images).glob('**/*-raw.png'))
input_images = all_input_images[:get_just_the_first_n_images]
output_images = [image.replace("-raw", "-growth") for image in input_images]
print("output_images: ")
print(" ".join(output_images))

rule all:
    input: output_images

rule run_amygda:
    input:
        "{prefix}-raw.png"
    output:
        "{prefix}-growth.png"
    shell:
         "python analyse-plate-with-amygda.py --image {input}"
