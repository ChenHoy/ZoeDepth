#!/bin/bash

# List of scenes
scenes=("P001" "P002" "P003" "P004" "P005" "P006" "P007" "P008" "P009")

# Loop over the scenes
for scene in "${scenes[@]}";do
	echo "Running inference on ${scene} ..."
	python demo.py --model ZoeD_NK --input_rgb_dir /media/data/TartanAir/seasidetown/Easy/${scene}/image_left/ --output_dir /media/data/TartanAir/seasidetown/Easy/${scene}/zoed_nk_left/
done
