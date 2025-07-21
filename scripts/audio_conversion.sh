#!/bin/bash 


male_dir="../.data/VoxCeleb_gender/males"
female_dir="../.data/VoxCeleb_gender/females"

output_dir="../.data/processed2"
convert_m4a_files() {
    local dir=$1  # Directory to process
    local gender=$2
    destination="${output_dir}/${gender}"

    # create the directory if not exists 
    mkdir -p "${destination}"
    echo $destination
    echo "Processing ${gender}"
    # Loop through all .m4a files in the given directory
    for file in "${dir}"/*.m4a; do
        filename=$(basename "$file" .m4a)
        # echo ${filename}
        ffmpeg -i "${file}" "${destination}/${filename}.wav" &
        while [ $(jobs | wc -l) -ge 5 ]; do
            wait -n  # Wait for any background process to finish before continuing
        done
    done 
}


convert_m4a_files "$male_dir" "male"
convert_m4a_files "$female_dir" "female"

