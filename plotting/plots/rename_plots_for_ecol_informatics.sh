#!/bin/bash

# Define the destination directory
DEST_DIR="ecological_informatics"

# Declare an associative array for direct file renaming from 'formalization'
declare -A formalization_files=(
    ["formalization/voronoi.pdf"]="Figure_3_1.pdf"
    ["formalization/delauny.pdf"]="Figure_3_2.pdf"
    ["formalization/delauny_with_subgrpahs.pdf"]="Figure_3_3.pdf"
)

# Define source directories and their corresponding file renaming patterns
declare -A src_dirs=(
    ["problem_issue"]="Figure_1_"
    ["evaluation/Anthus_trivialis"]="Figure_12_"
    ["evaluation/Certhia_brachydactyla"]="Figure_A_1_"
    ["evaluation/Erithacus_rubecula"]="Figure_10_"
    ["evaluation/Fringilla_coelebs"]="Figure_A_2_"
    ["evaluation/Muscicapa_striata"]="Figure_A_3_"
    ["evaluation/Phoenicurus_phoenicurus"]="Figure_A_4_"
    ["evaluation/Phylloscopus_collybita"]="Figure_A_5_"
    ["evaluation/Sylvia_atricapilla"]="Figure_7_"
    ["evaluation/Sylvia_borin"]="Figure_A_6_"
    ["evaluation/Troglodytes_troglodytes"]="Figure_8_"
    ["evaluation/Turdus_philomelos"]="Figure_11_"
    ["evaluation/Analysis/interference/Phoenicurus_phoenicurus"]="Figure_13_"
    ["evaluation/Analysis/interference/Turdus_philomelos"]="Figure_13_"
    ["evaluation/Analysis/methodological_errors"]="Figure_14_"
    ["evaluation/Troglodytes_troglodytes_short_timespans"]="Figure_9_"
)

# Function to copy and rename files
copy_and_rename_files() {
    local src_dir="$1"
    local file_prefix="$2"
    local counter=1

    if [ ! -d "$src_dir" ]; then
        echo "Warning: Source directory '$src_dir' not found, skipping."
        return
    fi

    for file in "$src_dir"/*.pdf; do
        [ -f "$file" ] || { echo "Warning: No PDF files found in '$src_dir'."; break; }
        new_name="${file_prefix}${counter}.pdf"
        cp -v "$file" "$DEST_DIR/$new_name"
        ((counter++))
    done
}

# Ensure destination directory exists
mkdir -p "$DEST_DIR"

# Copy formalization files with direct mapping
for src_file in "${!formalization_files[@]}"; do
    if [ -f "$src_file" ]; then
        cp -v "$src_file" "$DEST_DIR/${formalization_files[$src_file]}"
    else
        echo "Warning: File '$src_file' not found, skipping."
    fi
done

# Loop through all source directories and apply the renaming function
for dir in "${!src_dirs[@]}"; do
    copy_and_rename_files "$dir" "${src_dirs[$dir]}"
done

echo "All files successfully copied and renamed in '$DEST_DIR'."
