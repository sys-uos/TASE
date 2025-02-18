#!/bin/bash

# Define source directories
SRC_DIR_1="formalization"
SRC_DIR_2="problem_issue"
SRC_DIR_3="evaluation/Anthus_trivialis"
SRC_DIR_4="evaluation/Certhia_brachydactyla"
SRC_DIR_5="evaluation/Erithacus_rubecula"
SRC_DIR_6="evaluation/Fringilla_coelebs"
SRC_DIR_7="evaluation/Muscicapa_striata"
SRC_DIR_8="evaluation/Phoenicurus_phoenicurus"
SRC_DIR_9="evaluation/Phylloscopus_collybita"
SRC_DIR_10="evaluation/Sylvia_atricapilla"
SRC_DIR_11="evaluation/Sylvia_borin"
SRC_DIR_12="evaluation/Troglodytes_troglodytes"
SRC_DIR_13="evaluation/Turdus_philomelos"
SRC_DIR_14="evaluation/Analysis/interference/Phoenicurus_phoenicurus"
SRC_DIR_15="evaluation/Analysis/interference/Turdus_philomelos"
SRC_DIR_16="evaluation/Analysis/methodological_errors"
SRC_DIR_17="evaluation/Troglodytes_troglodytes_short_timespans"

DEST_DIR="ecological_informatics"

# Check if source directories exist
for dir in "$SRC_DIR_1" "$SRC_DIR_2" "$SRC_DIR_3"; do
    if [ ! -d "$dir" ]; then
        echo "Error: Source directory '$dir' not found!"
        exit 1
    fi
done

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Copy and rename files from 'formalization'
declare -A formalization_files=(
    ["voronoi.pdf"]="Figure_3_1.pdf"
    ["delauny.pdf"]="Figure_3_2.pdf"
    ["delauny_with_subgrpahs.pdf"]="Figure_3_3.pdf"
)

for file in "${!formalization_files[@]}"; do
    if [ -f "$SRC_DIR_1/$file" ]; then
        cp -v "$SRC_DIR_1/$file" "$DEST_DIR/${formalization_files[$file]}"
    else
        echo "Warning: File '$SRC_DIR_1/$file' not found, skipping."
    fi
done

# Copy and rename files from 'problem_issue'
counter=1
for file in "$SRC_DIR_2/"*.pdf; do
    if [[ -f "$file" ]]; then
        new_name="Figure_1_${counter}.pdf"
        cp -v "$file" "$DEST_DIR/$new_name"
        ((counter++))
    else
        echo "Warning: No PDF files found in '$SRC_DIR_2'."
        break
    fi
done

# Copy and rename files from 'evaluation/Anthus_trivialis'
eval_counter=1
for file in "$SRC_DIR_3/"*.pdf; do
    if [[ -f "$file" ]]; then
        new_name="Figure_12_${eval_counter}.pdf"
        cp -v "$file" "$DEST_DIR/$new_name"
        ((eval_counter++))
    else
        echo "Warning: No PDF files found in '$SRC_DIR_3'."
        break
    fi
done

# Copy and rename files from 'evaluation/Certhia_brachydactyla'
eval_counter=1
for file in "$SRC_DIR_4/"*.pdf; do
    if [[ -f "$file" ]]; then
        new_name="Figure_A_1_${eval_counter}.pdf"
        cp -v "$file" "$DEST_DIR/$new_name"
        ((eval_counter++))
    else
        echo "Warning: No PDF files found in '$SRC_DIR_4'."
        break
    fi
done

# Copy and rename files from 'evaluation/Erithacus_rubecula'
eval_counter=1
for file in "$SRC_DIR_5/"*.pdf; do
    if [[ -f "$file" ]]; then
        new_name="Figure_10_${eval_counter}.pdf"
        cp -v "$file" "$DEST_DIR/$new_name"
        ((eval_counter++))
    else
        echo "Warning: No PDF files found in '$SRC_DIR_5'."
        break
    fi
done

# Copy and rename files from 'evaluation/Fringilla_coelebs'
eval_counter=1
for file in "$SRC_DIR_6/"*.pdf; do
    if [[ -f "$file" ]]; then
        new_name="Figure_A_2_${eval_counter}.pdf"
        cp -v "$file" "$DEST_DIR/$new_name"
        ((eval_counter++))
    else
        echo "Warning: No PDF files found in '$SRC_DIR_6'."
        break
    fi
done

# Copy and rename files from 'evaluation/Muscicapa_striata'
eval_counter=1
for file in "$SRC_DIR_7/"*.pdf; do
    if [[ -f "$file" ]]; then
        new_name="Figure_A_3_${eval_counter}.pdf"
        cp -v "$file" "$DEST_DIR/$new_name"
        ((eval_counter++))
    else
        echo "Warning: No PDF files found in '$SRC_DIR_7'."
        break
    fi
done

# Copy and rename files from 'evaluation/Phoenicurus_phoenicurus'
eval_counter=1
for file in "$SRC_DIR_8/"*.pdf; do
    if [[ -f "$file" ]]; then
        new_name="Figure_A_4_${eval_counter}.pdf"
        cp -v "$file" "$DEST_DIR/$new_name"
        ((eval_counter++))
    else
        echo "Warning: No PDF files found in '$SRC_DIR_8'."
        break
    fi
done

# Copy and rename files from 'evaluation/Phylloscopus_collybita'
eval_counter=1
for file in "$SRC_DIR_9/"*.pdf; do
    if [[ -f "$file" ]]; then
        new_name="Figure_A_5_${eval_counter}.pdf"
        cp -v "$file" "$DEST_DIR/$new_name"
        ((eval_counter++))
    else
        echo "Warning: No PDF files found in '$SRC_DIR_9'."
        break
    fi
done

# Copy and rename files from 'evaluation/Sylvia_atricapilla'
eval_counter=1
for file in "$SRC_DIR_10/"*.pdf; do
    if [[ -f "$file" ]]; then
        new_name="Figure_7_${eval_counter}.pdf"
        cp -v "$file" "$DEST_DIR/$new_name"
        ((eval_counter++))
    else
        echo "Warning: No PDF files found in '$SRC_DIR_10'."
        break
    fi
done

# Copy and rename files from 'evaluation/Sylvia_borin'
eval_counter=1
for file in "$SRC_DIR_11/"*.pdf; do
    if [[ -f "$file" ]]; then
        new_name="Figure_A_6_${eval_counter}.pdf"
        cp -v "$file" "$DEST_DIR/$new_name"
        ((eval_counter++))
    else
        echo "Warning: No PDF files found in '$SRC_DIR_11'."
        break
    fi
done

# Copy and rename files from 'evaluation/Troglodytes_troglodytes'
eval_counter=1
for file in "$SRC_DIR_12/"*.pdf; do
    if [[ -f "$file" ]]; then
        new_name="Figure_8_${eval_counter}.pdf"
        cp -v "$file" "$DEST_DIR/$new_name"
        ((eval_counter++))
    else
        echo "Warning: No PDF files found in '$SRC_DIR_12'."
        break
    fi
done

# Copy and rename files from 'evaluation/Turdus_philomelos'
eval_counter=1
for file in "$SRC_DIR_13/"*.pdf; do
    if [[ -f "$file" ]]; then
        new_name="Figure_11_${eval_counter}.pdf"
        cp -v "$file" "$DEST_DIR/$new_name"
        ((eval_counter++))
    else
        echo "Warning: No PDF files found in '$SRC_DIR_13'."
        break
    fi
done


# Copy and rename files from 'evaluation/Analysis/interference/Phoenicurus_phoenicurus'
eval_counter=1
for file in "$SRC_DIR_14/"*.pdf; do
    if [[ -f "$file" ]]; then
        new_name="Figure_13_${eval_counter}.pdf"
        cp -v "$file" "$DEST_DIR/$new_name"
        ((eval_counter++))
    else
        echo "Warning: No PDF files found in '$SRC_DIR_14'."
        break
    fi
done

# Copy and rename files from 'evaluation/Analysis/interference/Turdus_philomelos'
eval_counter=2
for file in "$SRC_DIR_15/"*.pdf; do
    if [[ -f "$file" ]]; then
        new_name="Figure_13_${eval_counter}.pdf"
        cp -v "$file" "$DEST_DIR/$new_name"
        ((eval_counter++))
    else
        echo "Warning: No PDF files found in '$SRC_DIR_15'."
        break
    fi
done

# Copy and rename files from 'evaluation/Analysis/methodological_errors'
eval_counter=1
for file in "$SRC_DIR_16/"*.pdf; do
    if [[ -f "$file" ]]; then
        new_name="Figure_14_${eval_counter}.pdf"
        cp -v "$file" "$DEST_DIR/$new_name"
        ((eval_counter++))
    else
        echo "Warning: No PDF files found in '$SRC_DIR_16'."
        break
    fi
done

# Copy and rename files from 'evaluation/Troglodytes_troglodytes_short_timespans'
eval_counter=1
for file in "$SRC_DIR_17/"*.pdf; do
    if [[ -f "$file" ]]; then
        new_name="Figure_9_${eval_counter}.pdf"
        cp -v "$file" "$DEST_DIR/$new_name"
        ((eval_counter++))
    elsedd
        echo "Warning: No PDF files found in '$SRC_DIR_17'."
        break
    fi
done

echo "All files successfully copied and renamed in '$DEST_DIR'."

