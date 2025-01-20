#!/bin/bash

# Creating addresses
python 1_create_adds.py

# Creating Text Files
python 2_create_text_files.py

# Check if h5 is True
if [ "$h5" = true ] ; then
    # Creating h5 format
    python 3_create_h5_dataset.py
fi

# Check h5 format
# python 4_check_h5.py
