# Rock Climbing Coach

## Overview
This is the code repository for our CSE 237D Spring 2022 Project -- The Rock Climbing Coach.

The aim of this project is to generate a "post-climb report" from videos of rock climbers in action. The project makes use of ML and CV to detect holds on the rock wall as well as the position of the climber. We aggregate all of this information to produce 6 key metrics

1. Percentage of Route Completed
2. Climb Duration
3. Total Distance Moved
4. Number of Moves
5. Hold Usage Validity
6. Move Validity

More information on our approach and terminology can be found on our project [website](https://sites.google.com/view/rock-climbing-coach/).

## Requirements

To use this codebase for generating post-climb reports, one must have all the necessary requirements. We provide all libraries which are used in this project as part of the ```requirements.txt``` file. We recommend using a virtual environment during the setup. To properly set up the project run the following command from the root of this project:

```
pip install -r requirements.txt
```

# Usage

To run this project all requirements must have already been installed. The codebase expects an input directory which contains two files:

1. a .mp4 recording of the climber
2. a .jpg or .png image of the rock wall which was climbed

These are the only two files that must be contained within the input directory. To run the climb report follow these steps:

1. Switch to the ```src``` directory from the root via:
    
    ```cd src/```
2. Run the following command:

    ```python run.py -d test_data```

Here, we assume that ```test_data``` is the folder which contains the recording and the image, and that it is contained within ```src```. If this folder is stored elsewhere simply pass the path of the directory as the ```-d``` argument instead of ```test_data```.

That is all! We hope this tool is helpful to all those that use it!