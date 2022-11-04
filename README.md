# Limited Angle Challenge

The repository is created to participate at the Helsinki Tomography Challenge (HTC2022) presented by the Finnish Inverse Problems Society (FIPS)
which allows to test reconstruction algorithms on real-world data.


## Algorithm description

Our approach combines established reconstruction methods with deep learning methods. Starting from a sinogramm, we use the total variation regularization to obtain an
initial estimate of the scanned object. This reconstruction is then processed by a neural network using an U-net backbone architecture to return
a binary segmentation mask. To train the neural network, we generated 2.500 masks similar to the data provided by the HTC2022 challenge. Using the Python module *ASTRA-Toolbox*, we simulate the projection to 
to calculate the sinograms of the additionally generated masks.

## Requirements

In the repository we provide an environment.yml file as well as a requirements.txt file.

## Usage instructions

To run the code call the main.py function with from the command line: python3 main.py path/to/input/files path/to/output/files difficulty

## Examples


### Difficulty 1

<img src="https://github.com/matthi99/limited-angle-challenge-uibk/blob/main/results/Difficulty_1_example_3.png" alt="" title="">

### Difficulty 3

<img src="https://github.com/matthi99/limited-angle-challenge-uibk/blob/main/results/Difficulty_3_example_2.png" alt="" title="">

### Difficulty 5

<img src="https://github.com/matthi99/limited-angle-challenge-uibk/blob/main/results/Difficulty_5_example_3.png" alt="" title="">

### Difficulty 7

<img src="https://github.com/matthi99/limited-angle-challenge-uibk/blob/main/results/Difficulty_7_example_2.png" alt="" title="">

## Authors of Limited-Angle-Challenge

Matthias Schwab<sup>2</sup>, Simon Göppel<sup>1</sup>, Markus Tiefenthaler<sup>2</sup>, Christoph Angermann<sup>1</sup>

<sup>1</sup> Department of Mathematics, University of Innsbruck, Technikerstrasse 13, 6020 Innsbruck, Austria

<sup>2</sup> University Hospital for Radiology, Medical University Innsbruck, Anichstraße 35, 6020 Innsbruck, Austria
