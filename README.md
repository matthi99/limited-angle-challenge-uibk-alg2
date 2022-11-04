# Limited Angle Challenge

This repository is created for participating at the Helsinki Tomography Challenge (HTC2022) presented by the Finnish Inverse Problems Society (FIPS)
which allows to test reconstruction algorithms on real-world data.


## Algorithm description

In our presented approach we combined a well studied reconstruction technique with novel deep learning approaches. Judying from the given exemplary data and respective phantoms, it is safe to assume that the sought reconstruction can be interpreted as a piecewise constant function. We exploit this a-priori information and apply total variation regularization to obtain an inital estimate of the scanned object. This reconstruction is then processed by a neural network using an U-net backbone architecture to return a binary segmentation mask. To train the neural network, we generated 2.500 masks with similar shapes as the data provided by the HTC2022 challenge. Using the Python module *ASTRA-Toolbox*, we then simulated the projection 
to calculate the sinograms of the synthetic data set.

## Requirements

In the repository we provide an environment.yml file as well as a requirements.txt file.

## Usage instructions

To run the code, first install all necessary packages via
```
conda env create --file=env_lac.yaml
```

Then call the main.py function from the terminal with
```
python3 main.py path/to/input/files path/to/output/files difficulty
```

## Examples

The following images show our exemplary reconstructions for the phantoms provied by FIPS. The four provided phantoms are named "ta", "tb", "tc" and "td". From left to right, each column of images shows the reconstruction of these phantoms for each difficulty level. The left column always shows the original phantom ta, tb, tc or td, respectively. The right column of each block shows our binary reconstruction result for difficulty level 1, 2, ..., 7 in each respective row.

<p float="left">
  <img src="https://github.com/matthi99/limited-angle-challenge-uibk/blob/main/results/ex_ta.png" alt="" title="">
  <img src="https://github.com/matthi99/limited-angle-challenge-uibk/blob/main/results/ex_tb.png" alt="" title="">
  <img src="https://github.com/matthi99/limited-angle-challenge-uibk/blob/main/results/ex_tc.png" alt="" title="">
  <img src="https://github.com/matthi99/limited-angle-challenge-uibk/blob/main/results/ex_td.png" alt="" title="">
</p>


## Authors of Limited-Angle-Challenge

Matthias Schwab<sup>2</sup>, Simon Göppel<sup>1</sup>, Markus Tiefenthaler<sup>2</sup>, Christoph Angermann<sup>1</sup>

<sup>1</sup> Department of Mathematics, University of Innsbruck, Technikerstrasse 13, 6020 Innsbruck, Austria

<sup>2</sup> University Hospital for Radiology, Medical University Innsbruck, Anichstraße 35, 6020 Innsbruck, Austria
