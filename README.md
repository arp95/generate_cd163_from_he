# Generate CD163 IHC from H&E input using VISTA framework
---


## Summary
This code generates the CD163 IHC markings for the H&E image input (patch size can be 1024x1024, 2000x2000, 3000x3000 and should be in '.png' format). We have demonstrated the model performing better than the current state of the art frameworks.

![Screenshot](screenshots/Final_Figure1.png)


## Authors
Arpit Aggarwal, German Corredor, and Anant Madabhushi <br>


## Prerequisites
1. Python 3.8


## Packages Required
The packages required for running this code are PyTorch, PIL, OpenCV, Numpy.<br>


## Workflow for TAM identification
![Screenshot](screenshots/Final_Figure2.png)


## Running code
python3 test.py --dataroot 'dataset_path' --name cd163 --model pix2pix

('dataset_path' -> path which has a 'test' folder inside and the 'test' folder contains the H&E images used as input to the model. The CD163 IHC markings are stored in 'results' folder in local directory)


## License and Usage
Madabhushi Lab - This code is made available under Apache 2.0 with Commons Clause License and is available for non-commercial academic purposes.