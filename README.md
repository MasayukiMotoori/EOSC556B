# EOSC556B: Applied Geophysics, UBC,  2024 Winter session
## Induced Polarization effect on Time domain Electromagnetic for Sea floor hydrothermal deposit exploration. Synthetic Study inspired by real data.

This repository is about cthe ourse Project.  
Python code about Foward simulation and Inversion.
Induced Polarization(IP) models are based on cole-cole or pelton model.
Model are based on deep Sea Massive Sulfide exploration inspired by JOGMEC survey.

Foward simulation uses empymod. (See the instruction bellow.)
Jacobian is approximated by finite difference.
Steepest descend and Gauss-Newton method are available for optimization.
Jupyter note books demonstrate impact of IP parameter,
 one-dimensional inversion and objective function grids. 

## Components
1. python file.
2. Jupyter notebooks which import python file above.

## Installment instruction
1. Follow the instructions bellow to Install empymod.
https://empymod.emsig.xyz/en/stable/manual/installation.html
2. Import class "EMIP1D" from python file and run jupyter notebook


## Notebook Descriptions:
Project.py: ???

Project_EMIP_1D_multilayer.ipynb: ???

Project_EMIP_1D_objgrid.ipynb: ???

Project_EMIP_1D_test.ipynb: ???


## Installation:
???

## References
empymod  

Open-source full 3D electromagnetic modeller for 1D VTI media 
https://empymod.emsig.xyz/en/stable/gallery/tdomain/cole_cole_ip.html#sphx-glr-gallery-tdomain-cole-cole-ip-py   
"TDEM survey on Deep seamassive sulfide and IP impacts on data"
K. Nakayama,(2019), Application of Time-Domain Electromagnetic Survey for Seafloor Polymetallic Sulphides in the Okinawa Trough  
https://www.earthdoc.org/content/papers/10.3997/2214-4609.201902383

