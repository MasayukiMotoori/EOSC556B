# EOSC556B
EOSC556B: Applied Geophysics, UBC,  2024 Winter session

This is git hub repository for Project.  

This code is inversion for Induced Polarization parameter from TDEM.
Forward modelling uses a empymod.

Simulation model are based on deep Sea Massive Sulfide exploration.
Models are inspired by JOGMEC-WISTEM survey.

Induced polarization parameters are available either cole-cole model or the pelton model.
Steepest descend and Gauss-Newton method are available for optimization.
Jacobian is approximated by finite difference.
Plotting functions about objective value grids are also prepared.

Users are supposed to install Python file to use this code.
Jupyternotebooks are examples of how to use.

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
K. Nakayama,(2019), Application of Time-Domain Electromagnetic Survey for Seafloor Polymetallic Sulphides in the Okinawa Trough  
https://www.earthdoc.org/content/papers/10.3997/2214-4609.201902383

