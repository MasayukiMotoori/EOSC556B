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
1\_test

Test about forward modelling method. 
Comparison between empymod and some analytical solution.

2\_IP parameter impact on forward simulation

Notebooks to explore IP parameter impact on forward simulation. \\
This demonstrates how each of 4 parameters, resistivity, changeability, \\
time constant, relaxation parameter impacts on simulation. \\
You can see a slight difference between the Cole-Cole and the Pelton model.

3\_object function

Inversion is an ill-posed problem and there is a non-uniqueness. This notebook performs one-layer inversion and plots the objective function grid about all model parameters during the inversion process(inversion trajectory). r08/rm and tc grids are about resistivity/changeability and time-constant/relaxation parameters respectively. This notebook shows how inversion changes model parameters during iteration. 
This notebook studies the range of non-uniqueness and ill-conditioned problems about this IP parameter inversion in TDEM.

An example of source localization problems in two-dimensional space may help us understand this idea. Sub-folder contains additional notebooks and figures about this example which was an assignment of CPSC406: Computational Optimization.

4_\multilayer

Notebooks about multilayer inversion.

## References
empymod  

Open-source full 3D electromagnetic modeller for 1D VTI media 
https://empymod.emsig.xyz/en/stable/gallery/tdomain/cole_cole_ip.html#sphx-glr-gallery-tdomain-cole-cole-ip-py   
"TDEM survey on Deep seamassive sulfide and IP impacts on data"
K. Nakayama,(2019), Application of Time-Domain Electromagnetic Survey for Seafloor Polymetallic Sulphides in the Okinawa Trough  
https://www.earthdoc.org/content/papers/10.3997/2214-4609.201902383

