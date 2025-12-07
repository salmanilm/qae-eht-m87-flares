# QAE-EHT-M87-Flares
8 Qubit Quantum Autoencoder for Event Horizon Telescope M87* Flare Detection

**Description**: Compares quantum vs classical autoencoder anomaly detection on black hole accretion disk shadows using EHT M87* data.  

## Research Question
**Can a simple quantum program of 8 qubits spot flares from generated black hole M87 images as accurately as a regular AI, while using 5 times fewer adjustable settings?**

## Status/Changelog <br />
Week 1-4 - Data synthesis <br />
 downloaded FITS file of M 87* from https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/681/A79#/browse <br />
 should be EHT observations taken on 2018 April 21 at band 3 <br />
 did an initial exploration of the data and made some graphs using initial_data_exploration.ipynb, the graphs themselves are in the outputs of the notebooks<br />
 made data_synthesis.ipynb it uses: monte carlo sampling for image generation, gaussian flare rendering, uniform noise generation, and saves both kelvin and normalized version of the images<br />
 in data_synthesis.ipynb there is a graph comparing median peak brightness for all flare types, as well as how many ring flares are landing on the bright crescent and how many of them are faint to the human eye<br />
 also added a visual of some sample images created using the seed 67 (also in notebooks)
 


## License
[MIT License](LICENSE) © 2025 Salman Ali

## Contact
Salman Ali
s.alikvoth@gmail.com
QSYS 2024 Alumni | Gr.12 Student


## Citations
"This research has made use of the VizieR catalogue access tool, CDS,
 Strasbourg, France (DOI : 10.26093/cds/vizier). The original description 
 of the VizieR service was published in 2000, A&AS 143, 23"

 "Andrew Chael, Hybrid GRMHD and force-free simulations of black hole accretion, 
 Monthly Notices of the Royal Astronomical Society,
 Volume 532, Issue 3, August 2024, Pages 3198–3221, 
 https://doi.org/10.1093/mnras/stae1692"
