# QAE-EHT-M87-Flares
12 Qubit Quantum Autoencoder for Event Horizon Telescope M87* Flare Detection

**Description**: Compares quantum vs classical autoencoder anomaly detection on black hole accretion disk shadows using EHT M87* data.  

## Research Question
**Can a simple quantum program of 12 qubits spot flares from generated black hole M87 images as accurately as a regular AI, while using vastly fewer adjustable settings?**

## Status/Changelog <br />
There will be some gaps in the changelogs due to dense academic weeks so I apologize for that, just take the week timeline as how long each section has been worked on respectively <br />

## Results <br />

All data is synthetic, flares were artificially injected into M87* 2018 images. Real EHT data contains observational noise, calibration artifacts, and lower flare contrast. Consider a drastic performance degradation on real data (i.e 8-30%). <br />

Limitations: This is a methodology demonstration. Real EHT data requires: <br />
GRMHD simulation-based flares (physics-based) <br />
Noise injection from actual EHT arrays <br />
Calibration uncertainty quantification <br />
Error mitigation for NISQ hardware <br />
<br />

**ROC AUC**: Measures how good the model is at telling flares apart from normal images. <br />
**F1-Score**: Balances finding real flares versus false readings. <br />
**Parameters**: Counts how many adjustable parts the model needs to learn patterns. <br />
**Training Speed**: Shows how quickly the model learns from the data. <br />
<br />

| Metric             | Classical CNN | Quantum Hybrid | Advantage         |
| ------------------ | ------------- | -------------- | ----------------- |
| **ROC AUC**        | 0.923         | **0.991**      | **+6.8%**         |
| **F1-Score**       | 0.917          | **0.98**       | **+6%**         |
| **Parameters**     | 85,185        | **1052**        | **~81× reduction** |
| **Training Speed** | 20 epochs     | **5 epochs**   | **3.6× faster**   |

**Statistical Significance: Bootstrap analysis confirms quantum improvement (p < 0.05)**


## Changelog <br />
Week 1-4 - Data synthesis <br />
 Downloaded FITS file of M 87* from https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/681/A79#/browse <br />
 Should be EHT observations taken on 2018 April 21 at band 3 <br />
 Did an initial exploration of the data and made some graphs using initial_data_exploration.ipynb, the graphs themselves are in the outputs of the notebooks <br />
 Made data_synthesis.ipynb it uses: monte carlo sampling for image generation, gaussian flare rendering, uniform noise generation, and saves both kelvin and normalized version of the images <br />
 In data_synthesis.ipynb there is a graph comparing median peak brightness for all flare types, as well as how many ring flares are landing on the bright crescent and how many of them are faint to the human eye <br />
 Also added a visual of some sample images created using the seed 67 (also in notebooks) <br />
 <br />

Week 5-6 - Designing autoencoder <br />
 85,185 Paramaters <br />
 Compress 16 pixels to 64 features for easier handling, uses normalized images/features <br />
 Uses 4*4 patch input, this is done to avoid padding artifacts. I would have liked to use some more but I was unable to leave my system on for long periods <br />
 Uses MaxPool2d(2) to downsample to 2 pixels, then uses ConvTranspose2d upsampling, kernel_size=4, stride=2, to try and avoid interpolation artifacts that could smooth out flare signatures even further <br />
 Uses AdamW with a weight decay of 1e-4 to act as implicit regularization to avoid overfitting <br />
 <br />

Week 7-9 - Designing hybrid quantum autoencoder <br />
 ~1,052 Paramters <br />
 12-qubit amplitude encoding for 4*4 patches turned into 4096-dim Hilbert Space <br />
 Ring-entangled variational circuit (3 layers, 108 rotation spaces) <br />
 Hybrid decoder, 12->32->16 with Xavier init <br />
 Per-patch MSE reconstruction error metric <br />
 <br />
 For encoding, padding 16->4096 dims vs. 4-qubit minimal encoding <br />
 For entanglement, ring topology vs. all-to-all <br />
 For batching, manual sample loop  <br />
  <br />
 Limitations include: <br />
  - Inference: O(batch_size) serial execution; ~X ms/sample on CPU  <br />
  - Gradient: Parameter-shift rule doubles quantum circuit evaluations <br />
 
 
 


## License
[MIT License](LICENSE) © 2025 Salman Ali

## Contact
Salman Ali
s.alikvoth@gmail.com
QSYS 2024 Alumni | Gr.12 Student @ Queen Elizabeth School (Edmonton, Alberta)


## Citations
"This research has made use of the VizieR catalogue access tool, CDS,
 Strasbourg, France (DOI : 10.26093/cds/vizier). The original description 
 of the VizieR service was published in 2000, A&AS 143, 23"

 "Andrew Chael, Hybrid GRMHD and force-free simulations of black hole accretion, 
 Monthly Notices of the Royal Astronomical Society,
 Volume 532, Issue 3, August 2024, Pages 3198–3221, 
 https://doi.org/10.1093/mnras/stae1692"
