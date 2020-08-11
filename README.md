# Image Quality Assessment

In this repository we implement several visual quality algorithms (FR-IQA indices) and apply them to the images of 
the Tampere Image Database 2013 (TID2013) ([website] [https://qualinet.github.io/databases/image/tampere_image_database_tid2013/])
to generate a data set with the values each index quantify for each image.
Then calculate several correlation and concordance coefficients between the generated data and the Mean Opinion Score 
(MOS) values of TID2013 images. Finally we rank the indices according to their overall performance.

IQA_indices.py implements the Mean Square Error (MSE), Signal to Noise Ratio (SNR), Peak Signal to Noise Ratio (PSNR) and Weighted Signal to Noise 
Ratio (WSNR), the Universal Quality Index (UQI), the Structural Similarity Index (SSIM) and Multi-scale Structural Similarity Index (MSSIM), Noise 
Quality Measure (NQM), Visual Information Fidelity (VIF), the Gradient Magnitude Similarity Mean (GMSM), the Gradient Magnitude Similarity 
Deviation (GMSD) and the codispersion coefficient based CQ-Index

These algorithms compare an original image (reference) with a modified version of it (query).
All algorithms implemented in this package requires operate on two images of the same size. 
For reference image and query image with distinct sizes, the indices are out of scope. 
 
Usually, these implemented indices are available in Matlab. 
Since Matlab is a non-free, hard-integration and restrictive software, 
the use of this code can avoid the dependency of Matlab in open-sources projects. 

All authors of algorithms and them articles are properly cited in "References" section of this document.

In the Scripts folder are the scripts to generate the data, implementations of the correlation and concordance coefficients, 
routines to compute them with this data, as well as the index ranking.

Data folder contains all the generated dataframes.

In Results are the tables with the results each correlation and concordance coefficient produced.

In Plots there is a jupyter-notebook to make some graphs with the data.

## License
Released under [GNU GPL version 2.](http://www.gnu.org/licenses/old-licenses/gpl-2.0.txt)

## Requeriments
Python 3.6 or later must be installed. 
In addition, it requires the installation of Setuptools, 
Python Image Library (PIL), OpenCV, NumPy, Pandas, SciPy and scikit-learn libraries.

## Contact
Please send all comments, questions, reports and suggestions to **lucia.pappaterra@unc.edu.ar**

## Acknowledges
Special thanks to Pedro Garcia for his wonderful library [PyMetrikz](https://gitlab.com/gpds-unb/pymetrikz). 
Many of the functions used in this work can be found there, but we also incorporated new ones.


## References

Tampere Image Database 2013 (TID2013)
- N. Ponomarenko, O. Ieremeiev, V. Lukin, K. Egiazarian, L. Jin, J. Astola, B. Vozel, K. Chehdi, M. Carli, F. Battisti and C. Jay Kuo,
"Color image database tid2013: Peculiarities and preliminary results" 4th European Workshop on Visual Information Processing EUVIP2013 , pp.106-111, 2013

- N. Ponomarenko, L. Jin, O. Ieremeiev, V. Lukin, K. Egiazarian, J. Astola, B. Vozel, K. Chehdi, M. Carli, F. Battisti and C. Jay Kuo,
"Image database tid2013: Peculiarities, results and perspectives" Signal Processing: Image Communication 30, pp 57-77, 2014

Structural Similarity Index (SSIM)
Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: From error visibility to structural similarity" IEEE Transactions on Image Processing, vol. 13, no. 4, pp.600-612, Apr. 2004

Multi-scale SSIM Index (MSSIM)
Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: From error visibility to structural similarity" IEEE Transactions on Image Processing, vol. 13, no. 4, pp.600-612, Apr. 2004

Noise Quality Measure (NQM)
N. Damera-Venkata, T. Kite, W. Geisler, B. Evans and A. Bovik, "Image Quality Assessment Based on a Degradation Model", IEEE Trans. on Image Processing, Vol. 9, No. 4, Apr. 2000

Universal Image Quality Index (UQI)
Zhou Wang and Alan C. Bovik, "A Universal Image Quality Index", IEEE Signal Processing Letters, 2001

Visual Information Fidelity (VIF)
H. R. Sheikh and A. C. Bovik, "Image Information and Visual Quality"., IEEE Transactions on Image Processing, (to appear).

Weighted Signal-to-Noise Ratio (WSNR)
T. Mitsa and K. Varkur, "Evaluation of contrast sensitivity functions for the formulation of quality measures incorporated in halftoning algorithms", ICASSP '93-V, pp. 301-304.

Signal-to-Noise Ratio (SNR, PSNR)
J. Mannos and D. Sakrison, "The effects of a visual fidelity criterion on the encoding of images", IEEE Trans. Inf. Theory, IT-20(4), pp. 525-535, July 1974

CQ-index (CQ)
Ojeda Silvia M., Vallejos Ronny O., Lamberti Pedro W., 2012. Measure of similarity between images based on the codispersion coefficient. Journal of Electronic Imaging. 

Gradient magnitude similarity measures (GMSM, GMSD)
Xue W., Zhang L., Mou X., and Bovik A. C., 2014. Gradient magnitude similarity deviation: A highly efficient perceptual image quality index. IEEE Trans. Image Process., vol. 23, no. 2, pp. 684-695.
