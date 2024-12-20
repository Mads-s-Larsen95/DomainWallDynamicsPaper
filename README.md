# DomainWallDynamicsPaper
Data analysis based on the paper "Domain and domain wall structure of carbides in ferromagnetic steel revealed by off-axis electron holography and micromagnetic simulations", where all experimental data has been analyzed using HyperSpy, Numpy, SciPy, and Matplotlib.

Additional libraries used include glob, natsort, skimage, and cv2.

Codes available are:
 *DataAnalysis_April2024* : Data analysis of raw off-axis holography files to produce unwrapped phase and save as NP arrays
 *Holography_OnlyWorkOnNPArrays_UnwrappedPhase* : Work on the unwrapped phase images to produce phase profiles for analysis of DW dynamics etc.
 *OffAxisHolo_DataVisualization_LineProfile* : Visualization of the DW phase profiles and fitting of both the DW structure (Cosh function) and DW dynamics based on applied H field
 *TIE_Location1* : Solving the Transport of Intensity Equation (TIE) over 1 particle with various defocii
 *DataAnalysis_LocationC_Holo* : Off-axis holographic analysis on another particle, where the saturation magnetization was reached
 *TIE_And_Holo_Comparison* : Comparison between the TIE phase and off-axis holography phase profiles

Contact email: madsslarsen95@gmail.com
