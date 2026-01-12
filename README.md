This repository contains the Python scripts, model implementations, and documentation supporting the study:

PS-InSAR-supported deep learning and hybrid ensemble machine learning models for landslide susceptibility mapping in the upper part of the Sunkoshi River basin, Nepal.

The study integrates deep learning (DL), hybrid ensemble machine learning (ML), and PS-InSAR deformation analysis to generate high-resolution landslide susceptibility maps (LSMs) and to enhance model validation using ground deformation evidence.

This repository is made publicly available to ensure transparency, reproducibility, and reusability, in line with the data and code availability requirements of Computers & Geotechnics.

Repository Contents

Preprocessing scripts for landslide conditioning factors

Deep learning models (DNN, BPNN, RNN, LSTM)

Hybrid ensemble ML models (Voting, Stacking, Hybrid Bagging–Boosting, Meta-learning)

Model evaluation scripts including 10-fold stratified cross-validation and ROC–AUC analysis

🗺️ Study Area

The study focuses on the upper Sunkoshi River basin, Nepal, a tectonically active Himalayan region characterized by:

Steep topography

Complex lithology

High seismicity

Intense monsoonal rainfall

Frequent landslide activity along river corridors and major highways

🧠 Methodological Framework

The implemented workflow consists of the following major steps:

Landslide inventory preparation using field surveys and remote sensing interpretation

Derivation and preprocessing of landslide conditioning factors (LCFs)

Feature screening using Pearson correlation analysis, VIF, TOL, and SHAP

Development of predictive models:

Deep learning models: DNN, BPNN, RNN, LSTM

Hybrid ensemble ML models: Voting, Stacking, Hybrid Bagging–Boosting, Meta-learning

Model evaluation using accuracy, precision, recall, F1-score, Cohen’s Kappa, and ROC–AUC

Generation of landslide susceptibility maps

PS-InSAR deformation analysis and integration with susceptibility results for qualitative validation
