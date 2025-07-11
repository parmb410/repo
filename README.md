# Integrated DIVERSIFY with Automated K Estimation and Extensions

This repository contains the implementation and experiments related to extending the DIVERSIFY framework for out-of-distribution (OOD) time series classification. It integrates automated latent domain estimation (Automated K Estimation), Curriculum Learning, Graph Neural Networks (GNN), and SHAP-based explainability.

---

## Repository Overview

- **IntegratedDiversify_K.ipynb**:  
  A comprehensive Jupyter notebook demonstrating the integrated pipeline of DIVERSIFY with Automated K Estimation. Includes training, evaluation, and visualization of results.

- **Additional notebooks**:  
  Other attached notebooks explore various extensions and experiments related to the DIVERSIFY framework.

- **Code base**:  
  Python scripts implementing the DIVERSIFY model, extensions, data preprocessing, training routines, and explainability methods.

---

## Background

Time series classification in real-world scenarios is often challenged by non-stationary distributions and latent domain shifts. The DIVERSIFY method tackles this by adversarially discovering worst-case latent domains and aligning their distributions to improve generalization on unseen test domains.

This repository extends DIVERSIFY by:  
- Automating latent domain estimation (`Automated K Estimation`)  
- Incorporating Curriculum Learning for progressive exposure to domain difficulty  
- Integrating Graph Neural Networks (GNN) to model inter-variable relations in multivariate time series  
- Adding SHAP-based explainability for domain-specific feature insights

---

## Features

- **Automated K Estimation**: Eliminates manual tuning by dynamically estimating the optimal number of latent domains.  
- **Curriculum Learning**: Ranks latent domains by difficulty and schedules training to improve stability and accuracy.  
- **Graph Neural Networks**: Captures temporal and spatial dependencies in multivariate series for richer feature extraction.  
- **SHAP Explainability**: Provides interpretable insights on feature contributions across domains.

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/parmb410/repo.git
   cd repo
