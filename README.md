Integrated DIVERSIFY with Automated K Estimation and Extensions
This repository contains the implementation and experiments related to extending the DIVERSIFY framework for out-of-distribution (OOD) time series classification. It integrates automated latent domain estimation (Automated K Estimation), Curriculum Learning, Graph Neural Networks (GNN), and SHAP-based explainability.

Repository Overview
IntegratedDiversify_K.ipynb:
A comprehensive Jupyter notebook demonstrating the integrated pipeline of DIVERSIFY with Automated K Estimation. Includes training, evaluation, and visualization of results.

Additional notebooks:
Other attached notebooks explore various extensions and experiments related to the DIVERSIFY framework.

Code base:
Python scripts implementing the DIVERSIFY model, extensions, data preprocessing, training routines, and explainability methods.

Background
Time series classification in real-world scenarios is often challenged by non-stationary distributions and latent domain shifts. The DIVERSIFY method tackles this by adversarially discovering worst-case latent domains and aligning their distributions to improve generalization on unseen test domains.

This repository extends DIVERSIFY by:

Automating latent domain estimation (Automated K Estimation)

Incorporating Curriculum Learning for progressive exposure to domain difficulty

Integrating Graph Neural Networks (GNN) to model inter-variable relations in multivariate time series

Adding SHAP-based explainability for domain-specific feature insights

Features
Automated K Estimation: Eliminates manual tuning by dynamically estimating the optimal number of latent domains.

Curriculum Learning: Ranks latent domains by difficulty and schedules training to improve stability and accuracy.

Graph Neural Networks: Captures temporal and spatial dependencies in multivariate series for richer feature extraction.

SHAP Explainability: Provides interpretable insights on feature contributions across domains.

Installation
Clone the repository:

bash
Copy
git clone https://github.com/parmb410/repo.git
cd repo
(Optional) Create a virtual environment:

bash
Copy
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install required packages:

bash
Copy
pip install -r requirements.txt
Note: If requirements.txt is not available, install typical dependencies manually:

bash
Copy
pip install torch torchvision torchaudio
pip install torch-geometric
pip install scikit-learn
pip install shap
pip install matplotlib numpy pandas
Usage
Launch the Jupyter notebook server:

bash
Copy
jupyter notebook
Open IntegratedDiversify_K.ipynb and run the cells sequentially.

Modify parameters inside the notebook to experiment with different datasets, model hyperparameters, or extensions.

Datasets
The code supports multiple real-world time series datasets commonly used for OOD evaluation. Datasets may need to be downloaded and preprocessed separately. Instructions will be provided inside notebooks or scripts.

Results and Evaluation
The notebooks include training, validation, and target test set evaluation.

Metrics such as accuracy, H-divergence, and clustering quality are computed to assess generalization performance.

Visualizations illustrate latent domain partitions, curriculum progress, and SHAP explainability outcomes.

Citation
If you use this repository or the associated research in your work, please cite the original DIVERSIFY paper and any extensions you build upon:

bibtex
Copy
@inproceedings{lu2023ood,
  title={Out-of-Distribution Representation Learning for Time Series Classification},
  author={Lu, Tao and Liu, Cheng and Li, Sheng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}
Contributing
Contributions are welcome! Please open issues or pull requests for bug fixes, new features, or improvements.
