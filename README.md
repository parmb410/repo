# DIVERSIFY Framework with Automated K Estimation

This repository contains Jupyter notebooks and implementation scripts demonstrating the integration of automated K estimation into the DIVERSIFY framework for Out-of-Distribution (OOD) time series classification.

## Repository Contents

- **IntegratedDiversify\_K.ipynb**: This notebook illustrates the integration of the automated K estimation method into the DIVERSIFY pipeline. It includes data preprocessing, model training, evaluation, and visualization of results.

## Background

The original DIVERSIFY algorithm addresses the challenges posed by non-stationary distributions in time series data by adversarially modeling and aligning latent sub-domain distributions. This repository extends the algorithm's capabilities specifically by incorporating automated K estimation, which dynamically determines the optimal number of latent domains.

## Getting Started

### Installation

Clone this repository:

```bash
git clone https://github.com/parmb410/repo.git
cd repo
```

Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install torch torchvision torchaudio torch-geometric
pip install numpy pandas scikit-learn shap matplotlib jupyter
```

### Running Notebooks

Start the Jupyter notebook server:

```bash
jupyter notebook
```

Open the notebook directly in your browser and run the cells sequentially.

## Datasets

Ensure your datasets are placed according to paths specified in the notebook. Instructions for dataset preprocessing are detailed within the notebook.

## Evaluation Metrics

The notebook provides comprehensive evaluations using:

- Accuracy
- H-Divergence
- Clustering Quality

Visualizations include domain partitions and K estimation results.

## Citation

If you utilize this repository, please cite the original DIVERSIFY paper:

```bibtex
@inproceedings{lu2023ood,
  title={Out-of-Distribution Representation Learning for Time Series Classification},
  author={Lu, Tao and Liu, Cheng and Li, Sheng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}
```

## Contributions

Contributions and suggestions are welcome. Please open an issue or a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License.

