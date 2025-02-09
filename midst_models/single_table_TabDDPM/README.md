# TabDDPM for Single Table Data Overview

## Introduction
This folder contains the implementation of ClavaDDPM, a recent method that employs diffusion-based generative models for multi-relational tabular data generation. When configured for single-table synthesis, it leverages the TabDDPM implementation. The provided Jupyter notebooks and code examples focus on single-table scenarios, guiding you through data loading, model training, sampling, and metrics computation.

## Notebooks
Here you will find the following Jupyter notebooks:
1. **TabDDPM** - This notebook covers the tabddpm implementation for single table synthesis. It includes a brief description of the algorithm, its implementation, and a demonstration of how to use it for sampling synthetic data. **Use this notebook when you want to load and sample from models stored in the `train` folder for both the `White Box MIA on single table` and `Black-box MIA on single table` competitions.**
2. **TabDDPM_wb_dev_final** - **This notebook is specifically designed to load and sample from models stored in the `dev` and `final (eval)` folders for the `White Box MIA on single table` competition. It performs the same function as the `TabDDPM` notebook but is tailored for specific models used in the whitebox setting (dev and final).**
3. **evaluate_synthetic_data** - This notebook covers the evaluation of synthetic data generated by TabDDPM. In order to assess the quality of synthetic data, we report various evaluation metrics including low-dimensional statistics, high-dimensional statistics, evaluation of the synthetic data on downstream tasks and privacy metrics. This notebook includes a brief description of these evaluation metrics, their implementation, and a demonstration of how to use them to evaluate the quality of synthetic data.

## Code
This section includes code structure and description of the files:
* [figures/](./assets) ==> contains the figures to be used in the notebooks.
* [configs/](./configs) ==> contains the json file that provides details about dataset information and model configurations.
* [eval/](./eval) ==> contains the code to evaluate the synthetic data generated by TabDDPM algorithm.
    * [eval_dcr.py](./scripts/eval/eval_dcr.py) ==> contains the code to report privacy metrics.
    * [eval_density.py](./scripts/eval/eval_density.py) ==> contains the code to report low-order statistics.
    * [eval_detection.py](./scripts/eval/eval_detection.py) ==> contains the code to report whether the synthetic data and real data are distinguishable.
    * [eval_mle.py](./scripts/eval/eval_mle.py) ==> contains the code to report machine learning efficiency.
    * [eval_quality.py](./scripts/eval/eval_quality.py) ==> contains the code to report high-order statistics.
* [lib/](./lib) ==> contains the utility functions.
    * [data.py](./lib/data.py) ==> contains the code for data processing and loading.
    * [env.py](./lib/env.py) ==> contains the code for environment setup.
    * [metrics.py](./lib/metrics.py) ==> contains the code for metrics used in training.
    * [util.py](./lib/util.py) ==> contains the code for utility functions.
* [scripts/](./scripts) ==> contains the running scripts.
    * [train.py](./scripts/train.py) ==> contains the code to train the ClavaDDPM model.
    * [utils_train.py](./scripts/utils_train.py) ==> contains the utility functions used in training.
* [tab_ddpm/](./tab_ddpm) ==> contains the code for TabDDPM algorithm.
* [complex_pipeline.py](./complex_pipeline.py) ==> contains the functions used in the TabDDPM notebook.
* [gen_multi_report.py](./gen_multi_report.py) ==> contains the code to generate the evaluation report for multi-table data.
* [gen_single_report.py](./gen_single_report.py) ==> contains the code to generate the evaluation report for single-table data.
* [pipeline_modules.py](./pipeline_modules.py) ==> contains the code for training and sampling pipeline modules.
* [pipeline_utils.py](./pipeline_utils.py) ==> contains the code for utility functions used in the pipeline.
* [preprocess_utils.py](./preprocess_utils.py) ==> contains the code for data preprocessing.
* [report_utils.py](./report_utils.py) ==> contains the code for utility functions used in the evaluation report generation.
* [wb_complex_pipeline.py](./wb_complex_pipeline.py) ==> contains the functions used in the TabDDPM_wb_dev_final notebook.
* [wb_pipeline_modules.py](./wb_pipeline_modules.py) ==> contains the code for training and sampling modules in the white box (dev and final) pipeline.
* [wb_pipeline_utils.py](./wb_pipeline_utils.py) ==> contains the code for utility functions used in the white box (dev and final) pipeline.


## Getting Started
To get started with the materials in this topic:
1. Ensure you have followed the reference to the installation guide and environment setup in the `README` file.
2. `[New Update]` Choose the right Notebook:
- If you want to explore the TABDDPM method or load and sample from models stored in the `train` folder for both the `White Box MIA on single table` and `Black-box MIA on single table` competitions, use the `TabDDPM` notebook.
- If you want to load and sample from models stored in the `dev` and `final (eval)` folders for the `White Box MIA on single table` competition, use the `TabDDPM_wb_dev_final` notebook.
3. Finally using the notebook `evaluate_synthetic_data.ipynb` to evaluate the synthetic data generated by TabDDPM algorithm.
