# Phishing Website Detection Analysis: End-to-End MLOps Pipeline for Network Security

## Overview

This project is an end-to-end MLOps project designed to detect phishing websites using a dataset containing over 30
optimized features of phishing websites. The project involves building and deploying a machine learning model that can
identify malicious websites, thereby enhancing network security.

## Project Structure

The project is structured into several key parts:

1. **Data Ingestion**: Collecting and importing data from various sources.
2. **Data Validation**: Ensuring the quality and integrity of the data.
3. **Data Transformation**: Preparing the data for model training.
4. **Model Training**: Building and tuning machine learning models.
5. **Model Evaluation**: Assessing model performance.
6. **Model Deployment**: Deploying the model to a production environment.
7. **Monitoring and Maintenance**: Ensuring the model remains effective over time.

## Dataset

The dataset used in this project contains over 30 optimized features of phishing websites. These features are used to
train a machine learning model that can accurately classify websites as either phishing or legitimate.

The dataset used in this project is sourced
from [Phishing Website Dataset](https://www.kaggle.com/datasets/akashkr/phishing-website-dataset/data).

## Technologies Used

- **Python**: Primary programming language.
- **Docker**: Containerization for consistent environments.
- **GitHub Actions**: CI/CD pipeline automation.
- **AWS ECR/EC2/S3**: Cloud services for deployment and storage.
- **MLFlow**: Experiment tracking and model management.
- **MongoDB Atlas**: Data storage and management.

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:

```bash
   git clone https://github.com/PiusSunday/phishing-website-detection-analysis.git
```

2. Navigate to the project directory:

```bash
   cd path/to/project/root
```

3. Install the required dependencies:

```bash
   pip install -r requirements.txt
```

## License

This project is licensed under the MIT License. See
the [LICENSE](https://github.com/PiusSunday/phishing-website-detection-analysis/blob/main/LICENSE) file for details.
