# GPU Tests

A repository for GPU-related test scenarios in Python, utilizing libraries like PyTorch for deep learning operations. This project aims to provide intuitive and smooth testing processes for various GPU operations.

## Features

- **Simple Test Scenarios**: Easy-to-understand test cases for common GPU operations.
- **Reporting**: Generates HTML reports for test results using pytest-html and Allure.
- **Compatibility**: Designed to work with PyTorch and CUDA for GPU acceleration.

## Requirements

To run the tests, you'll need Python 3.6 or higher. The following dependencies are required:

- pytest
- pytest-html
- allure-pytest
- torch

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/hovhanns/gpu-tests.git
   cd gpu-tests

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt

## Running Tests

To run the tests and generate reports, use the following command:

```bash
python -m pytest --html=reports/report.html tests/
