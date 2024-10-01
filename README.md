# Machine Learning Model API

## Introduction

This repository provides a solution for deploying a Machine Learning model as a RESTful API. Using Flask, it implements a POST endpoint `/predict` that accepts JSON input matching the model's requirements. The API processes the input, performs inference with the ONNX model, and returns predictions in JSON format. The application is also dockerized for easy deployment.


## Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Dependency Management with Poetry](#dependency-management-with-poetry)
- [Project setup](#project-setup)
- [Testing the app](#testing)
- [Code Quality](#code-quality)
- [Unit tests](#unity-test)
- [Continuous Integration](#continuous-integration)


## Project Structure

- `/app`: Contains the main Flask application files.
- `/tests`: Unit tests for the API.
- `models`: Contains the Pre-trained ONNX model for inference.
- `docker/`: Docker configuration files.


## API Documentation

### `/predict` Endpoint
- **Method**: `POST`
- **Description**: This endpoint accepts input data in JSON format, runs predictions using the ONNX model, and returns the predictions in JSON format.

#### Request Body
```json
{
    "Material_A_Charged_Amount": [[10]],
    "Material_B_Charged_Amount": [[20]],
    "Reactor_Volume": [[30]],
    "Material_A_Final_Concentration_Previous_Batch": [[40]]
}
```

## Requirements

Before you begin, ensure you have the following prerequisites installed on your machine:

- [Python 3.9](https://www.python.org/downloads/) or higher
- [Poetry](https://python-poetry.org/docs/#installation)
- [Docker](https://www.docker.com/get-started)


## Dependency Management with Poetry

This project uses [Poetry](https://python-poetry.org/) for dependency management. Poetry simplifies the process of managing dependencies and packaging Python projects.

docOnce installed, you can easily manage dependencies by running `poetry add <dependency>` to add the new dependency and `poetry install`, which installs all required packages listed in the [`pyproject.toml`](pyproject.toml)
 file.


## Project setup

#### Clone the Repository
Clone this repository to your local machine:

`git clone https://github.com/renatoramossilva/ta-ml-model-api.git`


#### Install Dependencies
Navigate to the project directory and install the dependencies using Poetry:

```bash
cd ta-ml-model-api
poetry install
```


#### Activate the Virtual Environment
To activate the virtual environment created by Poetry, run:

`poetry shell`


#### Run the Application
You can now run the application using Docker or directly using Flask.

To run with Docker, use:

```bash
docker build -t ta-ml-model-api-image -f docker/Dockerfile .
docker run --rm -d -p 5001:5000 -p 8000:8000 --name ta-ml-model-api ta-ml-model-api-image
```

- `-p 5001:5000`: This maps port 5000 inside the container (where the Flask app is running) to port 5001 on your local machine. You can access the app locally at http://localhost:5001.
- `-p 8000:8000`: This maps port 8000 inside the container to port 8000 on your local machine, which is used for test coverage reports. You can view the test coverage at http://localhost:8000.
- `--rm`: Automatically removes the container when it is stopped.
- `-d`: Runs the container in detached mode (in the background).

Alternatively, to run the application directly, use:

`poetry run flask run`

Check if the container is running and the application is on:

```bash
docker ps
CONTAINER ID   IMAGE                   COMMAND                  CREATED          STATUS          PORTS                              NAMES
0197c0cc7247   ta-ml-model-api-image   "sh -c 'flask run --…"   52 seconds ago   Up 51 seconds   8000/tcp, 0.0.0.0:5001->5000/tcp   ta-ml-model-api
```

## Testing

The `/predict` endpoint can be tested locally using `curl`.

```bash
curl -X POST http://127.0.0.1:5001/predict \
-H "Content-Type: application/json" \
-d '{
    "Material_A_Charged_Amount": [[10]],
    "Material_B_Charged_Amount": [[20]],
    "Reactor_Volume": [[30]],
    "Material_A_Final_Concentration_Previous_Batch": [[40]]
}'
```

Expected output:
e.g.

```bash
{"prediction":["High"]}
```


## Code Quality

This project uses several tools to maintain code quality and enforce coding standards:

- **[black](https://black.readthedocs.io/)**: A code formatter that ensures consistent code style.
- **[pylint](https://pylint.pycqa.org/)**: A static code analysis tool to enforce coding standards and detect errors.
- **[isort](https://pycqa.github.io/isort/)**: A tool to sort and format imports automatically.
- **[mypy](http://mypy-lang.org/)**: A static type checker to ensure type safety in Python code.

These tools are integrated with [pre-commit](https://pre-commit.com/), ensuring that they are automatically run before each commit to maintain code quality.

To manually run these tools, you can use the following commands:

`pre-commit run --all-files` or `poetry run pre-commit run --all-files`

## Unit Tests

This project includes a comprehensive suite of unit tests to ensure the correctness of the API and underlying code. We aim for a [high test coverage, with results consistently close to 100%](http://localhost:8000). The unit tests are executed using `pytest` along with `pytest-cov` for test coverage reporting.

#### Running Unit Tests

To run the unit tests locally:

`pytest` or `poetry run pytest`


pytest --cov=ta_ml_model_api


#### Testing Across Python Versions
To verify compatibility with different versions of Python, the project is tested with Python 3.9, 3.10, 3.11, and 3.12. We use `tox` to automate testing across these Python versions.

`tox` or `poetry run tox`


## Continuous Integration

This project uses GitHub Actions to automate code quality checks and testing before merging into the master branch. The tools  included in the workflow are described in [Code Quality](#code-quality) session.


## Workflow Overview
Whenever a pull request is opened, the GitHub Actions workflow will trigger and perform the following checks:

- Code Formatting: Run `black` and `isort` to format the code.
- Static Analysis: Execute `pylint` and `mypy` to ensure code quality.
- Unit Testing: Run `pytest` to execute the unit tests and check for coverage. (specifically 3.9, 3.10, 3.11, and 3.12)

This setup ensures that only code that passes all checks is merged into the master branch, maintaining a high standard of code quality throughout the development process.
