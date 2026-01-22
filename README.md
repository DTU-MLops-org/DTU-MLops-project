# mlops

## Project Describtion

### Overall goal
Classifing playing cards into suit and rank.


### What framework are you going to use, and you do you intend to include the framework into your project?
PyTorch

### What data are you going to run on (initially, may change)
The dataset consists of images of playing cards, the dataset can be found here: https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification/code

### What models do you expect to use
We expect to use MobileNet.


## Project structure
````markdown

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
````


# How to use:

## Running and evaluating model
`uv run invoke preprocess-data`
`uv run invoke train`
`uv run invoke evaluate`

To run hyperparameter sweep with wandb:
`uv run wandb sweep configs/sweep.yaml`

## run tests to check model
`uv run invoke test`
## run load test - api backend needs to be running simultaniously on same port as deployed backend
`uv run locust -f tests/locust/locustfile.py --host=https://production-model-684678738707.europe-west1.run.app`



## Enable pre-commit
`uv run pre-commit install`

to ignore pre-commit use `--no-verify` flag when committing, e.g.
`git commit -m <message> --no-verify`

to run precommit manually use
`uv run pre-commit run --all-files`


## Docker

Requires the file dtu-mlops-group-48-1ddc4e04b98d.json with GOOGLE_APPLICATION_CREDENTIAL to be in root of project.
Requires the wand API key and GOOGLE_APPLICATION_CREDENTIAL to be in .env.

Build and run train.dockerfile:
```bash
docker build -f dockerfiles/train.dockerfile . -t train:latest
docker run --rm \
  --env-file .env \
  -v $(pwd)/dtu-mlops-group-48-1ddc4e04b98d.json:/app/credentials.json \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json \
  -e WANDB_API_KEY \
  train:latest
```

Build and run evaluate.dockerfile:
```bash
docker build -f dockerfiles/evaluate.dockerfile . -t evaluate:latest
docker run --rm \
  --env-file .env \
  -v $(pwd)/dtu-mlops-group-48-1ddc4e04b98d.json:/app/credentials.json \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json \
  -e WANDB_API_KEY \
  evaluate:latest
```

Network for the docker containers
`docker network create mlops-net`

Backend
`docker run --rm --name backend --network mlops-net -p 8002:8002 backend`

Frontend
`docker run --rm --name frontend --network mlops-net -p 8001:8001 -e BACKEND=http://backend:8002 frontend`

## Google Cloud
- Set our project as the default project: `gcloud config set project dtu-mlops-group-48`
- Artifact registry: `europe-west1-docker.pkg.dev/dtu-mlops-group-48/our-artifact-registry`
- Bucket for data and model: `dtu-mlops-group-48-data`
- Model is uploaded to the bucket when train is run.
- Automatic trigger that downloads data and latest model & builds the train and evaluate docker images when pushing to master branch.

To train the latest model using Vertex AI, run:

```bash
set -a && source .env && set +a && envsubst < configs/vertex_ai_config.yaml | gcloud ai custom-jobs create --region=europe-west1 --display-name=test-run --config=-
```

## API
Build and run docker locally:
- Backend:
```bash
docker build -f dockerfiles/backend.dockerfile . -t backend:latest

docker run --rm --name backend --network mlops-net -p 8002:8002 \
  -v $(pwd)/dtu-mlops-group-48-1ddc4e04b98d.json:/app/credentials.json \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json \
  backend
```

- Frontend
```bash
docker build -f dockerfiles/frontend.dockerfile . -t frontend:latest

docker run --rm --name frontend --network mlops-net -p 8001:8001 -e BACKEND=http://backend:8002 frontend
```

Also, make sure the Docker network exists before running:
```bash
docker network create mlops-net
```

Deploy backend and frontend in cloud:
`uv run invoke deploy-backend`
`uv run invoke deploy-frontend`


## API Monitoring
Build and run monitoring:
```bash
docker build -f dockerfiles/api_monitoring.dockerfile . -t api_monitoring:latest
```

```bash
docker run --rm --name api_monitoring --network mlops-net -p 8003:8003
  -v $(pwd)/dtu-mlops-group-48-1ddc4e04b98d.json:/app/credentials.json
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json   api_monitoring
```
Report is returned by the /report function but also saved to the bucket.
It is saved as reports/api_monitoring_report.html and can be opened at
```bash
https://storage.cloud.google.com/dtu-mlops-group-48-data/reports/api_monitoring_report.html
```



## Data Drifting (M27)
To run the data drifting analysis, first install development dependencies:
```bash
uv sync --dev
```
In this project, data drifting is simulated by rotating the images. TO run the analysis for a given rotation angle, use:
```bash
uvx invoke datadrift --angle 40
```

A report will be generated after the run, and stored in `reports/datadrift/rotation_{angle}_degrees.html`
