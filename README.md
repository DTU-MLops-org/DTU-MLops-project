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


### How to use:

## Train and test the model
`uvx invoke preprocess-data`

`uvx invoke train`

`uvx invoke evaluate`

## run tests to check model
`uvx invoke test`
## run load test - api backend needs to be running simultaniously on same port (host : 8002)
`locust -f tests/locust/locustfile.py`



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
- Automatic trigger that downloads data and latest model & builds and runs the train and evaluate docker images when pushing to master branch.
