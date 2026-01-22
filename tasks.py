import os

from dotenv import load_dotenv
from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "mlops"
PYTHON_VERSION = "3.13"


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def evaluate(ctx: Context) -> None:
    """Evaluate model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/evaluate.py", echo=True, pty=not WINDOWS)


@task
def datadrift(ctx: Context, angle) -> None:
    """Perform data drifting."""
    ctx.run(f"uv run src/{PROJECT_NAME}/datadrift.py" f" --angle {angle}", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run(
        'uv run coverage run --source=src --omit="tests/*,/tmp/*" -m pytest tests/unittests/',
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        'uv run coverage run --append --source=src --omit="tests/*,/tmp/*" -m pytest tests/integrationtests/',
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run('uv run coverage report -m -i --omit="tests/*,/tmp/*"', echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context) -> None:
    """Build docker images."""
    ctx.run(
        "docker build -f dockerfiles/train.dockerfile . -t train:latest",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run("docker build -f docker files/frontend.dockerfile . -t frontend:latest", echo=True, pty=not WINDOWS)
    ctx.run("docker run --env-file .env --name experiment-mlops-train train:latest", echo=True, pty=not WINDOWS)


@task
def docker_build_frontend(ctx: Context) -> None:
    """Build docker image for frontend."""
    ctx.run(
        "docker build -f dockerfiles/frontend.dockerfile . -t frontend:latest",
        echo=True,
        pty=not WINDOWS,
    )


@task
def docker_build_backend(ctx: Context) -> None:
    """Build docker image for backend."""
    ctx.run(
        "docker build -f dockerfiles/backend.dockerfile . -t backend:latest",
        echo=True,
        pty=not WINDOWS,
    )


@task
def profile_inference(ctx: Context) -> None:
    """Profile model inference."""
    ctx.run(f"uv run src/{PROJECT_NAME}/profiling.py", echo=True, pty=not WINDOWS)
    ctx.run("uv run tensorboard --logdir=./log", echo=True, pty=not WINDOWS)


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)


# Deployment commands
def _require_env(var_name: str) -> str:
    value = os.environ.get(var_name)
    if not value:
        msg = f"Missing required environment variable: {var_name}"
        raise RuntimeError(msg)
    return value


def _image_uri(image_name: str) -> str:
    """Build an Artifact Registry image URI."""
    artifact_repo = _require_env("ARTIFACT_REPO")
    tag = os.environ.get("IMAGE_TAG", "latest")
    return f"{artifact_repo}/{image_name}:{tag}"


@task
def deploy_backend(ctx: Context) -> None:
    load_dotenv()  # Load environment variables from .env file
    project_id = _require_env("PROJECT_ID")
    region = _require_env("REGION")
    service_name = _require_env("BACKEND_SERVICE_NAME")
    image_uri = _image_uri("backend")
    service_account = os.environ.get("BACKEND_SERVICE_ACCOUNT")

    ctx.run(f"docker build -f dockerfiles/backend.dockerfile . -t {image_uri}", echo=True, pty=not WINDOWS)
    ctx.run(f"docker push {image_uri}", echo=True, pty=not WINDOWS)

    deploy_cmd = (
        f"gcloud run deploy {service_name} --project {project_id} --region {region} "
        f"--image {image_uri} --port 8002 --allow-unauthenticated --memory 2Gi"
    )
    if service_account:
        deploy_cmd = f"{deploy_cmd} --service-account {service_account}"
    ctx.run(deploy_cmd, echo=True, pty=not WINDOWS)


@task
def deploy_frontend(ctx: Context) -> None:
    load_dotenv()  # Load environment variables from .env file
    project_id = _require_env("PROJECT_ID")
    region = _require_env("REGION")
    # backend_service = _require_env("BACKEND_SERVICE_NAME")
    frontend_service = os.environ.get("FRONTEND_SERVICE_NAME", "frontend")
    image_uri = _image_uri("frontend")
    backend_url = os.environ.get("BACKEND_URL")

    ctx.run(f"docker build -f dockerfiles/frontend.dockerfile . -t {image_uri}", echo=True, pty=not WINDOWS)
    ctx.run(f"docker push {image_uri}", echo=True, pty=not WINDOWS)

    deploy_cmd = (
        f"gcloud run deploy {frontend_service} --project {project_id} --region {region} "
        f"--image {image_uri} --port 8001 --allow-unauthenticated --memory 2Gi "
        f"--set-env-vars BACKEND={backend_url}"
    )
    ctx.run(deploy_cmd, echo=True, pty=not WINDOWS)
