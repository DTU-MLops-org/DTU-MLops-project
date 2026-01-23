# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

`![my_image](figures/<image>.<extension>)`

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

or

```bash
uv add typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [X] Create a git repository (M5)
* [X] Make sure that all team members have write access to the GitHub repository (M5)
* [X] Create a dedicated environment for you project to keep track of your packages (M2)
* [X] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [X] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [X] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [X] Remember to either fill out the `requirements.txt`/`requirements_dev.txt` files or keeping your `pyproject.toml`/`uv.lock` up-to-date with whatever dependencies that you are using (M2+M6)
* [X] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [X] Do a bit of code typing and remember to document essential parts of your code (M7)
* [ ] Setup version control for your data or part of your data (M8)
* [X] Add command line interfaces and project commands to your code where it makes sense (M9)
* [X] Construct one or multiple docker files for your code (M10)
* [X] Build the docker files locally and make sure they work as intended (M10)
* [X] Write one or multiple configurations files for your experiments (M11)
* [X] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [X] Use profiling to optimize your code (M12)
* [X] Use logging to log important events in your code (M14)
* [X] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [X] Consider running a hyperparameter optimization sweep (M14)
* [ ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [X] Write unit tests related to the data part of your code (M16)
* [X] Write unit tests related to model construction and or model training (M16)
* [X] Calculate the code coverage (M16)
* [X] Get some continuous integration running on the GitHub repository (M17)
* [X] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [X] Add a linting step to your continuous integration (M17)
* [X] Add pre-commit hooks to your version control setup (M18)
* [ ] Add a continues workflow that triggers when data changes (M19)
* [ ] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [X] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [X] Create a trigger workflow for automatically building your docker images (M21)
* [X] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [X] Create a FastAPI application that can do inference using your model (M22)
* [X] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [X] Write API tests for your application and setup continues integration for these (M24)
* [X] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [X] Create a frontend for your API (M26)

### Week 3

* [X] Check how robust your model is towards data drifting (M27)
* [X] Setup collection of input-output data from your deployed application (M27)
* [X] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [X] Revisit your initial project description. Did the project turn out as you wanted?
* [X] Create an architectural diagram over your MLOps pipeline
* [X] Make sure all group members have an understanding about all parts of the project
* [X] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

Group 48

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s245822, s234136, s245261, s243069, s253167

### Question 3
> **A requirement to the project is that you include a third-party package not covered in the course. What framework**
> **did you choose to work with and did it help you complete the project?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

We are using `kagglehub` package to download and cache the dataset `gpiosenka/cards-image-datasetclassification` from Kaggle. We also use `tqdm` package to visualize progress bars when downloading and preprocessing the data. The `kagglehub` package made it easy to download and cache the dataset directly from Kaggle without needing to manually download and upload the raw dataset to our GCP bucket (processed data is stored in the GCP bucket). This streamlined our data acquisition process and ensured that all team members could easily access the same version of the dataset. The `tqdm` package provided a simple way to add progress bars to our data processing loops, which was helpful for monitoring the progress of long-running tasks and improving the user experience during data preprocessing.
We used the third-party frameworks `tqdm` and `kagglehub`, which were not covered in the course. `kagglehub.dataset_download`was used to download the dataset from [Kaggle](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification/datadownload) and `tqm` was used to display a progress bar during the data loading and processing.

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

We are managing dependencies using `uv`, which allows us to create and manage virtual environments for our project. The dependencies are specified in the `pyproject.toml` file, which is automatically generated and updated by `uv` whenever we add or remove packages. To set up the development environment, a new team member would first need to clone the GitHub repository. Then, they would install `uv` if they don't have it already. After that, they can run `uv sync` in the project directory, which will create a virtual environment and install all the dependencies listed in the `pyproject.toml` and `uv.lock` file. This ensures that all team members are working with the same versions of packages, reducing compatibility issues and making collaboration smoother. Aside from `uv` the new member should also have Docker installed to build and run the docker images provided in the `dockerfiles` directory.

To get a complete copy of our development environment, one would first have to install a compatible version of Python and the `uv` tool and then run the following command:


```
bash
uv sync --dev
````

To install only the dependencies required to run the code, use:
```bash
uv sync
```

To add new dependencies, one would write the following command:
```
bash
# Regular dependency
uv add <dependency>

# Dependency needed for development
uv add --dev <dependency>
```

To run something within the uv environment, one would write the following command:
```bash
uv run <command>
```

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

We initialized the project with the mlops_template cookiecutter and kept the standard layout mostly. We filled out the core modules in `src/mlops` for data loading, model definition, training, evaluation, and added application modules for the backend, frontend, data‑drift analysis and profiling. The `tests/` folder contains unit tests for data, training, model, and API behavior. Configuration is in `configs/` (defaults, sweep, Cloud Build, and Vertex AI), and we use `dockerfiles/` for separate train, evaluate, backend, and frontend images. We also populated `docs/` (mkdocs) and `reports/` (figures and analysis outputs).

Deviations from the template are mostly additions: extra dockerfiles, cloud configs, and runtime artifact folders (`outputs/`, `log/`, `wandb/`).

We have removed the `notebooks` folder because we did not use Jupyter notebooks, as well as the file `src/mlops/visualizations.py` since we did not do any data visualizations aside from the graph displayed in the frontend. We added the folder `reports\datadrift` to store the autogenerated datadrift reports. Additionally, we added several files within the `src/mlops` directly to run the datadrift, frontend, backend, monitoring and profiling components, which are named accordingly. Moreover, we added an `.env` file to store API keys needed to connect with services.

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

We set up code quality and formatting rules with `ruff` (linting + formatting, 120‑char line length) and run checks via `pre-commit` to keep style consistent. Testing uses `pytest`, and we run tasks through `invoke`/`uv` to keep the workflow reproducible. For typing, we use Python type hints throughout the codebase and include `mypy` as a dev dependency for static checks.

These practices matter in larger projects because they reduce ambiguity and make collaboration safer. They ensure good code quality, organization, and long-term maintainability and scalability, especially when many people are working together on the same project.

Linting/formatting avoids style debates and keeps diffs small. Linting also helps identify unused imports and unreachable code, which reduces clutter and improves readability. Typing makes interfaces explicit and helps tools catch mistakes early (e.g., wrong tensor shapes or config types) and makes it easier for team members how the functions are intended to be used.


## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

We have implemented a test of the model, which tests that the model is initialized correctly, with the correct dimensions. We also test the training, where we test that the training can run without issues and that the model is saved. We also test the data preprocessing, where it is tested that the processed data is saved in the correct place when the function is run. We also check that we get datasets of the correct sizes when we load the preprocessed data. We have also done API tests.

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

The total code coverage of our project is 45%, which includes all our source code in the repository. This number is relatively low because the codebase also contains components such as data drift detection, profiling, and API monitoring, for which we did not implement tests, as they were not central to the core functionality of the project. When focusing only on the backend, model, data handling and training scripts, the code coverage is significantly higher, exceeding 75%. 

Even if our code coverage were close to 100%, we would not necessarily trust the code to be completely error free. Code coverage only measures how much of the code is executed during testing, not how well it is tested. Tests may execute code paths without properly checking edge cases, incorrect outputs, or unexpected behaviour. Therefore, while code coverage is a useful indicator, it should be combined with well-designed tests and code reviews to increase overall reliability.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

We made use of both branches and pull requests in our workflow. We configured rules in our GitHub repository that prevented direct pushes to the `master` branch. Instead, all changes had to be introduced through pull requests. This ensured that new features or fixes were developed in separate branches before being merged into the main codebase. Each time a new feature was implemented, a new branch was created specifically for that task.


To maintain code quality and stability, we followed a rule of not merging pull requests unless all automated tests passed successfully. This helped prevent broken code from being introduced into the main branch and encouraged early detection of errors.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We did not implement DVC in our project but we did implement model versioning using Weights and Biases (W&B) and storing trained models as artifacts after each training run. However, we did not have the same level of version control for our dataset.

Version control of data becomes quite important when datasets or preprocessing steps change over time. Using DVC would allow such changes to be tracked systematically, making it easier to reproduce experiments and compare results.

If we were to continue working on this project, we would prioritize implementing data augmentation. In that case, data versioning would be essential for keeping track of changes to understand their impact on model performance.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

Our CI is implemented with two GitHub Actions workflows: one for unit tests and one for code quality. The test workflow (`.github/workflows/tests.yaml`) runs on every push and pull request to `master`. It uses a matrix across three operating systems (`ubuntu-latest`, `windows-latest`, and `macos-latest`) with Python 3.13. Dependencies are installed via `uv sync --locked --dev`, and tests are executed through `uv run invoke test` (which wraps `pytest` and any task setup we defined)to verify that new changes do not break existing functionality. The linting workflow (`.github/workflows/linting.yaml`) also runs on push/PR to `master`, but only on Ubuntu for speed. It enforces style and formatting with `ruff check .` and `ruff format . --check`, and runs `mypy` for static type checking (currently marked `continue-on-error` so type issues do not fail the entire CI but still can be reviewed).

We do use caching: `astral-sh/setup-uv` is configured with `enable-cache: true`, which caches dependencies and speeds up repeated runs.

These workflows give us fast feedback on correctness and style and ensure the same checks run consistently in PRs. A link to the workflows in this repo: [linting](https://github.com/DTU-MLops-org/DTU-MLops-project/blob/master/.github/workflows/linting.yaml) and [tests](https://github.com/DTU-MLops-org/DTU-MLops-project/blob/master/.github/workflows/tests.yaml).

We configured our workflows to run automatically on every pull request to the master branch. This ensures that tests must pass before code can be merged. An example of a triggered workflow can be seen at: [here](https://github.com/DTU-MLops-org/DTU-MLops-project/actions/runs/21207700763)

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We configured our experiments using configuration files managed with Hydra and Weights & Biases (W&B) sweeps, while using Typer to provide a simple command-line interface for starting training runs. Hyperparameters such as learning rate, batch size, and number of epochs were defined in YAML configuration files and could be overwritten from the command line or optimized automatically using W&B sweeps. This allowed us to run multiple experiments in a reproducible and structured way. A training run can be started using the following command:


```
bash
uvx invoke train
```

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

Whenever an experiment is run, all configuration parameters are resolved and recorded using Hydra, ensuring that the exact settings used for the run are preserved. During training, metrics such as loss and accuracy are logged continuously to Weights&Biases (W&B), and trained models are stored as artifacts after completion. This ensures that no information is lost during an experiment and that all results, configurations, and outputs are centrally stored and traceable.


To reproduce an experiment, one has to identify the corresponding W&B run and retrieve the recorded configuration and model artifact. By checking out the matching version of the code and recreating the software environment, the experiments could be rerun using the same settings.


### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

For the train file, we are tracking the loss and accuracy for suit, rank, and overall training accuracy, as is seen in the following picture below. This is to observe whether or not the model is improving and learning during the training. Here we see that the model decreases training loss and increases training accuracy, showing that it is continually learning.
![wandb_train](figures/wandb_train.png)

For evaluation, we are tracking the validation loss in order to ensure that the accuracy of the model is improving during evaluation. Reaching an optimal accuracy would then prevent overfitting for our model as well. This is showcased below in the image. It is seen in the image that generally for all runs, the accuracy increases quickly but plateaus / rises very slowly after 1 to 2 thousand steps, while the training loss still decreases. This means that it is still learning, but the new model does not perform much better in the test.
![wandb_eval](figures/wandb_eval.png)

In the hyperparameter sweep, which is presented in the final image, it is important to track the configuration of hyperparameters. Furthermore, the validation accuracy is vital as well. This way, we can explore the most optimal configuration of hyperparameters for our specific model. We generally see that the runs with more epochs and lower learning rate perform better.

![wandb_sweep](figures/wandb_sweep.png)

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

For our project we developed several Docker images, including images for training, API, API monitoring, backend, frontend and evaluation. This separation allowed us to isolate responsibilities and ensure that each component runs in a controlled and reproducible environment. Docker was essential for guaranteeing that experiments could be executed consistently across local machines, CI pipelines, and cloud infrastructure, independent of the host system configuration. 


To build and run the training Docker image, the following commands can be used:
```bash
docker build -f dockerfiles/train.dockerfile . -t train:latest
docker run --rm \
 --env-file .env \
 -v $(pwd)/dtu-mlops-group-48-1ddc4e04b98d.json:/app/credentials.json \
 -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json \
 -e WANDB_API_KEY \
 train:latest
```
Environment variables and credentials are injected at runtime, allowing secure access to external services such as W&B and Google Cloud Storage. All Dockerfiles used in the project can be found at: 
[here](https://github.com/DTU-MLops-org/DTU-MLops-project/tree/master/dockerfiles)



### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

During the first days we used debugging when creating `train.py` and `data.py` scripts, since most bugs were in core logic. We relied on small reproducible runs, basic logging, and stack traces to isolate issues quickly. Once the core code was established, most remaining bugs were configuration related (paths, credentials, Hydra configs, Docker/CI settings), where debugging itself did not help much. Instead proper logging and careful reading of error messages was the key to fixing the issues.

We do not consider the code perfect. We added a small profiling utility (`profiling.py`) and used lightweight checks to look for obvious bottlenecks.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

We used several Google Cloud services throughout the project. Buckets were used to store trained models, datasets and experiment artifacts. When the application is running, the backend downloads the trained model from the bucket and uploads user-provided data for inference or evaluation.

We also used Artifact Registry to store and manage our Docker images, which were used for training, backend services, and deployment. In addition, we used Compute engine to run virtual machines for training and hosting Docker containers, and Vertex AI to run managed training jobs in the cloud.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

We did not directly manage Compute Engine instances ourselves, but instead used them indirectly by training the model using Vertex AI. Vertex AI relies on Compute Engine under the hood to provision virtual machines that execute training jobs.


When submitting a custom training job, we specified the machine type and Docker container image to use. Vertex AI then automatically created the required Compute Engine instances, pulled the container, executed the training script, and tore down the instances once the job finished.


We used a CPU-based virtual machine with the machine type `n1-highmem-2`, which provided sufficient memory for training while keeping costs manageable. Training was run using a custom Docker container built from our training Dockerfile and stored in Artifact Registry.

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

![Used buckets](figures/q19-data-bucket.png)

In the bucket we have stored our data, the best model from training which was updated with the newest model directly from the train script.

We also stored the input-output that was given to the deployed API (input image, output prediction, output probabilities and timestamp).

At last, we have stored the data drift report from the data monitoring API.

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

![Registry repositories](figures/q20-registry-p1.png)
![Registry repository details](figures/q20-registry-p2.png)


We have stored dockerfiles for train, evaluate, backend, frontend and api_monitoring.

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

![Build history](figures/q21-build.png)


In our cloud build history, you can see all the building of train and evaluate dockerfiles. These are triggered every time we push to the master branch.

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

We trained the model in the cloud using Vertex AI. We chose Vertex AI instead of directly using Compute Engine because it takes away much of the manual setup, such as creating and managing virtual machines, while still allowing us to run our own Docker containers. This made it easier to focus on the training process rather than infrastructure management.


To run training jobs, we created a configuration file (`configs/vertex_ai_config.yaml`) that specifies parameters such as the container image, machine type, and environment variable names that are passed to the training script. When a job is launched, Vertex AI automatically creates a VM, pulls the specified container, and executes the training code.


Instead of using DVC in the cloud, we relied on Google Cloud Storage for storing and accessing the data. This allowed the training job to read the data directly from cloud storage without downloading it locally first, making the process more efficient and easier to scale.

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

We implemented a model API using FastAPI in `src/mlops/backend.py`. The service loads the trained model on startup (via FastAPI's lifespan hook) by downloading the checkpoint from a GCP bucket using a service account or default credentials. The main endpoint is `POST /classify`, which accepts an uploaded image file, applies the same preprocessing as training (resize + tensor), runs inference with PyTorch, and returns the predicted suit/rank plus the full probability vectors.


One thing we did that is slightly special is logging predictions to GCP in the background: the endpoint uses `BackgroundTasks` to save both the input image and a JSON file with probabilities and predictions to the bucket without blocking the response, so that this data can be used later for the drift detection service and monitoring. We containerized the backend and exposed it through a Streamlit frontend, but the API is fully usable on its own.

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

Yes, we did manage to deploy the model in the cloud for continuous deployment.
Firstly, we tried to deploy the model locally with docker images, where we performed debugging and got it to work. Then we tried to deploy the model in the cloud by pushing the docker images to the artifact registry and used the images to deploy in the cloud with cloud run. 


We also wrote tasks with invoke to be able to deploy both frontend and backend in the cloud. A user can write:


`uv run invoke deploy-backend`
`uv run invoke deploy-frontend`


In the terminal. This builds the images, pushes them to the artifact registry and deploys them. However, we have already deployed the front- and backend in the cloud:


Links for the deployed backend and frontend:
[backend](https://production-model-684678738707.europe-west1.run.app)
&
[frontend](https://frontend-684678738707.europe-west1.run.app)

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

We performed both unit testing and load testing for the API. 

For unit testing, we did two small API tests. The initial test mainly focused on entering the host properly. The second test checks that the api returns the correct prediction based on the calculated probabilities, ie. not the “correct” prediction, simply the one it is supposed to predict. 

For load testing we used locust and tested using a test which increased by 5 users per second. The API model managed to run with limited failure, less than one percent, until around 350 clients per second were testing the API. The model was able to handle around 380 clients per second before the failure rate increased tremendously. The average response time was 5906 ms.


### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

We implemented monitoring of data drifting. We downloaded the processed training data from the gcp bucket and the latest images given to the model in the frontend. The images were processed the same way as the training data was, and all images were passed to the model, where we extracted the image features. Finally, we created an evidently report with the DataDriftPreset comparing the latest images passed to the model to the training data images. We created a docker image which was deployed to the cloud and can be accessed at: [here](https://api-monitoring-684678738707.europe-west1.run.app/report?n=20)


Otherwise, it can also be deployed locally by:
```bash
docker build -f dockerfiles/api_monitoring.dockerfile . -t api_monitoring:latest
```


```bash
docker run --rm --name api_monitoring --network mlops-net -p 8003:8003
  -v $(pwd)/dtu-mlops-group-48-1ddc4e04b98d.json:/app/credentials.json
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json   api_monitoring
```

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

Group member 1 used $3.34 overall and the most on cloud engine. Second most was Cloud Run.
Group member 2 used $0.81
Group member 3 used $0
Group member 4 used $1.55. Mostly on cloud run and artifact registry.
Group member 5 used $0.61


Cloud engine was the most expensive due to running virtual machines in the cloud to train the model. Nextmost was cloud run due to deploying the frontend and backend of our API and also the monitoring API.


Artifact registry was also expensive due to pushing images to it many times. Especially with the backend and frontend images.


### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

We implemented a frontend for our API using StreamLit and deployed it. This made it easier for us to visualize the actual user experience of our API.

Using the frontend, the user can upload their own image. Our model will do a prediction on the uploaded image. The overall predicted label will be displayed together with bar charts of all the probabilities for rank and suit.

![frontend](figures/frontend.png)

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 29 fill here ---

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

During the project some of the biggest challenges was with implementing cloud. We got the model training locally, but getting it to run in the cloud was very difficult. We had issues with our .env file and getting it to work in the cloud with the Secret Manager. It was difficult ensuring the correct API keys and Google Credentials were working in the docker images and in the cloud. We also had to ensure that every member of the group had the necessary information and variables in the .env, since we could not just upload that file to the repository.


Deploying the API (backend and front-end) was also difficult. Every time we had an issue or an error, it took a long time debugging it and fixing it. Then, we had to wait a long time to see if the error was actually fixed, because it took so long to deploy the API again. Again, it was difficult working with the secret variables.


Furthermore, we had a few issues when merging to the master branch, since we had a lot of branches active at the same time, especially to our pyproject.toml and uv files, since they were updated often. To combat this, it was important to pull from the master often, especially before attempting to merge. We encountered a lot of merge conflicts and ensuring all the changes were correct took a long time.

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

Student s245261 was in charge of:
- Building the train & evaluating docker files
- Getting training and evaluation to work in the cloud + continuous integration with GitHub Actions.
- Writing the code for analyzing data drifting.
- Writing the code for monitoring data drifting in the Data Drift API.
- Writing parts of the report.

Student s234136 was in charge of:
- Backend API
- API testing and load testing
- Writing the initial model setup. 

Student s243069 was in charge of:
- Data loading, preprocessing and writing unit tests for this module (data.py). 
- Development of Backend API and Frontend, local deployment using containers and cloud deployment of Frontend and Backend.
- Configuring pre-commit
- Configuring Github Actions CI: automatic tests, code coverage and linting
- Code profiling
- Writing parts of report

Student s253167 was in charge of:
- Model/training scripts and some of the tests for those components
- Configuring hydra and implementing W&B sweeps
- Analyzing data drifting 
- Writing the report

Student s245822 was in charge of:
-  Writing code and dockerfile for monitoring data drift API 
-  Deployment of monitoring data drift API in the cloud
-  Model, training and evaluation scripts 
-  Tests of model and training
-  Logging with WandB
-  Backend API

We have used ChatGPT and GitHub Copilot to help debug our code and rewrite some of it.