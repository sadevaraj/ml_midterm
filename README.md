# ml_midterm

This project uses Poetry to manage Python dependencies and its virtual
environment.

## Problem and how a model could help

Employee burnout can affect well-being and work, and identifying possible
burnout risk early may help organizations offer support. This project explores
whether information in the developer dataset can help identify employees who
may be at risk. The dataset does not provide a direct burnout label: the
notebook uses `mentally_exhausted` as its target, so the model estimates the
likelihood of reported mental exhaustion as a possible burnout-related signal,
not burnout itself or a clinical diagnosis.

The notebook trains a logistic regression model on employee/developer
attributes and their recorded target labels. For a new employee record, the
trained model can return an estimated probability of mental exhaustion.
Organizations could use that estimate as one tentative signal to offer a
voluntary well-being check-in or additional support. It should not be treated
as a burnout diagnosis or used on its own for employment decisions, and its
predictions should be checked for accuracy and fairness before any real-world
use. Predicting burnout directly would require data with a suitable burnout
label.

## Model flow: dictionaries, vectorization, and evaluation

1. **Create dictionaries:** `to_dict(orient="records")` converts each feature
   row into a dictionary mapping column names to values.
2. **Vectorize the features:** `DictVectorizer` is a scikit-learn transformer
   that converts these dictionaries into a numeric feature matrix. It
   one-hot encodes string categories and keeps numeric feature values.
   `fit_transform` learns the feature mapping from the training data and
   applies it. Reuse the fitted vectorizer's `transform` method for validation
   and new records so their feature columns match the training data.
   Here, "transformer" means a data-preprocessing component, not a Transformer
   neural network.
3. **Train and predict:** despite its name, logistic regression is used here
   for binary classification (`No` or `Yes`). The notebook's `predict`
   function uses `model.predict_proba`, which returns estimated class
   probabilities between 0 and 1, not percentages. For example, `0.8`
   represents an estimated 80% probability. Selecting `[:, 1]` returns the
   probability of `model.classes_[1]`; check that this class is `Yes`.
4. **Evaluate on validation data:** compare the true labels with the predicted
   positive-class probabilities. Varying the decision threshold gives the
   true positive rate (TPR) and false positive rate (FPR). An ROC curve plots
   TPR against FPR; plotting each rate against the threshold instead produces
   a threshold plot. ROC AUC is the area under the ROC curve, not a separate
   curve. Use the probability scores, rather than labels rounded at a single
   threshold, with `roc_auc_score`. Both classes must be present in the
   validation labels. An AUC of 1 indicates perfect ranking, while 0.5
   indicates chance-level ranking.

Probability predictions are suitable scores for ROC/AUC, but probabilities
are not required: other continuous classifier scores can also be used.

### Why true positives and true negatives apply here

True positives and true negatives are classification measures, not standard
regression measures. This is a classification problem because the target is
a category (`Yes` or `No`), rather than a continuous quantity such as a price.
Despite its name, logistic regression is a classification model;
`predict_proba` returns its estimated class probabilities but does not make
the task classification by itself.

To count classification outcomes, choose a threshold such as 0.5 and predict
`Yes` when its estimated probability is at least that threshold:

- **True positive (TP):** the actual label is `Yes` and the prediction is `Yes`.
- **True negative (TN):** the actual label is `No` and the prediction is `No`.
- **False positive (FP):** the actual label is `No` but the prediction is `Yes`.
- **False negative (FN):** the actual label is `Yes` but the prediction is `No`.

The true positive rate is `TP / (TP + FN)`, and the false positive rate is
`FP / (FP + TN)`. Computing these rates across different thresholds gives
the points on the ROC curve. Regression models instead typically use measures
such as mean absolute error (MAE) or mean squared error (MSE).

TP and TN do not directly apply to regression because a continuous target
has no predefined positive or negative class. For example, if the actual
price is 100 and the predicted price is 95, the prediction has an error of
5; it is not a true positive or a true negative. Regression evaluation
measures how far predictions are from actual values rather than counting
correct class assignments. A negative number or a negative prediction error
is not the same as a negative class.

You can define a rule such as "price at least 100 is positive" and apply it
to both actual and predicted values. TP and TN then become meaningful, but
they evaluate that derived classification task, not the original regression
task, and no longer capture the size of the numerical errors.

General learning resource for dictionaries, vectorization, and transformers:
[YouTube video](https://www.youtube.com/watch?v=114i2Kz-LZA).

## Set up

Install Poetry 2.0 or later if it is not already available, for example:

```bash
pipx install poetry
```

From the project directory, install the dependencies:

```bash
poetry install
```

Poetry creates the virtual environment in `.venv`. To choose a specific Python
interpreter, run `poetry env use /path/to/python` before `poetry install`.
In VS Code, select `.venv/bin/python` as the Python interpreter. Make sure the
Python and Jupyter extensions are installed, then open `ml_midtremp.ipynb` and
select the `.venv` environment from **Select Kernel**. Restart the kernel if
the notebook was already open.

## Run the notebook

Open `ml_midtremp.ipynb` in VS Code and choose **Run All** (or run its cells
individually). The notebook loads `indian_developer_burnout_2026.csv` from the
project directory, so keep the CSV alongside the notebook and run it with the
project as the workspace.

## Train the model

Run `poetry run python train.py` to select the logistic regression regularization
parameter with five-fold ROC AUC and fit the final model on the full dataset.
By default, it reads `indian_developer_burnout_2026.csv` and writes the
prediction artifact to `burn_out_model.bin`. Use `--data` or `--model` to
override either path.

## Run the prediction web service

Install the project dependencies, then start the app:

```bash
poetry install
poetry run uvicorn predict:app --host 0.0.0.0 --port 9696
```

Open `http://localhost:9696` to enter model inputs and view the estimated
probability of reported mental exhaustion. Blank fields use the defaults shown
as gray hints. The API remains available at `/predict`, and `/health` reports
service status.

## Build and run with Docker

Install Docker and make sure it is running. From the repository root, build
the image and start the web service:

```bash
docker build -t ml-midterm .
docker run --rm -p 9696:9696 ml-midterm
```

Open `http://localhost:9696` in your browser. Select values from the text
dropdowns, enter numeric values, and click **Estimate probability** to see the
probability and verdict. Leave fields at their gray defaults to use `NA` for
text and `0` for numbers.

Check the running service with:

```bash
curl http://localhost:9696/health
```

Press **Ctrl+C** in the terminal running the container to stop it. Rebuild the
image after changing the code or model artifact.

The container listens on port `9696` by default. To use a different local
container port, set `PORT` and publish that port, for example:

```bash
docker run --rm -e PORT=10000 -p 10000:10000 ml-midterm
```

### Deploy to Render

In the [Render dashboard](https://dashboard.render.com/), create
**New > Web Service**, select **Git Provider**, and connect this repository.
Use these settings:

| Setting | Value |
| --- | --- |
| Branch | `main` |
| Language / runtime | `Docker` |
| Root directory | Leave blank (repository root) |
| Dockerfile path | `./Dockerfile` |
| Docker build context | Repository root (`.`), if shown |
| Docker Command override | Leave blank; use the image's `CMD` |
| Health check path | `/health` |

Choose a region and instance plan, then deploy. No custom build command,
environment variables, database, or persistent disk are required. Render
provides `PORT` (normally `10000`); the Docker command listens on that port
and binds to `0.0.0.0`. Do not replace it with a localhost-only command.

Render builds and hosts the Docker image directly from Git, so you do not
need to upload an image to Docker Hub. After deployment, open the service's
HTTPS URL (for example, `https://your-service.onrender.com/`) to use the page;
do not append a port number. The same service serves both the webpage and
prediction API, so no separate frontend deployment or CORS configuration is
needed.

For deployment details, see [Render's Docker documentation](https://render.com/docs/docker).