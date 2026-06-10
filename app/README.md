# MedGemma Glaucoma Control Room

A Flask-based research interface for uploading retinal fundus images, running a
simulated or live MedGemma glaucoma assessment, viewing model explanations and
XAI heatmaps, and recording human confirmation or override decisions.

> **Important:** This project is a research/demo interface. It must not be used
> as a clinical diagnostic system.

## Features

- Simulation mode that works without an external model API
- Optional live MedGemma API connection through a Colab/ngrok tunnel
- PNG and JPEG retinal image uploads
- Model confidence, explanation, and XAI heatmap display
- Human confirmation and override audit records stored in SQLite
- Docker Compose setup with persistent data volumes

## Quick Start With Docker

### Requirements

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) or Docker
  Engine with Docker Compose
- Git, if cloning the project from GitHub

### 1. Clone and enter the project

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

If you already have the project folder locally, open a terminal in that folder
and continue with the next step.

### 2. Create the environment file

```bash
cp .env.example .env
```

Open `.env` and replace:

```env
SECRET_KEY=replace-with-a-long-random-value
```

Generate a suitable value with:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

The MedGemma API settings can remain empty when using simulation mode.

### 3. Build and start the app

```bash
docker compose up --build
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in a browser. The app
starts in **Simulation Mode (Static Run)**, so it can be tested immediately.

To run the containers in the background:

```bash
docker compose up --build -d
```

To stop the app:

```bash
docker compose down
```

## Using the App

1. Keep **Simulation Mode (Static Run)** selected and click **Apply Source**.
2. Upload a PNG or JPEG retinal fundus image.
3. Set the image quality score to `5` or higher and click **Save Case**.
4. Run the inference pipeline.
5. Review the result, then confirm it or submit an override rationale.

Simulation mode returns a fixed demonstration result. Use live mode for
image-specific model output.

## Live MedGemma API Mode

Live mode requires a compatible API that:

- Responds to `GET /` with JSON containing `"status": "ok"` or
  `"status": "online"`
- Accepts an image in the multipart field `image` at `POST /process`
- Accepts the API key in the `X-API-Key` header

Configure these values in `.env`:

```env
MEDGEMMA_API_URL=https://your-current-ngrok-url.ngrok-free.app
MEDGEMMA_API_KEY=the-same-key-configured-by-the-api
MEDGEMMA_API_TIMEOUT=600
```

Do not add `/process` to `MEDGEMMA_API_URL`. Restart the app after changing the
environment:

```bash
docker compose up --build -d
```

Then select **Live Colab API Tunnel** in the interface and click
**Apply Source**. The remote API or Colab notebook must remain running while
live mode is in use.

## Run Locally Without Docker

Python 3.12 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
python app1.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

For live mode, export `MEDGEMMA_API_URL` and `MEDGEMMA_API_KEY` before starting
the app.

## Configuration

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `SECRET_KEY` | Docker Compose: yes | Development fallback | Signs Flask session cookies |
| `MEDGEMMA_API_URL` | Live mode only | Empty | Base URL of the MedGemma API, without `/process` |
| `MEDGEMMA_API_KEY` | Live mode only | Empty | API key sent in the `X-API-Key` header |
| `MEDGEMMA_API_TIMEOUT` | No | `600` | Live inference timeout in seconds |
| `APP_PORT` | No | `5000` | Host port used by Docker Compose |
| `GUNICORN_WORKERS` | No | `1` | Number of Gunicorn worker processes |
| `GUNICORN_THREADS` | No | `4` | Threads per Gunicorn worker |
| `GUNICORN_TIMEOUT` | No | API timeout + 60 | Gunicorn request timeout |

If port `5000` is already in use, set another port in `.env`, for example:

```env
APP_PORT=8080
```

Then open `http://127.0.0.1:8080`.

## Data and Storage

Docker Compose stores uploads, inference results, and `assessments.db` in named
volumes. Normal shutdown preserves this data:

```bash
docker compose down
```

To stop the app and permanently delete its Docker volumes:

```bash
docker compose down --volumes
```

The `.gitignore` excludes local secrets, uploaded images, generated inference
results, virtual environments, and the SQLite database from Git.

## Run Tests

After installing the Python dependencies:

```bash
python -m unittest discover -s tests
```

## Troubleshooting

- **Compose says `SECRET_KEY` is not set:** copy `.env.example` to `.env` and
  replace the example secret.
- **Port `5000` is already allocated:** change `APP_PORT` in `.env`.
- **Live pipeline shows offline:** verify the API URL and key, and confirm that
  the Colab runtime/ngrok tunnel is still running.
- **Inference will not run:** upload a supported image and set its quality score
  to at least `5`.
- **View container logs:** run `docker compose logs -f app`.

## Publish This Project to GitHub

Create a new empty repository on GitHub without adding a README, license, or
`.gitignore`. Then run these commands from this project folder:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Before committing, verify that `.env` and generated patient or assessment data
are not included:

```bash
git status
```

Never commit API keys, `.env`, uploaded medical images, or assessment records
to a public repository.
