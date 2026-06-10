# MGDS

A Flask-based research system for analyzing glaucoma and classifying images using medgemma-1.5-4b-it.

> **Important:** This project is a demo system. It must not be used
> as a clinical diagnostic system.

## Features

- Live MedGemma API connection through ngrok tunnel
- PNG and JPEG retinal image uploads
- Model confidence, explanation, and XAI heatmap display
- Human confirmation and override audit records stored in SQLite
- Docker Compose setup

## Start GUI app

```bash
cd app
docker compose up --build -d
```
