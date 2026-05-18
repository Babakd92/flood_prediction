# 14-Day Flood Prediction for Huron River, Ohio

This repository runs a daily Random Forest flood prediction model for the Huron River basin and generates an interactive forecast dashboard.

## Main Files

- `version_4.py` - daily flood prediction model and dashboard generator
- `input.csv` - historical daily water-level and precipitation input data
- `huron_river_basin.jpg` - basin map used in the dashboard
- `index.html` - generated interactive dashboard for GitHub Pages
- `requirements.txt` - Python dependencies for GitHub Actions
- `.github/workflows/daily-flood-forecast.yml` - scheduled GitHub Actions workflow

## Automation

GitHub Actions runs the model every day at 1:00 AM Eastern during daylight saving time, regenerates the dashboard and output files, and commits updated outputs back to the repository.

The workflow can also be started manually from the repository's Actions tab.
