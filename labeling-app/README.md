---
title: Bouncing Target
emoji: 📊
colorFrom: gray
colorTo: purple
sdk: gradio
sdk_version: 5.35.0
app_file: app.py
pinned: false
short_description: draw boxes around the objects
python_version: 3.13
hf_oauth: true
---

The above YAML block configures the app for deployment on HuggingFace Spaces.

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Labeling app

This is a Gradio-based application for viewing and labeling MR sequences from the TrackRAD2025 Grand Challenge.

## Running locally

This project uses the [uv package manager](https://docs.astral.sh/uv/).

To run the app locally, first install depedencies:

```
uv sync
```

Then:

```
uv run gradio app.py
```

Note that submitting labeled images will not work, as they require pushing to a private HuggingFace dataset. But for the sake of demonstration, the app should otherwise work locally.
