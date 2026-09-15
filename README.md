# Nexus_Bot
AI chatbot in the actuarial sense

## Run on Render

Create a Render Web Service from this directory, or use the included `render.yaml` blueprint.
Render will install `requirements.txt` and start the app with Gunicorn.

Required environment variable:

- `GOOGLE_API_KEY`: Google GenAI API key

The application serves the UI and API from the same origin. After deployment, open the Render URL
and use `/health` to verify the service is ready.
