"""Root entrypoint for the HealthMonitor FastAPI service."""

from healthmonitor.main import *  # noqa: F401,F403


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("healthmonitor.main:app", host="127.0.0.1", port=8000)
