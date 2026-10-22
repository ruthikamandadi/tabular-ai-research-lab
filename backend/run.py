"""Run the API using the backend port configured in the project .env file."""
import uvicorn

from app.core.config import BACKEND_PORT


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=BACKEND_PORT, reload=True)
