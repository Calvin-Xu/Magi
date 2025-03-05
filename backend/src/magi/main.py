# src/magi/main.py
import uvicorn

from .api.server import create_app

app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "magi.main:app",
        host="0.0.0.0",
        port=1998,
        reload=True,
    )
