#!/usr/bin/env sh

# Optionally, ensure we execute from the /app directory
cd /app

# Start the Uvicorn server (or relevant startup command)
exec uvicorn magi.main:app --host 0.0.0.0 --port 8000
