#!/bin/bash
# start.sh — Run frontend and backend together
# Usage: bash start.sh

echo "Starting NeuroSearch Dashboard..."
echo "  Backend  → http://localhost:8000"
echo "  Frontend → http://localhost:5173"
echo ""

# Run backend and frontend concurrently
concurrently \
  --names "BACKEND,FRONTEND" \
  --prefix-colors "cyan,magenta" \
  "python3 -m uvicorn backend.main:app --reload --port 8000" \
  "cd frontend && npm run dev"
