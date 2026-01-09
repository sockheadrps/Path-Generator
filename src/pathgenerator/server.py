from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ValidationError
from .generator import PDPathGenerator
import logging

log = logging.getLogger(__name__)


ROOT = Path(__file__).resolve().parent
TEMPLATES_DIR = ROOT / "templates"
STATIC_DIR = ROOT / "static"


app = FastAPI(title="Path Generator API", version="1.0.0")

# Serve static assets (CSS, JS)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# Serve the Generator at root
@app.get("/")
async def index():
    playground_path = TEMPLATES_DIR / "playground.html"
    if not playground_path.exists():
        raise HTTPException(status_code=404, detail="playground.html not found")
    return FileResponse(playground_path)

@app.get("/playground")
async def playground():
    return await index()


class GenerateRequest(BaseModel):
    start: Tuple[float, float]
    target: Tuple[float, float]
    screen_w: int
    screen_h: int
    
    # Simplified PD knobs
    mouse_velocity: float = 0.35
    kp_start: float = 0.010
    kp_end: float = 0.010
    stabilization: float = 0.15
    noise: float = 0.0
    keep_prob_start: float = 0.70
    keep_prob_end: float = 0.98
    arc_strength: float = 0.0
    variance: float = 0.0
    overshoot_prob: float = 0.0  # Probability of overshooting target


@app.get("/api/ping")
async def ping():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    try:
        # Unpack all fields from GenerateRequest for the generator
        generator = PDPathGenerator()
        path_fixed, prog, steps, actual_params = generator.generate_path(
            start_x=req.start[0],
            start_y=req.start[1],
            end_x=req.target[0],
            end_y=req.target[1],
            canvas_width=req.screen_w,
            canvas_height=req.screen_h,
            **req.model_dump(exclude={'start', 'target', 'screen_w', 'screen_h'})
        )
        return {
            "path": path_fixed.tolist(),
            "prog": prog,
            "steps": steps,
            "actual_params": actual_params
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", 8002)))

if __name__ == "__main__":
    main()
