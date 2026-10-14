import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router, cleanup_old_files
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    async def cleanup_loop():
        while True:
            try:
                await run_in_threadpool(cleanup_old_files)
            except Exception as e:          # never let housekeeping kill the app
                print(f"[cleanup] failed: {e}")
            await asyncio.sleep(settings.cleanup_interval_hours * 3600)

    task = asyncio.create_task(cleanup_loop())
    yield
    task.cancel()


app = FastAPI(title="TennisVision AI", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def reject_oversized_uploads(request: Request, call_next):
    """Refuse too-large uploads from the Content-Length header, before the body is read."""
    if request.method == "POST" and request.url.path == "/api/analyze":
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > (settings.max_upload_mb + 1) * 1024 * 1024:
            return JSONResponse({"detail": f"Video is larger than {settings.max_upload_mb} MB"},
                                status_code=413)
    return await call_next(request)


app.include_router(router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------------------------
# Built frontend (production). In development the Vite dev server serves the UI instead.
# ---------------------------------------------------------------------------------------------
_dist = Path(settings.frontend_dist)
if (_dist / "index.html").exists():
    if (_dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        # Real files in dist/ (favicon etc.), otherwise the SPA entry point
        candidate = (_dist / full_path).resolve()
        if full_path and candidate.is_file() and _dist.resolve() in candidate.parents:
            return FileResponse(candidate)
        if full_path.startswith("api/"):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        return FileResponse(_dist / "index.html")
