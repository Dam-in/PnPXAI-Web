import matplotlib
matplotlib.use("Agg")

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.core.job_manager import start_cleanup_task, stop_cleanup_task


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_cleanup_task()
    yield
    stop_cleanup_task()


app = FastAPI(
    title="XAI Demo Platform v2",
    description="Multi-modal eXplainable AI demo platform powered by PnPXAI",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
