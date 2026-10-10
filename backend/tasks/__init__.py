from backend.tasks.image import ImageTaskHandler
from backend.tasks.text import TextTaskHandler
from backend.tasks.timeseries import TimeSeriesTaskHandler

_task_handlers = {
    "image": ImageTaskHandler(),
    "text": TextTaskHandler(),
    "timeseries": TimeSeriesTaskHandler(),
}


def get_task_handler(task_name: str):
    handler = _task_handlers.get(task_name)
    if handler is None:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(_task_handlers.keys())}")
    return handler


def list_tasks() -> list[dict]:
    return [
        {"name": "image", "display_name": "Image Classification", "description": "Explain image classification models with heatmap visualizations"},
        {"name": "text", "display_name": "Text Classification", "description": "Explain text classification models with token-level attribution"},
        {"name": "timeseries", "display_name": "Time-Series Classification", "description": "Explain time-series classification models with temporal attribution"},
    ]
