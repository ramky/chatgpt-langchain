import os
from langfuse.client import Langfuse

langfuse = Langfuse(
    os.getenv("LANGFUSE_PUBLIC_KEY"),
    os.getenv("LANGFUSE_SECRET_KEY"),
    host="https://prod-langfuse.fly.dev",
)
