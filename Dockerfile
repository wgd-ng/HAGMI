FROM python:3.13

WORKDIR /app
RUN pip install uv --no-cache-dir

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-cache-dir

RUN uv run playwright install firefox --with-deps && uv run camoufox fetch

# Copy the rest of the application code
COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the application
CMD ["uv", "run", "python", "app.py"]
