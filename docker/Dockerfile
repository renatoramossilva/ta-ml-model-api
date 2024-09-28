FROM python:3.12-slim

WORKDIR /app

# Copy the requirements file to the working directory
COPY pyproject.toml poetry.lock* ./

# Install Poetry and dependencies
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

# Copy the rest of the application code to the working directory
COPY . .

# Run tests
RUN coverage run -m pytest

# Export test coverage to HTML file
RUN coverage html

# Expose the ports that the app/coverage runs on
EXPOSE 5000 8000

# Star flask and coverage page
CMD ["sh", "-c", "flask run --host=0.0.0.0 & python -m http.server 8000 --directory htmlcov"]
