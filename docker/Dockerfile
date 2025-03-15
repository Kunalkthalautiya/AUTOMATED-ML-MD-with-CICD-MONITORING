# Base Image
FROM python:3.8

# Set working directory
WORKDIR /app

# Copy project files
COPY app.py model.pkl vectorizer.pkl requirements.txt /app/

# Install dependencies
RUN pip install -r requirements.txt

# Expose port
EXPOSE 5000

# Run Flask App
CMD ["python", "app.py"]
