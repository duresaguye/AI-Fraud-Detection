# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the model file into the container
COPY model/fraud_random_forest_model.pkl /app/model/fraud_random_forest_model.pkl

# Copy the current directory contents into the container
COPY . .

# Install required packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000
EXPOSE 5000

# Run the application
CMD ["python", "serve_model.py"]