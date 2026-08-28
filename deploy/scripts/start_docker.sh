#!/bin/bash
# Log everything to start_docker.log
exec > /home/ubuntu/start_docker.log 2>&1

# AWS account ID and region are resolved at deploy time by CI (or set here)
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-YOUR_AWS_ACCOUNT_ID}"
AWS_REGION="${AWS_REGION:-us-east-1}"
IMAGE="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/tweet-sentiment-plugin:latest"

echo "Logging in to ECR..."
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

echo "Pulling Docker image..."
docker pull "$IMAGE"

echo "Checking for existing container..."
if [ "$(docker ps -q -f name=campusx-app)" ]; then
    echo "Stopping existing container..."
    docker stop campusx-app
fi

if [ "$(docker ps -aq -f name=campusx-app)" ]; then
    echo "Removing existing container..."
    docker rm campusx-app
fi

echo "Starting new container..."
docker run -d -p 80:5000 --name campusx-app "$IMAGE"

echo "Container started successfully."
