#!/bin/bash

set -e

docker compose -f docker-compose.yml kill || echo "no containers to kill"
docker compose -f docker-compose.yml down -v || echo "no volumes to remove"
docker compose -f docker-compose.yml rm -s -f -v || echo "no containers to remove"
docker compose -f docker-compose.yml build
docker compose -f docker-compose.yml up -d