.PHONY: help install dev build test clean deploy docker-up docker-down

APP_NAME := $(shell grep '^name = ' pyproject.toml | sed 's/name = "\(.*\)"/\1/')
APP_VERSION := $(shell grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')

setup:
	@echo "🛠️ Iniciando setup en modo desarrollo ..."
	./scripts/setup.sh

dev:
	@echo "▶️ Iniciando app en modo desorrollo ..."
	./scripts/run-app.sh

test:
	@echo "🧪 Ejecutando tests ..."
	./scripts/run-tests.sh

docker-up:
	@echo "🐳 Iniciando servicios con Docker ..."
	./scripts/run-docker.sh up

docker-up-all:
	@echo "🐳 Iniciando servicios con Docker ..."
	./scripts/run-docker.sh up-all

docker-down:
	@echo "⏹️ Deteniendo servicios con Docker ..."
	./scripts/run-docker.sh down

docker-clean:
	@echo "🧹 Limpiando recursos de Docker ..."
	./scripts/run-docker.sh clean

clean:
	@echo "🧹 Limpiando archivos temporales ..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/
	@echo "✅ Limpieza completa"

info:
	@echo "📋 Informacion del Proyecto: "
	@echo " Nombre: ${APP_NAME}"
	@echo " Version: ${APP_VERSION}"


.DEFAULT_GOAL := info