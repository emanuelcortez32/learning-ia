.PHONY: help install dev build test clean deploy docker-up docker-down

APP_NAME := $(shell grep '^name = ' pyproject.toml | sed 's/name = "\(.*\)"/\1/')
APP_VERSION := $(shell grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')

setup:
	@echo "🛠️ Iniciando setup en modo desarrollo ..."
	./scripts/setup.sh

info:
	@echo "📋 Informacion del Proyecto: "
	@echo " Nombre: ${APP_NAME}"
	@echo " Version: ${APP_VERSION}"


.DEFAULT_GOAL := info