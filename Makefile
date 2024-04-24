install:
	poetry install --sync --no-root

install-dev:
	poetry install --no-root
	pre-commit install && pre-commit autoupdate
