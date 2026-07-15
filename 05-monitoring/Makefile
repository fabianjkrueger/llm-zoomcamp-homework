# Makefile for homework 5
# I deliberately kept it in this subdirectory
# it won't be possible to run this while you're in the repo root directory
# however, this is on purpose, because the entire homework repo is organized
# like this, and each homework is a different subdirectory independent of others

# run the RAG agent from the command line
rag_cli:
	uv run python assistant.py

chat:
	uv run streamlit run app.py

network:
	docker network create monitoring

postgres: network
	docker run -it \
		--name course-assistant-pg \
		--network monitoring \
		-e POSTGRES_USER=user \
		-e POSTGRES_PASSWORD=password \
		-e POSTGRES_DB=course_assistant \
		-p 5432:5432 \
		-v pgdata:/var/lib/postgresql/data \
		postgres:17