# Makefile for Fast RAG Math Learning System
# ===========================================
# IMPORTANT: Makefile requires TAB characters for indentation, not spaces!

.PHONY: help setup clean ingest train-gnn serve test compare-performance status health frontend-install frontend-dev frontend-build frontend-clean

# Colors for pretty output
GREEN := \033[0;32m
BLUE := \033[0;34m
YELLOW := \033[1;33m
RED := \033[0;31m
CYAN := \033[0;36m
NC := \033[0m # No Color

# Default target
help:
	@echo "$(BLUE)🚀 Fast RAG Math Learning System$(NC)"
	@echo "=================================="
	@echo ""
	@echo "$(GREEN)Setup Commands:$(NC)"
	@echo "  make setup                - Setup the fast system (one-time)"
	@echo "  make clean                - Clean cache and temporary files"
	@echo ""
	@echo "$(GREEN)Frontend Commands:$(NC)"
	@echo "  make frontend-install     - Install frontend dependencies"
	@echo "  make frontend-dev         - Start frontend development server"
	@echo "  make frontend-build       - Build frontend for production"
	@echo "  make frontend-clean       - Clean frontend build artifacts"
	@echo ""
	@echo "$(GREEN)Phase 1 - Offline Processing:$(NC)"
	@echo "  make update-preamble      - Update master preamble with new commands from latest .tex file"
	@echo "  make ingest               - Process new documents (fast parallel)"
	@echo "  make train-gnn            - Train GNN for graph embeddings (optional)"
	@echo "  make process-all          - Run both ingest and train-gnn"
	@echo ""
	@echo "$(GREEN)Phase 2 - Online Server:$(NC)"
	@echo "  make serve                - Start optimized API server"
	@echo "  make serve-dev            - Start server in development mode"
	@echo "  make app                  - Start interactive RAG application"
	@echo ""
	@echo "$(GREEN)Testing & Monitoring:$(NC)"
	@echo "  make test                 - Run all pytest tests"
	@echo "  make compare-performance  - Compare original vs optimized performance"
	@echo "  make status               - Check system status"
	@echo "  make health               - Health check (server must be running)"
	@echo ""
	@echo "$(GREEN)Database Management:$(NC)"
	@echo "  make db-stats             - Show database statistics"
	@echo "  make check-fragments      - Check for data fragments"
	@echo "  make clean-db             - Clean all databases (with confirmation)"
	@echo "  make clean-weaviate       - Clean only Weaviate (with confirmation)"
	@echo "  make clean-graph          - Clean graph and embeddings"
	@echo "  make reset                - Complete system reset"
	@echo ""
	@echo "$(GREEN)Quick Start:$(NC)"
	@echo "  make setup && make ingest && make serve"
	@echo "  make frontend-install && make frontend-dev"

# Setup the fast system
setup:
	@echo "$(BLUE)🔧 Setting up Fast RAG System...$(NC)"
	@mkdir -p data/embeddings/cache/
	@mkdir -p data/performance_logs/
	@mkdir -p data/latex_temp/logs/
	@mkdir -p src/retrieval/
	@mkdir -p scripts/
	@echo "$(GREEN)✓ Created directory structure$(NC)"
	@python -m compileall src/ --quiet
	@echo "$(GREEN)✓ Pre-compiled Python files$(NC)"
	@echo "$(BLUE)🎉 Setup complete! You can now run 'make ingest'$(NC)"

# Clean cache and temporary files
clean:
	@echo "$(YELLOW)🧹 Cleaning cache and temporary files...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf data/embeddings/cache/* 2>/dev/null || true
	@rm -rf data/performance_logs/* 2>/dev/null || true
	@rm -rf data/latex_temp/logs/* 2>/dev/null || true
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

# Phase 1: Offline Processing
update-preamble:
	@echo "$(BLUE)📚 Updating preamble...$(NC)"
	@python scripts/update_preamble.py
	@echo "$(GREEN)✓ Preamble updated!$(NC)"

ingest:
	@echo "$(BLUE)📚 Starting fast document ingestion...$(NC)"
	@echo "$(YELLOW)This processes new LaTeX documents with parallel embedding$(NC)"
	@python -m src.pipeline ingest
	@echo "$(GREEN)✓ Document ingestion complete!$(NC)"

train-gnn:
	@echo "$(BLUE)🧠 Training GNN for graph-aware embeddings...$(NC)"
	@echo "$(YELLOW)This creates advanced embeddings using the knowledge graph$(NC)"
	@python -m src.pipeline train-gnn
	@echo "$(GREEN)✓ GNN training complete!$(NC)"

process-all: ingest train-gnn
	@echo "$(GREEN)🎉 All offline processing complete!$(NC)"
	@echo "$(BLUE)Next step: Run 'make serve' to start the API server$(NC)"

# Phase 2: Online Server
serve:
	@echo "$(BLUE)🖥️  Starting optimized API server...$(NC)"
	@echo "$(YELLOW)Server will be available at: http://localhost:8000$(NC)"
	@echo "$(YELLOW)API docs at: http://localhost:8000/docs$(NC)"
	@echo "$(YELLOW)Press Ctrl+C to stop the server$(NC)"
	@python -m src.pipeline serve

serve-dev:
	@echo "$(BLUE)🛠️  Starting API server in development mode...$(NC)"
	@echo "$(YELLOW)Development mode includes auto-reload and debug info$(NC)"
	@uvicorn src.api.fast_api:app --host 0.0.0.0 --port 8000 --reload

app:
	@echo "$(BLUE)🎯 Starting interactive RAG application...$(NC)"
	@echo "$(YELLOW)This starts the interactive learning session$(NC)"
	@echo "$(YELLOW)Press Ctrl+C to exit$(NC)"
	@python -m src.app

# Testing & Performance
test:
	@echo "🏃 Running tests with timeout protection..."
	@docker-compose run --rm app timeout 300 pytest -v --tb=short

pytest: test

# System status and health
status:
	@echo "$(BLUE)📋 System Status Check$(NC)"
	@echo "======================"
	@echo ""
	@echo "$(GREEN)Directory Structure:$(NC)"
	@ls -la data/ 2>/dev/null || echo "$(RED)❌ data/ directory not found$(NC)"
	@ls -la data/embeddings/ 2>/dev/null || echo "$(YELLOW)⚠️  No embeddings directory$(NC)"
	@ls -la data/graph_db/ 2>/dev/null || echo "$(YELLOW)⚠️  No graph database$(NC)"
	@echo ""
	@echo "$(GREEN)Key Files:$(NC)"
	@if [ -f "data/embeddings/initial_text_embeddings.pkl" ]; then \
		echo "$(GREEN)✓ Initial embeddings found$(NC)"; \
	else \
		echo "$(RED)❌ Initial embeddings missing - run 'make ingest'$(NC)"; \
	fi
	@if [ -f "data/embeddings/gnn_embeddings.pkl" ]; then \
		echo "$(GREEN)✓ GNN embeddings found$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  GNN embeddings missing - run 'make train-gnn'$(NC)"; \
	fi
	@if [ -f "data/graph_db/knowledge_graph.graphml" ]; then \
		echo "$(GREEN)✓ Knowledge graph found$(NC)"; \
	else \
		echo "$(RED)❌ Knowledge graph missing - run 'make ingest'$(NC)"; \
	fi
	@echo ""
	@echo "$(GREEN)Python Environment:$(NC)"
	@python --version
	@python -c "import src.pipeline; print('✓ Fast pipeline module available')" 2>/dev/null || echo "$(RED)❌ Fast pipeline not available$(NC)"

health:
	@echo "$(BLUE)🏥 Health Check (server must be running)$(NC)"
	@curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/v1/health | grep 200 > /dev/null && \
	curl -s http://localhost:8000/api/v1/health | python -m json.tool || \
	echo "$(RED)❌ Server not responding at localhost:8000$(NC)"

# Frontend Commands
frontend-install:
	@echo "$(BLUE)📦 Installing frontend dependencies...$(NC)"
	@cd frontend && npm install
	@echo "$(GREEN)✓ Frontend dependencies installed$(NC)"

frontend-dev:
	@echo "$(BLUE)🖥️  Starting frontend development server...$(NC)"
	@echo "$(YELLOW)Frontend will be available at: http://localhost:3000$(NC)"
	@cd frontend && npm run dev

frontend-build:
	@echo "$(BLUE)🏗️  Building frontend for production...$(NC)"
	@cd frontend && npm run build
	@echo "$(GREEN)✓ Frontend build complete!$(NC)"
	@echo "$(YELLOW)Build output is in frontend/build$(NC)"

frontend-clean:
	@echo "$(YELLOW)🧹 Cleaning frontend build artifacts...$(NC)"
	@rm -rf frontend/.next 2>/dev/null || true
	@rm -rf frontend/node_modules 2>/dev/null || true
	@echo "$(GREEN)✓ Frontend cleanup complete$(NC)"

# Database Management Commands
db-stats:
	@echo "$(BLUE)📊 Database Statistics$(NC)"
	@python scripts/clean_databases.py --stats-only

check-fragments:
	@echo "$(BLUE)🔍 Checking for data fragments...$(NC)"
	@echo ""
	@echo "$(CYAN)Weaviate Status:$(NC)"
	@make weaviate-count 2>/dev/null || echo "$(RED)Weaviate not accessible$(NC)"
	@echo ""
	@echo "$(CYAN)Graph Database:$(NC)"
	@if [ -f "data/graph_db/knowledge_graph.graphml" ]; then \
		echo "$(GREEN)✓ Graph file exists$(NC)"; \
		ls -lh data/graph_db/knowledge_graph.graphml; \
	else \
		echo "$(YELLOW)No graph file found$(NC)"; \
	fi
	@echo ""
	@echo "$(CYAN)Embeddings:$(NC)"
	@ls -lh data/embeddings/*.pkl 2>/dev/null || echo "$(YELLOW)No embedding files found$(NC)"
	@echo ""
	@echo "$(CYAN)Processed Documents:$(NC)"
	@if [ -f "data/processed_documents_log.txt" ]; then \
		wc -l data/processed_documents_log.txt; \
	else \
		echo "$(YELLOW)No processed documents log$(NC)"; \
	fi

clean-db:
	@echo "$(RED)⚠️  WARNING: This will delete all data in the databases!$(NC)"
	@echo "$(YELLOW)This includes: Weaviate data, knowledge graph, embeddings$(NC)"
	@read -p "Are you sure? (y/N): " confirm && \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		python scripts/clean_databases.py --confirm; \
	else \
		echo "$(YELLOW)Cancelled$(NC)"; \
	fi

clean-weaviate:
	@echo "$(RED)⚠️  WARNING: This will delete all data in Weaviate!$(NC)"
	@read -p "Are you sure? (y/N): " confirm && \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		echo "$(YELLOW)🗑️  Cleaning Weaviate database only...$(NC)"; \
		python -c "from scripts.clean_databases import clean_weaviate_data; clean_weaviate_data()"; \
		echo "$(GREEN)✓ Weaviate cleaned$(NC)"; \
	else \
		echo "$(YELLOW)Cancelled$(NC)"; \
	fi

clean-graph:
	@echo "$(YELLOW)🗑️  Cleaning graph database...$(NC)"
	@rm -f data/graph_db/knowledge_graph.graphml
	@rm -f data/embeddings/initial_text_embeddings.pkl
	@rm -f data/embeddings/gnn_embeddings.pkl
	@echo "$(GREEN)✓ Graph and embeddings cleaned$(NC)"

reset: clean
	@echo "$(RED)🗑️  Resetting system - removing all generated data...$(NC)"
	@python scripts/clean_databases.py --confirm
	@rm -f data/processed_documents_log.txt
	@rm -f data/learner_profiles.sqlite3
	@echo "$(GREEN)🔄 System reset complete!$(NC)"
	@echo "$(YELLOW)Add documents to data/raw_latex/ and run 'make ingest'$(NC)"