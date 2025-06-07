# Makefile for Fast RAG Math Learning System
# ===========================================

.PHONY: help setup clean ingest train-gnn serve test compare-performance status health

# Colors for pretty output
GREEN := \033[0;32m
BLUE := \033[0;34m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Default target
help:
	@echo "$(BLUE)🚀 Fast RAG Math Learning System$(NC)"
	@echo "=================================="
	@echo ""
	@echo "$(GREEN)Setup Commands:$(NC)"
	@echo "  make setup           - Setup the fast system (one-time)"
	@echo "  make clean           - Clean cache and temporary files"
	@echo ""
	@echo "$(GREEN)Phase 1 - Offline Processing:$(NC)"
	@echo "  make ingest          - Process new documents (fast parallel)"
	@echo "  make train-gnn       - Train GNN for graph embeddings (optional)"
	@echo "  make process-all     - Run both ingest and train-gnn"
	@echo ""
	@echo "$(GREEN)Phase 2 - Online Server:$(NC)"
	@echo "  make serve           - Start optimized API server"
	@echo "  make serve-dev       - Start server in development mode"
	@echo ""
	@echo "$(GREEN)Testing & Monitoring:$(NC)"
	@echo "  make test            - Run performance tests"
	@echo "  make compare         - Compare original vs optimized performance"
	@echo "  make status          - Check system status"
	@echo "  make health          - Health check (server must be running)"
	@echo ""
	@echo "$(GREEN)Quick Start:$(NC)"
	@echo "  make setup && make ingest && make serve"

# Setup the fast system
setup:
	@echo "$(BLUE)🔧 Setting up Fast RAG System...$(NC)"
	@mkdir -p data/embeddings/cache/
	@mkdir -p data/performance_logs/
	@mkdir -p src/retrieval/
	@mkdir -p scripts/
	@echo "$(GREEN)✓ Created directory structure$(NC)"
	
	@if [ -f "src/api/main_api.py" ]; then \
		cp src/api/main_api.py src/api/main_api.py.backup; \
		echo "$(GREEN)✓ Backed up main_api.py$(NC)"; \
	fi
	
	@if [ -f "src/pipeline.py" ]; then \
		cp src/pipeline.py src/pipeline.py.backup; \
		echo "$(GREEN)✓ Backed up pipeline.py$(NC)"; \
	fi
	
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
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

# Phase 1: Offline Processing
ingest:
	@echo "$(BLUE)📚 Starting fast document ingestion...$(NC)"
	@echo "$(YELLOW)This processes new LaTeX documents with parallel embedding$(NC)"
	@python -m src.fast_pipeline ingest
	@echo "$(GREEN)✓ Document ingestion complete!$(NC)"

train-gnn:
	@echo "$(BLUE)🧠 Training GNN for graph-aware embeddings...$(NC)"
	@echo "$(YELLOW)This creates advanced embeddings using the knowledge graph$(NC)"
	@python -m src.fast_pipeline train-gnn
	@echo "$(GREEN)✓ GNN training complete!$(NC)"

# Process both ingest and train-gnn
process-all: ingest train-gnn
	@echo "$(GREEN)🎉 All offline processing complete!$(NC)"
	@echo "$(BLUE)Next step: Run 'make serve' to start the API server$(NC)"

# Phase 2: Online Server
serve:
	@echo "$(BLUE)🖥️  Starting optimized API server...$(NC)"
	@echo "$(YELLOW)Server will be available at: http://localhost:8000$(NC)"
	@echo "$(YELLOW)API docs at: http://localhost:8000/docs$(NC)"
	@echo "$(YELLOW)Press Ctrl+C to stop the server$(NC)"
	@python -m src.fast_pipeline serve

serve-dev:
	@echo "$(BLUE)🛠️  Starting API server in development mode...$(NC)"
	@echo "$(YELLOW)Development mode includes auto-reload and debug info$(NC)"
	@uvicorn src.api.fast_api:app --host 0.0.0.0 --port 8000 --reload

# Testing & Performance
test:
	@echo "$(BLUE)🏃 Running performance tests...$(NC)"
	@python -m src.fast_pipeline test

compare:
	@echo "$(BLUE)📊 Comparing original vs optimized performance...$(NC)"
	@python scripts/compare_performance.py

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
	@python -c "import src.fast_pipeline; print('✓ Fast pipeline module available')" 2>/dev/null || echo "$(RED)❌ Fast pipeline not available$(NC)"

health:
	@echo "$(BLUE)🏥 Health Check (server must be running)$(NC)"
	@curl -s http://localhost:8000/api/v1/health | python -m json.tool 2>/dev/null || echo "$(RED)❌ Server not responding at localhost:8000$(NC)"



# Monitor server performance (requires server to be running)
monitor:
	@echo "$(BLUE)📈 Monitoring server performance...$(NC)"
	@echo "$(YELLOW)Press Ctrl+C to stop monitoring$(NC)"
	@while true; do \
		echo "$(BLUE)[$(shell date '+%H:%M:%S')] Performance Check:$(NC)"; \
		curl -s http://localhost:8000/api/v1/performance | python -m json.tool 2>/dev/null || echo "$(RED)Server not responding$(NC)"; \
		echo ""; \
		sleep 5; \
	done

# Quick development workflow
dev-workflow: clean setup ingest serve-dev

# Production deployment workflow  
prod-workflow: clean setup process-all
	@echo "$(GREEN)🚀 Production setup complete!$(NC)"
	@echo "$(BLUE)Run 'make serve' to start the production server$(NC)"

# Show performance tips
tips:
	@echo "$(BLUE)💡 Performance Tips$(NC)"
	@echo "==================="
	@echo ""
	@echo "$(GREEN)For Best Performance:$(NC)"
	@echo "  • Run 'make process-all' before starting the server"
	@echo "  • Use 'make serve' (not serve-dev) for production"
	@echo "  • Monitor cache hit rates with 'make health'"
	@echo "  • Run 'make clean' if you encounter issues"
	@echo ""
	@echo "$(GREEN)Cache Management:$(NC)"
	@echo "  • Clear cache: curl -X POST http://localhost:8000/api/v1/clear_cache"
	@echo "  • Check cache stats: curl http://localhost:8000/api/v1/performance"
	@echo ""
	@echo "$(GREEN)Troubleshooting:$(NC)"
	@echo "  • Check status: make status"
	@echo "  • View logs: Check terminal output from 'make serve'"
	@echo "  • Reset everything: make clean setup process-all"

# Install development dependencies (optional)
install-dev:
	@echo "$(BLUE)📦 Installing optional development dependencies...$(NC)"
	@pip install psutil uvloop redis-py 2>/dev/null || echo "$(YELLOW)⚠️  Some optional packages failed to install$(NC)"
	@echo "$(GREEN)✓ Development dependencies installed$(NC)"
