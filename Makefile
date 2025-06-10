# Makefile for DSPy Mastery Series Presentations

# Tools
PANDOC := pandoc
MERMAID_FILTER := mermaid-filter

# Directories
SRC_DIR := sessions
DIST_DIR := dist

# Find all session markdown files named presentation.md
SOURCES := $(wildcard $(SRC_DIR)/session-*/presentation.md)
# Generate corresponding HTML output file names in the dist directory
TARGETS := $(patsubst $(SRC_DIR)/%/presentation.md,$(DIST_DIR)/%/presentation.html,$(SOURCES))

# Pandoc flags for reveal.js presentation with mermaid support.
# This configuration uses reveal.js from a public CDN.
# You must install the mermaid-filter: npm install -g mermaid-filter
PANDOC_FLAGS := -t revealjs -s \
                --filter=$(MERMAID_FILTER) \
                -V theme=black \
                -V revealjs-url=https://unpkg.com/reveal.js@4/dist/

# Phony targets do not represent files.
.PHONY: all clean help

# Default target: build all presentations.
all: $(TARGETS)

# Pattern rule to build a single presentation.
# Creates the destination directory if it doesn't exist.
$(DIST_DIR)/session-%/presentation.html: $(SRC_DIR)/session-%/presentation.md
	@mkdir -p $(@D)
	@echo "Building $< -> $@"
	@$(PANDOC) $(PANDOC_FLAGS) -o $@ $<

# Clean up all generated files.
clean:
	@echo "Cleaning up generated files in $(DIST_DIR)..."
	@rm -rf $(DIST_DIR)

# Display help information.
help:
	@echo "Available commands:"
	@echo "  all    - Build all presentations (default)."
	@echo "  clean  - Remove all generated files from the '$(DIST_DIR)' directory."
	@echo "  help   - Show this help message."

