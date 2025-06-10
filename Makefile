# Makefile for DSPy Mastery Series Presentations

# Tools
PANDOC := pandoc

# Directories
SRC_DIR := sessions
DIST_DIR := dist

# Find all session markdown files named presentation.md
SOURCES := $(wildcard $(SRC_DIR)/session-*/presentation.md)
# Generate corresponding PDF output file names in the dist directory
TARGETS := $(patsubst $(SRC_DIR)/%/presentation.md,$(DIST_DIR)/%/presentation.pdf,$(SOURCES))

# Pandoc flags for Beamer PDF presentation.
# Uses LaTeX for math and code highlighting.
# Example theme: Warsaw with colortheme seahorse.
PANDOC_FLAGS := -s -t beamer \
                --pdf-engine=xelatex \
                -V mainfont="Noto Sans" \
                -V theme:Warsaw \
                -V colortheme:seahorse \
                --listings \
                -F mermaid-filter

# Phony targets do not represent files.
.PHONY: all clean help

# Default target: build all presentations.
all: $(TARGETS)

# Pattern rule to build a single presentation.
# Creates the destination directory if it doesn't exist.
$(DIST_DIR)/session-%/presentation.pdf: $(SRC_DIR)/session-%/presentation.md
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

