# Makefile for DSPy Mastery Series Presentations

# Tools
PANDOC := pandoc

# Directories
SRC_DIR := sessions
DIST_DIR := docs

# Find all session markdown files named presentation.md
SOURCES := $(wildcard $(SRC_DIR)/session-*/presentation.md)
# Generate corresponding HTML output file names in the dist directory
TARGETS := $(patsubst $(SRC_DIR)/%/presentation.md,$(DIST_DIR)/%/presentation.html,$(SOURCES))

# Pandoc flags for HTML presentation using pandoc-plot.
PANDOC_FLAGS := --filter pandoc-plot \
                --embed-resources \
                --standalone \
                -c assets/css/main-style.css

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
	@echo "Cleaning up generated files in $(DIST_DIR) and plots/ directory..."
	@rm -rf $(DIST_DIR)
	@rm -rf plots/

## This help screen
help:
	@printf "Available targets:\n\n"
	@awk '/^[a-zA-Z\-_0-9%:\\]+/ { \
		helpMessage = match(lastLine, /^## (.*)/); \
		if (helpMessage) { \
			helpCommand = $$1; \
			helpMessage = substr(lastLine, RSTART + 3, RLENGTH); \
			gsub("\\\\", "", helpCommand); \
			gsub(":+$$", "", helpCommand); \
			printf "  \x1b[32;01m%-35s\x1b[0m %s\n", helpCommand, helpMessage; \
		} \
	} \
	{ lastLine = $$0 }' $(MAKEFILE_LIST) | sort -u
	@printf "\n"

