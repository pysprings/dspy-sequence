# Makefile for DSPy Mastery Series Presentations

# Tools
PANDOC := pandoc
PDFLATEX := pdflatex
CONVERT := convert

# Directories
SRC_DIR := sessions
DIST_DIR := docs
IMG_DIR := assets/images

# Find all session markdown files named presentation.md
SOURCES := $(wildcard $(SRC_DIR)/session-*/presentation.md)
# Generate corresponding HTML output file names in the dist directory
TARGETS := $(patsubst $(SRC_DIR)/%/presentation.md,$(DIST_DIR)/%/presentation.html,$(SOURCES))

# Case study sources and targets
CASE_STUDY_SOURCES := $(wildcard case-studies/*/TOUR.md)
CASE_STUDY_TARGETS := $(patsubst case-studies/%/TOUR.md,$(DIST_DIR)/case-studies/%/tour.html,$(CASE_STUDY_SOURCES))

# TikZ diagram sources and targets
TIKZ_SOURCES := $(wildcard $(IMG_DIR)/*.tex)
TIKZ_PDFS := $(TIKZ_SOURCES:.tex=.pdf)
TIKZ_PNGS := $(TIKZ_SOURCES:.tex=.png)

# Pandoc flags for HTML presentation using pandoc-plot.
PANDOC_FLAGS := --filter pandoc-plot \
                --embed-resources \
                --standalone \
                -c assets/css/main-style.css

# Phony targets do not represent files.
.PHONY: all clean help

## Build all presentations and diagrams (default).
all: $(TIKZ_PNGS) $(TARGETS) $(CASE_STUDY_TARGETS)

# Pattern rule to build a single presentation.
# Creates the destination directory if it doesn't exist.
$(DIST_DIR)/session-%/presentation.html: $(SRC_DIR)/session-%/presentation.md
	@mkdir -p $(@D)
	@echo "Building $< -> $@"
	@$(PANDOC) $(PANDOC_FLAGS) -o $@ $<

# Pattern rule to build case study tours.
$(DIST_DIR)/case-studies/%/tour.html: case-studies/%/TOUR.md
	@mkdir -p $(@D)
	@echo "Building $< -> $@"
	@$(PANDOC) $(PANDOC_FLAGS) -o $@ $<

# Pattern rule to build TikZ diagrams.
$(IMG_DIR)/%.pdf: $(IMG_DIR)/%.tex
	@echo "Building $< -> $@"
	@cd $(IMG_DIR) && $(PDFLATEX) -interaction=nonstopmode $(notdir $<) > /dev/null 2>&1
	@rm -f $(IMG_DIR)/$*.aux $(IMG_DIR)/$*.log

$(IMG_DIR)/%.png: $(IMG_DIR)/%.pdf
	@echo "Converting $< -> $@"
	@$(CONVERT) -density 300 $< -quality 95 $@

## Clean up all generated files.
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

