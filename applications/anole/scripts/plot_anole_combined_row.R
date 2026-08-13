#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(cowplot)
  library(ggplot2)
  library(ragg)
  library(rsvg)
  library(svglite)
})

script_path <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg) > 0) {
    return(normalizePath(sub("^--file=", "", file_arg[[1]]), mustWork = TRUE))
  }
  fallback <- file.path("scripts", "plot_anole_combined_row.R")
  if (file.exists(fallback)) {
    return(normalizePath(fallback, mustWork = TRUE))
  }
  normalizePath("plot_anole_combined_row.R", mustWork = FALSE)
}

cli_value <- function(name, default) {
  pattern <- paste0("^--", name, "=")
  value <- grep(pattern, commandArgs(trailingOnly = TRUE), value = TRUE)
  if (length(value) > 0) {
    sub(pattern, "", value[[length(value)]])
  } else {
    default
  }
}

SCRIPT_DIR <- dirname(script_path())
PROJECT_ROOT <- normalizePath(file.path(SCRIPT_DIR, ".."), mustWork = TRUE)
OUTPUT_DIR <- cli_value("output-dir", file.path(PROJECT_ROOT, "output"))
OUTPUT_NAME <- cli_value("output-name", "anole_phylogeny_drift_matrix_row_10m_fresh_2m")
HEATMAP_INPUT <- cli_value(
  "input",
  file.path(PROJECT_ROOT, "results", "anole_selection_strength_posterior_median_10m_fresh_2m.tsv")
)

FIG_WIDTH_IN <- 10.92
FIG_HEIGHT_IN <- 4.82
TREE_SCALE <- 1.04

muffle_postscript_font_warnings <- function(expr) {
  withCallingHandlers(
    expr,
    warning = function(warning_condition) {
      if (grepl("not found in PostScript font database", conditionMessage(warning_condition), fixed = TRUE)) {
        invokeRestart("muffleWarning")
      }
    }
  )
}

source_plot_script <- function(path) {
  env <- new.env(parent = globalenv())
  old_skip <- Sys.getenv("ANOLIS_SKIP_COMPONENT_MAIN", unset = NA_character_)
  Sys.setenv(ANOLIS_SKIP_COMPONENT_MAIN = "true")
  on.exit({
    if (is.na(old_skip)) {
      Sys.unsetenv("ANOLIS_SKIP_COMPONENT_MAIN")
    } else {
      Sys.setenv(ANOLIS_SKIP_COMPONENT_MAIN = old_skip)
    }
  }, add = TRUE)
  sys.source(path, envir = env)
  env
}

make_combined_plot <- function() {
  tree_env <- source_plot_script(file.path(SCRIPT_DIR, "plot_anole_phylogeny.R"))
  heat_env <- source_plot_script(file.path(SCRIPT_DIR, "plot_anole_drift_matrix_heatmap.R"))

  title_font <- tree_env$FIG_FONT
  title_pt <- tree_env$ANOLIS_TITLE_PT

  tree_panel <- tree_env$tree_plot +
    labs(title = NULL) +
    theme(
      plot.title = element_blank(),
      plot.margin = margin(0, 2, 2, 2, "pt")
    )

  heat_input <- if (nzchar(HEATMAP_INPUT)) HEATMAP_INPUT else heat_env$INPUT
  heat_panel <- heat_env$plot_drift_matrix(heat_env$read_drift_matrix(heat_input)) +
    labs(title = NULL) +
    theme(
      plot.title = element_blank(),
      plot.margin = margin(0, 8, 8, 8, "pt")
    )

  title_row <- cowplot::plot_grid(
    cowplot::ggdraw() + cowplot::draw_label(
      "Anolis phylogeny",
      fontfamily = title_font,
      fontface = "bold",
      size = title_pt,
      x = 0.5,
      y = 0.5,
      hjust = 0.5,
      vjust = 0.5
    ),
    cowplot::ggdraw() + cowplot::draw_label(
      "Anolis drift matrix",
      fontfamily = title_font,
      fontface = "bold",
      size = title_pt,
      x = 0.464,
      y = 0.5,
      hjust = 0.5,
      vjust = 0.5
    ),
    nrow = 1,
    rel_widths = c(tree_env$FIG_WIDTH_IN, heat_env$FIG_WIDTH_IN)
  )

  panel_row <- cowplot::plot_grid(
    tree_panel,
    heat_panel,
    nrow = 1,
    rel_widths = c(tree_env$FIG_WIDTH_IN, heat_env$FIG_WIDTH_IN),
    scale = c(TREE_SCALE, 1)
  )

  cowplot::plot_grid(
    title_row,
    panel_row,
    ncol = 1,
    rel_heights = c(0.12, 1)
  )
}

save_plot_pdf <- function(plot, path, width, height) {
  svg_path <- tempfile(fileext = ".svg")
  svglite::svglite(svg_path, width = width, height = height)
  on.exit({
    if (dev.cur() > 1L) dev.off()
    if (file.exists(svg_path)) unlink(svg_path)
  }, add = TRUE)
  suppressWarnings(print(plot))
  dev.off()
  rsvg::rsvg_pdf(svg_path, path)
}

main <- function() {
  dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
  combined_plot <- muffle_postscript_font_warnings(make_combined_plot())
  pdf_path <- file.path(OUTPUT_DIR, paste0(OUTPUT_NAME, ".pdf"))
  png_path <- file.path(OUTPUT_DIR, paste0(OUTPUT_NAME, ".png"))

  muffle_postscript_font_warnings(save_plot_pdf(combined_plot, pdf_path, FIG_WIDTH_IN, FIG_HEIGHT_IN))
  muffle_postscript_font_warnings(ggsave(
    filename = png_path,
    plot = combined_plot,
    width = FIG_WIDTH_IN,
    height = FIG_HEIGHT_IN,
    units = "in",
    dpi = 300,
    device = ragg::agg_png,
    bg = "white"
  ))

  message("Saved ", pdf_path)
  message("Saved ", png_path)
}

main()
