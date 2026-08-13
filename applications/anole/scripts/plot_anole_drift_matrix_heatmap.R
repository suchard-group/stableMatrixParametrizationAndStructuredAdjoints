#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(ragg)
  library(rsvg)
  library(scales)
  library(showtext)
  library(sysfonts)
  library(svglite)
})

script_path <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg) > 0) {
    return(normalizePath(sub("^--file=", "", file_arg[[1]]), mustWork = TRUE))
  }
  normalizePath("plot_anole_drift_matrix_heatmap.R", mustWork = FALSE)
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
INPUT <- cli_value("input", file.path(PROJECT_ROOT, "results", "anole_selection_strength_posterior_median_10m_fresh_2m.tsv"))
OUTPUT_DIR <- cli_value("output-dir", file.path(PROJECT_ROOT, "output"))
OUTPUT_NAME <- cli_value("output-name", "anole_drift_matrix_heatmap_full_10m_fresh_2m")

TRAIT_KEYS <- c("SVL", "HL", "HLL", "FLL", "LAM", "TL")
TRAIT_LABELS <- c(
  SVL = "Snout-vent\nlength",
  HL = "Head\nlength",
  HLL = "Hindlimb\nlength",
  FLL = "Forelimb\nlength",
  LAM = "Lamella\nnumber",
  TL = "Tail\nlength"
)

FIG_WIDTH_IN <- 5.52
FIG_HEIGHT_IN <- 4.82
TEXT_WIDTH_IN <- 6.5
MINIPAGE_WIDTH <- 0.48
ANOLIS_SCALE_DOWN <- MINIPAGE_WIDTH * TEXT_WIDTH_IN / FIG_WIDTH_IN
ANOLIS_FONT_SCALE <- (12 / ANOLIS_SCALE_DOWN) / 9
ANOLIS_TITLE_PT <- 8 * ANOLIS_FONT_SCALE
ANOLIS_AXIS_PT <- ANOLIS_TITLE_PT
ANOLIS_TICK_PT <- 4.65 * ANOLIS_FONT_SCALE
ANOLIS_CELL_PT <- 10.0
ANOLIS_COLORBAR_PT <- 6.2 * ANOLIS_FONT_SCALE

find_first_font <- function(paths) {
  paths <- path.expand(paths)
  existing <- paths[file.exists(paths)]
  if (length(existing) > 0) existing[[1]] else NA_character_
}

configure_fonts <- function() {
  cmu_font_candidates <- list(
    regular = c("~/Library/Fonts/cmunrm.otf", "/usr/share/fonts/opentype/cmu/cmunrm.otf", "/usr/share/fonts/truetype/cmu/cmunrm.otf"),
    italic = c("~/Library/Fonts/cmunti.otf", "/usr/share/fonts/opentype/cmu/cmunti.otf", "/usr/share/fonts/truetype/cmu/cmunti.otf"),
    bold = c("~/Library/Fonts/cmunbx.otf", "/usr/share/fonts/opentype/cmu/cmunbx.otf", "/usr/share/fonts/truetype/cmu/cmunbx.otf"),
    bolditalic = c("~/Library/Fonts/cmunbi.otf", "/usr/share/fonts/opentype/cmu/cmunbi.otf", "/usr/share/fonts/truetype/cmu/cmunbi.otf")
  )
  cmu_paths <- lapply(cmu_font_candidates, find_first_font)
  if (!is.na(cmu_paths$regular)) {
    sysfonts::font_add(
      "CMU Serif",
      regular = cmu_paths$regular,
      italic = cmu_paths$italic,
      bold = cmu_paths$bold,
      bolditalic = cmu_paths$bolditalic
    )
  }
  showtext::showtext_auto(enable = TRUE)
  if ("CMU Serif" %in% sysfonts::font_families()) "CMU Serif" else "serif"
}

FIG_FONT <- configure_fonts()

read_drift_matrix <- function(path) {
  selection_strength <- as.matrix(read.delim(path, check.names = FALSE))
  storage.mode(selection_strength) <- "double"
  -selection_strength
}

plot_drift_matrix <- function(drift_matrix) {
  if (!identical(colnames(drift_matrix), TRAIT_KEYS)) {
    warning("Unexpected trait columns in ", INPUT, call. = FALSE)
  }
  vmax <- max(1.0, max(abs(drift_matrix), na.rm = TRUE))
  row_count <- nrow(drift_matrix)
  col_count <- ncol(drift_matrix)
  plot_data <- expand.grid(
    row_index = seq_len(row_count),
    col_index = seq_len(col_count)
  )
  plot_data$value <- drift_matrix[cbind(plot_data$row_index, plot_data$col_index)]
  plot_data$target <- factor(
    TRAIT_LABELS[TRAIT_KEYS[plot_data$row_index]],
    levels = rev(unname(TRAIT_LABELS[TRAIT_KEYS]))
  )
  plot_data$source <- factor(
    TRAIT_LABELS[TRAIT_KEYS[plot_data$col_index]],
    levels = unname(TRAIT_LABELS[TRAIT_KEYS])
  )
  plot_data$label <- sprintf("%.2f", plot_data$value)
  plot_data$text_color <- ifelse(abs(plot_data$value) > 0.5, "white", "black")

  ggplot(plot_data, aes(.data$source, .data$target, fill = .data$value)) +
    geom_tile(color = "white", linewidth = 0.45) +
    geom_text(
      aes(label = .data$label, colour = .data$text_color),
      size = ANOLIS_CELL_PT / ggplot2::.pt,
      family = FIG_FONT,
      show.legend = FALSE
    ) +
    coord_fixed(clip = "off") +
    scale_colour_identity() +
    scale_x_discrete(expand = expansion(add = 0)) +
    scale_y_discrete(expand = expansion(add = 0)) +
    scale_fill_gradient2(
      name = "Posterior median",
      low = "#3B4CC0",
      mid = "white",
      high = "#B40426",
      midpoint = 0,
      limits = c(-vmax, vmax),
      oob = scales::squish,
      guide = guide_colourbar(
        title.position = "right",
        title.theme = element_text(
          angle = 90,
          size = ANOLIS_COLORBAR_PT,
          family = FIG_FONT,
          colour = "black",
          hjust = 0.5,
          vjust = 0.5,
          margin = margin(0, 0, 0, 8, "pt")
        ),
        label.theme = element_text(size = ANOLIS_COLORBAR_PT, family = FIG_FONT, colour = "black"),
        barheight = grid::unit(2.66, "inches"),
        barwidth = grid::unit(0.16, "inches")
      )
    ) +
    labs(
      title = "Anolis drift matrix",
      x = "Source trait",
      y = "Target trait"
    ) +
    theme_classic(base_size = ANOLIS_AXIS_PT, base_family = FIG_FONT) +
    theme(
      axis.line = element_blank(),
      axis.ticks = element_blank(),
      axis.text.x = element_text(
        size = ANOLIS_TICK_PT,
        family = FIG_FONT,
        colour = "black",
        angle = 45,
        hjust = 1,
        vjust = 1,
        lineheight = 0.78
      ),
      axis.text.y = element_text(size = ANOLIS_TICK_PT, family = FIG_FONT, colour = "black", lineheight = 0.78),
      axis.title.x = element_text(size = ANOLIS_AXIS_PT, family = FIG_FONT, colour = "black", margin = margin(t = -4, unit = "pt")),
      axis.title.y = element_text(size = ANOLIS_AXIS_PT, family = FIG_FONT, colour = "black", margin = margin(r = 4, unit = "pt")),
      legend.position = "right",
      legend.box.margin = margin(t = 0, r = 0, b = 0, l = 0, unit = "pt"),
      legend.title = element_text(size = ANOLIS_COLORBAR_PT, family = FIG_FONT, colour = "black"),
      legend.text = element_text(size = ANOLIS_COLORBAR_PT, family = FIG_FONT, colour = "black"),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.35),
      panel.background = element_rect(fill = "white", colour = NA),
      panel.grid = element_blank(),
      plot.background = element_rect(fill = "white", colour = NA),
      plot.title = element_text(face = "bold", hjust = 0.5, family = FIG_FONT, size = ANOLIS_TITLE_PT),
      plot.margin = margin(-12, 8, 8, 8, "pt")
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
  drift_matrix <- read_drift_matrix(INPUT)
  plot <- plot_drift_matrix(drift_matrix)
  dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
  pdf_path <- file.path(OUTPUT_DIR, paste0(OUTPUT_NAME, ".pdf"))
  png_path <- file.path(OUTPUT_DIR, paste0(OUTPUT_NAME, ".png"))
  save_plot_pdf(plot, pdf_path, FIG_WIDTH_IN, FIG_HEIGHT_IN)
  ggsave(
    filename = png_path,
    plot = plot,
    width = FIG_WIDTH_IN,
    height = FIG_HEIGHT_IN,
    units = "in",
    dpi = 300,
    device = ragg::agg_png,
    bg = "white"
  )
  message("Saved ", pdf_path)
  message("Saved ", png_path)
}

if (!identical(Sys.getenv("ANOLIS_SKIP_COMPONENT_MAIN"), "true")) {
  main()
}
