#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE)

required <- c("dplyr", "ggplot2", "patchwork", "readr", "scales")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0L) {
  stop("Install missing R packages: ", paste(missing, collapse = ", "), call. = FALSE)
}

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(patchwork)
  library(readr)
  library(scales)
})

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(prefix, default) {
  hit <- args[startsWith(args, prefix)]
  if (length(hit) == 0L) default else sub(prefix, "", hit[[1L]], fixed = TRUE)
}

input <- get_arg("--input=", "simulations/results/processed/summary_for_plot_best_likelihood.csv")
bound_input <- get_arg("--bound-input=", "simulations/results/processed/shear_certified_bound_rmse_scale.csv")
output <- get_arg("--output=", "simulations/figures/output/supp_simulations_combined_rmse_final_battery_lambda0p1_r25_best_likelihood.pdf")
orthogonal_label <- get_arg("--orthogonal-label=", "Orthogonal H-SSBP")
dense_label <- get_arg("--dense-label=", "Dense-R H-SSBP")

find_first_font <- function(paths) {
  existing <- paths[file.exists(paths)]
  if (length(existing) > 0L) existing[[1L]] else NA_character_
}

font_family <- "serif"
if (requireNamespace("showtext", quietly = TRUE) && requireNamespace("sysfonts", quietly = TRUE)) {
  cmu_paths <- list(
    regular = find_first_font(c(
      file.path(path.expand("~"), "Library", "Fonts", "cmunrm.otf"),
      "/usr/share/fonts/opentype/cmu/cmunrm.otf",
      "/usr/share/fonts/truetype/cmu/cmunrm.otf"
    )),
    italic = find_first_font(c(
      file.path(path.expand("~"), "Library", "Fonts", "cmunti.otf"),
      "/usr/share/fonts/opentype/cmu/cmunti.otf",
      "/usr/share/fonts/truetype/cmu/cmunti.otf"
    )),
    bold = find_first_font(c(
      file.path(path.expand("~"), "Library", "Fonts", "cmunbx.otf"),
      "/usr/share/fonts/opentype/cmu/cmunbx.otf",
      "/usr/share/fonts/truetype/cmu/cmunbx.otf"
    )),
    bolditalic = find_first_font(c(
      file.path(path.expand("~"), "Library", "Fonts", "cmunbi.otf"),
      "/usr/share/fonts/opentype/cmu/cmunbi.otf",
      "/usr/share/fonts/truetype/cmu/cmunbi.otf"
    ))
  )
  if (!is.na(cmu_paths$regular)) {
    sysfonts::font_add(
      "CMU Serif",
      regular = cmu_paths$regular,
      italic = cmu_paths$italic,
      bold = cmu_paths$bold,
      bolditalic = cmu_paths$bolditalic
    )
    font_family <- "CMU Serif"
  }
  showtext::showtext_auto(enable = TRUE)
}

summary_data <- readr::read_csv(input, show_col_types = FALSE) |>
  mutate(
    x_value = as.numeric(.data$x_value),
    rmse = as.numeric(.data$rmse_median_completed),
    rmse_q1 = as.numeric(.data$rmse_q1_completed),
    rmse_q3 = as.numeric(.data$rmse_q3_completed),
    completed_rate = as.numeric(.data$completion_rate),
    method = case_when(
      grepl("^orthogonal", .data$fit_family) ~ orthogonal_label,
      grepl("^generic", .data$fit_family) ~ dense_label,
      TRUE ~ .data$fit_family
    ),
    panel = factor(.data$panel, levels = c(
      "Real/complex boundary",
      "Non-orthogonal shear",
      "Jordan(4,1) coupling"
    )),
    method = factor(.data$method, levels = c(orthogonal_label, dense_label))
  ) |>
  filter(!is.na(.data$rmse))

bound_data <- NULL
if (file.exists(bound_input)) {
  bound_data <- readr::read_csv(bound_input, show_col_types = FALSE) |>
    filter(.data$panel == "Non-orthogonal shear") |>
    transmute(x_value = as.numeric(.data$x_value), rmse = as.numeric(.data$bound_rmse_scale)) |>
    arrange(.data$x_value)
}

method_colors <- setNames(c("#D4960A", "#2F67B1"), c(orthogonal_label, dense_label))
method_linetypes <- setNames(c("solid", "solid"), c(orthogonal_label, dense_label))
method_linewidths <- setNames(c(0.85, 0.85), c(orthogonal_label, dense_label))
method_shapes <- setNames(c(16L, 15L), c(orthogonal_label, dense_label))
rmse_limits <- c(0, max(summary_data$rmse_q3, summary_data$rmse, na.rm = TRUE) * 1.10)

axis_labels <- function(x) {
  vapply(x, function(value) formatC(value, format = "fg", digits = 3), character(1))
}

boundary_labels <- function(x) {
  labels <- axis_labels(x)
  labels[abs(x - 1) < sqrt(.Machine$double.eps)] <- ""
  labels
}

figure_width <- 13
focused_scale_down <- 0.98 * 6.5 / figure_width
focused_font_scale <- (12 / focused_scale_down) / 9
focused_title_pt <- 8 * focused_font_scale
focused_text_scale <- focused_title_pt / 9
focused_plot_title_pt <- focused_title_pt - 1 / focused_scale_down

theme_paper <- function(base_size = 9) {
  theme_classic(base_size = base_size, base_family = "serif") +
    theme(
      axis.line = element_line(linewidth = 0.3),
      axis.ticks = element_line(linewidth = 0.3),
      axis.text = element_text(color = "black"),
      legend.position = "bottom",
      legend.title = element_text(color = "black"),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.35),
      strip.background = element_rect(fill = "grey93", color = "black", linewidth = 0.3),
      strip.text = element_text(face = "bold"),
      plot.title = element_text(face = "bold", hjust = 0),
      plot.subtitle = element_text(color = "grey25")
    )
}

panel_theme <- theme_paper(base_size = focused_title_pt) +
  theme(
    legend.position = "none",
    panel.grid.major.y = element_line(color = "grey93", linetype = "33", linewidth = 0.4),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    plot.title = element_text(hjust = 0.5, size = focused_plot_title_pt, face = "bold"),
    axis.title = element_text(size = focused_title_pt),
    plot.margin = margin(2, 4, 2, 4, "pt")
  )

rmse_panel <- function(data, title, x_lab, x_breaks, x_labels = axis_labels,
                       show_y = TRUE, show_legend = FALSE, vline = NULL,
                       x_text_angle = 0, bound = NULL) {
  p <- ggplot(
    data,
    aes(
      x_value, rmse,
      colour = method,
      fill = method,
      linetype = method,
      linewidth = method,
      shape = method,
      group = method
    )
  )

  if (!is.null(bound) && nrow(bound) > 0L) {
    p <- p +
      geom_line(
        data = bound,
        aes(x_value, rmse),
        inherit.aes = FALSE,
        colour = method_colors[[orthogonal_label]],
        alpha = 0.5,
        linetype = "42",
        linewidth = 0.55
      )
  }

  p <- p +
    geom_ribbon(aes(ymin = rmse_q1, ymax = rmse_q3), alpha = 0.15, colour = NA, linewidth = 0) +
    geom_line() +
    geom_point(size = 1.65, stroke = 0) +
    scale_x_continuous(breaks = x_breaks, labels = x_labels, expand = expansion(mult = c(0.05, 0.10))) +
    scale_y_continuous(limits = rmse_limits, labels = label_number(accuracy = 0.1), expand = expansion(mult = c(0, 0.02))) +
    scale_colour_manual(values = method_colors, name = NULL) +
    scale_fill_manual(values = method_colors, name = NULL) +
    scale_linetype_manual(values = method_linetypes, name = NULL) +
    scale_linewidth_manual(values = method_linewidths, name = NULL) +
    scale_shape_manual(values = method_shapes, name = NULL) +
    labs(x = x_lab, y = NULL, title = title) +
    panel_theme +
    theme(
      axis.text.y = if (show_y) element_text(color = "black") else element_blank(),
      axis.text.x = element_text(
        size = focused_title_pt * 0.68,
        angle = x_text_angle,
        hjust = if (x_text_angle == 0) 0.5 else 1,
        vjust = if (x_text_angle == 0) 0.5 else 1
      ),
      axis.ticks.y = if (show_y) element_line(linewidth = 0.3) else element_blank(),
      legend.position = if (show_legend) "inside" else "none",
      legend.position.inside = if (show_legend) c(0.03, 0.97) else c(0.5, 0.5),
      legend.justification = c(0, 1),
      legend.direction = "vertical",
      legend.background = element_rect(fill = "white", colour = "black", linewidth = 0.3),
      legend.margin = margin(3, 5, 3, 4, "pt"),
      legend.key.size = grid::unit(10, "pt"),
      legend.key.width = grid::unit(18, "pt"),
      legend.title = element_text(size = 6.2 * focused_font_scale),
      legend.text = element_text(size = 6.2 * focused_font_scale)
    ) +
    guides(
      colour = guide_legend(
        override.aes = list(
          linewidth = unname(method_linewidths),
          size = 1.55,
          alpha = 1,
          shape = unname(method_shapes)
        )
      ),
      fill = "none",
      linetype = "none",
      linewidth = "none",
      shape = "none"
    )

  if (!is.null(vline)) {
    p <- p + geom_vline(xintercept = vline, color = "grey45", linetype = "22", linewidth = 0.35)
  }

  if (!is.null(bound) && nrow(bound) > 0L) {
    p <- p +
      annotate(
        "text",
        x = 7,
        y = 0.59,
        label = "Certified lower bound\nfor orthog. H-SSBP",
        hjust = 0.5,
        vjust = 0,
        lineheight = 0.9,
        size = 1.5 * focused_text_scale,
        family = font_family,
        colour = "black"
      )
  }

  p
}

shared_rmse_label_panel <- function(label = "Drift RMSE") {
  ggplot() +
    annotate(
      "text",
      x = 0,
      y = 0,
      label = label,
      angle = 90,
      hjust = 0.5,
      vjust = 0.5,
      size = 3.2 * focused_text_scale,
      family = font_family,
      colour = "black"
    ) +
    scale_x_continuous(limits = c(-1, 1)) +
    scale_y_continuous(limits = c(-1, 1)) +
    coord_cartesian(clip = "off") +
    theme_void() +
    theme(plot.margin = margin(1, 2, 1, 10, "pt"))
}

panels <- split(summary_data, summary_data$panel)
figure <- wrap_plots(
  shared_rmse_label_panel(),
  rmse_panel(
    panels[["Real/complex boundary"]],
    "Real/complex boundary",
    expression(omega[1] / omega[1]^"*"),
    sort(unique(panels[["Real/complex boundary"]]$x_value)),
    boundary_labels,
    show_y = TRUE,
    show_legend = TRUE,
    vline = 1,
    x_text_angle = 45
  ),
  rmse_panel(
    panels[["Non-orthogonal shear"]],
    "Non-orthogonal shear",
    "Shear multiplier",
    sort(unique(panels[["Non-orthogonal shear"]]$x_value)),
    show_y = FALSE,
    bound = bound_data
  ),
  rmse_panel(
    panels[["Jordan(4,1) coupling"]],
    "Jordan coupling",
    "Jordan coupling",
    sort(unique(panels[["Jordan(4,1) coupling"]]$x_value)),
    show_y = FALSE
  ),
  design = "ABCD",
  widths = c(0.11, 1, 1, 1)
)

dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
ggsave(output, figure, width = figure_width, height = 4.0, units = "in", dpi = 300)
message("wrote ", output)
