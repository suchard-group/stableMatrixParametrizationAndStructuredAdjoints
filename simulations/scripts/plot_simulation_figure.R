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

summary_data <- readr::read_csv(input, show_col_types = FALSE) |>
  mutate(
    x_value = as.numeric(.data$x_value),
    rmse = as.numeric(.data$rmse_median_completed),
    rmse_q1 = as.numeric(.data$rmse_q1_completed),
    rmse_q3 = as.numeric(.data$rmse_q3_completed),
    method = case_when(
      grepl("^orthogonal", .data$fit_family) ~ "Orthogonal H-SSBP",
      grepl("^generic", .data$fit_family) ~ "Dense-R H-SSBP",
      TRUE ~ .data$fit_family
    ),
    panel = factor(.data$panel, levels = c(
      "Real/complex boundary",
      "Non-orthogonal shear",
      "Jordan(4,1) coupling"
    )),
    method = factor(.data$method, levels = c("Orthogonal H-SSBP", "Dense-R H-SSBP"))
  )

bound_data <- readr::read_csv(bound_input, show_col_types = FALSE) |>
  filter(.data$panel == "Non-orthogonal shear") |>
  transmute(x_value = as.numeric(.data$x_value), rmse = as.numeric(.data$bound_rmse_scale))

method_colors <- c("Orthogonal H-SSBP" = "#D4960A", "Dense-R H-SSBP" = "#2F67B1")
method_shapes <- c("Orthogonal H-SSBP" = 16, "Dense-R H-SSBP" = 15)
rmse_limits <- c(0, max(summary_data$rmse_q3, summary_data$rmse, na.rm = TRUE) * 1.10)

axis_labels <- function(x) formatC(x, format = "fg", digits = 3)
boundary_labels <- function(x) {
  labels <- axis_labels(x)
  labels[abs(x - 1) < sqrt(.Machine$double.eps)] <- ""
  labels
}

theme_paper <- function() {
  theme_classic(base_size = 9, base_family = "serif") +
    theme(
      axis.line = element_line(linewidth = 0.3),
      axis.ticks = element_line(linewidth = 0.3),
      axis.text = element_text(color = "black"),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.35),
      panel.grid.major.y = element_line(color = "grey93", linetype = "33", linewidth = 0.4),
      panel.grid.major.x = element_blank(),
      plot.title = element_text(face = "bold", hjust = 0.5),
      legend.position = "none"
    )
}

rmse_panel <- function(data, title, x_lab, x_breaks, x_labels = axis_labels,
                       show_y = TRUE, show_legend = FALSE, vline = NULL,
                       bound = NULL) {
  p <- ggplot(data, aes(
    x_value, rmse,
    colour = method, fill = method, shape = method, group = method
  ))
  if (!is.null(bound)) {
    p <- p + geom_line(
      data = bound, aes(x_value, rmse),
      inherit.aes = FALSE,
      colour = method_colors[["Orthogonal H-SSBP"]],
      alpha = 0.5,
      linetype = "42",
      linewidth = 0.55
    )
  }
  p <- p +
    geom_ribbon(aes(ymin = rmse_q1, ymax = rmse_q3), alpha = 0.15, colour = NA) +
    geom_line(linewidth = 0.85) +
    geom_point(size = 1.65, stroke = 0) +
    scale_x_continuous(breaks = x_breaks, labels = x_labels, expand = expansion(mult = c(0.05, 0.10))) +
    scale_y_continuous(limits = rmse_limits, labels = label_number(accuracy = 0.1), expand = expansion(mult = c(0, 0.02))) +
    scale_colour_manual(values = method_colors, name = NULL) +
    scale_fill_manual(values = method_colors, name = NULL) +
    scale_shape_manual(values = method_shapes, name = NULL) +
    labs(x = x_lab, y = if (show_y) "Drift RMSE" else NULL, title = title) +
    theme_paper() +
    theme(
      axis.text.y = if (show_y) element_text(color = "black") else element_blank(),
      axis.ticks.y = if (show_y) element_line(linewidth = 0.3) else element_blank(),
      legend.position = if (show_legend) "inside" else "none",
      legend.position.inside = c(0.03, 0.97),
      legend.justification = c(0, 1),
      legend.background = element_rect(fill = "white", colour = "black", linewidth = 0.3)
    ) +
    guides(fill = "none")
  if (!is.null(vline)) {
    p <- p + geom_vline(xintercept = vline, color = "grey45", linetype = "22", linewidth = 0.35)
  }
  if (!is.null(bound)) {
    p <- p + annotate(
      "text",
      x = 7,
      y = 0.59,
      label = "Certified lower bound\nfor orthog. H-SSBP",
      hjust = 0.5,
      vjust = 0,
      lineheight = 0.9,
      size = 2.7,
      family = "serif",
      colour = "black"
    )
  }
  p
}

panels <- split(summary_data, summary_data$panel)
figure <- rmse_panel(
  panels[["Real/complex boundary"]],
  "Real/complex boundary",
  expression(omega / omega^"*"),
  sort(unique(panels[["Real/complex boundary"]]$x_value)),
  boundary_labels,
  show_y = TRUE,
  show_legend = TRUE,
  vline = 1
) +
  rmse_panel(
    panels[["Non-orthogonal shear"]],
    "Non-orthogonal shear",
    "Shear multiplier",
    sort(unique(panels[["Non-orthogonal shear"]]$x_value)),
    show_y = FALSE,
    bound = bound_data
  ) +
  rmse_panel(
    panels[["Jordan(4,1) coupling"]],
    "Jordan coupling",
    "Jordan coupling",
    sort(unique(panels[["Jordan(4,1) coupling"]]$x_value)),
    show_y = FALSE
  ) +
  plot_layout(widths = c(1, 1, 1))

dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
ggsave(output, figure, width = 13, height = 4.0, units = "in", dpi = 300)
message("wrote ", output)
