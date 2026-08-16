#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(cowplot)
  library(dplyr)
  library(ggplot2)
  library(jsonlite)
  library(readr)
  library(ragg)
  library(rsvg)
  library(scales)
  library(showtext)
  library(sysfonts)
  library(svglite)
  library(tidyr)
})

script_path <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg) > 0) {
    return(normalizePath(sub("^--file=", "", file_arg[[1]]), mustWork = TRUE))
  }
  normalizePath("plot_bitmex_trade_drifts.R", mustWork = FALSE)
}

SCRIPT_DIR <- dirname(script_path())
PROJECT_ROOT <- normalizePath(file.path(SCRIPT_DIR, ".."), mustWork = TRUE)
FIGURES <- file.path(PROJECT_ROOT, "output")
DIMENSION <- 16L

BASE_COLORS <- c(
  XBT = "#2F67B1",
  ETH = "#D4960A",
  SOL = "#228B22",
  DOGE = "#D67C2F",
  DOT = "#9B4356",
  UNI = "#6F7B82",
  LTC = "#228B22",
  BCH = "#9B4356",
  XRP = "#B85E72",
  SPCX = "#E0A75A",
  HYPE = "#86A9CF",
  SUI = "#5E9D5E",
  ADA = "#C66F3F",
  FIL = "#7E8A90",
  BNB = "#B7892C"
)
FALLBACK_COLORS <- c(
  "#2F67B1", "#228B22", "#D4960A", "#9B4356", "#D67C2F", "#6F7B82",
  "#86A9CF", "#5E9D5E", "#B7892C", "#B85E72", "#E0A75A", "#7E8A90"
)

# Match the computational-complexity figures' wide-figure font scaling.  The
# native BitMEX row figure is 12.5in wide and is included at \textwidth in the
# manuscript, so raw ggplot text has to be larger than its printed size.
FIG_WIDTH_IN <- 12.5
FIG_HEIGHT_IN <- 4.25
TEXT_WIDTH_IN <- 6.5
BITMEX_SCALE_DOWN <- TEXT_WIDTH_IN / FIG_WIDTH_IN
BITMEX_FONT_SCALE <- (12 / BITMEX_SCALE_DOWN) / 9
BITMEX_TITLE_PT <- 8 * BITMEX_FONT_SCALE
BITMEX_AXIS_PT <- BITMEX_TITLE_PT
BITMEX_LEGEND_PT <- 6.2 * BITMEX_FONT_SCALE
BITMEX_INSIDE_LEGEND_PT <- 8.3
BITMEX_HEATMAP_ENTRY_PT <- 8.0
BITMEX_HEATMAP_CELL_PT <- 10.0
BITMEX_COLORBAR_TEXT_PT <- 6.2 * BITMEX_FONT_SCALE
PAPER_ROW_LAYOUTS <- c("paper-row", "paper-row-wide-timeseries")

paper_row_rel_widths <- function(layout) {
  if (layout == "paper-row") {
    return(c(1.0, 1.0))
  }
  if (layout == "paper-row-wide-timeseries") {
    return(c(2.35, 1.0))
  }
  stop("Unknown paper-row layout: ", layout, call. = FALSE)
}

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

paper_theme <- function(base_size = BITMEX_AXIS_PT) {
  theme_classic(base_size = base_size, base_family = FIG_FONT) +
    theme(
      axis.line = element_line(linewidth = 0.3),
      axis.ticks = element_line(linewidth = 0.3),
      axis.text = element_text(color = "black", family = FIG_FONT),
      axis.title = element_text(color = "black", family = FIG_FONT, size = BITMEX_AXIS_PT),
      legend.position = "bottom",
      legend.title = element_blank(),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.35),
      panel.background = element_rect(fill = "white", colour = NA),
      panel.grid.major = element_line(color = "grey93", linetype = "33", linewidth = 0.4),
      panel.grid.minor = element_blank(),
      plot.background = element_rect(fill = "white", colour = NA),
      plot.title = element_text(face = "bold", hjust = 0.5, family = FIG_FONT, size = BITMEX_TITLE_PT),
      plot.margin = margin(2, 4, 2, 4, "pt")
    )
}

parse_args <- function(argv = commandArgs(trailingOnly = TRUE)) {
  out <- list(
    window = "jan2022_J20_trade_7h",
    layout = "paper-row",
    burnin_fraction = 0.10,
    invertible_log = "",
    events = "data/jan2022_J20_trade_7h_context/events.csv",
    metadata = "data/jan2022_J20_trade_7h_context/metadata.json",
    matrix_csv = "results/compact/jan2022_J20_trade_7h/matrices/jan2022_J20_trade_7h_lambda0p1_random02_manuscript_drift_matrix.csv",
    matrix_kind = "drift",
    show_diagonal = TRUE,
    out_prefix = "bitmex_jan2022_j20_trade_7h_cartesian_map_row"
  )

  i <- 1L
  while (i <= length(argv)) {
    arg <- argv[[i]]
    if (grepl("^--[^=]+=", arg)) {
      pieces <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1]]
      key <- pieces[[1]]
      value <- paste(pieces[-1], collapse = "=")
    } else if (startsWith(arg, "--")) {
      key <- sub("^--", "", arg)
      if (i == length(argv)) stop("Missing value for --", key, call. = FALSE)
      i <- i + 1L
      value <- argv[[i]]
    } else {
      stop("Unexpected argument: ", arg, call. = FALSE)
    }

    if (key == "window") {
      out$window <- value
    } else if (key == "layout") {
      out$layout <- value
    } else if (key == "burnin-fraction") {
      out$burnin_fraction <- as.numeric(value)
    } else if (key == "invertible-log") {
      out$invertible_log <- value
    } else if (key == "events") {
      out$events <- value
    } else if (key == "metadata") {
      out$metadata <- value
    } else if (key == "matrix-csv") {
      out$matrix_csv <- value
    } else if (key == "matrix-kind") {
      out$matrix_kind <- value
    } else if (key == "show-diagonal") {
      out$show_diagonal <- tolower(value) %in% c("1", "true", "yes", "y")
    } else if (key == "out-prefix") {
      out$out_prefix <- value
    } else {
      stop("Unknown argument: --", key, call. = FALSE)
    }
    i <- i + 1L
  }

  if (!out$window %in% c("30", "60", "jan2022_J2_4h", "jan2022_J20_trade_7h") && !nzchar(out$events)) {
    stop("--window must be 30, 60, jan2022_J2_4h, or jan2022_J20_trade_7h unless --events is provided", call. = FALSE)
  }
  if (!out$layout %in% PAPER_ROW_LAYOUTS) {
    stop(
      "The R plotting script supports --layout ",
      paste(PAPER_ROW_LAYOUTS, collapse = " or "),
      ".",
      call. = FALSE
    )
  }
  if (!is.finite(out$burnin_fraction) || out$burnin_fraction < 0 || out$burnin_fraction >= 1) {
    stop("--burnin-fraction must be in [0, 1)", call. = FALSE)
  }
  if (!out$matrix_kind %in% c("selection", "drift")) {
    stop("--matrix-kind must be selection or drift", call. = FALSE)
  }
  out
}

default_paths <- function(window) {
  if (window == "jan2022_J20_trade_7h") {
    return(list(
      events = file.path(PROJECT_ROOT, "data", "jan2022_J20_trade_7h_context", "events.csv"),
      metadata = file.path(PROJECT_ROOT, "data", "jan2022_J20_trade_7h_context", "metadata.json"),
      orthogonal_log = ""
    ))
  }
  if (window == "jan2022_J2_4h") {
    return(list(
      events = file.path(PROJECT_ROOT, "data", "jan2022_day_context", "events.csv"),
      metadata = file.path(PROJECT_ROOT, "data", "jan2022_day_context", "metadata.json"),
      orthogonal_log = ""
    ))
  }
  run <- paste0("bitmex_orthogonal_", window, "min_shared_top16")
  list(
    events = file.path(PROJECT_ROOT, "runs", run, "inputs", paste0("bitmex_real_", window, "min_shared_top16_events.csv")),
    metadata = file.path(PROJECT_ROOT, "runs", run, "inputs", paste0("bitmex_real_", window, "min_shared_top16_metadata.json")),
    orthogonal_log = file.path(PROJECT_ROOT, "results", "raw", run, "output", "job_001", paste0(run, ".log"))
  )
}

absolute_path <- function(path) {
  grepl("^/", path) || grepl("^[A-Za-z]:", path)
}

resolve_input_path <- function(path) {
  if (!nzchar(path)) return("")
  if (absolute_path(path)) return(normalizePath(path, mustWork = TRUE))
  if (file.exists(path)) return(normalizePath(path, mustWork = TRUE))
  normalizePath(file.path(PROJECT_ROOT, path), mustWork = TRUE)
}

instrument_base <- function(symbol) {
  collapsed <- gsub("_", "", symbol)
  sub("(USDT|USD)$", "", collapsed)
}

quote_variant <- function(symbol) {
  ifelse(
    grepl("_USDT$", symbol), "_USDT",
    ifelse(grepl("USDT$", symbol), "USDT", ifelse(grepl("USD$", symbol), "USD", "other"))
  )
}

repeated_bases <- function(symbols) {
  counts <- table(instrument_base(symbols))
  names(counts)[counts > 1]
}

line_colors <- function(symbols) {
  bases <- instrument_base(symbols)
  ordered_bases <- unique(bases)
  out <- vapply(seq_along(symbols), function(index) {
    base <- bases[[index]]
    if (base %in% names(BASE_COLORS)) {
      BASE_COLORS[[base]]
    } else {
      FALLBACK_COLORS[[((match(base, ordered_bases) - 1L) %% length(FALLBACK_COLORS)) + 1L]]
    }
  }, character(1))
  names(out) <- symbols
  out
}

line_types <- function(symbols) {
  out <- rep("solid", length(symbols))
  names(out) <- symbols
  out
}

read_events <- function(path, symbols) {
  raw <- readr::read_csv(path, show_col_types = FALSE)
  time_column <- if ("time_minutes" %in% names(raw)) {
    "time_minutes"
  } else if ("time_hours" %in% names(raw)) {
    "time_hours"
  } else {
    stop("Event CSV must contain time_minutes or time_hours: ", path, call. = FALSE)
  }
  events <- raw |>
    filter(.data$symbol %in% symbols) |>
    mutate(
      symbol = factor(.data$symbol, levels = symbols),
      time_display = .data[[time_column]]
    ) |>
    arrange(.data$instrument_index, .data[[time_column]])
  attr(events, "time_axis_label") <- if (time_column == "time_hours") {
    "Hours from window start"
  } else {
    "Minutes from window start"
  }
  events
}

read_beast_log <- function(path) {
  raw <- readLines(path, warn = FALSE)
  usable <- raw[nzchar(trimws(raw)) & !startsWith(raw, "#")]
  header_index <- which(grepl("^state\\t", usable))[1]
  if (is.na(header_index)) {
    stop("No BEAST trace header found in ", path, call. = FALSE)
  }
  header <- strsplit(usable[[header_index]], "\t", fixed = TRUE)[[1]]
  data_lines <- usable[(header_index + 1L):length(usable)]
  data_lines <- data_lines[grepl("^-?[0-9]+(\\t|$)", data_lines)]
  if (length(data_lines) == 0L) {
    stop("No BEAST trace rows found in ", path, call. = FALSE)
  }

  rows <- strsplit(data_lines, "\t", fixed = TRUE)
  full_rows <- rows[lengths(rows) >= length(header)]
  values <- unlist(lapply(full_rows, function(row) row[seq_along(header)]), use.names = FALSE)
  table <- matrix(as.numeric(values), ncol = length(header), byrow = TRUE)
  list(header = header, table = table)
}

vector_from_columns <- function(header, table, prefix, count) {
  wanted <- paste0(prefix, seq_len(count))
  indices <- match(wanted, header)
  if (anyNA(indices)) {
    stop("Missing log columns: ", paste(wanted[is.na(indices)], collapse = ", "), call. = FALSE)
  }
  table[, indices, drop = FALSE]
}

givens_pairs <- function(dim) {
  out <- vector("list", dim * (dim - 1L) / 2L)
  index <- 1L
  for (i in seq_len(dim - 1L)) {
    for (j in seq.int(i + 1L, dim)) {
      out[[index]] <- c(i, j)
      index <- index + 1L
    }
  }
  out
}

givens_rotation <- function(dim, angles) {
  out <- diag(dim)
  pairs <- givens_pairs(dim)
  for (index in seq_along(pairs)) {
    angle <- angles[[index]]
    pair <- pairs[[index]]
    i <- pair[[1]]
    j <- pair[[2]]
    cval <- cos(angle)
    sval <- sin(angle)
    old_i <- out[, i]
    old_j <- out[, j]
    out[, i] <- old_i * cval + old_j * sval
    out[, j] <- -old_i * sval + old_j * cval
  }
  out
}

block_matrix <- function(rho, theta, t_values) {
  dim <- 2L * length(rho)
  out <- matrix(0, nrow = dim, ncol = dim)
  for (block in seq_along(rho)) {
    row <- 2L * (block - 1L) + 1L
    cval <- cos(theta[[block]])
    sval <- sin(theta[[block]])
    out[row:(row + 1L), row:(row + 1L)] <- matrix(
      c(
        rho[[block]] * cval, rho[[block]] * sval - t_values[[block]],
        rho[[block]] * sval + t_values[[block]], rho[[block]] * cval
      ),
      nrow = 2L,
      byrow = TRUE
    )
  }
  out
}

dense_rotation_sample <- function(header, row) {
  col_indices <- grep("^ou\\.A\\.rotation\\.col", header)
  if (length(col_indices) > 0L) {
    if (length(col_indices) != DIMENSION * DIMENSION) {
      stop("Expected ", DIMENSION * DIMENSION, " dense rotation columns, found ", length(col_indices), call. = FALSE)
    }
    return(matrix(row[col_indices], nrow = DIMENSION, ncol = DIMENSION, byrow = TRUE))
  }

  rotation <- matrix(NA_real_, nrow = DIMENSION, ncol = DIMENSION)
  missing <- character()
  for (i in seq_len(DIMENSION)) {
    for (j in seq_len(DIMENSION)) {
      name <- paste0("ou.A.rotation", i, j)
      index <- match(name, header)
      if (is.na(index)) {
        missing <- c(missing, name)
      } else {
        rotation[i, j] <- row[[index]]
      }
    }
  }
  if (length(missing) > 0L) {
    stop("Missing dense rotation columns: ", paste(head(missing, 5L), collapse = ", "), call. = FALSE)
  }
  rotation
}

median_drift <- function(path, burnin_fraction) {
  parsed <- read_beast_log(path)
  header <- parsed$header
  table <- parsed$table
  burnin_rows <- floor(nrow(table) * burnin_fraction)
  post <- table[seq.int(burnin_rows + 1L, nrow(table)), , drop = FALSE]
  if (nrow(post) == 0L) {
    stop("No post-burnin rows in ", path, call. = FALSE)
  }

  block_count <- DIMENSION / 2L
  rho <- vector_from_columns(header, post, "ou.A.rho", block_count)
  theta <- vector_from_columns(header, post, "ou.A.theta", block_count)
  t_values <- vector_from_columns(header, post, "ou.A.t", block_count)
  matrices <- array(NA_real_, dim = c(nrow(post), DIMENSION, DIMENSION))

  if ("ou.A.angles1" %in% header) {
    angles <- vector_from_columns(header, post, "ou.A.angles", DIMENSION * (DIMENSION - 1L) / 2L)
    for (sample_index in seq_len(nrow(post))) {
      rotation <- givens_rotation(DIMENSION, angles[sample_index, ])
      matrices[sample_index, , ] <- rotation %*%
        block_matrix(rho[sample_index, ], theta[sample_index, ], t_values[sample_index, ]) %*%
        t(rotation)
    }
    chart <- "orthogonal"
  } else if (any(grepl("^ou\\.A\\.rotation\\.col", header)) || "ou.A.rotation11" %in% header) {
    for (sample_index in seq_len(nrow(post))) {
      rotation <- dense_rotation_sample(header, post[sample_index, ])
      matrices[sample_index, , ] <- rotation %*%
        block_matrix(rho[sample_index, ], theta[sample_index, ], t_values[sample_index, ]) %*%
        solve(rotation)
    }
    chart <- "invertible"
  } else {
    stop("Could not detect stable matrix chart from ", path, call. = FALSE)
  }

  list(matrix = apply(matrices, c(2, 3), median), samples = nrow(post), chart = chart)
}

read_matrix_csv <- function(path, dimension) {
  data <- readr::read_csv(path, show_col_types = FALSE)
  if (all(c("row", "col", "value") %in% names(data))) {
    out <- matrix(NA_real_, nrow = dimension, ncol = dimension)
    for (i in seq_len(nrow(data))) {
      out[data$row[[i]], data$col[[i]]] <- data$value[[i]]
    }
    if (anyNA(out)) {
      stop("Sparse matrix CSV did not define every entry: ", path, call. = FALSE)
    }
    return(out)
  }
  numeric <- as.matrix(data)
  storage.mode(numeric) <- "double"
  if (nrow(numeric) != dimension || ncol(numeric) != dimension) {
    stop("Dense matrix CSV has wrong dimension: ", path, call. = FALSE)
  }
  numeric
}

off_diagonal_vmax <- function(matrix) {
  off_diagonal <- matrix[row(matrix) != col(matrix)]
  finite <- abs(off_diagonal[is.finite(off_diagonal)])
  if (length(finite) == 0L) 0.05 else max(0.01, max(finite))
}

display_vmax <- function(display) {
  finite <- abs(display[is.finite(display)])
  if (length(finite) == 0L) return(0.05)
  max(0.01, max(finite))
}

format_heatmap_label <- function(value) {
  rounded <- round(value, 2)
  rounded[abs(rounded) < 0.005] <- 0
  sprintf("%.2f", rounded)
}

json_or <- function(value, default) {
  if (is.null(value)) default else value
}

series_plot <- function(events, symbols, window, legend_style = "bottom", legend_rows = 3L, inside_legend_rows = 4L) {
  symbol_linetypes <- line_types(symbols)
  if (!legend_style %in% c("bottom", "inside", "none")) {
    stop("Unknown legend style: ", legend_style, call. = FALSE)
  }
  inside_legend <- legend_style == "inside"
  x_axis_label <- attr(events, "time_axis_label")
  if (is.null(x_axis_label)) x_axis_label <- "Minutes from window start"
  ggplot(events, aes(.data$time_display, .data$centered_log_price, colour = .data$symbol, linetype = .data$symbol, group = .data$symbol)) +
    geom_line(linewidth = 0.85, alpha = 0.95) +
    scale_colour_manual(values = line_colors(symbols), name = NULL) +
    scale_linetype_manual(values = symbol_linetypes, name = NULL) +
    labs(
      title = "BitMEX prices",
      x = x_axis_label,
      y = "Centered log price"
    ) +
    paper_theme() +
    theme(
      legend.position = if (legend_style == "none") "none" else if (inside_legend) "inside" else "bottom",
      legend.position.inside = if (inside_legend) c(0.94, 0.035) else c(0.5, 0.5),
      legend.justification = if (inside_legend) c(1, 0) else "center",
      legend.direction = "horizontal",
      legend.background = element_rect(fill = "transparent", colour = NA),
      legend.box.background = element_rect(fill = "transparent", colour = NA),
      legend.text = element_text(size = if (inside_legend) BITMEX_INSIDE_LEGEND_PT else BITMEX_LEGEND_PT, family = FIG_FONT),
      legend.key.width = grid::unit(if (inside_legend) 0.18 else 0.22, "inches"),
      legend.key.height = grid::unit(if (inside_legend) 0.075 else 0.10, "inches"),
      legend.spacing.x = grid::unit(if (inside_legend) 0.38 else 0.04, "inches"),
      legend.key.spacing.x = grid::unit(if (inside_legend) 0.38 else 0.04, "inches"),
      legend.spacing.y = grid::unit(0, "pt"),
      legend.margin = margin(0, 0, 0, 0, "pt"),
      legend.box.margin = if (inside_legend) margin(0, 0, 0, 0, "pt") else margin(0, 18, 0, 18, "pt"),
      plot.margin = margin(2, 4, 4, 4, "pt")
    ) +
    guides(
      colour = guide_legend(
        nrow = if (inside_legend) inside_legend_rows else legend_rows,
        byrow = TRUE,
        override.aes = list(linewidth = if (inside_legend) 0.8 else 1.0, linetype = unname(symbol_linetypes))
      ),
      linetype = "none"
    )
}

heatmap_plot <- function(matrix, symbols, vmax, matrix_kind = "selection", show_diagonal = FALSE) {
  display <- if (matrix_kind == "selection") -matrix else matrix
  if (!show_diagonal) {
    diag(display) <- NA_real_
  }
  heatmap_data <- expand_grid(
    target_index = seq_along(symbols),
    source_index = seq_along(symbols)
  ) |>
    mutate(
      target = factor(symbols[.data$target_index], levels = rev(symbols)),
      source = factor(symbols[.data$source_index], levels = symbols),
      value = display[cbind(.data$target_index, .data$source_index)],
      label = ifelse(is.na(.data$value), "", format_heatmap_label(.data$value)),
      text_color = ifelse(abs(.data$value) > 0.5 * vmax, "white", "black")
    )

  annotate_values <- length(symbols) <= 6L
  plot <- ggplot(heatmap_data, aes(.data$source, .data$target, fill = .data$value)) +
    geom_tile(color = "white", linewidth = 0.14)
  if (annotate_values) {
    plot <- plot +
      geom_text(
        aes(label = .data$label, colour = .data$text_color),
        family = FIG_FONT,
        size = BITMEX_HEATMAP_CELL_PT / ggplot2::.pt,
        show.legend = FALSE
      )
  }
  plot +
    coord_fixed(clip = "off") +
    scale_colour_identity() +
    scale_x_discrete(expand = expansion(add = 0)) +
    scale_y_discrete(expand = expansion(add = 0)) +
    scale_fill_gradient2(
      name = if (matrix_kind == "selection") "Posterior median" else "MAP estimate",
      low = "#3B4CC0",
      mid = "white",
      high = "#B40426",
      midpoint = 0,
      limits = c(-vmax, vmax),
      oob = scales::squish,
      na.value = "white",
      guide = guide_colourbar(
        title.position = "right",
        title.theme = element_text(
          angle = 90,
          size = BITMEX_AXIS_PT,
          family = FIG_FONT,
          colour = "black",
          hjust = 0.5,
          vjust = 0.5,
          margin = margin(0, 0, 0, 8, "pt")
        ),
        label.theme = element_text(size = BITMEX_COLORBAR_TEXT_PT, family = FIG_FONT, colour = "black"),
        barheight = grid::unit(3.19, "inches"),
        barwidth = grid::unit(0.13, "inches")
      )
    ) +
    labs(
      title = "BitMEX drift matrix",
      x = "Source instrument",
      y = "Target instrument"
    ) +
    paper_theme() +
    theme(
      axis.line = element_blank(),
      axis.ticks = element_blank(),
      axis.text.x = element_text(
        size = BITMEX_HEATMAP_ENTRY_PT,
        family = FIG_FONT,
        colour = "black",
        angle = 45,
        hjust = 1,
        vjust = 1,
        margin = margin(t = 0, unit = "pt"),
        lineheight = 0.78
      ),
      axis.text.y = element_text(
        size = BITMEX_HEATMAP_ENTRY_PT,
        family = FIG_FONT,
        colour = "black",
        margin = margin(r = -1.0, unit = "pt")
      ),
      axis.title.x = element_text(margin = margin(t = 14, unit = "pt")),
      axis.title.y = element_text(margin = margin(r = 20, unit = "pt")),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.35),
      panel.grid = element_blank(),
      legend.position = "right",
      legend.title = element_text(size = BITMEX_AXIS_PT, family = FIG_FONT, colour = "black"),
      legend.text = element_text(size = BITMEX_COLORBAR_TEXT_PT, family = FIG_FONT, colour = "black"),
      legend.margin = margin(0, 0, 0, 0, "pt"),
      plot.margin = margin(2, 6, 8, 12, "pt")
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

extract_legend <- function(plot) {
  grob <- ggplotGrob(plot)
  grob_names <- vapply(grob$grobs, function(child) {
    if (is.null(child$name)) "" else child$name
  }, character(1))
  guide_indices <- grep("^guide-box", grob_names)
  if (length(guide_indices) == 0L) {
    return(grid::nullGrob())
  }
  bottom <- which(grob_names[guide_indices] == "guide-box-bottom")
  if (length(bottom) > 0L) {
    return(grob$grobs[[guide_indices[[bottom[[1L]]]]]])
  }
  grob$grobs[[guide_indices[[1L]]]]
}

make_paper_row_figure <- function(window, events, symbols, drift_matrix, out_prefix, layout, matrix_kind = "selection", show_diagonal = FALSE, time_context = NULL) {
  display <- if (matrix_kind == "selection") -drift_matrix else drift_matrix
  if (!show_diagonal) diag(display) <- NA_real_
  vmax <- display_vmax(display)
  legend_style <- if (layout == "paper-row") "inside" else "bottom"
  inside_legend_rows <- if (!is.null(time_context$legend_rows)) as.integer(time_context$legend_rows) else 4L
  time_series <- series_plot(events, symbols, window, legend_style = legend_style, inside_legend_rows = inside_legend_rows)
  if (!is.null(time_context)) {
    time_series <- time_series +
      geom_vline(xintercept = time_context$separator, linewidth = 0.45, linetype = "22", colour = "black") +
      scale_x_continuous(
        breaks = time_context$breaks,
        labels = time_context$labels,
        expand = expansion(mult = c(0, 0.01))
      ) +
      coord_cartesian(xlim = time_context$xlim) +
      theme(
        legend.position = "inside",
        legend.position.inside = time_context$legend_position,
        legend.justification = time_context$legend_justification
      )
  }
  plot <- suppressWarnings(
    cowplot::plot_grid(
      time_series,
      heatmap_plot(drift_matrix, symbols, vmax, matrix_kind = matrix_kind, show_diagonal = show_diagonal),
      nrow = 1,
      rel_widths = paper_row_rel_widths(layout),
      align = "h",
      axis = "tb"
    )
  )

  dir.create(FIGURES, recursive = TRUE, showWarnings = FALSE)
  pdf_path <- file.path(FIGURES, paste0(out_prefix, ".pdf"))
  png_path <- file.path(FIGURES, paste0(out_prefix, ".png"))
  suppressWarnings(save_plot_pdf(plot, pdf_path, FIG_WIDTH_IN, FIG_HEIGHT_IN))
  suppressWarnings(
    ggsave(
      png_path,
      plot = plot,
      width = FIG_WIDTH_IN,
      height = FIG_HEIGHT_IN,
      units = "in",
      dpi = 300,
      device = ragg::agg_png,
      bg = "white"
    )
  )
  c(pdf_path, png_path)
}

main <- function() {
  args <- parse_args()
  paths <- default_paths(args$window)
  metadata_path <- if (nzchar(args$metadata)) resolve_input_path(args$metadata) else paths$metadata
  metadata <- jsonlite::fromJSON(metadata_path)
  symbols <- as.character(metadata$symbols)
  DIMENSION <<- length(symbols)
  events_path <- if (nzchar(args$events)) resolve_input_path(args$events) else paths$events
  events <- read_events(events_path, symbols)
  time_context <- NULL
  if (!is.null(metadata$vertical_separator_time_hours)) {
    attr(events, "time_axis_label") <- "UTC hour on 24 January 2022"
    display_xlim <- as.numeric(json_or(metadata$display_xlim_hours, c(8, 17)))
    display_breaks <- as.numeric(json_or(metadata$display_breaks_hours, c(8, 10, 12, 13, 15, 17)))
    display_labels <- as.character(json_or(metadata$display_break_labels, display_breaks))
    legend_position <- as.numeric(json_or(metadata$legend_position, c(0.94, 0.965)))
    legend_justification <- as.numeric(json_or(metadata$legend_justification, c(1, 1)))
    time_context <- list(
      separator = as.numeric(metadata$vertical_separator_time_hours),
      xlim = display_xlim,
      breaks = display_breaks,
      labels = display_labels,
      legend_position = legend_position,
      legend_justification = legend_justification,
      legend_rows = as.integer(json_or(metadata$legend_rows, 4L))
    )
  }

  drift_matrix <- NULL
  matrix_kind <- args$matrix_kind
  if (nzchar(args$matrix_csv)) {
    drift_matrix <- read_matrix_csv(resolve_input_path(args$matrix_csv), DIMENSION)
  } else {
    orthogonal <- median_drift(paths$orthogonal_log, args$burnin_fraction)
    if (orthogonal$chart != "orthogonal") {
      stop("Expected orthogonal log, found ", orthogonal$chart, ": ", paths$orthogonal_log, call. = FALSE)
    }
    drift_matrix <- orthogonal$matrix
    matrix_kind <- "selection"
  }
  out_prefix <- if (nzchar(args$out_prefix)) {
    args$out_prefix
  } else if (args$window == "jan2022_J2_4h") {
    "bitmex_jan2022_j2_4h_map_row"
  } else if (args$window == "jan2022_J20_trade_7h") {
    "bitmex_jan2022_j20_trade_7h_cartesian_map_row"
  } else {
    paste0("bitmex_", args$window, "min_trade_data_orthogonal_row")
  }

  if (nzchar(args$invertible_log)) {
    invertible_log <- resolve_input_path(args$invertible_log)
    invertible <- median_drift(invertible_log, args$burnin_fraction)
    if (invertible$chart != "invertible") {
      stop("Expected invertible log, found ", invertible$chart, ": ", invertible_log, call. = FALSE)
    }
    drift_matrix <- invertible$matrix
    matrix_kind <- "selection"
  }

  paths <- make_paper_row_figure(args$window, events, symbols, drift_matrix, out_prefix, args$layout, matrix_kind = matrix_kind, show_diagonal = args$show_diagonal, time_context = time_context)
  cat("wrote=", paths[[1]], "\n", sep = "")
  cat("wrote=", paths[[2]], "\n", sep = "")
}

main()
