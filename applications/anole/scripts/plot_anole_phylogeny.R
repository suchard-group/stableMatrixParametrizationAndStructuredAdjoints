#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ape)
  library(ggplot2)
  library(ggtree)
  library(ragg)
  library(rsvg)
  library(showtext)
  library(sysfonts)
  library(svglite)
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
script_dir <- if (length(file_arg) == 1) {
  dirname(normalizePath(sub("^--file=", "", file_arg)))
} else {
  getwd()
}

tree_path <- file.path(script_dir, "..", "data", "Anolis.tre")
output_path <- file.path(script_dir, "..", "output", "anole_phylogeny.pdf")
png_output_path <- file.path(script_dir, "..", "output", "anole_phylogeny.png")

FIG_WIDTH_IN <- 5.4
FIG_HEIGHT_IN <- 3.95
TEXT_WIDTH_IN <- 6.5
MINIPAGE_WIDTH <- 0.48
ANOLIS_SCALE_DOWN <- MINIPAGE_WIDTH * TEXT_WIDTH_IN / FIG_WIDTH_IN
ANOLIS_FONT_SCALE <- (12 / ANOLIS_SCALE_DOWN) / 9
ANOLIS_TITLE_PT <- 8 * ANOLIS_FONT_SCALE

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

dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)

tree <- ladderize(read.tree(tree_path))

children_by_parent <- split(tree$edge[, 2], tree$edge[, 1])
node_depth <- node.depth.edgelength(tree)
tip_count <- Ntip(tree)
all_nodes <- seq_len(tip_count + tree$Nnode)
cut_depth <- 0.12

descendant_tips <- function(node) {
  if (node <= tip_count) {
    return(node)
  }
  children <- children_by_parent[[as.character(node)]]
  unlist(lapply(children, descendant_tips), use.names = FALSE)
}

descendant_nodes <- function(node) {
  children <- children_by_parent[[as.character(node)]]
  if (is.null(children)) {
    return(integer(0))
  }
  c(children, unlist(lapply(children, descendant_nodes), use.names = FALSE))
}

clade_roots <- tree$edge[
  node_depth[tree$edge[, 1]] < cut_depth &
    node_depth[tree$edge[, 2]] >= cut_depth,
  2
]

clade_sizes <- vapply(clade_roots, function(node) length(descendant_tips(node)), integer(1))
major_clade_roots <- clade_roots[clade_sizes >= 5]
major_clade_roots <- major_clade_roots[order(vapply(
  major_clade_roots,
  function(node) min(descendant_tips(node)),
  numeric(1)
))]

node_group <- setNames(rep("Backbone", length(all_nodes)), all_nodes)
for (index in seq_along(major_clade_roots)) {
  root <- major_clade_roots[index]
  clade_nodes <- c(root, descendant_nodes(root))
  node_group[as.character(clade_nodes)] <- paste0("Clade ", index)
}

branch_palette <- c(
  "Backbone" = "#6f7b82",
  "Clade 1" = "#0072B2",
  "Clade 2" = "#009E73",
  "Clade 3" = "#D55E00",
  "Clade 4" = "#CC79A7",
  "Clade 5" = "#E69F00"
)

tree_plot <- suppressMessages(
  ggtree(
    tree,
    layout = "fan",
    open.angle = 18,
    size = 0
  )
)
tree_plot$data$clade_group <- node_group[as.character(tree_plot$data$node)]

tree_plot <- tree_plot +
  geom_tree(aes(color = clade_group), size = 0.34) +
  geom_tippoint(aes(color = clade_group), size = 0.58, alpha = 0.9) +
  scale_color_manual(values = branch_palette, guide = "none") +
  labs(title = "Anolis phylogeny") +
  theme_void() +
  theme(
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA),
    plot.title = element_text(
      face = "bold",
      hjust = 0.5,
      family = FIG_FONT,
      size = ANOLIS_TITLE_PT,
      margin = margin(b = 0, unit = "pt")
    ),
    plot.margin = margin(4, 2, 2, 2, "pt")
  )

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

save_tree_plot <- function(pdf_path, png_path) {
  save_plot_pdf(tree_plot, pdf_path, FIG_WIDTH_IN, FIG_HEIGHT_IN)
  ggsave(
    filename = png_path,
    plot = tree_plot,
    width = FIG_WIDTH_IN,
    height = FIG_HEIGHT_IN,
    units = "in",
    dpi = 300,
    device = ragg::agg_png,
    bg = "white"
  )
}

if (!identical(Sys.getenv("ANOLIS_SKIP_COMPONENT_MAIN"), "true")) {
  save_tree_plot(output_path, png_output_path)

  message("Saved ", output_path)
  message("Saved ", png_output_path)
}
