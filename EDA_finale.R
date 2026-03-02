################################################################################
# 04_tesim/02_eda_gofundme_full_with_plots_daily1_20_FULL.R  (FULL REWRITE)
#
# OBIETTIVO (come richiesto)
# - EDA completa (dall'inizio alla fine), coerente con il tuo stile precedente
# - Outcome di progetto: y = log(1 + share_complete)  (NO cap a 1)
# - Mantiene goal_performance capped SOLO per EDA descrittiva / binning
#
# - Early-traction: TUTTI i giorni 1..20 (1,2,...,20) con 4 famiglie di variabili:
#   A) CUMULATIVE AMOUNT:  amount_raised_cum_d{d}
#   B) CUMULATIVE COUNT :  donors_cum_d{d}  ( = #donazioni cumulate entro giorno d)
#   C) DAILY (non cumul):  amount_raised_day_d{d} (solo giorno d)
#   D) DAILY (non cumul):  donors_day_d{d}        (solo giorno d)
#   + percentuali:
#   E) pct_goal_cum_d{d}   = amount_raised_cum_d{d} / goal_amount
#   F) pct_goal_day_d{d}   = amount_raised_day_d{d} / goal_amount
#
# NOTA DEFINIZIONE "DONATORI"
# - Nel tuo dataset, la colonna donations è una lista di donazioni (con name).
# - Non hai un donor_id univoco affidabile: quindi "donors" qui = numero di donazioni.
#   (coerente con il tuo vecchio n_donations_*). Se vuoi "unique donors per day"
#   posso aggiungere donors_unique_* usando 'name' (ma è più rumoroso).
#
# OUTPUT
# - ~/04_tesim/Data_processed/dataset_full_preprocessed.csv
# - ~/04_tesim/Data_processed/dataset_eda_clean.csv
# - ~/04_tesim/Tables/*.csv
# - ~/04_tesim/Figures/*.png
# - ~/04_tesim/Logs/eda_log_*.txt
################################################################################

# =========================
# 0) CONFIG
# =========================
setwd("~/04_tesim")

DATA_PATH <- "~/04_tesim/Data_raw/dataframe_gofundme_complete_nomissingGoalAmount.csv"

P_LOWER <- 0.005
P_UPPER <- 0.995

# Outliers: includo base + *tutti* i 20 giorni per le percentuali cum e daily + counts
# (se vuoi meno aggressivo, puoi commentare le liste ALL_*)
DAYS <- 1:20
OUTLIER_PCT_CUM  <- paste0("pct_goal_cum_d", DAYS)
OUTLIER_PCT_DAY  <- paste0("pct_goal_day_d", DAYS)
OUTLIER_DON_CUM  <- paste0("donors_cum_d", DAYS)
OUTLIER_DON_DAY  <- paste0("donors_day_d", DAYS)

VARS_OUTLIERS <- c(
  "donor_count", "n_caratteri", "goal_amount", "duration_days",
  "share_complete", "goal_performance", "log_share_complete",
  OUTLIER_PCT_CUM, OUTLIER_PCT_DAY, OUTLIER_DON_CUM, OUTLIER_DON_DAY
)

# =========================
# 1) LIBRERIE
# =========================
packages <- c("dplyr","tidyr","ggplot2","scales","viridis","readr","stringr","forcats",
              "lubridate","janitor","jsonlite","corrplot","purrr")

install_if_missing <- function(pkgs){
  for(p in pkgs){
    if(!requireNamespace(p, quietly = TRUE)){
      install.packages(p, dependencies = TRUE)
    }
  }
}
install_if_missing(packages)

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(ggplot2); library(scales); library(viridis)
  library(readr); library(stringr); library(forcats); library(lubridate); library(janitor)
  library(jsonlite); library(corrplot); library(purrr)
})

# =========================
# THEME + PALETTE (coerente e "pulita")
# =========================
library(ggplot2)
library(scales)

theme_eda <- function(base_size = 13, base_family = "sans") {
  theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
      plot.title.position = "plot",
      plot.title = element_text(face = "bold", size = base_size + 4, margin = margin(b = 6)),
      plot.subtitle = element_text(size = base_size, color = "grey30", margin = margin(b = 10)),
      axis.title = element_text(face = "bold", color = "grey15"),
      axis.text = element_text(color = "grey20"),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_line(color = "grey88", linewidth = 0.35),
      legend.title = element_text(face = "bold"),
      legend.position = "right",
      plot.margin = margin(12, 14, 12, 14)
    )
}

theme_set(theme_eda())

# Palette fissa (stessa “identità” in tutto l’EDA)
PAL_SUCCESS <- c("Insuccesso" = "#3B0F70", "Successo" = "#6ECE58")


# =========================
# 2) CARTELLE OUTPUT (EDA_files)
# =========================
EDA_ROOT <- "~/04_tesim/EDA_files"

dir_create <- function(path) if (!dir.exists(path)) dir.create(path, recursive = TRUE, showWarnings = FALSE)

dirs <- list(
  root    = EDA_ROOT,
  raw     = file.path(EDA_ROOT, "Data_raw_link"),     # opzionale (solo link/nota)
  proc    = file.path(EDA_ROOT, "Data_processed"),
  tables  = file.path(EDA_ROOT, "Tables"),
  figures = file.path(EDA_ROOT, "Figures"),
  logs    = file.path(EDA_ROOT, "Logs")
)
purrr::walk(dirs, dir_create)

timestamp <- format(Sys.time(), "%Y-%m-%d_%H%M%S")
LOG_FILE <- file.path(dirs$logs, paste0("eda_log_", timestamp, ".txt"))

log_line <- function(...) {
  msg <- paste0(...)
  cat(msg, "\n")
  cat(msg, "\n", file = LOG_FILE, append = TRUE)
}

save_plot <- function(p, filename, w = 10, h = 6, dpi = 360){
  ggplot2::ggsave(
    filename = file.path(dirs$figures, filename),
    plot = p, width = w, height = h, dpi = dpi,
    bg = "white"
  )
  log_line("   ✓ Salvata figura: ", filename)
}

save_csv <- function(df, filename){
  readr::write_csv(df, file.path(dirs$tables, filename))
  log_line("   ✓ Salvata tabella: ", filename)
}

# =========================
# 3) HELPERS
# =========================
parse_date_safe <- function(x) {
  x <- trimws(as.character(x))
  out <- suppressWarnings(as.Date(x))
  idx <- is.na(out) & !is.na(x) & nchar(x) >= 10
  if (any(idx)) out[idx] <- suppressWarnings(as.Date(substr(x[idx], 1, 10)))
  out
}

count_words <- function(x) {
  x <- ifelse(is.na(x), "", x)
  stringr::str_count(x, "\\b[[:alnum:]]+\\b")
}

count_sentences <- function(x) {
  x <- ifelse(is.na(x), "", x)
  n <- stringr::str_count(x, "[.!?]+")
  n <- ifelse(n == 0 & stringr::str_trim(x) != "", 1, n)
  n
}

topk_emotions <- function(mat, k = 3) {
  stopifnot(is.matrix(mat), !is.null(colnames(mat)))
  nm <- colnames(mat)
  ord <- t(apply(mat, 1, function(v) order(v, decreasing = TRUE)[1:k]))
  top_names <- matrix(nm[ord], nrow = nrow(mat), ncol = k)
  top_vals <- matrix(NA_real_, nrow = nrow(mat), ncol = k)
  for (i in seq_len(nrow(mat))) top_vals[i, ] <- mat[i, ord[i, ]]
  list(names = top_names, values = top_vals)
}

flag_outliers_percentile <- function(df, vars, p_low, p_up){
  vars <- vars[vars %in% names(df)]
  flags <- df %>% transmute(row_id = row_number())
  for(v in vars){
    ql <- quantile(df[[v]], p_low, na.rm = TRUE)
    qu <- quantile(df[[v]], p_up,  na.rm = TRUE)
    f  <- (df[[v]] < ql) | (df[[v]] > qu)
    flags[[paste0("outlier_", v)]] <- ifelse(is.na(f), FALSE, f)
    log_line(sprintf("%-35s: [%.3f, %.3f] -> %d outliers", v, ql, qu, sum(f, na.rm = TRUE)))
  }
  flags
}

# JSON “python-like” -> JSON
fix_to_json <- function(x) {
  if (is.na(x) || !nzchar(x)) return(NA_character_)
  y <- x
  y <- gsub("\\bTrue\\b", "true", y)
  y <- gsub("\\bFalse\\b", "false", y)
  y <- gsub("\\bNone\\b", "null", y)
  y <- gsub("'", "\"", y)  # best effort
  y
}

# --- parse donations string -> tibble(amount, created_at) ---
parse_donations_tbl <- function(don_str) {
  if (is.na(don_str) || !nzchar(don_str)) return(NULL)
  js <- fix_to_json(don_str)
  if (is.na(js)) return(NULL)
  
  out <- tryCatch({
    lst <- jsonlite::fromJSON(js, simplifyDataFrame = TRUE)
    if (is.null(lst) || nrow(as.data.frame(lst)) == 0) return(NULL)
    dd <- as_tibble(lst)
    if (!all(c("amount","created_at") %in% names(dd))) return(NULL)
    
    dd %>%
      transmute(
        amount = suppressWarnings(as.numeric(amount)),
        created_at = suppressWarnings(lubridate::ymd_hms(created_at, tz = "UTC"))
      ) %>%
      filter(!is.na(created_at))
  }, error = function(e) NULL)
  
  out
}

# --- compute daily + cumulative day1..20 in one pass (efficiente) ---
# Ritorna tibble con:
# amount_raised_cum_d1..d20, donors_cum_d1..d20,
# amount_raised_day_d1..d20, donors_day_d1..d20
compute_daily_and_cum_1_20 <- function(don_str, created_date, max_day = 20) {
  if (is.na(created_date)) {
    # NA created_date -> tutto NA
    out <- tibble()
    for (d in 1:max_day) {
      out[[paste0("amount_raised_cum_d", d)]] <- NA_real_
      out[[paste0("donors_cum_d", d)]] <- NA_integer_
      out[[paste0("amount_raised_day_d", d)]] <- NA_real_
      out[[paste0("donors_day_d", d)]] <- NA_integer_
    }
    return(out)
  }
  
  dd <- parse_donations_tbl(don_str)
  if (is.null(dd)) {
    out <- tibble()
    for (d in 1:max_day) {
      out[[paste0("amount_raised_cum_d", d)]] <- 0
      out[[paste0("donors_cum_d", d)]] <- 0L
      out[[paste0("amount_raised_day_d", d)]] <- 0
      out[[paste0("donors_day_d", d)]] <- 0L
    }
    return(out)
  }
  
  # day index (1..): floor diff in days + 1
  cd <- as.POSIXct(created_date, tz = "UTC")
  dd <- dd %>%
    mutate(
      day_index = as.integer(floor(as.numeric(difftime(created_at, cd, units = "days"))) + 1)
    ) %>%
    filter(!is.na(day_index), day_index >= 1, day_index <= max_day)
  
  # DAILY aggregates
  daily <- dd %>%
    group_by(day_index) %>%
    summarise(
      amount_day = sum(amount, na.rm = TRUE),
      donors_day = n(),
      .groups = "drop"
    )
  
  # complete 1..20 (missing -> 0)
  daily_full <- tibble(day_index = 1:max_day) %>%
    left_join(daily, by = "day_index") %>%
    mutate(
      amount_day = ifelse(is.na(amount_day), 0, amount_day),
      donors_day = ifelse(is.na(donors_day), 0L, as.integer(donors_day)),
      amount_cum = cumsum(amount_day),
      donors_cum = cumsum(donors_day)
    )
  
  out <- tibble()
  for (d in 1:max_day) {
    out[[paste0("amount_raised_day_d", d)]] <- daily_full$amount_day[d]
    out[[paste0("donors_day_d", d)]] <- daily_full$donors_day[d]
    out[[paste0("amount_raised_cum_d", d)]] <- daily_full$amount_cum[d]
    out[[paste0("donors_cum_d", d)]] <- daily_full$donors_cum[d]
  }
  out
}

# =========================
# 4) LOAD DATA
# =========================
log_line(strrep("=", 78))
log_line("EDA RUN: ", timestamp)
log_line("Data path: ", DATA_PATH)
log_line(strrep("=", 78))

stopifnot(file.exists(DATA_PATH))
raw <- read.csv(DATA_PATH, stringsAsFactors = FALSE) %>%
  janitor::clean_names()

log_line("Dataset caricato. N = ", nrow(raw), " | P = ", ncol(raw))

df <- raw %>%
  mutate(
    goal_amount    = suppressWarnings(as.numeric(goal_amount)),
    current_amount = suppressWarnings(as.numeric(current_amount)),
    donor_count    = suppressWarnings(as.numeric(donor_count)),
    share_complete = suppressWarnings(as.numeric(share_complete)),
    scraped_date   = parse_date_safe(scraped_date),
    created_date   = parse_date_safe(created_date)
  )

# =========================
# 5) TEXT (title + description) + feature testuali
# =========================
if (!("description" %in% names(df))) df$description <- ""
if (!("title" %in% names(df))) df$title <- ""

df <- df %>%
  mutate(
    description = ifelse(is.na(description), "", description),
    title       = ifelse(is.na(title), "", title),
    text        = str_squish(paste(title, description))
  ) %>%
  mutate(
    n_caratteri = nchar(text),
    n_parole    = count_words(text),
    n_frasi     = count_sentences(text),
    parole_per_frase     = ifelse(n_frasi > 0, n_parole / n_frasi, NA_real_),
    caratteri_per_parola = ifelse(n_parole > 0, n_caratteri / n_parole, NA_real_)
  )

# durata “snapshot”: giorni tra created e scraped
df <- df %>%
  mutate(duration_days = as.integer(scraped_date - created_date))

# Compatibilità con grafici/feature del tuo script
df <- df %>%
  mutate(
    campaign_duration = duration_days,
    text_length = n_caratteri
  )

# =========================
# 6) OUTCOME (response di progetto)
# =========================
df <- df %>%
  mutate(
    performance = share_complete,                          # % grezza
    goal_performance = pmin(pmax(performance, 0), 1),       # capped solo descrittivo
    log_share_complete = ifelse(!is.na(performance) & performance >= 0, log1p(performance), NA_real_),
    y = log_share_complete,
    goal_reached = ifelse(!is.na(performance) & performance >= 1, 1L,
                          ifelse(!is.na(performance), 0L, NA_integer_))
  ) %>%
  mutate(
    threshold = if ("threshold" %in% names(df)) ifelse(is.na(goal_reached), threshold, goal_reached) else goal_reached
  )

# =========================
# 7) Money features
# =========================
df <- df %>%
  mutate(
    money_difference = current_amount - goal_amount,
    money_missing    = pmax(goal_amount - current_amount, 0),
    money_excess     = pmax(current_amount - goal_amount, 0),
    donors_per_goal  = ifelse(goal_amount > 0, donor_count / goal_amount, NA_real_),
    avg_donation     = ifelse(donor_count > 0, current_amount / donor_count, NA_real_)
  )

# =========================
# 8) Emozioni -> aggregati + emotional_balance
# =========================
emotion_cols <- intersect(
  c("neutral","desire","sadness","disappointment","optimism","approval","caring","realization",
    "annoyance","disapproval","grief","love","remorse","disgust","admiration","gratitude",
    "fear","joy","nervousness","anger","excitement","relief","curiosity","pride",
    "embarrassment","surprise","confusion","amusement"),
  names(df)
)
for (cc in emotion_cols) df[[cc]] <- suppressWarnings(as.numeric(df[[cc]]))

positive_set <- intersect(c("joy","gratitude","optimism","love","admiration","caring","relief","pride","approval","excitement"), emotion_cols)
negative_set <- intersect(c("sadness","anger","fear","disgust","grief","remorse","disappointment","annoyance","disapproval","confusion","embarrassment","nervousness"), emotion_cols)

df <- df %>%
  mutate(
    positive_emotions = if (length(positive_set) > 0) rowSums(across(all_of(positive_set)), na.rm = TRUE) else NA_real_,
    negative_emotions = if (length(negative_set) > 0) rowSums(across(all_of(negative_set)), na.rm = TRUE) else NA_real_,
    emotional_balance = ifelse(!is.na(positive_emotions) & !is.na(negative_emotions),
                               positive_emotions - negative_emotions, NA_real_)
  )

if (length(emotion_cols) > 0) {
  df <- df %>%
    mutate(
      intensita_emotiva_totale = rowSums(across(all_of(emotion_cols)), na.rm = TRUE),
      densita_emotiva = ifelse(n_parole > 0, intensita_emotiva_totale / n_parole, NA_real_),
      intensita_std = as.numeric(scale(intensita_emotiva_totale))
    )
  mat <- as.matrix(df[, emotion_cols, drop = FALSE])
  mat[is.na(mat)] <- -Inf
  top3 <- topk_emotions(mat, k = 3)
  df$emo_top1 <- top3$names[, 1]; df$emo_top1_value <- top3$values[, 1]
  df$emo_top2 <- top3$names[, 2]; df$emo_top2_value <- top3$values[, 2]
  df$emo_top3 <- top3$names[, 3]; df$emo_top3_value <- top3$values[, 3]
}

# =========================
# 9) EARLY TRACTION SETS (SOLO CUMULATE, GIORNI 1..20)
# =========================
# Set:
# A) amount_raised_cum_d{d}
# B) donors_cum_d{d}
# C) pct_goal_cum_d{d}

log_line("Calcolo EARLY TRACTION CUMULATA 1..20 (amount, donors, pct).")

# ---- SET LISTS (sempre definite)
SET_TRACTION_AMOUNT_CUM <- paste0("amount_raised_cum_d", 1:20)
SET_TRACTION_DONORS_CUM <- paste0("donors_cum_d", 1:20)
SET_TRACTION_PCT_CUM    <- paste0("pct_goal_cum_d", 1:20)

ALL_TRACTION_COLS <- c(
  SET_TRACTION_AMOUNT_CUM,
  SET_TRACTION_DONORS_CUM,
  SET_TRACTION_PCT_CUM
)

# ---- helper robusta: SOLO CUMULATO
safe_compute_cum_1_20 <- function(don_str, created_date, max_day = 20) {
  
  # caso created_date mancante
  if (is.na(created_date)) {
    return(list(
      amount_cum = rep(NA_real_, max_day),
      donors_cum = rep(NA_integer_, max_day)
    ))
  }
  
  out <- tryCatch({
    
    dd <- parse_donations_tbl(don_str)
    if (is.null(dd) || nrow(dd) == 0) {
      return(list(
        amount_cum = rep(0, max_day),
        donors_cum = rep(0L, max_day)
      ))
    }
    
    cd <- as.POSIXct(created_date, tz = "UTC")
    
    dd <- dd %>%
      mutate(
        day_index = as.integer(floor(as.numeric(difftime(created_at, cd, units = "days"))) + 1L)
      ) %>%
      filter(!is.na(day_index), day_index >= 1L, day_index <= max_day)
    
    if (nrow(dd) == 0) {
      return(list(
        amount_cum = rep(0, max_day),
        donors_cum = rep(0L, max_day)
      ))
    }
    
    daily <- dd %>%
      group_by(day_index) %>%
      summarise(
        amount_day = sum(amount, na.rm = TRUE),
        donors_day = n(),
        .groups = "drop"
      )
    
    daily_full <- tibble(day_index = 1:max_day) %>%
      left_join(daily, by = "day_index") %>%
      mutate(
        amount_day = ifelse(is.na(amount_day), 0, amount_day),
        donors_day = ifelse(is.na(donors_day), 0L, as.integer(donors_day)),
        amount_cum = cumsum(amount_day),
        donors_cum = cumsum(donors_day)
      )
    
    list(
      amount_cum = as.numeric(daily_full$amount_cum),
      donors_cum = as.integer(daily_full$donors_cum)
    )
    
  }, error = function(e) {
    list(
      amount_cum = rep(NA_real_, max_day),
      donors_cum = rep(NA_integer_, max_day)
    )
  })
  
  out
}

# ---- MAIN
if (all(c("donations", "created_date") %in% names(df))) {
  
  n <- nrow(df)
  
  amt_cum_mat <- matrix(NA_real_, nrow = n, ncol = 20)
  don_cum_mat <- matrix(NA_integer_, nrow = n, ncol = 20)
  
  for (i in seq_len(n)) {
    res <- safe_compute_cum_1_20(df$donations[i], df$created_date[i], max_day = 20)
    
    amt_cum_mat[i, ] <- res$amount_cum
    don_cum_mat[i, ] <- res$donors_cum
    
    if (i %% 500 == 0) log_line("   ... righe processate: ", i, "/", n)
  }
  
  colnames(amt_cum_mat) <- SET_TRACTION_AMOUNT_CUM
  colnames(don_cum_mat) <- SET_TRACTION_DONORS_CUM
  
  df <- bind_cols(
    df,
    as.data.frame(amt_cum_mat),
    as.data.frame(don_cum_mat)
  )
  
  # percentuali cumulative su goal
  for (d in 1:20) {
    a_c <- SET_TRACTION_AMOUNT_CUM[d]
    p_c <- SET_TRACTION_PCT_CUM[d]
    
    df[[p_c]] <- ifelse(
      !is.na(df$goal_amount) & df$goal_amount > 0,
      df[[a_c]] / df$goal_amount,
      NA_real_
    )
  }
  
  log_line("✓ EARLY TRACTION CUMULATA 1..20 completata.")
  
} else {
  
  log_line("ATTENZIONE: mancano donations e/o created_date -> early traction = NA.")
  for (nm in SET_TRACTION_AMOUNT_CUM) df[[nm]] <- NA_real_
  for (nm in SET_TRACTION_DONORS_CUM) df[[nm]] <- NA_integer_
  for (nm in SET_TRACTION_PCT_CUM)    df[[nm]] <- NA_real_
}

# =========================
# 10) FEATURE ENGINEERING EDA (come prima)
# =========================
df <- df %>%
  mutate(
    performance_level = cut(
      goal_performance,
      breaks = c(0, 0.25, 0.5, 0.75, 1),
      labels = c("Basso (<25%)", "Medio (25-50%)", "Alto (50-75%)", "Molto alto (75-100%)"),
      include.lowest = TRUE
    ),
    goal_category = cut(
      goal_amount,
      breaks = c(0, 5000, 20000, 50000, Inf),
      labels = c("<5K", "5K-20K", "20K-50K", ">50K"),
      include.lowest = TRUE
    ),
    duration_category = cut(
      campaign_duration,
      breaks = c(0, 30, 60, 90, Inf),
      labels = c("<30gg", "30-60gg", "60-90gg", ">90gg"),
      include.lowest = TRUE
    ),
    donor_quintile = paste0("Q", dplyr::ntile(donor_count, 5))
  ) %>%
  mutate(donor_quintile = factor(donor_quintile, levels = paste0("Q", 1:5), ordered = TRUE))

# =========================
# 11) SALVA FULL
# =========================
full_out <- file.path(dirs$proc, "dataset_full_preprocessed.csv")
write.csv(df, full_out, row.names = FALSE)
log_line("✓ FULL salvato: ", full_out)

# =========================
# 12) OUTLIERS (solo per EDA clean)
# =========================
log_line("\n", strrep("-", 78))
log_line("OUTLIERS: percentili estremi (P_LOWER=", P_LOWER, ", P_UPPER=", P_UPPER, ")")
log_line(strrep("-", 78))

flags <- flag_outliers_percentile(df, VARS_OUTLIERS, P_LOWER, P_UPPER)
outlier_final <- flags %>% select(starts_with("outlier_")) %>% as.matrix() %>% rowSums() > 0
log_line("Outliers totali (>=1 variabile): ", sum(outlier_final), " (", round(mean(outlier_final)*100,2), "%)")

dmgf <- df[!outlier_final, ]
log_line("Dataset EDA clean. N = ", nrow(dmgf))

eda_out <- file.path(dirs$proc, "dataset_eda_clean.csv")
write.csv(dmgf, eda_out, row.names = FALSE)
log_line("✓ EDA clean salvato: ", eda_out)

# =========================
# 13) TABELLE
# =========================
tab_outcome <- tibble(
  N = nrow(dmgf),
  share_mean = mean(dmgf$share_complete, na.rm = TRUE),
  share_median = median(dmgf$share_complete, na.rm = TRUE),
  log_share_mean = mean(dmgf$log_share_complete, na.rm = TRUE),
  log_share_median = median(dmgf$log_share_complete, na.rm = TRUE),
  performance_mean_capped = mean(dmgf$goal_performance, na.rm = TRUE),
  performance_median_capped = median(dmgf$goal_performance, na.rm = TRUE),
  success_rate = mean(dmgf$goal_reached, na.rm = TRUE)
)
save_csv(tab_outcome, "tab_00_outcome_summary.csv")

if ("category_map" %in% names(dmgf)) {
  tab_success_cat <- dmgf %>%
    group_by(category_map) %>%
    summarise(n = n(), success_rate = mean(goal_reached, na.rm = TRUE), .groups = "drop") %>%
    arrange(desc(success_rate), desc(n))
  save_csv(tab_success_cat, "tab_01_success_by_category.csv")
}

tab_success_quint <- dmgf %>%
  group_by(donor_quintile) %>%
  summarise(
    n = n(),
    success_rate = mean(goal_reached, na.rm = TRUE),
    perf_mean_capped = mean(goal_performance, na.rm = TRUE),
    donors_mean = mean(donor_count, na.rm = TRUE),
    .groups = "drop"
  )
save_csv(tab_success_quint, "tab_02_success_by_donor_quintile.csv")

# Tabella: summary daily+cum per TUTTI i giorni 1..20
library(dplyr)
library(purrr)
library(tibble)

# helper: calcola mean/median solo se la colonna esiste
safe_stat <- function(data, colname, fun = mean) {
  if (!colname %in% names(data)) return(NA_real_)
  x <- data[[colname]]
  x <- suppressWarnings(as.numeric(x))
  fun(x, na.rm = TRUE)
}

tab_early_daily <- tibble(
  day = 1:20,
  pct_cum_mean      = map_dbl(1:20, ~ safe_stat(dmgf, paste0("pct_goal_cum_d", .x), mean)),
  pct_cum_median    = map_dbl(1:20, ~ safe_stat(dmgf, paste0("pct_goal_cum_d", .x), median)),
  pct_day_mean      = map_dbl(1:20, ~ safe_stat(dmgf, paste0("pct_goal_day_d", .x), mean)),
  pct_day_median    = map_dbl(1:20, ~ safe_stat(dmgf, paste0("pct_goal_day_d", .x), median)),
  donors_cum_mean   = map_dbl(1:20, ~ safe_stat(dmgf, paste0("donors_cum_d", .x), mean)),
  donors_cum_median = map_dbl(1:20, ~ safe_stat(dmgf, paste0("donors_cum_d", .x), median)),
  donors_day_mean   = map_dbl(1:20, ~ safe_stat(dmgf, paste0("donors_day_d", .x), mean)),
  donors_day_median = map_dbl(1:20, ~ safe_stat(dmgf, paste0("donors_day_d", .x), median))
)

# Tabella operativa: soglie (qui uso cum, perché è quella "quanto entro quando")
days_check <- 1:20
thr_check  <- c(0.10,0.25,0.50,0.75)

tab_ops <- purrr::map_dfr(days_check, function(d){
  col <- paste0("pct_goal_cum_d", d)
  purrr::map_dfr(thr_check, function(t){
    tmp <- dmgf %>% filter(!is.na(goal_reached), !is.na(.data[[col]]))
    reached <- tmp[[col]] >= t
    tibble(
      day = d,
      threshold = t,
      n = nrow(tmp),
      share_reaching_threshold = mean(reached),
      success_if_reached = mean(tmp$goal_reached[reached], na.rm = TRUE),
      success_if_not = mean(tmp$goal_reached[!reached], na.rm = TRUE)
    )
  })
}) %>%
  mutate(
    threshold = scales::percent(threshold, accuracy = 1)
  )
save_csv(tab_ops, "tab_04_operational_thresholds_day1_20.csv")

# =========================
# 14) FIGURE
# =========================
log_line("\n", strrep("-", 78))
log_line("FIGURE: base + early traction (DAY & CUM) su TUTTI i giorni 1..20")
log_line(strrep("-", 78))

# FIG 01a — share_complete
# n overfunded (share>1)
d_overfund <- sum(dmgf$share_complete > 1, na.rm = TRUE)

p1 <- ggplot(dmgf, aes(x = share_complete)) +
  geom_histogram(bins = 40, fill = "#4B1D6B", color = "white", linewidth = 0.25, alpha = 0.92) +
  coord_cartesian(xlim = c(0, 2)) +
  scale_x_continuous(breaks = seq(0, 2, 0.25)) +
  labs(
    title = "Distribuzione di share_complete",
    subtitle = paste0("Visualizzata fino a 2 | Overfunded (share>1): ", d_overfund),
    x = "share_complete", y = "Frequenza"
  )

save_plot(p1, "fig_01a_share_complete.png", w = 10, h = 5.8)


# FIG 01b — goal_performance capped
p2 <- ggplot(dmgf, aes(x = goal_performance)) +
  geom_histogram(bins = 40, fill = "#234E70", color = "white", linewidth = 0.25, alpha = 0.92) +
  scale_x_continuous(labels = percent_format(accuracy = 1)) +
  labs(
    title = "Distribuzione goal_performance (capped a 1)",
    subtitle = paste0("Mean=", round(mean(dmgf$goal_performance, na.rm=TRUE),3),
                      " | Median=", round(median(dmgf$goal_performance, na.rm=TRUE),3)),
    x = "Goal performance [0,1]", y = "Frequenza"
  )
save_plot(p2, "fig_01b_goal_performance.png", w = 10, h = 5.8)


# --- FIG 01c: outcome log ---
p_logperf <- ggplot(dmgf, aes(x = log_share_complete)) +
  geom_histogram(bins = 50, fill = viridis(1, option = "B"), color = "white", alpha = 0.9) +
  labs(title = "Distribuzione outcome: log(1 + share_complete)",
       subtitle = paste0("Mean=", round(mean(dmgf$log_share_complete, na.rm=TRUE),3),
                         " | Median=", round(median(dmgf$log_share_complete, na.rm=TRUE),3)),
       x = "log(1 + share_complete)", y = "Frequenza")
save_plot(p_logperf, "fig_01c_log_share_complete.png", w = 10, h = 6)

# --- FIG 02: goal_reached imbalance ---
# Assumiamo che tab_goal contenga:
# goal_reached (factor o 0/1), n (conteggio), e prop (proporzione)
# e che label sia del tipo "n (xx%)"
tab_goal <- df %>%
  dplyr::filter(!is.na(goal_reached)) %>%
  dplyr::count(goal_reached, name = "n") %>%
  dplyr::mutate(
    prop  = n / sum(n),
    label = paste0(n, " (", scales::percent(prop, accuracy = 0.1), ")")
  )

max_y <- max(tab_goal$n, na.rm = TRUE)

p_goal <- ggplot(tab_goal, aes(x = factor(goal_reached),
                               y = n,
                               fill = factor(goal_reached))) +
  geom_col(width = 0.65,
           color = "white",
           linewidth = 0.3,
           alpha = 0.95) +
  geom_text(aes(label = label),
            vjust = -0.4,
            fontface = "bold",
            size = 4.2) +
  scale_fill_manual(values = PAL_SUCCESS, guide = "none") +
  scale_y_continuous(labels = scales::comma) +
  labs(
    title = "Distribuzione della variabile target (goal_reached)",
    subtitle = "Il dataset presenta un marcato sbilanciamento tra campagne di successo e insuccesso",
    x = NULL,
    y = "Numero di campagne"
  ) +
  expand_limits(y = max_y * 1.15) +
  theme(
    plot.title = element_text(face = "bold"),
    plot.subtitle = element_text(size = 11),
    axis.text.x = element_text(face = "bold")
  )

save_plot(p_goal, "fig_02_goal_reached_imbalance.png", w = 8.2, h = 5.0)



# --- FIG 03a: donor_count vs goal_performance (descrittivo) ---
p_donor_scatter <- ggplot(dmgf, aes(x = donor_count, y = goal_performance, color = factor(goal_reached))) +
  geom_point(alpha = 0.14, size = 1.1) +
  geom_smooth(method = "loess", se = FALSE, linewidth = 1.1, color = "grey10") +
  scale_x_continuous(trans = "log1p") +
  scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1)) +
  scale_color_manual(values = c("0" = PAL_SUCCESS["Insuccesso"], "1" = PAL_SUCCESS["Successo"]),
                     name = "Goal reached", labels = c("0"="No","1"="Sì")) +
  labs(
    title = "Donatori (finale) e performance (capped)",
    subtitle = paste0("Scala x: log(1+donor_count) | Pearson r = ",
                      round(cor(dmgf$donor_count, dmgf$goal_performance, use="complete.obs"), 3)),
    x = "donor_count (log1p)", y = "goal_performance [0,1]"
  ) +
  guides(color = guide_legend(override.aes = list(alpha = 1, size = 2.8)))
save_plot(p_donor_scatter, "fig_03a_donor_vs_performance.png", w = 10.5, h = 6.2)


# --- FIG 03b: success rate by donor quintile ---
tab_quint <- dmgf %>%
  group_by(donor_quintile) %>%
  summarise(success_rate = mean(goal_reached, na.rm=TRUE), n=n(), .groups="drop")

p_quint <- ggplot(tab_quint, aes(x = donor_quintile, y = success_rate, fill = donor_quintile)) +
  geom_col(color="white", alpha=0.95) +
  geom_text(aes(label=scales::percent(success_rate, accuracy=0.1)), vjust=-0.4) +
  scale_fill_viridis_d(option="C", end=0.9, guide="none") +
  scale_y_continuous(labels=scales::percent_format(accuracy=1),
                     limits=c(0, max(tab_quint$success_rate, na.rm=TRUE)*1.2)) +
  labs(title="Tasso di successo per quintile di donatori (finali)",
       x="Quintile donatori", y="Success rate")
save_plot(p_quint, "fig_03b_success_by_donor_quintile.png", w = 8, h = 5)

# --- FIG 03c: donor_count vs log outcome ---
p_donor_log <- ggplot(dmgf, aes(x = donor_count, y = log_share_complete, color = factor(goal_reached))) +
  geom_point(alpha = 0.18, size = 1.2) +
  geom_smooth(method="loess", se=FALSE, linewidth=1, color="black") +
  scale_color_viridis_d(option="D", end=0.9, name="Goal reached",
                        labels=c("0"="No","1"="Sì")) +
  labs(title="Numero donatori (finale) vs outcome logaritmico",
       subtitle=paste0("Pearson r = ",
                       round(cor(dmgf$donor_count, dmgf$log_share_complete, use="complete.obs"), 3)),
       x="donor_count (finale)", y="log(1 + share_complete)")
save_plot(p_donor_log, "fig_03c_donor_vs_log_outcome.png", w = 10, h = 6)

# --- FIG 04a: text length vs goal_performance (descrittivo) ---
p_text <- ggplot(dmgf, aes(x = text_length, y = goal_performance, color = goal_performance)) +
  geom_point(alpha=0.18, size=1.1) +
  geom_smooth(method="loess", se=FALSE, linewidth=1, color="black") +
  scale_color_viridis_c(option="B", end=0.9, guide="none") +
  labs(title="Lunghezza testo vs goal_performance (capped) [descrittivo]",
       subtitle=paste0("Pearson r = ",
                       round(cor(dmgf$text_length, dmgf$goal_performance, use="complete.obs"), 3)),
       x="Lunghezza testo (char)", y="goal_performance [0,1]") +
  coord_cartesian(xlim=c(0, quantile(dmgf$text_length, 0.99, na.rm=TRUE)))
save_plot(p_text, "fig_04a_text_length_vs_performance.png", w = 10, h = 6)

# --- FIG 04b: heatmap text x emotion ---
if (all(c("text_length","emotional_balance") %in% names(dmgf))) {
  
  dmgf_h <- dmgf %>%
    dplyr::filter(
      !is.na(text_length),
      !is.na(emotional_balance),
      !is.na(goal_reached)
    ) %>%
    dplyr::mutate(
      text_q = paste0("Q", dplyr::ntile(text_length, 4)),
      emo_tone = factor(
        dplyr::ntile(emotional_balance, 3),
        levels = 1:3,
        labels = c("Neg","Neu","Pos"),
        ordered = TRUE
      )
    ) %>%
    dplyr::group_by(text_q, emo_tone) %>%
    dplyr::summarise(
      success_rate = mean(goal_reached, na.rm = TRUE),
      n = dplyr::n(),
      .groups = "drop"
    ) %>%
    dplyr::mutate(
      text_q = factor(text_q, levels = paste0("Q", 1:4), ordered = TRUE)
    )
  
  p_heat <- ggplot2::ggplot(dmgf_h, ggplot2::aes(x = emo_tone, y = text_q, fill = success_rate)) +
    ggplot2::geom_tile(color = "white") +
    ggplot2::geom_text(
      ggplot2::aes(label = scales::percent(success_rate, accuracy = 0.1)),
      size = 3,
      color = "black"
    ) +
    ggplot2::scale_fill_viridis_c(option = "D", labels = scales::percent_format(accuracy = 1)) +
    ggplot2::labs(
      title = "Successo: combinazione testo × tono emotivo (binning per quantili)",
      x = "Tono emotivo (tercili emotional_balance)",
      y = "Quartile lunghezza testo",
      fill = "Success"
    )
  
  save_plot(p_heat, "fig_04b_heatmap_text_x_emotion.png", w = 8.5, h = 5.5)
  
} else {
  log_line("   (skip) fig_04b: mancano text_length o emotional_balance.")
}

# --- FIG 05: correlation matrix (includo una selezione, NON tutte le 120 variabili) ---
# (Per non avere un corrplot enorme/illeggibile, uso milestone + outcome + base)
corr_vars <- c(
  "log_share_complete", "goal_reached",
  "goal_amount", "donor_count", "campaign_duration", "text_length",
  "positive_emotions", "negative_emotions", "emotional_balance",
  # milestone cum/day (1,3,7,14,20) per leggibilità
  "pct_goal_cum_d1","pct_goal_cum_d3","pct_goal_cum_d7","pct_goal_cum_d14","pct_goal_cum_d20",
  "pct_goal_day_d1","pct_goal_day_d3","pct_goal_day_d7","pct_goal_day_d14","pct_goal_day_d20",
  "donors_cum_d1","donors_cum_d3","donors_cum_d7","donors_cum_d14","donors_cum_d20",
  "donors_day_d1","donors_day_d3","donors_day_d7","donors_day_d14","donors_day_d20"
)
corr_vars <- corr_vars[corr_vars %in% names(dmgf)]

if (length(corr_vars) >= 6) {
  corr_mat <- cor(dmgf[, corr_vars], use = "complete.obs")
  png(file.path(dirs$figures, "fig_05_correlation_matrix.png"), width = 2400, height = 1900, res = 250)
  corrplot::corrplot(
    corr_mat, method="color", type="upper",
    col = viridis::viridis(200),
    tl.col="black", tl.srt=45, tl.cex=0.75,
    addCoef.col="black", number.cex=0.5
  )
  dev.off()
  log_line("   ✓ Salvata figura: fig_05_correlation_matrix.png")
} else {
  log_line("   (skip) fig_05: troppe poche variabili.")
}

# =========================
# 14b) FIGURE “OPERATIVE” su TUTTI i giorni 1..20
# =========================
# --- FIG 06: Traiettorie cum % (mediana + IQR) giorno 1..20 per successo/insuccesso ---

# 1) Colonne percentuale cum d1..d20
pct_cols <- paste0("pct_goal_cum_d", 1:20)
pct_cols <- pct_cols[pct_cols %in% names(df)]
if (length(pct_cols) < 5) stop("Non trovo abbastanza colonne pct_goal_cum_d1..d20 nel df.")

# 2) Wide -> Long (senza tidyr::pivot_longer, così sei sicuro di averlo anche in ambienti “scarni”)
df_long <- df %>%
  dplyr::select(goal_reached, dplyr::all_of(pct_cols)) %>%
  dplyr::mutate(goal_reached = as.integer(goal_reached)) %>%
  tidyr::pivot_longer(
    cols = dplyr::all_of(pct_cols),
    names_to = "day",
    values_to = "pct_goal_cum"
  ) %>%
  dplyr::mutate(
    day = as.integer(gsub("pct_goal_cum_d", "", day))
  ) %>%
  dplyr::filter(day >= 1, day <= 20)

# 3) Aggregazione: mediana + IQR per giorno e outcome
traj <- df_long %>%
  dplyr::group_by(day, goal_reached) %>%
  dplyr::summarise(
    pct_med = median(pct_goal_cum, na.rm = TRUE),
    pct_p25 = stats::quantile(pct_goal_cum, 0.25, na.rm = TRUE),
    pct_p75 = stats::quantile(pct_goal_cum, 0.75, na.rm = TRUE),
    n = dplyr::n(),
    .groups = "drop"
  )

# 4) Label più leggibili (0/1 -> Insuccesso/Successo)
traj_plot <- traj %>%
  dplyr::mutate(
    goal_reached = factor(goal_reached, levels = c(0, 1),
                          labels = c("Insuccesso", "Successo"))
  )

# 5) Plot “da tesi”
p_traj <- ggplot(traj_plot,
                 aes(x = day, y = pct_med,
                     color = goal_reached, fill = goal_reached)) +
  geom_ribbon(aes(ymin = pct_p25, ymax = pct_p75),
              alpha = 0.18, linewidth = 0) +
  geom_line(linewidth = 1.25) +
  scale_color_manual(values = PAL_SUCCESS, name = NULL) +
  scale_fill_manual(values = PAL_SUCCESS, guide = "none") +
  scale_x_continuous(breaks = seq(1, 20, by = 2), minor_breaks = NULL) +
  scale_y_continuous(
    labels = scales::percent_format(accuracy = 1),
    expand = expansion(mult = c(0.02, 0.06))
  ) +
  labs(
    title = "Traiettorie nei primi 20 giorni: percentuale cumulata del goal",
    subtitle = "Mediana e intervallo interquartile (IQR) — confronto tra campagne di successo e insuccesso",
    x = "Giorno",
    y = "% del goal raccolta (cumulata)"
  ) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "top"
  )

save_plot(p_traj, "fig_06_trajectory_pct_cum_day1_20.png", w = 10.8, h = 6.2)


# --- FIG 07: Traiettorie DAILY % (mediana) giorno 1..20 per successo/insuccesso ---
pct_day_cols_all <- paste0("pct_goal_day_d", 1:20)
pct_day_cols <- intersect(pct_day_cols_all, names(dmgf))

if (length(pct_day_cols) == 0) {
  log_line("   (skip) fig_07: nessuna colonna pct_goal_day_d1..d20 trovata.")
} else {
  
  traj_day <- dmgf %>%
    dplyr::select(goal_reached, dplyr::all_of(pct_day_cols)) %>%
    tidyr::pivot_longer(
      cols = dplyr::all_of(pct_day_cols),
      names_to = "day",
      values_to = "pct_day"
    ) %>%
    dplyr::mutate(
      day = as.integer(gsub("^pct_goal_day_d", "", day)),
      pct_day = suppressWarnings(as.numeric(pct_day)),
      goal_reached = dplyr::case_when(
        isTRUE(is.logical(goal_reached)) ~ ifelse(goal_reached, 1, 0),
        TRUE ~ as.integer(goal_reached)
      )
    ) %>%
    dplyr::group_by(goal_reached, day) %>%
    dplyr::summarise(
      pct_day_med = median(pct_day, na.rm = TRUE),
      pct_day_p25 = as.numeric(quantile(pct_day, 0.25, na.rm = TRUE, names = FALSE)),
      pct_day_p75 = as.numeric(quantile(pct_day, 0.75, na.rm = TRUE, names = FALSE)),
      .groups = "drop"
    ) %>%
    dplyr::filter(!is.na(day)) %>%
    dplyr::mutate(
      goal_reached = factor(goal_reached, levels = c(0, 1), labels = c("Insuccesso", "Successo"))
    )
  
  p_traj_day <- ggplot2::ggplot(
    traj_day,
    ggplot2::aes(x = day, y = pct_day_med, color = goal_reached, fill = goal_reached)
  ) +
    ggplot2::geom_ribbon(ggplot2::aes(ymin = pct_day_p25, ymax = pct_day_p75),
                         alpha = 0.18, color = NA) +
    ggplot2::geom_line(linewidth = 1.1) +
    ggplot2::scale_color_viridis_d(option = "C", end = 0.9) +
    ggplot2::scale_fill_viridis_d(option = "C", end = 0.9, guide = "none") +
    ggplot2::scale_y_continuous(labels = scales::percent_format(accuracy = 0.1)) +
    ggplot2::labs(
      title = "Traiettorie 1..20: % goal giornaliera (non cumulata) (mediana + IQR)",
      subtitle = "Dove avvengono gli 'spike' di raccolta? Confronto Successo vs Insuccesso",
      x = "Giorno", y = "% goal raccolta nel giorno d", color = NULL
    )
  
  save_plot(p_traj_day, "fig_07_trajectory_pct_day_day1_20.png", w = 10, h = 6)
}

# --- FIG 08: Success rate se raggiungi soglia entro il giorno d (TUTTI 1..20) ---
thresholds <- c(0.10, 0.25, 0.50, 0.75)

sr_grid <- purrr::map_dfr(1:20, function(d){
  col <- paste0("pct_goal_cum_d", d)
  purrr::map_dfr(thresholds, function(t){
    tmp <- dmgf %>% filter(!is.na(goal_reached), !is.na(.data[[col]]))
    reached <- tmp[[col]] >= t
    tibble(
      day = d,
      threshold = t,
      n = nrow(tmp),
      success_if_reached = mean(tmp$goal_reached[reached], na.rm=TRUE)
    )
  })
}) %>%
  mutate(threshold_label = paste0(scales::percent(threshold, accuracy=1)))

p_thr <- ggplot(sr_grid, aes(x=day, y=success_if_reached, color=threshold_label)) +
  geom_line(linewidth=1.1) +
  geom_point(size=1.6, alpha=0.9) +
  scale_color_viridis_d(option="D", end=0.9) +
  scale_y_continuous(labels=scales::percent_format(accuracy=1), limits=c(0,1)) +
  labs(title="Probabilità di successo se raggiungi una soglia entro il giorno d (1..20)",
       subtitle="Risposta operativa: entro quanto conviene raggiungere 10/25/50/75% del goal?",
       x="Giorno", y="Success rate | condizionato a (cum pct >= soglia)", color="Soglia (%)")
save_plot(p_thr, "fig_08_success_rate_thresholds_day1_20.png", w=10, h=6)

# --- FIG 09: Primo giorno di raggiungimento soglie (1..20) ---
first_day_reach <- function(df_in, thr){
  cols <- paste0("pct_goal_cum_d", 1:20)
  mat <- as.matrix(df_in[, cols])
  idx <- apply(mat, 1, function(v){
    w <- which(!is.na(v) & v >= thr)
    if (length(w) == 0) return(NA_integer_)
    min(w)
  })
  idx
}

fd <- purrr::map_dfr(thresholds, function(t){
  tibble(
    threshold = t,
    first_day = first_day_reach(dmgf, t),
    goal_reached = dmgf$goal_reached
  )
}) %>%
  filter(!is.na(goal_reached)) %>%
  mutate(
    threshold_label = paste0(scales::percent(threshold, accuracy=1)),
    outcome = factor(goal_reached, levels=c(0,1), labels=c("Insuccesso","Successo"))
  )

p_fd <- ggplot(fd, aes(x=first_day, fill=outcome)) +
  geom_histogram(bins=20, position="identity", alpha=0.55, color="white") +
  facet_wrap(~ threshold_label, ncol=2) +
  scale_fill_viridis_d(option="D", end=0.9) +
  labs(title="Primo giorno di raggiungimento soglia (nei primi 20 giorni)",
       subtitle="Confronto distribuzioni: i successi raggiungono prima le soglie?",
       x="Primo giorno (1..20) in cui cum pct >= soglia", y="Conteggio", fill=NULL)
save_plot(p_fd, "fig_09_first_day_reach_thresholds.png", w=11, h=7)

# --- FIG 10: Heatmap giorno x soglia (success rate) (dashboard compatta) ---
# (molto utile per la tesi: una sola figura che riassume tutto)
p_heat_ops <- ggplot(sr_grid, aes(x=day, y=threshold_label, fill=success_if_reached)) +
  geom_tile(color="white", linewidth=0.35) +
  geom_text(aes(label=percent(success_if_reached, accuracy=1)),
            size=3.2, color="black", fontface="bold") +
  scale_fill_viridis_c(option="C", labels=percent_format(accuracy=1), limits=c(0,1)) +
  scale_x_continuous(breaks = seq(1, 20, 1)) +
  labs(
    title="Heatmap operativa: success rate condizionato",
    subtitle="Probabilità di successo se la campagna raggiunge X% entro il giorno Y",
    x="Giorno", y="Soglia (% goal) raggiunta", fill="Success rate"
  ) +
  theme(panel.grid = element_blank())
save_plot(p_heat_ops, "fig_10_heatmap_success_threshold_by_day.png", w=11.5, h=5.8)

# =========================
# 15) CHIUSURA
# =========================
log_line("\n✓ Fine run.")
log_line(" - FULL: ", full_out)
log_line(" - EDA : ", eda_out)
log_line(" - Figures: ", dirs$figures)
log_line(" - Tables : ", dirs$tables)
log_line(" - Log    : ", LOG_FILE)
################################################################################
# FINE
################################################################################
