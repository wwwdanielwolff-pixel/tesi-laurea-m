################################################################################
# 05_tesim/01_bart_daily_models_day1_20_report_with_DCA_ROBUST.R
#
# VERSIONE CORRETTA (robusta + metodologicamente coerente + runnabile)
#
# OBIETTIVO
# (A) 20 modelli BART (regressione), uno per ciascun giorno d = 1..20:
#     y_final = log(1 + share_complete_finale)
#   usando SOLO informazione disponibile entro il giorno d:
#     - feature statiche note a t0 (goal, testo, emozioni, categoria, durata osservata…)
#     - early traction cumulata fino al giorno d (pct_goal_cum_d*, donors_cum_d*)
#   (NO leakage: NON uso current_amount finale, donor_count finale, money_* ecc.)
#
# (B) Valutazione out-of-sample: K-fold CV
#     - metriche su y_final (RMSE, MAE, R2)
#     - metriche su scala interpretabile: pct_final = min(100, 100*share_final)
#     - calibrazione: coverage & width intervalli predittivi (80%, 95%)
#
# (C) Scelta non arbitraria dei giorni:
#     - metrica primaria: RMSE_pct (scala capped)
#     - smoothing + guadagno marginale Δ(d)
#     - peak days, near-optimal day, plateau day
#
# (D) Decision rule non arbitraria: DCA “bayesiana”
#     - evento operativo: pct_final >= T (T in {75, 100})
#     - π_i(T) = P(pct_final >= T | X_<=d) stimata dai draw posterior predittivi BART
#     - NB(τ) su τ in [0.05..0.95], bande 90% via Bayesian bootstrap
#     - τ_star = argmax NB(τ); τ_robust = argmax lower bound 90%
#
# (E) Decisione multistadio (Value of Information / Stop-waiting)
#     - Utility per day: max_tau NB (o NB_low90 robust)
#     - VOI(d,k)=U(d+k)-U(d)-k*C_wait
#     - stop_day: primo d con min_k VOI(d,k) <= 0
#
# OUTPUT
# - ~/04_tesim/Models_BART/
#   - Tables/ Figures/ Report/ Logs/
################################################################################

# =========================
# 0) CONFIG
# =========================
setwd("~/04_tesim")

DATA_PATH <- "~/04_tesim/Data_processed/dataset_full_preprocessed.csv"

OUT_DIR <- "~/04_tesim/Models_BART"
DIRS <- list(
  root    = OUT_DIR,
  tables  = file.path(OUT_DIR, "Tables"),
  figures = file.path(OUT_DIR, "Figures"),
  report  = file.path(OUT_DIR, "Report"),
  logs    = file.path(OUT_DIR, "Logs")
)

DAYS <- 1:20

# CV
K_FOLDS <- 5
SEED <- 123

# BART (dbarts)
BART_NSKIP   <- 400
BART_NDPOST  <- 400
BART_NTREES  <- 300
BART_VERBOSE <- FALSE

# Se TRUE: aggiunge rumore ~N(0, sigma^2) ai draw yhat.test (solo se sigma presente).
# NOTA: in dbarts spesso yhat.test sono già "predictive draws". Tienilo FALSE di default.
ADD_NOISE_USING_SIGMA <- FALSE

# Giorni chiave (data-driven)
EPS_REL_NEAR <- 0.02   # entro 2% dal best
K_PLATEAU    <- 3
PEAK_Q       <- 0.75
PLATEAU_Q    <- 0.25

# DCA
TARGET_T_LIST <- c(75, 100)          # target percentuali finali (capped)
TAUS <- seq(0.05, 0.95, by = 0.05)   # soglie su probabilità
DCA_BB_REPS <- 300                   # Bayesian bootstrap reps
SAVE_DCA_CURVES_FOR_FOLDS <- c(1)    # salva curve solo per alcuni fold

# MULTISTAGE / VOI
WAIT_K_LIST   <- c(1, 2, 3)  # attendi k giorni (d -> d+k)
C_WAIT_PER_DAY <- 0.002      # costo attesa per giorno (unità NB); tarabile
STOP_RULE <- "NB_ROBUST"     # "NB_STAR" oppure "NB_ROBUST"

timestamp <- format(Sys.time(), "%Y-%m-%d_%H%M%S")
LOG_FILE <- file.path(DIRS$logs, paste0("bart_log_", timestamp, ".txt"))

dir_create <- function(path) if (!dir.exists(path)) dir.create(path, recursive = TRUE, showWarnings = FALSE)
invisible(lapply(DIRS, dir_create))

log_line <- function(...) {
  msg <- paste0(...)
  cat(msg, "\n")
  cat(msg, "\n", file = LOG_FILE, append = TRUE)
}

# =========================
# 1) LIBRERIE
# =========================
packages <- c("dplyr","tidyr","readr","ggplot2","stringr","purrr","tibble","dbarts")
install_if_missing <- function(pkgs){
  for(p in pkgs){
    if(!requireNamespace(p, quietly = TRUE)){
      install.packages(p, dependencies = TRUE)
    }
  }
}
install_if_missing(packages)

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(readr); library(ggplot2)
  library(stringr); library(purrr); library(tibble); library(dbarts)
})

theme_set(theme_minimal(base_size = 12))

# =========================
# 2) HELPERS: trasformazioni + metriche + robustezza
# =========================

# ---- Trasformazione stabile: da y=log1p(share) a pct capped senza overflow
# input: matrix [S x N] (o vettore), output: matrix [S x N]
y_to_pct_cap_safe <- function(y_draws) {
  y_mat <- as.matrix(y_draws)
  storage.mode(y_mat) <- "double"
  # clamp per evitare expm1 overflow
  share <- expm1(pmin(y_mat, 50))
  share[!is.finite(share)] <- NA_real_
  share <- pmax(share, 0)
  pct <- 100 * pmin(share, 1)
  pct[!is.finite(pct)] <- NA_real_
  pct
}

# per costruire pct_true dal vettore share_complete
share_to_pct_cap <- function(share) {
  share <- suppressWarnings(as.numeric(share))
  out <- 100 * pmax(share, 0)
  out[!is.finite(out)] <- NA_real_
  pmin(100, out)
}

# ---- Metriche
rmse <- function(a, p) sqrt(mean((a - p)^2, na.rm = TRUE))
mae  <- function(a, p) mean(abs(a - p), na.rm = TRUE)
r2   <- function(a, p){
  ok <- is.finite(a) & is.finite(p)
  a <- a[ok]; p <- p[ok]
  ss_res <- sum((a - p)^2)
  ss_tot <- sum((a - mean(a))^2)
  if (!is.finite(ss_tot) || ss_tot <= 0) return(NA_real_)
  1 - ss_res/ss_tot
}

coverage <- function(y_true, lo, hi){
  ok <- is.finite(y_true) & is.finite(lo) & is.finite(hi)
  if (sum(ok) < 10) {
    log_line("Coverage skipped: insufficient finite values (", sum(ok), ")")
    return(NA_real_)
  }
  mean(y_true[ok] >= lo[ok] & y_true[ok] <= hi[ok])
}

mean_or_na <- function(x){
  x <- x[is.finite(x)]
  if (length(x) == 0) return(NA_real_)
  mean(x)
}

width_int <- function(lo, hi){
  ok <- is.finite(lo) & is.finite(hi)
  if (!any(ok)) return(NA_real_)
  mean(hi[ok] - lo[ok])
}

# ---- Robustezza draw posterior predittivi (dbarts può restituire vettori)
as_draws_matrix <- function(draws, ndpost) {
  if (is.matrix(draws)) {
    if (nrow(draws) == ndpost) return(draws)
    if (ncol(draws) == ndpost) return(t(draws))
    log_line("WARN: unexpected draws matrix shape: ", nrow(draws), " x ", ncol(draws))
    return(draws)
  }
  v <- as.numeric(draws)
  if (length(v) == ndpost) return(matrix(v, nrow = ndpost, ncol = 1))
  matrix(v, nrow = 1, ncol = length(v))
}

col_means_safe <- function(mat) colMeans(as.matrix(mat), na.rm = TRUE)
col_quantile <- function(mat, probs) apply(as.matrix(mat), 2, quantile, probs = probs, na.rm = TRUE)

# ---- Output helpers
save_csv <- function(df, filename){
  fp <- file.path(DIRS$tables, filename)
  readr::write_csv(df, fp)
  log_line("✓ Salvata tabella: ", fp)
  fp
}
save_plot <- function(p, filename, w = 10, h = 6, dpi = 320){
  fp <- file.path(DIRS$figures, filename)
  ggsave(fp, p, width = w, height = h, dpi = dpi)
  log_line("✓ Salvata figura: ", fp)
  fp
}

# =========================
# 2b) HELPERS: selezione giorni chiave (non arbitraria)
# =========================
is_local_max <- function(x, i) {
  if (length(x) < 3) return(FALSE)
  if (i <= 1 || i >= length(x)) return(FALSE)
  if (is.na(x[i-1]) || is.na(x[i]) || is.na(x[i+1])) return(FALSE)
  x[i] > x[i-1] && x[i] > x[i+1]
}

get_deltas_and_peaks <- function(day_df, metric = "rmse_pct", span = 0.6, peak_quantile = 0.75){
  stopifnot(all(c("day", metric) %in% names(day_df)))
  ds <- day_df %>% arrange(day)
  
  smooth_col <- paste0(metric, "_smooth")
  
  # smoothing robusto (loess può fallire)
  ds[[smooth_col]] <- ds[[metric]]
  lo_ok <- FALSE
  if (sum(is.finite(ds[[metric]])) >= 5) {
    try({
      lo <- loess(as.formula(paste0(metric, " ~ day")), data = ds, span = span)
      ds[[smooth_col]] <- as.numeric(predict(lo, newdata = ds$day))
      lo_ok <- TRUE
    }, silent = TRUE)
  }
  if (!lo_ok) {
    # fallback: nessuno smoothing (usa metrica grezza)
    ds[[smooth_col]] <- ds[[metric]]
  }
  
  # Δ(d) = M(d-1) - M(d)
  ds$delta <- NA_real_
  for (i in 2:nrow(ds)) ds$delta[i] <- ds[[smooth_col]][i-1] - ds[[smooth_col]][i]
  
  deltas <- ds$delta[is.finite(ds$delta)]
  if (length(deltas) == 0) {
    ds$peak <- FALSE
    return(list(ds = ds, peak_threshold = NA_real_))
  }
  
  thr <- as.numeric(quantile(deltas, probs = peak_quantile, na.rm = TRUE))
  if (!is.finite(thr)) {
    ds$peak <- FALSE
    return(list(ds = ds, peak_threshold = NA_real_))
  }
  
  ds$peak <- FALSE
  for (i in seq_len(nrow(ds))) {
    if (!is.na(ds$delta[i]) &&
        is.finite(ds$delta[i]) &&
        ds$delta[i] >= thr &&
        isTRUE(is_local_max(ds$delta, i))) {
      ds$peak[i] <- TRUE
    }
  }
  
  list(ds = ds, peak_threshold = thr)
}

select_key_days <- function(ds, metric = "rmse_pct", eps_rel = 0.02, k_plateau = 3, plateau_quantile = 0.25){
  stopifnot(all(c("day", metric, "delta", "peak") %in% names(ds)))
  ds <- ds %>% arrange(day)
  
  best_day <- ds$day[which.min(ds[[metric]])]
  best_val <- min(ds[[metric]], na.rm = TRUE)
  
  eps <- eps_rel * best_val
  near_day <- min(ds$day[ds[[metric]] <= best_val + eps], na.rm = TRUE)
  
  deltas <- ds$delta[is.finite(ds$delta)]
  tau <- as.numeric(quantile(deltas, plateau_quantile, na.rm = TRUE))
  
  plateau_day <- NA_integer_
  for (d in ds$day[ds$day >= 2]) {
    idx <- which(ds$day %in% d:(d + k_plateau - 1))
    if (length(idx) == k_plateau) {
      cond <- all(ds$delta[idx] <= tau, na.rm = TRUE)
      if (isTRUE(cond)) { plateau_day <- d; break }
    }
  }
  
  list(
    best_day = best_day,
    near_day = near_day,
    plateau_day = plateau_day,
    peak_days = ds$day[ds$peak],
    eps = eps,
    tau = tau
  )
}

# =========================
# 2c) HELPERS: DCA + Bayesian bootstrap
# =========================
posterior_event_prob <- function(pct_draws, T) col_means_safe(pct_draws >= T)

net_benefit <- function(event_true, decision_pos, tau) {
  N <- length(event_true)
  if (N == 0) return(NA_real_)
  TP <- sum(decision_pos == 1 & event_true == 1)
  FP <- sum(decision_pos == 1 & event_true == 0)
  (TP / N) - (FP / N) * (tau / (1 - tau))
}

dca_curve <- function(pi_hat, event_true, taus) {
  bind_rows(lapply(taus, function(tau){
    dec <- as.integer(pi_hat >= tau)
    tibble(
      tau = tau,
      NB = net_benefit(event_true, dec, tau),
      treat_rate = mean(dec),
      TP = sum(dec == 1 & event_true == 1),
      FP = sum(dec == 1 & event_true == 0)
    )
  }))
}

dca_baselines <- function(event_true, taus) {
  N <- length(event_true)
  TP_all <- sum(event_true == 1)
  FP_all <- sum(event_true == 0)
  NB_all <- sapply(taus, function(tau) (TP_all/N) - (FP_all/N) * (tau/(1-tau)))
  tibble(tau = taus, NB_none = 0, NB_all = NB_all, prevalence = mean(event_true==1))
}

dca_bayes_bootstrap <- function(pi_hat, event_true, taus, B = 300, seed = 123) {
  set.seed(seed)
  N <- length(event_true)
  if (N == 0) return(tibble(tau = taus, NB_low90 = NA_real_, NB_high90 = NA_real_))
  
  nb_weighted <- function(event_true, decision_pos, tau, w) {
    TPw <- sum(w * (decision_pos==1 & event_true==1))
    FPw <- sum(w * (decision_pos==1 & event_true==0))
    TPw - FPw * (tau/(1-tau))
  }
  
  nb_draws <- matrix(NA_real_, nrow = B, ncol = length(taus))
  for (b in 1:B) {
    g <- rgamma(N, shape = 1, rate = 1)
    w <- g / sum(g)
    for (j in seq_along(taus)) {
      tau <- taus[j]
      dec <- as.integer(pi_hat >= tau)
      nb_draws[b, j] <- nb_weighted(event_true, dec, tau, w)
    }
  }
  
  tibble(
    tau = taus,
    NB_low90  = apply(nb_draws, 2, quantile, probs = 0.05, na.rm = TRUE),
    NB_high90 = apply(nb_draws, 2, quantile, probs = 0.95, na.rm = TRUE)
  )
}

pick_tau_star   <- function(dca_df) dca_df$tau[which.max(dca_df$NB)]
pick_tau_robust <- function(dca_ci_df) dca_ci_df$tau[which.max(dca_ci_df$NB_low90)]

get_nb_star <- function(dca_point) {
  tau_star <- pick_tau_star(dca_point)
  NB_star  <- dca_point$NB[dca_point$tau == tau_star][1]
  list(tau_star = tau_star, NB_star = NB_star)
}

get_nb_robust <- function(dca_point, dca_ci) {
  tau_rob <- pick_tau_robust(dca_ci)
  NB_rob  <- dca_point$NB[dca_point$tau == tau_rob][1]
  NB_low  <- dca_ci$NB_low90[dca_ci$tau == tau_rob][1]
  list(tau_robust = tau_rob, NB_robust = NB_rob, NB_low90_at_robust = NB_low)
}

# =========================
# 3) LOAD + PREP BASE
# =========================
log_line(strrep("=", 78))
log_line("BART DAILY MODELS + DCA RUN: ", timestamp)
log_line("Data path: ", DATA_PATH)
log_line(strrep("=", 78))

stopifnot(file.exists(DATA_PATH))
df <- read.csv(DATA_PATH, stringsAsFactors = FALSE)

# outcome finale
df <- df %>%
  mutate(
    share_complete = suppressWarnings(as.numeric(share_complete)),
    y = suppressWarnings(as.numeric(y)),
    y = ifelse(is.na(y) & is.finite(share_complete) & share_complete >= 0, log1p(share_complete), y),
    pct_true = share_to_pct_cap(share_complete)
  ) %>%
  filter(is.finite(y), is.finite(pct_true))

log_line("N dopo filtro outcome: ", nrow(df))

# =========================
# 4) FEATURE SET (NO LEAKAGE) + durata osservata
# =========================
pct_cols_all <- paste0("pct_goal_cum_d", DAYS)
don_cols_all <- paste0("donors_cum_d", DAYS)

if (!all(pct_cols_all %in% names(df))) log_line("ATTENZIONE: mancano colonne pct_goal_cum_d*.")
if (!all(don_cols_all %in% names(df))) log_line("ATTENZIONE: mancano colonne donors_cum_d*.")

static_num_candidates <- c(
  "goal_amount",
  "campaign_duration","duration_days",
  "text_length","n_caratteri","n_parole","n_frasi",
  "parole_per_frase","caratteri_per_parola",
  "positive_emotions","negative_emotions","emotional_balance",
  "intensita_emotiva_totale","densita_emotiva",
  "disaster"
)
static_cat_candidates <- c("category_map","goal_category","duration_category")

leakage_candidates <- c(
  "current_amount","donor_count","avg_donation",
  "money_difference","money_missing","money_excess","donors_per_goal",
  "performance","goal_performance","goal_reached","threshold",
  "share_complete","log_share_complete","y"
)
leakage_cols <- intersect(leakage_candidates, names(df))
if (length(leakage_cols) > 0) {
  log_line("INFO: colonne potenzialmente leakage presenti (IGNORATE): ", paste(leakage_cols, collapse=", "))
}

static_num <- intersect(static_num_candidates, names(df))
static_cat <- intersect(static_cat_candidates, names(df))

if ("campaign_duration" %in% static_num && "duration_days" %in% static_num) {
  static_num <- setdiff(static_num, "duration_days")
}

df <- df %>% mutate(across(all_of(static_num), ~ suppressWarnings(as.numeric(.x))))
for (cc in static_cat) {
  df[[cc]] <- as.factor(ifelse(is.na(df[[cc]]) | df[[cc]]=="", "Unknown", df[[cc]]))
}

impute_median <- function(x){
  x <- suppressWarnings(as.numeric(x))
  med <- median(x, na.rm = TRUE)
  x[!is.finite(x)] <- med
  x
}
df <- df %>% mutate(across(all_of(static_num), impute_median))

# =========================
# 5) CV SPLITS (stratificati e robusti)
# =========================
set.seed(SEED)

make_stratified_folds <- function(data, k, strata_col = NULL, seed = 123) {
  set.seed(seed)
  n <- nrow(data)
  fold <- rep(NA_integer_, n)
  
  if (!is.null(strata_col) && strata_col %in% names(data)) {
    tmp <- data %>%
      mutate(.row_id = row_number()) %>%
      group_by(.data[[strata_col]]) %>%
      summarise(ids = list(sample(.row_id)), .groups="drop") %>%
      pull(ids)
    
    for (grp in tmp) {
      kk <- length(grp)
      ass <- rep(1:k, length.out = kk)
      fold[grp] <- sample(ass, size = kk, replace = FALSE)
    }
  } else {
    fold <- sample(rep(1:k, length.out = n))
  }
  
  if (any(is.na(fold))) {
    miss <- which(is.na(fold))
    fold[miss] <- sample(rep(1:k, length.out = length(miss)))
  }
  fold
}

STRATA <- if ("category_map" %in% names(df)) "category_map" else NULL
df$fold <- make_stratified_folds(df, K_FOLDS, strata_col = STRATA, seed = SEED)
log_line("CV folds: ", K_FOLDS, " | seed=", SEED, " | strata=", ifelse(is.null(STRATA),"none",STRATA))

# =========================
# 6) DESIGN MATRIX (solo info fino al giorno d)
# =========================
# include:
# - static features
# - early traction cumulata in finestra [d-window+1, d]
# - speed = differenza tra d e d-1 (entro d)
# extra robustezza:
# - early cumulate NA->0
# - forzo monotonicità (cummax) sulle cumulate dentro la finestra

force_row_cummax <- function(mat) {
  m <- as.matrix(mat)
  if (ncol(m) <= 1) return(m)
  for (j in 2:ncol(m)) m[, j] <- pmax(m[, j], m[, j-1], na.rm = TRUE)
  m
}

build_design <- function(data, day, static_num, static_cat, window = 3) {
  d0 <- max(1, day - (window - 1))
  days_w <- d0:day
  
  pct_cols <- paste0("pct_goal_cum_d", days_w)
  don_cols <- paste0("donors_cum_d", days_w)
  
  early_cols <- intersect(c(pct_cols, don_cols), names(data))
  if (length(early_cols) < 2) stop("Mancano colonne early traction per day=", day)
  
  tmp <- data
  
  # early: cast numeric + NA->0
  for (ec in early_cols) {
    tmp[[ec]] <- suppressWarnings(as.numeric(tmp[[ec]]))
    tmp[[ec]][!is.finite(tmp[[ec]])] <- 0
  }
  
  # monotonicità sulle cumulate (se la finestra include più giorni)
  pct_mat <- as.matrix(tmp[, intersect(pct_cols, names(tmp)), drop=FALSE])
  don_mat <- as.matrix(tmp[, intersect(don_cols, names(tmp)), drop=FALSE])
  if (ncol(pct_mat) >= 2) tmp[, intersect(pct_cols, names(tmp))] <- force_row_cummax(pct_mat)
  if (ncol(don_mat) >= 2) tmp[, intersect(don_cols, names(tmp))] <- force_row_cummax(don_mat)
  
  # speed (delta entro d)
  pct_d  <- paste0("pct_goal_cum_d", day)
  pct_dm <- paste0("pct_goal_cum_d", max(1, day-1))
  don_d  <- paste0("donors_cum_d", day)
  don_dm <- paste0("donors_cum_d", max(1, day-1))
  
  tmp$pct_goal_speed <- tmp[[pct_d]] - tmp[[pct_dm]]
  tmp$donors_speed   <- tmp[[don_d]] - tmp[[don_dm]]
  
  tmp$pct_goal_speed[!is.finite(tmp$pct_goal_speed)] <- 0
  tmp$donors_speed[!is.finite(tmp$donors_speed)] <- 0
  
  rhs <- c(static_num, static_cat, early_cols, "pct_goal_speed", "donors_speed")
  rhs <- rhs[rhs %in% names(tmp)]
  
  # safety: leak check finale
  leak_in_rhs <- intersect(rhs, c("current_amount","donor_count","money_difference","money_missing","money_excess","avg_donation"))
  if (length(leak_in_rhs) > 0) stop("LEAKAGE DETECTED in RHS: ", paste(leak_in_rhs, collapse=", "))
  
  fml <- as.formula(paste0("~ ", paste(rhs, collapse = " + ")))
  X <- model.matrix(fml, data = tmp)
  X[!is.finite(X)] <- 0
  if ("(Intercept)" %in% colnames(X)) X <- X[, colnames(X)!="(Intercept)", drop=FALSE]
  
  list(X = X, rhs = rhs, fml = fml, early_cols = early_cols, colnames = colnames(X))
}

# =========================
# 7) LOOP: 20 giorni x K-fold (CV) + DCA
# =========================
log_line(strrep("-", 78))
log_line("Fit BART per giorni 1..20 con CV + DCA + VOI")
log_line(strrep("-", 78))

all_fold_metrics <- vector("list", length(DAYS))
all_day_summary  <- vector("list", length(DAYS))
pred_store <- list()
dca_fold_store <- list()
feature_store <- list()

for (di in seq_along(DAYS)) {
  d <- DAYS[di]
  log_line("\n[DAY ", d, "]")
  
  des <- build_design(df, d, static_num, static_cat, window = 3)
  X_all <- des$X
  y_all <- df$y
  pct_true_all <- df$pct_true
  
  # salva features per day
  feature_store[[di]] <- tibble(day = d, feature = des$colnames)
  
  fold_metrics <- vector("list", K_FOLDS)
  
  for (k in 1:K_FOLDS) {
    idx_test  <- which(df$fold == k)
    idx_train <- which(df$fold != k)
    
    if (length(idx_test) == 0 || length(idx_train) == 0) {
      log_line("  (skip) fold ", k, " vuoto (test o train).")
      next
    }
    
    X_train <- X_all[idx_train, , drop=FALSE]
    X_test  <- X_all[idx_test,  , drop=FALSE]
    y_train <- y_all[idx_train]
    y_test  <- y_all[idx_test]
    pct_test <- pct_true_all[idx_test]
    
    fit <- dbarts::bart(
      x.train = X_train,
      y.train = y_train,
      x.test  = X_test,
      nskip   = BART_NSKIP,
      ndpost  = BART_NDPOST,
      ntree   = BART_NTREES,
      verbose = BART_VERBOSE
    )
    
    # posterior draws
    y_draws <- as_draws_matrix(fit$yhat.test, ndpost = BART_NDPOST)
    
    # opzionale: aggiungi rumore se vuoi "predictive" e sei sicuro yhat.test è latente
    if (isTRUE(ADD_NOISE_USING_SIGMA) && !is.null(fit$sigma)) {
      S <- nrow(y_draws)
      Ntest <- ncol(y_draws)
      
      sigma_vec <- fit$sigma
      if (length(sigma_vec) == 1) sigma_vec <- rep(sigma_vec, S)
      if (length(sigma_vec) != S) sigma_vec <- rep(sigma_vec[1], S)
      
      set.seed(SEED + d*1000 + k*10)
      eps <- matrix(rnorm(S*Ntest, mean = 0, sd = rep(sigma_vec, times = Ntest)), nrow = S, ncol = Ntest)
      y_draws <- y_draws + eps
    }
    
    y_pred_mean <- col_means_safe(y_draws)
    
    y_lo80 <- col_quantile(y_draws, probs = 0.10)
    y_hi80 <- col_quantile(y_draws, probs = 0.90)
    y_lo95 <- col_quantile(y_draws, probs = 0.025)
    y_hi95 <- col_quantile(y_draws, probs = 0.975)
    
    # scala interpretabile: pct capped
    pct_draws <- y_to_pct_cap_safe(y_draws)
    pct_pred_mean <- col_means_safe(pct_draws)
    
    pct_lo80 <- col_quantile(pct_draws, probs = 0.10)
    pct_hi80 <- col_quantile(pct_draws, probs = 0.90)
    pct_lo95 <- col_quantile(pct_draws, probs = 0.025)
    pct_hi95 <- col_quantile(pct_draws, probs = 0.975)
    
    # metriche
    m <- tibble(
      day = d,
      fold = k,
      n_test = length(idx_test),
      
      rmse_y = rmse(y_test, y_pred_mean),
      mae_y  = mae(y_test, y_pred_mean),
      r2_y   = r2(y_test, y_pred_mean),
      
      rmse_pct = rmse(pct_test, pct_pred_mean),
      mae_pct  = mae(pct_test, pct_pred_mean),
      
      cov80_y   = coverage(y_test, y_lo80, y_hi80),
      cov95_y   = coverage(y_test, y_lo95, y_hi95),
      wid80_y   = width_int(y_lo80, y_hi80),
      wid95_y   = width_int(y_lo95, y_hi95),
      
      cov80_pct = coverage(pct_test, pct_lo80, pct_hi80),
      cov95_pct = coverage(pct_test, pct_lo95, pct_hi95),
      wid80_pct = width_int(pct_lo80, pct_hi80),
      wid95_pct = width_int(pct_lo95, pct_hi95)
    )
    fold_metrics[[k]] <- m
    
    # DCA + bande + salva curve (solo per alcuni fold)
    for (T_target in TARGET_T_LIST) {
      pi_hat <- posterior_event_prob(pct_draws, T = T_target)
      event_true <- as.integer(pct_test >= T_target)
      
      dca_point <- dca_curve(pi_hat, event_true, TAUS)
      dca_base  <- dca_baselines(event_true, TAUS)
      dca_ci    <- dca_bayes_bootstrap(
        pi_hat, event_true, TAUS,
        B = DCA_BB_REPS,
        seed = SEED + d*1000 + k*10 + T_target
      )
      
      star <- get_nb_star(dca_point)
      rob  <- get_nb_robust(dca_point, dca_ci)
      
      # massimi (per decisional day)
      max_NB_point <- max(dca_point$NB, na.rm = TRUE)
      max_NB_low90 <- max(dca_ci$NB_low90, na.rm = TRUE)
      
      dca_fold_store[[length(dca_fold_store) + 1]] <- tibble(
        day = d, fold = k, T = T_target,
        
        tau_star   = star$tau_star,
        tau_robust = rob$tau_robust,
        
        NB_star   = star$NB_star,
        NB_robust = rob$NB_robust,
        NB_low90_at_robust = rob$NB_low90_at_robust,
        
        max_NB_point = max_NB_point,
        max_NB_low90 = max_NB_low90,
        
        NB_all_at_star = dca_base$NB_all[dca_base$tau == star$tau_star][1],
        NB_all_at_robust = dca_base$NB_all[dca_base$tau == rob$tau_robust][1],
        
        prevalence = mean(event_true == 1),
        treat_rate_star = dca_point$treat_rate[dca_point$tau == star$tau_star][1],
        treat_rate_rob  = dca_point$treat_rate[dca_point$tau == rob$tau_robust][1]
      )
      
      if (k %in% SAVE_DCA_CURVES_FOR_FOLDS) {
        cur <- dca_point %>%
          left_join(dca_ci, by = "tau") %>%
          left_join(dca_base %>% select(tau, NB_all, NB_none), by = "tau") %>%
          mutate(day = d, fold = k, T = T_target)
        save_csv(cur, paste0("dca_curve_day", d, "_T", T_target, "_fold", k, ".csv"))
      }
    }
    
    # salva predizioni (solo fold 1)
    if (k == 1) {
      pred_store[[paste0("day_", d)]] <- tibble(
        day = d, fold = k,
        y_true = y_test, y_pred = y_pred_mean,
        pct_true = pct_test, pct_pred = pct_pred_mean
      )
    }
  }
  
  day_fold_df <- bind_rows(fold_metrics)
  all_fold_metrics[[di]] <- day_fold_df
  
  day_sum <- day_fold_df %>%
    group_by(day) %>%
    summarise(
      folds = n_distinct(fold),
      n_test_tot = sum(n_test),
      
      rmse_y = mean(rmse_y, na.rm=TRUE),
      mae_y  = mean(mae_y,  na.rm=TRUE),
      r2_y   = mean(r2_y,   na.rm=TRUE),
      
      rmse_pct = mean(rmse_pct, na.rm=TRUE),
      mae_pct  = mean(mae_pct,  na.rm=TRUE),
      
      cov80_y = mean_or_na(cov80_y),
      cov95_y = mean_or_na(cov95_y),
      wid80_y = mean_or_na(wid80_y),
      wid95_y = mean_or_na(wid95_y),
      
      cov80_pct = mean_or_na(cov80_pct),
      cov95_pct = mean_or_na(cov95_pct),
      wid80_pct = mean_or_na(wid80_pct),
      wid95_pct = mean_or_na(wid95_pct),
      .groups = "drop"
    )
  all_day_summary[[di]] <- day_sum
  
  log_line("  rmse_pct=", round(day_sum$rmse_pct,3),
           " | rmse_y=", round(day_sum$rmse_y,3),
           " | cov95_pct=", round(day_sum$cov95_pct,3))
}

cv_fold_metrics <- bind_rows(all_fold_metrics)
cv_day_metrics  <- bind_rows(all_day_summary) %>% arrange(day)
dca_fold_metrics <- bind_rows(dca_fold_store)
features_by_day <- bind_rows(feature_store)

save_csv(cv_fold_metrics, "cv_fold_metrics_day1_20.csv")
save_csv(cv_day_metrics,  "cv_day_metrics_summary_day1_20.csv")
save_csv(dca_fold_metrics,"dca_fold_metrics_day1_20.csv")
save_csv(features_by_day, "features_used_by_day.csv")

# =========================
# 8) DCA: decisional day + utility per day + VOI
# =========================
dca_day_dec <- dca_fold_metrics %>%
  group_by(day, T) %>%
  summarise(
    max_NB_low90_mean = mean(max_NB_low90, na.rm = TRUE),
    max_NB_point_mean = mean(max_NB_point, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(T, day) %>%
  group_by(T) %>%
  summarise(
    d_dec = {
      dd <- day[max_NB_low90_mean > 0]
      if (length(dd) == 0) NA_integer_ else min(dd)
    },
    .groups = "drop"
  )
save_csv(dca_day_dec, "decisional_day_by_target.csv")

# Utility per giorno = max NB (o max NB_low90), media sui fold
dca_day_utility <- dca_fold_metrics %>%
  group_by(day, T) %>%
  summarise(
    NB_star_mean       = mean_or_na(NB_star),
    NB_robust_mean     = mean_or_na(NB_robust),
    NB_low90_rob_mean  = mean_or_na(NB_low90_at_robust),
    max_NB_point_mean  = mean_or_na(max_NB_point),
    max_NB_low90_mean  = mean_or_na(max_NB_low90),
    prevalence_mean    = mean_or_na(prevalence),
    .groups = "drop"
  ) %>%
  arrange(T, day)
save_csv(dca_day_utility, "dca_day_utility_summary.csv")

compute_voi <- function(utility_df, wait_k_list, c_wait_per_day, use = c("max_NB_point_mean","max_NB_low90_mean")) {
  use <- match.arg(use)
  out <- list()
  for (Tt in sort(unique(utility_df$T))) {
    uT <- utility_df %>% filter(T == Tt) %>% arrange(day)
    for (k in wait_k_list) {
      tmp <- uT %>%
        mutate(
          day_next = day + k,
          U_now  = .data[[use]],
          U_next = uT[[use]][match(day + k, uT$day)],
          VOI = U_next - U_now - k * c_wait_per_day
        ) %>%
        filter(day_next %in% uT$day) %>%
        mutate(wait_k = k, utility_used = use, T = Tt)
      out[[length(out)+1]] <- tmp
    }
  }
  bind_rows(out)
}

voi_star  <- compute_voi(dca_day_utility, WAIT_K_LIST, C_WAIT_PER_DAY, use="max_NB_point_mean")
voi_rob   <- compute_voi(dca_day_utility, WAIT_K_LIST, C_WAIT_PER_DAY, use="max_NB_low90_mean")

save_csv(voi_star, "voi_summary_maxNBpoint.csv")
save_csv(voi_rob,  "voi_summary_maxNBlow90.csv")

pick_stop_day <- function(voi_df, T_target) {
  v <- voi_df %>% filter(T == T_target)
  agg <- v %>%
    group_by(day) %>%
    summarise(VOI_min = min(VOI, na.rm=TRUE), .groups="drop") %>%
    arrange(day)
  stop_day <- agg$day[which(agg$VOI_min <= 0)[1]]
  tibble(T = T_target, stop_day = stop_day)
}

stop_tbl <- bind_rows(lapply(TARGET_T_LIST, function(Tt){
  if (STOP_RULE == "NB_STAR") pick_stop_day(voi_star, Tt) else pick_stop_day(voi_rob, Tt)
}))
save_csv(stop_tbl, "stop_waiting_day.csv")
log_line("STOP-WAITING days (rule=", STOP_RULE, ", c_wait=", C_WAIT_PER_DAY, "): ",
         paste0("T", stop_tbl$T, "->d", stop_tbl$stop_day, collapse=" | "))

# =========================
# 9) SELEZIONE GIORNI CHIAVE (data-driven su RMSE_pct)
# =========================
tmp  <- get_deltas_and_peaks(cv_day_metrics, metric = "rmse_pct", span = 0.6, peak_quantile = PEAK_Q)
ds   <- tmp$ds
keys <- select_key_days(ds, metric = "rmse_pct", eps_rel = EPS_REL_NEAR, k_plateau = K_PLATEAU, plateau_quantile = PLATEAU_Q)

key_days_tbl <- tibble(
  best_day = keys$best_day,
  near_day = keys$near_day,
  plateau_day = keys$plateau_day,
  peak_days = paste(keys$peak_days, collapse = ","),
  eps_rel = EPS_REL_NEAR,
  k_plateau = K_PLATEAU,
  peak_quantile = PEAK_Q,
  plateau_quantile = PLATEAU_Q
)
save_csv(key_days_tbl, "key_days_selection.csv")

log_line("\nKEY DAYS (RMSE_pct):")
log_line("  best_day=", keys$best_day,
         " | near_day=", keys$near_day,
         " | plateau_day=", keys$plateau_day,
         " | peak_days=", paste(keys$peak_days, collapse=", "))

# =========================
# 10) FIGURE: qualità predittiva
# =========================
p_rmse <- ggplot(ds, aes(x = day, y = rmse_pct)) +
  geom_line(linewidth = 1) +
  geom_point(aes(shape = peak), size = 2) +
  geom_line(aes(y = rmse_pct_smooth), linewidth = 1.1, linetype = "dashed") +
  geom_vline(xintercept = keys$near_day, linetype = "dotted") +
  labs(
    title = "BART Day 1..20: RMSE su % completamento (capped a 100)",
    subtitle = paste0("near=", keys$near_day,
                      " | best=", keys$best_day,
                      " | plateau=", keys$plateau_day,
                      " | peaks=", paste(keys$peak_days, collapse=", ")),
    x = "Giorno", y = "RMSE (%-points)"
  )
save_plot(p_rmse, "fig_01_rmse_pct_by_day.png", w=10, h=6)

p_delta <- ggplot(ds %>% filter(day>=2), aes(x=day, y=delta)) +
  geom_line(linewidth = 1) +
  geom_point(aes(shape = peak), size = 2) +
  geom_hline(yintercept = tmp$peak_threshold, linetype="dotted") +
  labs(
    title = "Guadagno informativo marginale Δ(d) su RMSE_pct (smoothed)",
    subtitle = "Δ(d) = M(d-1) - M(d). Picchi = giorni con maggiore valore informativo.",
    x="Giorno", y="Δ(d) (positivo = migliora)"
  )
save_plot(p_delta, "fig_02_delta_gain_rmse_pct.png", w=10, h=6)

p_cov <- ggplot(cv_day_metrics, aes(x=day, y=cov95_pct)) +
  geom_line(linewidth=1) +
  geom_point(size=2) +
  scale_y_continuous(limits=c(0,1)) +
  labs(
    title = "Copertura intervallo predittivo 95% (scala % capped)",
    subtitle = "Valori ~0.95 indicano incertezza ben calibrata (media sui fold).",
    x="Giorno", y="Coverage 95%"
  )
save_plot(p_cov, "fig_03_coverage95_pct_by_day.png", w=10, h=6)

# Scatter per giorni chiave (fold 1)
pick_day <- function(x) if (is.na(x) || length(x)==0) NA_integer_ else as.integer(x)
d_peak1 <- pick_day(keys$peak_days[1])
d_near  <- pick_day(keys$near_day)
d_plat  <- pick_day(keys$plateau_day)

scatter_days <- unique(na.omit(c(d_peak1, d_near, d_plat)))
scatter_df <- bind_rows(pred_store[paste0("day_", scatter_days)])

if (nrow(scatter_df) > 0) {
  scatter_df <- scatter_df %>% mutate(day = factor(day, levels = scatter_days))
  p_sc <- ggplot(scatter_df, aes(x = pct_true, y = pct_pred)) +
    geom_point(alpha = 0.25) +
    geom_abline(slope=1, intercept=0, linetype="dashed") +
    facet_wrap(~ day, ncol = length(scatter_days)) +
    coord_cartesian(xlim=c(0,100), ylim=c(0,100)) +
    labs(
      title = "Predetto vs Vero (% completamento capped) - giorni chiave (fold 1)",
      subtitle = "Linea tratteggiata = perfetta previsione (y=x).",
      x = "Vero (%)", y = "Predetto (%)"
    )
  save_plot(p_sc, "fig_04_scatter_key_days_pct.png", w=11, h=4.5)
} else {
  log_line("Skip scatter: nessuna predizione salvata.")
}

# =========================
# 10b) FIGURE: DCA (decision rule)
# =========================
plot_dca_from_csv <- function(day, T_target, fold_to_plot = 1) {
  fp <- file.path(DIRS$tables, paste0("dca_curve_day", day, "_T", T_target, "_fold", fold_to_plot, ".csv"))
  if (!file.exists(fp)) return(NULL)
  dd <- readr::read_csv(fp, show_col_types = FALSE)
  
  ggplot(dd, aes(x = tau)) +
    geom_line(aes(y = NB), linewidth=1) +
    geom_ribbon(aes(ymin = NB_low90, ymax = NB_high90), alpha=0.15) +
    geom_line(aes(y = NB_all), linetype="dashed") +
    geom_hline(yintercept = 0, linetype="dotted") +
    labs(
      title = paste0("Decision Curve (Day ", day, ", Target ", T_target, "%)"),
      subtitle = "NB modello (linea), banda 90% (Bayesian bootstrap), treat-all (tratteggiata), treat-none (0).",
      x = "Soglia τ su π = P(pct_final >= T | X_<=d)", y = "Net Benefit"
    )
}

dca_days_to_plot <- unique(na.omit(c(keys$near_day, keys$plateau_day)))
for (dday in dca_days_to_plot) {
  for (T_target in TARGET_T_LIST) {
    p <- plot_dca_from_csv(dday, T_target, fold_to_plot = SAVE_DCA_CURVES_FOR_FOLDS[1])
    if (!is.null(p)) save_plot(p, paste0("fig_05_dca_day", dday, "_T", T_target, ".png"), w=10, h=6)
  }
}

# =========================
# 10c) FIGURE: VOI
# =========================
plot_voi <- function(voi_df, T_target, utility_label){
  dd <- voi_df %>% filter(T == T_target)
  ggplot(dd, aes(x=day, y=VOI, linetype=factor(wait_k))) +
    geom_hline(yintercept=0, linetype="dashed") +
    geom_line(linewidth=1) +
    geom_point(size=1.5) +
    labs(
      title = paste0("Value of Waiting (T=", T_target, "%) using ", utility_label),
      subtitle = paste0("VOI(d,k)=U(d+k)-U(d)-k*C_wait,  C_wait=", C_WAIT_PER_DAY),
      x="Giorno d", y="VOI (Δ utilità attesa)", linetype="attendi k giorni"
    )
}

for (Tt in TARGET_T_LIST) {
  save_plot(plot_voi(voi_star, Tt, "max NB point"), paste0("fig_06_voi_T", Tt, "_maxNBpoint.png"), w=10, h=6)
  save_plot(plot_voi(voi_rob,  Tt, "max NB low90"), paste0("fig_07_voi_T", Tt, "_maxNBlow90.png"), w=10, h=6)
}

# =========================
# 11) REPORT (Markdown + HTML opzionale)
# =========================
report_path_md <- file.path(DIRS$report, paste0("report_bart_day1_20_", timestamp, ".md"))

top_days <- cv_day_metrics %>%
  arrange(rmse_pct) %>%
  slice(1:10) %>%
  mutate(across(where(is.numeric), ~ round(.x, 4)))

dca_key_summary <- dca_fold_metrics %>%
  filter(day %in% dca_days_to_plot) %>%
  group_by(day, T) %>%
  summarise(
    tau_star_mean   = mean(tau_star, na.rm = TRUE),
    tau_rob_mean    = mean(tau_robust, na.rm = TRUE),
    NB_star_mean    = mean(NB_star, na.rm = TRUE),
    NB_rob_mean     = mean(NB_robust, na.rm = TRUE),
    prevalence_mean = mean(prevalence, na.rm = TRUE),
    treat_rate_star = mean(treat_rate_star, na.rm = TRUE),
    treat_rate_rob  = mean(treat_rate_rob, na.rm = TRUE),
    .groups="drop"
  ) %>%
  mutate(across(where(is.numeric), ~ round(.x, 4)))
save_csv(dca_key_summary, "dca_key_days_summary.csv")

# One-glance table: giorni chiave + performance
key_set <- unique(na.omit(c(keys$near_day, keys$best_day, keys$plateau_day, keys$peak_days[1])))
key_perf <- cv_day_metrics %>%
  filter(day %in% key_set) %>%
  select(day, rmse_pct, mae_pct, r2_y, cov95_pct, wid95_pct) %>%
  arrange(day) %>%
  mutate(across(where(is.numeric), ~ round(.x, 4)))
save_csv(key_perf, "key_days_performance_one_glance.csv")

lines <- c(
  paste0("# Report BART day 1..20 + DCA + VOI (", timestamp, ")"),
  "",
  "## 1) Setup",
  paste0("- Dataset: `", DATA_PATH, "`"),
  "- Target: `y_final = log(1 + share_complete_finale)` (finale allo scraped_date)",
  "- Scala interpretativa: `pct_final = min(100, 100*(exp(y)-1))`",
  paste0("- CV: ", K_FOLDS, "-fold (seed=", SEED, ")"),
  paste0("- BART: ntree=", BART_NTREES, ", nskip=", BART_NSKIP, ", ndpost=", BART_NDPOST),
  paste0("- ADD_NOISE_USING_SIGMA: ", ADD_NOISE_USING_SIGMA),
  "",
  "## 2) Vincolo temporale (no leakage)",
  "- Per ogni giorno d uso SOLO feature statiche + early traction cumulata entro d (finestra [d-2,d]) + speed Δ(d).",
  "- Forzo monotonicità sulle cumulate (cummax per riga) per correggere eventuali incoerenze da scraping.",
  "",
  "## 3) Scelta giorni (non arbitraria)",
  "- Metrica primaria: RMSE su % capped.",
  paste0("- Best day: **", keys$best_day, "**"),
  paste0("- Near-optimal day: **", keys$near_day, "**"),
  paste0("- Plateau day: **", ifelse(is.na(keys$plateau_day), "NA", keys$plateau_day), "**"),
  paste0("- Peak days: **", paste(keys$peak_days, collapse=", "), "**"),
  "",
  "### Key results (one glance)",
  "```",
  paste(capture.output(print(key_perf)), collapse = "\n"),
  "```",
  "",
  "## 3b) Primo giorno utile per decidere (criterio DCA robusto)",
  "Definizione: d_dec = min_d { E_fold[max_tau NB_low90(d,tau)] > 0 }.",
  "```",
  paste(capture.output(print(dca_day_dec)), collapse="\n"),
  "```",
  "",
  "## 4) Figure qualità predittiva",
  paste0("![](", file.path("..","Figures","fig_01_rmse_pct_by_day.png"), ")"),
  "",
  paste0("![](", file.path("..","Figures","fig_02_delta_gain_rmse_pct.png"), ")"),
  "",
  paste0("![](", file.path("..","Figures","fig_03_coverage95_pct_by_day.png"), ")"),
  "",
  if (file.exists(file.path(DIRS$figures,"fig_04_scatter_key_days_pct.png")))
    paste0("![](", file.path("..","Figures","fig_04_scatter_key_days_pct.png"), ")")
  else "",
  "",
  "## 5) Decision Curve Analysis (DCA) con contributo bayesiano",
  "- π_i(T)=P(pct_final>=T | X_<=d) dai draw posterior predittivi.",
  "- NB(τ) e banda 90% via Bayesian bootstrap.",
  "",
  "### Riassunto DCA (media sui fold) per i giorni chiave",
  "```",
  paste(capture.output(print(dca_key_summary)), collapse = "\n"),
  "```",
  "",
  "## 6) Decisione multistadio (VOI / stop-waiting)",
  paste0("- C_wait per giorno: ", C_WAIT_PER_DAY),
  paste0("- k considerati: ", paste(WAIT_K_LIST, collapse=", ")),
  paste0("- Regola stop basata su: ", STOP_RULE),
  "",
  "### Stop-waiting day (per target T)",
  "```",
  paste(capture.output(print(stop_tbl)), collapse = "\n"),
  "```",
  ""
)

# aggiungo immagini DCA
for (dday in dca_days_to_plot) {
  for (T_target in TARGET_T_LIST) {
    fig <- file.path(DIRS$figures, paste0("fig_05_dca_day", dday, "_T", T_target, ".png"))
    if (file.exists(fig)) lines <- c(lines, paste0("![](", file.path("..","Figures", basename(fig)), ")"), "")
  }
}

# aggiungo immagini VOI
for (Tt in TARGET_T_LIST) {
  fig1 <- file.path(DIRS$figures, paste0("fig_06_voi_T", Tt, "_maxNBpoint.png"))
  fig2 <- file.path(DIRS$figures, paste0("fig_07_voi_T", Tt, "_maxNBlow90.png"))
  if (file.exists(fig1)) lines <- c(lines, paste0("![](", file.path("..","Figures", basename(fig1)), ")"), "")
  if (file.exists(fig2)) lines <- c(lines, paste0("![](", file.path("..","Figures", basename(fig2)), ")"), "")
}

lines <- c(
  lines,
  "## 7) Top 10 giorni (RMSE_pct più basso)",
  "```",
  paste(capture.output(print(top_days)), collapse = "\n"),
  "```",
  "",
  "## 8) File prodotti",
  paste0("- CV fold metrics: `", file.path(DIRS$tables, "cv_fold_metrics_day1_20.csv"), "`"),
  paste0("- CV day summary  : `", file.path(DIRS$tables, "cv_day_metrics_summary_day1_20.csv"), "`"),
  paste0("- Key days        : `", file.path(DIRS$tables, "key_days_selection.csv"), "`"),
  paste0("- DCA fold metrics: `", file.path(DIRS$tables, "dca_fold_metrics_day1_20.csv"), "`"),
  paste0("- DCA utility     : `", file.path(DIRS$tables, "dca_day_utility_summary.csv"), "`"),
  paste0("- VOI max NB point: `", file.path(DIRS$tables, "voi_summary_maxNBpoint.csv"), "`"),
  paste0("- VOI max NB low90: `", file.path(DIRS$tables, "voi_summary_maxNBlow90.csv"), "`"),
  paste0("- Stop-waiting    : `", file.path(DIRS$tables, "stop_waiting_day.csv"), "`"),
  paste0("- Features by day : `", file.path(DIRS$tables, "features_used_by_day.csv"), "`"),
  paste0("- Log             : `", LOG_FILE, "`"),
  ""
)

writeLines(lines, report_path_md)
log_line("✓ Report markdown scritto: ", report_path_md)

# HTML opzionale
if (requireNamespace("rmarkdown", quietly = TRUE)) {
  rmd_path <- file.path(DIRS$report, paste0("report_bart_day1_20_", timestamp, ".Rmd"))
  rmd_lines <- c(
    "---",
    paste0("title: \"BART Day 1..20 + DCA + VOI Report (", timestamp, ")\""),
    "output: html_document",
    "---",
    "",
    paste(lines, collapse = "\n")
  )
  writeLines(rmd_lines, rmd_path)
  try({
    rmarkdown::render(rmd_path, output_dir = DIRS$report, quiet = TRUE)
    log_line("✓ Report HTML renderizzato in: ", DIRS$report)
  }, silent = TRUE)
}

log_line("\n✓ Fine run.")
log_line("Output root: ", OUT_DIR)

################################################################################
# FINE
################################################################################
