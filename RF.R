################################################################################
# 04_tesim/02_modeling_rf_mcc_auc_exante_day20.R
#
# Tesi Magistrale — Random Forest per classificazione (successo vs insuccesso)
# Obiettivo: dimostrare il "guadagno informativo" passando da
#   (A) EX-ANTE  : solo info disponibili al lancio + durata osservata (campaign_duration)
#   (B) DAY-20   : ex-ante + early traction cumulata entro 20 giorni (pct_goal_cum_d20, donors_cum_d20)
#
# Coerenza con la tua pipeline:
# - INPUT e preprocessing coerenti con EDA_finale.R (stesse colonne/nomenclatura)
# - Data leakage: rimozione esplicita di proxy finali (current_amount, donor_count, share_complete, performance, money_*, ecc.)
# - Validazione: split temporale robusto se created_date presente (come stile EDA/BART), altrimenti split stratificato
# - Class imbalance: class weights + soglia ottimizzata su MCC (Mattews Corr. Coef., robusta a sbilanciamento)
# - Metriche chiave per "informazione acquisita": MCC (con soglia), AUC ROC (threshold-free), PR-AUC, LogLoss/Brier
# - Feature importance: permutation importance (ranger) + stabilità su più run
# - Reporting: confronto EX-ANTE vs DAY-20; importanza per feature e per gruppi (testo, emozioni, monetarie/goal, tempo, traction)
#
# OUTPUT (in ~/04_tesim/RF_files):
#   RF_files/
#     Figures/
#     Tables/
#     Models/
#     Logs/
#     Reports/
################################################################################

# =========================
# 0) CONFIG
# =========================
setwd("~/04_tesim")

INPUT_DATA <- "~/04_tesim/Data_processed/dataset_full_preprocessed.csv"

# Cartella unica richiesta
OUT_ROOT <- "~/04_tesim/RF_files"

# Parsimonia (come nel vecchio file)
MAX_FEATS_EXANTE <- 10
MAX_FEATS_DAY20  <- 12

# Split
TEST_FRAC <- 0.20

# CV nel training (per tuning + OOF threshold)
K_FOLDS <- 5

# Stabilità importanze
SEEDS_STABILITY <- 20
SEED_BASE <- 20260210

# Ranger
NUM_TREES <- 1000
MTRY_CAND <- c(2, 3, 4, 5)
MIN_NODE_CAND <- c(1, 5, 10, 20)
CLASS_W_MULT <- c(1, 2, 3, 5, 8, 10)  # peso classe 1 vs 0

# Soglie
THRESH_GRID <- seq(0.01, 0.80, by = 0.01)

# =========================
# 1) LIBRERIE
# =========================
packages <- c(
  "dplyr","tidyr","readr","stringr","forcats","ggplot2","scales","lubridate",
  "ranger","rsample","purrr",
  "pROC"          # AUC ROC + CI (DeLong)
)

install_if_missing <- function(pkgs){
  for(p in pkgs){
    if(!requireNamespace(p, quietly = TRUE)){
      install.packages(p, dependencies = TRUE)
    }
  }
}
install_if_missing(packages)

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(readr); library(stringr); library(forcats)
  library(ggplot2); library(scales); library(lubridate)
  library(ranger); library(rsample); library(purrr)
  library(pROC)
})

theme_set(theme_minimal(base_size = 12))

# =========================
# 2) CARTELLE OUTPUT (RF_files)
# =========================
dir_create <- function(path) if (!dir.exists(path)) dir.create(path, recursive = TRUE, showWarnings = FALSE)

dirs <- list(
  root   = OUT_ROOT,
  figs   = file.path(OUT_ROOT, "Figures"),
  tables = file.path(OUT_ROOT, "Tables"),
  models = file.path(OUT_ROOT, "Models"),
  logs   = file.path(OUT_ROOT, "Logs"),
  reps   = file.path(OUT_ROOT, "Reports")
)
purrr::walk(dirs, dir_create)

timestamp <- format(Sys.time(), "%Y-%m-%d_%H%M%S")
LOG_FILE <- file.path(dirs$logs, paste0("rf_log_", timestamp, ".txt"))

log_line <- function(...) {
  msg <- paste0(...)
  cat(msg, "\n")
  cat(msg, "\n", file = LOG_FILE, append = TRUE)
}

save_csv <- function(df, filename){
  readr::write_csv(df, file.path(dirs$tables, filename))
  log_line("   ✓ Salvata tabella: ", filename)
}

save_plot <- function(p, filename, w = 10, h = 6, dpi = 320){
  ggsave(filename = file.path(dirs$figs, filename), plot = p, width = w, height = h, dpi = dpi)
  log_line("   ✓ Salvata figura: ", filename)
}

log_line("R sessionInfo():")
log_line(paste(capture.output(sessionInfo()), collapse = "\n"))

# =========================
# 3) METRICHE: confusion, MCC, ecc.
# =========================
confusion_counts <- function(y_true, y_pred){
  stopifnot(length(y_true) == length(y_pred))
  y_true <- as.integer(y_true)
  y_pred <- as.integer(y_pred)
  tibble(
    TP = sum(y_true == 1 & y_pred == 1, na.rm = TRUE),
    TN = sum(y_true == 0 & y_pred == 0, na.rm = TRUE),
    FP = sum(y_true == 0 & y_pred == 1, na.rm = TRUE),
    FN = sum(y_true == 1 & y_pred == 0, na.rm = TRUE)
  )
}

mcc_from_counts <- function(TP, TN, FP, FN){
  TP <- as.numeric(TP); TN <- as.numeric(TN)
  FP <- as.numeric(FP); FN <- as.numeric(FN)
  num <- (TP * TN) - (FP * FN)
  den <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
  if (is.na(den) || den == 0) return(NA_real_)
  num / den
}

metrics_from_counts <- function(ct){
  TP <- ct$TP; TN <- ct$TN; FP <- ct$FP; FN <- ct$FN
  mcc <- mcc_from_counts(TP,TN,FP,FN)
  acc <- (TP + TN) / max((TP+TN+FP+FN), 1)
  prec <- ifelse((TP+FP) == 0, NA_real_, TP/(TP+FP))
  rec  <- ifelse((TP+FN) == 0, NA_real_, TP/(TP+FN))
  spec <- ifelse((TN+FP) == 0, NA_real_, TN/(TN+FP))
  bal_acc <- mean(c(rec, spec), na.rm = TRUE)
  f1 <- ifelse(is.na(prec) | is.na(rec) | (prec+rec)==0, NA_real_, 2*prec*rec/(prec+rec))
  tibble(
    MCC = mcc, Accuracy = acc, Balanced_Accuracy = bal_acc,
    Precision = prec, Recall = rec, Specificity = spec, F1 = f1
  )
}

log_loss <- function(y, p, eps = 1e-15){
  p <- pmin(pmax(p, eps), 1 - eps)
  -mean(y * log(p) + (1-y) * log(1-p), na.rm = TRUE)
}

brier_score <- function(y, p){
  mean((y - p)^2, na.rm = TRUE)
}

best_threshold_mcc <- function(y_true, prob, grid = THRESH_GRID){
  res <- purrr::map_dfr(grid, function(t){
    pred <- ifelse(prob >= t, 1L, 0L)
    ct <- confusion_counts(y_true, pred)
    met <- metrics_from_counts(ct)
    tibble(threshold = t, met, ct)
  })
  res <- res %>% arrange(desc(MCC), desc(Recall), desc(Precision))
  list(best = res[1,], curve = res)
}

# =========================
# 4) COERENZA CON EDA_finale.R: "MODEL READY" + ANTI-LEAKAGE
# =========================
# Qui NON re-ingegnerizziamo tutto: assumiamo che dataset_full_preprocessed.csv
# sia prodotto da EDA_finale.R, e ci limitiamo a:
# - cast numeriche/fattori
# - creare log_goal_amount
# - drop colonne di leakage (snapshot finali / outcome-contaminated)
make_model_ready_like_eda <- function(df){
  
  # --- target
  if (!("goal_reached" %in% names(df))) stop("Manca goal_reached nel dataset (atteso da EDA_finale.R).")
  df <- df %>% mutate(goal_reached = as.integer(goal_reached))
  
  # --- date
  if ("created_date" %in% names(df)) {
    df <- df %>% mutate(created_date_parsed = suppressWarnings(as.Date(substr(as.character(created_date), 1, 10))))
  }
  
  # --- numeriche strutturali (EDA)
  for (v in c("goal_amount","campaign_duration","text_length","parole_per_frase","caratteri_per_parola",
              "emotional_balance","emotional_intensity","emotions_positive","emotions_negative")) {
    if (v %in% names(df)) df[[v]] <- suppressWarnings(as.numeric(df[[v]]))
  }
  
  # --- alcune emozioni raw (se presenti in EDA)
  emo_raw <- c("gratitude","sadness","caring","optimism","joy","fear","anger","love","remorse","disgust",
               "admiration","approval","disapproval","grief","surprise","relief")
  emo_raw <- intersect(emo_raw, names(df))
  for (v in emo_raw) df[[v]] <- suppressWarnings(as.numeric(df[[v]]))
  
  # --- fattori
  if ("category_map" %in% names(df)) df <- df %>% mutate(category_map = as.factor(category_map))
  
  # --- log goal
  if ("goal_amount" %in% names(df)) {
    df <- df %>% mutate(log_goal_amount = log1p(pmax(goal_amount, 0)))
  } else {
    df$log_goal_amount <- NA_real_
  }
  
  # --- ANTI-LEAKAGE: rimuovi proxy finali e variabili non ammesse
  leak_patterns <- c(
    "^current_amount$", "^donor_count$", "^share_complete$", "^performance$",
    "^money_", "^amount_raised_", "^donors_",      # ATTENZIONE: include donors_cum_d20 ecc.
    "^pct_goal_day_",                               # giornalieri
    "^pct_goal_cum_d(2[1-9]|[3-9][0-9])$"            # oltre d20 (se esistono)
  )
  
  # NB: NON vogliamo droppare donors_cum_d20 / pct_goal_cum_d20 nello scenario DAY20.
  # Quindi droppiamo donors_/amount_raised_ SOLO se non sono *_cum_d1..d20.
  # Gestiamo con lista esplicita sotto.
  
  # Drop espliciti noti (safe)
  drop_explicit <- intersect(c("share_complete","performance","goal_performance"), names(df))
  if (length(drop_explicit) > 0) df <- df %>% select(-all_of(drop_explicit))
  
  # Drop snapshot finali tipici GoFundMe (se presenti)
  drop_snapshot <- intersect(c("current_amount","donor_count","raised_amount","raised","donors"), names(df))
  if (length(drop_snapshot) > 0) df <- df %>% select(-all_of(drop_snapshot))
  
  # Drop money_* (se presenti)
  money_cols <- grep("^money_", names(df), value = TRUE)
  if (length(money_cols) > 0) df <- df %>% select(-all_of(money_cols))
  
  # Drop amount_raised_final-ish (se presenti), ma NON gli amount_raised_cum_d*
  amt_bad <- setdiff(grep("^amount_raised_", names(df), value = TRUE),
                     grep("^amount_raised_cum_d\\d+$", names(df), value = TRUE))
  if (length(amt_bad) > 0) df <- df %>% select(-all_of(amt_bad))
  
  # Drop donors_final-ish (se presenti), ma NON donors_cum_d*
  don_bad <- setdiff(grep("^donors_", names(df), value = TRUE),
                     grep("^donors_cum_d\\d+$", names(df), value = TRUE))
  if (length(don_bad) > 0) df <- df %>% select(-all_of(don_bad))
  
  # Drop pct_goal_day_d* (daily non cumulata) se presente
  pct_day <- grep("^pct_goal_day_d\\d+$", names(df), value = TRUE)
  if (length(pct_day) > 0) df <- df %>% select(-all_of(pct_day))
  
  df
}

# Imputazione conservativa: mediana su train per numeriche
fit_imputer <- function(train_df, num_cols){
  meds <- purrr::map_dbl(num_cols, ~median(train_df[[.x]], na.rm = TRUE))
  list(medians = meds, num_cols = num_cols)
}
apply_imputer <- function(df, imp){
  for (v in imp$num_cols) df[[v]] <- ifelse(is.na(df[[v]]), imp$medians[[v]], df[[v]])
  df
}

# =========================
# 5) FEATURE SETS (coerenti con EDA_finale.R)
# =========================
# EX-ANTE: struttura + durata + testo + emozioni (ma parsimonioso)
# DAY-20 : ex-ante + pct_goal_cum_d20 + donors_cum_d20 (se presenti)
get_feature_sets_like_eda <- function(df){
  
  # Candidati ex-ante
  exante_pool <- c(
    "log_goal_amount",
    "category_map",
    "campaign_duration",
    "text_length",
    "parole_per_frase",
    "caratteri_per_parola",
    "emotional_balance",
    "emotional_intensity",
    "gratitude",
    "sadness"
  )
  exante_pool <- exante_pool[exante_pool %in% names(df)]
  exante_feats <- exante_pool[seq_len(min(length(exante_pool), MAX_FEATS_EXANTE))]
  
  # DAY-20 pool
  day20_add <- intersect(c("pct_goal_cum_d20","donors_cum_d20"), names(df))
  day20_feats <- unique(c(exante_feats, day20_add))
  day20_feats <- day20_feats[seq_len(min(length(day20_feats), MAX_FEATS_DAY20))]
  
  list(exante = exante_feats, day20 = day20_feats)
}

# =========================
# 6) SPLIT TEMPORALE ROBUSTO (stile tua pipeline)
# =========================
make_split_like_pipeline <- function(df, test_frac = TEST_FRAC){
  
  if ("created_date_parsed" %in% names(df)) {
    
    na_rate <- mean(is.na(df$created_date_parsed))
    if (is.na(na_rate)) na_rate <- 1
    
    if (na_rate <= 0.20) {
      df_non_na <- df %>% filter(!is.na(created_date_parsed)) %>% arrange(created_date_parsed)
      df_na     <- df %>% filter(is.na(created_date_parsed))
      
      n <- nrow(df_non_na)
      n_test <- ceiling(n * test_frac)
      test_idx <- seq.int(n - n_test + 1, n)
      
      test_df  <- df_non_na[test_idx, , drop = FALSE]
      train_df <- bind_rows(df_non_na[-test_idx, , drop = FALSE], df_na)
      
      log_line("Split temporale (robusto): train=", nrow(train_df), " | test=", nrow(test_df),
               " | NA-date nel train=", nrow(df_na))
      return(list(train = train_df, test = test_df, method = "temporal_holdout_robust"))
    }
    
    log_line("ATTENZIONE: created_date_parsed NA rate=", round(na_rate,3),
             " (>0.20) -> fallback random stratificato.")
  }
  
  split <- rsample::initial_split(df, prop = 1 - test_frac, strata = goal_reached)
  log_line("Split random stratificato.")
  list(split = split, method = "random_stratified")
}

# =========================
# 7) TRAIN RF (ranger) + PROB
# =========================
fit_rf_predict <- function(train_df, test_df, features, seed, mtry, min_node, class_w_mult){
  
  set.seed(seed)
  
  # factor richiesto per classificazione
  train_df <- train_df %>% mutate(goal_reached = factor(goal_reached, levels = c(0,1)))
  test_df  <- test_df  %>% mutate(goal_reached = factor(goal_reached, levels = c(0,1)))
  
  # class weights: 0=1, 1=mult
  cw <- c("0" = 1, "1" = class_w_mult)
  
  fml <- as.formula(paste("goal_reached ~", paste(features, collapse = " + ")))
  
  model <- ranger::ranger(
    formula = fml,
    data = train_df,
    probability = TRUE,
    num.trees = NUM_TREES,
    mtry = mtry,
    min.node.size = min_node,
    class.weights = cw,
    importance = "permutation",
    seed = seed
  )
  
  prob_test <- predict(model, data = test_df)$predictions[, "1"]
  
  list(model = model, prob_test = prob_test)
}

# =========================
# 8) TUNING (CV nel training): selezione hyperparam su MCC (soglia ottimizzata fold-wise)
# =========================
cv_tune_rf <- function(train_df, features, seed_base){
  
  set.seed(seed_base)
  folds <- rsample::vfold_cv(train_df, v = K_FOLDS, strata = goal_reached)
  
  grid <- expand.grid(
    mtry = MTRY_CAND,
    min_node = MIN_NODE_CAND,
    class_w_mult = CLASS_W_MULT,
    stringsAsFactors = FALSE
  )
  
  # numeriche per imputazione
  num_cols <- features[features %in% names(train_df) &
                         sapply(train_df[, features, drop=FALSE], is.numeric)]
  
  res_all <- vector("list", nrow(grid))
  
  for (gi in seq_len(nrow(grid))) {
    g <- grid[gi, ]
    mtry <- g$mtry; min_node <- g$min_node; cw_mult <- g$class_w_mult
    
    fold_stats <- purrr::map_dfr(seq_len(nrow(folds)), function(i){
      
      split_i <- folds$splits[[i]]
      tr <- rsample::analysis(split_i)
      va <- rsample::assessment(split_i)
      
      # imputazione fit su tr
      imp <- fit_imputer(tr, num_cols)
      tr2 <- apply_imputer(tr, imp)
      va2 <- apply_imputer(va, imp)
      
      # category_map: Unknown per NA + allineamento livelli fold-wise
      if ("category_map" %in% features) {
        tr2$category_map <- as.character(tr2$category_map)
        va2$category_map <- as.character(va2$category_map)
        tr2$category_map[is.na(tr2$category_map)] <- "Unknown"
        va2$category_map[is.na(va2$category_map)] <- "Unknown"
        levs <- sort(unique(tr2$category_map))
        if (!("Unknown" %in% levs)) levs <- c(levs, "Unknown")
        va2$category_map[!(va2$category_map %in% levs)] <- "Unknown"
        tr2$category_map <- factor(tr2$category_map, levels = levs)
        va2$category_map <- factor(va2$category_map, levels = levs)
      }
      
      fit <- fit_rf_predict(tr2, va2, features,
                            seed = seed_base + 1000*i + gi,
                            mtry = mtry, min_node = min_node, class_w_mult = cw_mult)
      
      y_true <- as.integer(as.character(va2$goal_reached))
      prob <- fit$prob_test
      
      bt <- best_threshold_mcc(y_true, prob, grid = THRESH_GRID)
      bt$best %>% mutate(fold = i)
    })
    
    agg <- fold_stats %>%
      summarise(
        MCC_mean = mean(MCC, na.rm = TRUE),
        MCC_sd   = sd(MCC, na.rm = TRUE),
        thr_mean = mean(threshold, na.rm = TRUE),
        Recall_mean = mean(Recall, na.rm = TRUE),
        Precision_mean = mean(Precision, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      mutate(mtry = mtry, min_node = min_node, class_w_mult = cw_mult)
    
    res_all[[gi]] <- agg
    log_line("Grid ", gi, "/", nrow(grid),
             " | mtry=", mtry, " min_node=", min_node, " cw=", cw_mult,
             " -> MCC=", round(agg$MCC_mean, 4), " (sd=", round(agg$MCC_sd,4), ")")
  }
  
  res <- bind_rows(res_all) %>%
    arrange(desc(MCC_mean), desc(Recall_mean), desc(Precision_mean))
  
  list(grid_results = res, best = res[1,])
}

# =========================
# 9) OOF per soglia "honest" (MCC-opt) + FIT finale + metriche su TEST (MCC & AUC)
# =========================
cv_oof_probs_rf <- function(train_df, features, best_params, seed_base){
  
  set.seed(seed_base + 123)
  folds <- rsample::vfold_cv(train_df, v = K_FOLDS, strata = goal_reached)
  
  num_cols <- features[features %in% names(train_df) &
                         sapply(train_df[, features, drop=FALSE], is.numeric)]
  
  oof <- purrr::map_dfr(seq_len(nrow(folds)), function(i){
    
    split_i <- folds$splits[[i]]
    tr <- rsample::analysis(split_i)
    va <- rsample::assessment(split_i)
    
    imp <- fit_imputer(tr, num_cols)
    tr2 <- apply_imputer(tr, imp)
    va2 <- apply_imputer(va, imp)
    
    if ("category_map" %in% features) {
      tr2$category_map <- as.character(tr2$category_map)
      va2$category_map <- as.character(va2$category_map)
      tr2$category_map[is.na(tr2$category_map)] <- "Unknown"
      va2$category_map[is.na(va2$category_map)] <- "Unknown"
      levs <- sort(unique(tr2$category_map))
      if (!("Unknown" %in% levs)) levs <- c(levs, "Unknown")
      va2$category_map[!(va2$category_map %in% levs)] <- "Unknown"
      tr2$category_map <- factor(tr2$category_map, levels = levs)
      va2$category_map <- factor(va2$category_map, levels = levs)
    }
    
    fit <- fit_rf_predict(
      train_df = tr2,
      test_df  = va2,
      features = features,
      seed = seed_base + 1000*i + 77,
      mtry = best_params$mtry,
      min_node = best_params$min_node,
      class_w_mult = best_params$class_w_mult
    )
    
    tibble(
      fold = i,
      y_true = as.integer(as.character(va2$goal_reached)),
      prob1  = fit$prob_test
    )
  })
  
  oof
}

evaluate_final_honest <- function(train_df, test_df, features, best_params, seed_base){
  
  # 1) soglia su OOF
  oof <- cv_oof_probs_rf(train_df, features, best_params, seed_base)
  bt  <- best_threshold_mcc(oof$y_true, oof$prob1, grid = THRESH_GRID)
  t_star <- bt$best$threshold
  thr_curve <- bt$curve
  
  # 2) imputazione train -> test
  num_cols <- features[features %in% names(train_df) &
                         sapply(train_df[, features, drop=FALSE], is.numeric)]
  imp <- fit_imputer(train_df, num_cols)
  train2 <- apply_imputer(train_df, imp)
  test2  <- apply_imputer(test_df, imp)
  
  if ("category_map" %in% features) {
    train2$category_map <- as.character(train2$category_map)
    test2$category_map  <- as.character(test2$category_map)
    train2$category_map[is.na(train2$category_map)] <- "Unknown"
    test2$category_map[is.na(test2$category_map)]   <- "Unknown"
    levs <- sort(unique(train2$category_map))
    if (!("Unknown" %in% levs)) levs <- c(levs, "Unknown")
    test2$category_map[!(test2$category_map %in% levs)] <- "Unknown"
    train2$category_map <- factor(train2$category_map, levels = levs)
    test2$category_map  <- factor(test2$category_map,  levels = levs)
  }
  
  # 3) fit finale + prob test
  fit <- fit_rf_predict(
    train_df = train2,
    test_df  = test2,
    features = features,
    seed = seed_base + 999,
    mtry = best_params$mtry,
    min_node = best_params$min_node,
    class_w_mult = best_params$class_w_mult
  )
  
  y_test <- as.integer(as.character(test2$goal_reached))
  prob_test <- fit$prob_test
  pred_test <- ifelse(prob_test >= t_star, 1L, 0L)
  
  ct <- confusion_counts(y_test, pred_test)
  met <- metrics_from_counts(ct)
  
  # AUC ROC + CI DeLong (threshold-free)
  roc_obj <- pROC::roc(response = y_test, predictor = prob_test, quiet = TRUE, direction = "<")
  auc_val <- as.numeric(pROC::auc(roc_obj))
  auc_ci  <- as.numeric(pROC::ci.auc(roc_obj))
  
  # LogLoss / Brier
  ll <- log_loss(y_test, prob_test)
  br <- brier_score(y_test, prob_test)
  
  out <- bind_cols(
    tibble(
      threshold_star = t_star,
      AUC = auc_val,
      AUC_CI_low = auc_ci[1],
      AUC_CI_high = auc_ci[3],
      LogLoss = ll,
      Brier = br
    ),
    met,
    ct
  )
  
  list(
    model = fit$model,
    threshold_star = t_star,
    threshold_curve_train = thr_curve,
    test_results = out,
    prob_test = prob_test,
    pred_test = pred_test,
    y_test = y_test,
    roc_obj = roc_obj,
    oof = oof
  )
}

# =========================
# 10) STABILITY IMPORTANCE + FEATURE SELECTION
# =========================
stability_importance <- function(train_df, features, best_params, seed_base,
                                 B = SEEDS_STABILITY, topk = 10){
  
  num_cols <- features[features %in% names(train_df) &
                         sapply(train_df[, features, drop=FALSE], is.numeric)]
  
  # imputazione su tutto train (ok: siamo nel training)
  imp <- fit_imputer(train_df, num_cols)
  train2 <- apply_imputer(train_df, imp)
  
  if ("category_map" %in% features) {
    train2$category_map <- fct_explicit_na(as.factor(train2$category_map), na_level = "Unknown")
  }
  
  imps <- purrr::map_dfr(seq_len(B), function(b){
    
    set.seed(seed_base + 5000 + b)
    train2 <- train2 %>% mutate(goal_reached = factor(goal_reached, levels = c(0,1)))
    fml <- as.formula(paste("goal_reached ~", paste(features, collapse = " + ")))
    
    cw <- c("0" = 1, "1" = best_params$class_w_mult)
    
    mod <- ranger::ranger(
      formula = fml,
      data = train2,
      probability = TRUE,
      num.trees = NUM_TREES,
      mtry = best_params$mtry,
      min.node.size = best_params$min_node,
      class.weights = cw,
      importance = "permutation",
      seed = seed_base + 5000 + b
    )
    
    impv <- mod$variable.importance
    tibble(run = b, feature = names(impv), importance = as.numeric(impv))
  })
  
  freq_top <- imps %>%
    group_by(run) %>%
    arrange(desc(importance)) %>%
    mutate(rank = row_number()) %>%
    filter(rank <= topk) %>%
    ungroup() %>%
    count(feature, name = "topk_count") %>%
    mutate(topk_freq = topk_count / B) %>%
    arrange(desc(topk_freq), desc(topk_count))
  
  imp_summary <- imps %>%
    group_by(feature) %>%
    summarise(
      importance_mean = mean(importance, na.rm = TRUE),
      importance_sd   = sd(importance, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    left_join(freq_top, by = "feature") %>%
    mutate(topk_freq = tidyr::replace_na(topk_freq, 0)) %>%
    arrange(desc(topk_freq), desc(importance_mean))
  
  list(imp_runs = imps, imp_summary = imp_summary)
}

# =========================
# 11) GRUPPI FEATURE (per dimostrare "emozioni poco rilevanti")
# =========================
feature_group <- function(x){
  if (x %in% c("log_goal_amount","goal_amount")) return("Strutturali/Goal")
  if (x %in% c("campaign_duration")) return("Tempo (durata)")
  if (x %in% c("category_map")) return("Categoria")
  if (grepl("^pct_goal_cum_d", x) || grepl("^donors_cum_d", x) || grepl("^amount_raised_cum_d", x)) return("Early traction (cumulata)")
  if (x %in% c("text_length","n_parole","n_frasi","parole_per_frase","caratteri_per_parola","n_caratteri")) return("Testo (stile/lunghezza)")
  if (x %in% c("emotional_balance","emotional_intensity","emotions_positive","emotions_negative")) return("Emozioni (indici)")
  if (x %in% c("gratitude","sadness","caring","optimism","joy","fear","anger","love","remorse","disgust","admiration","approval","disapproval","grief","surprise","relief")) return("Emozioni (raw)")
  return("Altro")
}

# =========================
# 12) PIPELINE SCENARIO
# =========================
run_scenario <- function(df_all, scenario_name, features){
  
  log_line(strrep("=", 78))
  log_line("SCENARIO: ", scenario_name)
  log_line("Features (n=", length(features), "): ", paste(features, collapse = ", "))
  log_line(strrep("=", 78))
  
  df <- df_all %>% filter(!is.na(goal_reached))
  
  # scenario day20: serve la traction d20 (almeno pct)
  if (scenario_name == "DAY20") {
    if ("pct_goal_cum_d20" %in% features) df <- df %>% filter(!is.na(pct_goal_cum_d20))
  }
  
  # split
  sp <- make_split_like_pipeline(df, test_frac = TEST_FRAC)
  if (!is.null(sp$split)) {
    train_df <- rsample::training(sp$split)
    test_df  <- rsample::testing(sp$split)
  } else {
    train_df <- sp$train
    test_df  <- sp$test
  }
  
  log_line("Class balance train: p(success)=", round(mean(train_df$goal_reached==1, na.rm=TRUE), 4))
  log_line("Class balance test : p(success)=", round(mean(test_df$goal_reached==1, na.rm=TRUE), 4))
  
  # tuning
  log_line("\nTUNING RF (CV) per MCC con class weights...")
  tuned <- cv_tune_rf(train_df, features, seed_base = SEED_BASE)
  best_params <- tuned$best
  save_csv(best_params %>% mutate(scenario=scenario_name),
           paste0("tab_best_params_", tolower(scenario_name), ".csv"))
  save_csv(tuned$grid_results,
           paste0("tab_tuning_grid_", tolower(scenario_name), ".csv"))
  
  log_line("BEST PARAMS: mtry=", best_params$mtry,
           " | min_node=", best_params$min_node,
           " | class_w_mult=", best_params$class_w_mult,
           " | MCC_mean=", round(best_params$MCC_mean, 4))
  
  # stability importance
  log_line("\nSTABILITY permutation importance (", SEEDS_STABILITY, " runs)...")
  topk <- min(ifelse(scenario_name=="EXANTE", MAX_FEATS_EXANTE, MAX_FEATS_DAY20), length(features))
  stab <- stability_importance(train_df, features, best_params,
                               seed_base = SEED_BASE, B = SEEDS_STABILITY, topk = topk)
  imp_sum <- stab$imp_summary %>%
    mutate(group = vapply(feature, feature_group, character(1)))
  save_csv(imp_sum, paste0("tab_importance_stability_", tolower(scenario_name), ".csv"))
  
  # feature selection (conservativa): topk_freq >= 0.70 + always-keep strutturali
  keep <- imp_sum %>% filter(topk_freq >= 0.70) %>% pull(feature)
  
  must_keep <- intersect(c("log_goal_amount","category_map","campaign_duration"), features)
  keep <- unique(c(must_keep, keep))
  keep <- keep[keep %in% features]
  
  # se troppo poche: completa con importance_mean
  if (length(keep) < min(6, length(features))) {
    add <- imp_sum %>% arrange(desc(importance_mean)) %>% pull(feature)
    keep <- unique(c(keep, add))
    keep <- keep[seq_len(min(topk, length(keep)))]
  } else {
    keep <- keep[seq_len(min(topk, length(keep)))]
  }
  
  log_line("Feature finali post-stability (n=", length(keep), "): ", paste(keep, collapse=", "))
  save_csv(tibble(scenario=scenario_name, feature=keep),
           paste0("tab_final_features_", tolower(scenario_name), ".csv"))
  
  # evaluate honest + test metrics (MCC + AUC ecc.)
  log_line("\nFIT finale + soglia MCC su OOF + valutazione su TEST (MCC, AUC)...")
  final <- evaluate_final_honest(train_df, test_df, keep, best_params, seed_base = SEED_BASE)
  
  # salva output base
  save_csv(final$oof, paste0("tab_oof_predictions_", tolower(scenario_name), ".csv"))
  save_csv(final$threshold_curve_train, paste0("tab_threshold_curve_train_", tolower(scenario_name), ".csv"))
  
  res_test <- final$test_results %>%
    mutate(
      scenario = scenario_name,
      n_train = nrow(train_df),
      n_test  = nrow(test_df)
    )
  save_csv(res_test, paste0("tab_test_results_", tolower(scenario_name), ".csv"))
  
  # prob/pred su test
  save_csv(
    tibble(y_test = final$y_test, prob_test = final$prob_test, pred_test = final$pred_test),
    paste0("tab_test_prob_pred_", tolower(scenario_name), ".csv")
  )
  
  # plot MCC vs threshold (train OOF)
  p_thr <- final$threshold_curve_train %>%
    ggplot(aes(x = threshold, y = MCC)) +
    geom_line(linewidth = 1) +
    geom_vline(xintercept = final$threshold_star, linetype = "dashed") +
    labs(title = paste0("MCC vs soglia decisionale (Training OOF) — ", scenario_name),
         subtitle = paste0("threshold* = ", round(final$threshold_star, 3)),
         x = "Soglia", y = "MCC")
  save_plot(p_thr, paste0("fig_mcc_vs_threshold_train_", tolower(scenario_name), ".png"), w = 10, h = 5)
  
  # plot ROC (test)
  roc_df <- tibble(
    tpr = rev(final$roc_obj$sensitivities),
    fpr = rev(1 - final$roc_obj$specificities)
  )
  p_roc <- ggplot(roc_df, aes(x = fpr, y = tpr)) +
    geom_line(linewidth = 1) +
    geom_abline(linetype = "dashed") +
    coord_equal() +
    labs(
      title = paste0("ROC (TEST) — ", scenario_name),
      subtitle = paste0("AUC = ", round(res_test$AUC, 3),
                        " [", round(res_test$AUC_CI_low, 3), ", ", round(res_test$AUC_CI_high, 3), "]"),
      x = "False Positive Rate", y = "True Positive Rate"
    )
  save_plot(p_roc, paste0("fig_roc_test_", tolower(scenario_name), ".png"), w = 7, h = 6)
  
  # importance plot (top 12) — mostrare che emozioni stanno in basso
  top_show <- min(12, nrow(imp_sum))
  imp_top <- imp_sum %>%
    arrange(desc(topk_freq), desc(importance_mean)) %>%
    slice_head(n = top_show)
  
  p_imp <- ggplot(imp_top, aes(x = reorder(feature, importance_mean), y = importance_mean, fill = group)) +
    geom_col(color = "white") +
    coord_flip() +
    labs(title = paste0("Permutation importance (media) — ", scenario_name),
         subtitle = "Top feature (importance_mean). Colore = gruppo (testo/emozioni/traction/struttura).",
         x = NULL, y = "Importance (permutation)")
  save_plot(p_imp, paste0("fig_importance_top_", tolower(scenario_name), ".png"), w = 11, h = 7)
  
  # importance per gruppi (somma/mean)
  imp_group <- imp_sum %>%
    group_by(group) %>%
    summarise(importance_mean_sum = sum(importance_mean, na.rm = TRUE),
              importance_mean_avg = mean(importance_mean, na.rm = TRUE),
              .groups = "drop") %>%
    arrange(desc(importance_mean_sum))
  save_csv(imp_group, paste0("tab_importance_by_group_", tolower(scenario_name), ".csv"))
  
  p_grp <- ggplot(imp_group, aes(x = reorder(group, importance_mean_sum), y = importance_mean_sum)) +
    geom_col(color = "white") +
    coord_flip() +
    labs(title = paste0("Importanza aggregata per gruppo — ", scenario_name),
         subtitle = "Somma delle permutation importance (media) per gruppo di feature.",
         x = NULL, y = "Somma importance_mean")
  save_plot(p_grp, paste0("fig_importance_group_", tolower(scenario_name), ".png"), w = 10, h = 6)
  
  # salva modello
  model_path <- file.path(dirs$models, paste0("rf_", tolower(scenario_name), "_", timestamp, ".rds"))
  saveRDS(final$model, model_path)
  log_line("✓ Modello salvato: ", model_path)
  
  list(
    scenario = scenario_name,
    split_method = sp$method,
    train = train_df,
    test = test_df,
    features_initial = features,
    features_final = keep,
    best_params = best_params,
    test_results = res_test,
    model_path = model_path
  )
}

# =========================
# 13) MAIN RUN
# =========================
log_line(strrep("=", 78))
log_line("RUN RF: ", timestamp)
log_line("INPUT: ", INPUT_DATA)
log_line("OUT: ", OUT_ROOT)
log_line(strrep("=", 78))

stopifnot(file.exists(INPUT_DATA))
df_all <- read.csv(INPUT_DATA, stringsAsFactors = FALSE)

# coerenza con EDA_finale.R
df_all <- make_model_ready_like_eda(df_all)

# sanity: target
if (!all(df_all$goal_reached %in% c(0,1,NA))) stop("goal_reached deve essere 0/1/NA")

# feature sets coerenti con EDA
fs <- get_feature_sets_like_eda(df_all)

# check presenza feature
req_cols <- unique(c(fs$exante, fs$day20, "goal_reached"))
missing_cols <- setdiff(req_cols, names(df_all))
if (length(missing_cols) > 0) {
  stop("Mancano colonne nel dataset (attese da EDA_finale.R o da questo script): ",
       paste(missing_cols, collapse = ", "))
}

# ---- Scenario A: EX-ANTE
res_exante <- run_scenario(df_all, "EXANTE", fs$exante)

# ---- Scenario B: DAY-20
res_day20  <- run_scenario(df_all, "DAY20",  fs$day20)

# =========================
# 14) CONFRONTO "GUADAGNO INFORMATIVO" (MCC + AUC + altre metriche)
# =========================
cmp <- bind_rows(
  res_exante$test_results %>% mutate(scenario_label = "EX-ANTE"),
  res_day20$test_results  %>% mutate(scenario_label = "DAY-20")
) %>%
  select(
    scenario_label, n_train, n_test,
    threshold_star, MCC, AUC, AUC_CI_low, AUC_CI_high,
    Balanced_Accuracy, Precision, Recall, Specificity, F1,
    LogLoss, Brier,
    TP, TN, FP, FN
  )

save_csv(cmp, "tab_compare_exante_vs_day20_test_metrics.csv")

# Delta (DAY20 - EXANTE): “miglioramento informativo”
delta <- cmp %>%
  select(-n_train, -n_test, -AUC_CI_low, -AUC_CI_high, -TP, -TN, -FP, -FN) %>%
  pivot_longer(cols = -scenario_label, names_to = "metric", values_to = "value") %>%
  pivot_wider(names_from = scenario_label, values_from = value) %>%
  mutate(delta_day20_minus_exante = `DAY-20` - `EX-ANTE`) %>%
  arrange(desc(abs(delta_day20_minus_exante)))

save_csv(delta, "tab_delta_day20_minus_exante_metrics.csv")

# Plot comparativo metriche chiave (MCC + AUC + PR proxy: qui mostriamo almeno LogLoss/Brier)
cmp_long <- cmp %>%
  select(scenario_label, MCC, AUC, Balanced_Accuracy, LogLoss, Brier) %>%
  pivot_longer(cols = -scenario_label, names_to = "metric", values_to = "value")

p_cmp <- ggplot(cmp_long, aes(x = metric, y = value, fill = scenario_label)) +
  geom_col(position = "dodge", color = "white") +
  labs(
    title = "Confronto prestazioni (TEST): EX-ANTE vs DAY-20",
    subtitle = "MCC (threshold), AUC (threshold-free), e metriche probabilistiche (LogLoss/Brier).",
    x = NULL, y = "Valore"
  ) +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))
save_plot(p_cmp, "fig_compare_metrics_exante_vs_day20.png", w = 11, h = 6)

# =========================
# 15) REPORT testuale (stile tesi, pronto da citare)
# =========================
REPORT_FILE <- file.path(dirs$reps, paste0("report_rf_results_", timestamp, ".txt"))
write_report <- function(...) {
  msg <- paste0(...)
  cat(msg, "\n", file = REPORT_FILE, append = TRUE)
}

write_report(strrep("=", 90))
write_report("REPORT — Random Forest (MCC + AUC) — ", timestamp)
write_report(strrep("=", 90), "\n")

write_report("A) Vincolo informativo / anti-leakage (coerente con la tesi)")
write_report("- Due scenari separati:")
write_report("  EX-ANTE: feature note al lancio + campaign_duration (come nel dataset EDA).")
write_report("  DAY-20 : EX-ANTE + early traction cumulata a 20 giorni (pct_goal_cum_d20, donors_cum_d20).")
write_report("- Variabili proxy dell'outcome o snapshot finali rimosse (current_amount, donor_count, share_complete, performance, money_*...).\n")

write_report("B) Validazione e sbilanciamento")
write_report("- Holdout temporale robusto se created_date disponibile; altrimenti split stratificato.")
write_report("- Class imbalance gestito con class weights nel RF.")
write_report("- Soglia decisionale stimata massimizzando MCC su predizioni OOF (K-fold) nel training (stima 'honest').\n")

write_report("C) Risultati su TEST (MCC e AUC come 'informazione acquisita')")
write_report(capture.output(print(cmp)), "\n")

write_report("D) Delta DAY-20 minus EX-ANTE (guadagno informativo)")
write_report("Le righe con delta maggiore (in valore assoluto) quantificano l'incremento di informazione dato dall'early traction.")
write_report(capture.output(print(delta %>% slice_head(n = 10))), "\n")

write_report("E) Interpretazione mirata alla tua domanda di ricerca")
write_report("- AUC: migliora se il ranking probabilistico è più informativo (misura threshold-free).")
write_report("- MCC: migliora se, fissata una regola operativa (soglia), aumenta la qualità della classificazione sotto sbilanciamento.")
write_report("- Se DAY-20 mostra incremento simultaneo di AUC e MCC, l'evidenza è coerente con: 'la trazione iniziale contiene informazione addizionale sostanziale rispetto alle sole feature testuali/strutturali'.\n")

write_report("F) Importanza delle variabili (per mostrare che le emozioni sono marginali)")
write_report("- Tabelle: tab_importance_stability_exante.csv / day20.csv")
write_report("- Aggregazione per gruppi: tab_importance_by_group_exante.csv / day20.csv")
write_report("  Atteso: traction (pct_goal_cum_d20, donors_cum_d20) domina in DAY-20; emozioni (raw/indici) hanno importanza bassa e instabile.\n")

write_report(strrep("=", 90))
write_report("FINE REPORT")
write_report(strrep("=", 90))

log_line("✓ Fine run. Output in: ", OUT_ROOT)
log_line("✓ Log: ", LOG_FILE)
log_line("✓ Report: ", REPORT_FILE)

