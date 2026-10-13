
# ============================================================
# QM 640 CAPSTONE
# 03 - NHANES SURVEY DESIGN VALIDATION
#
# NHANES August 2021 - August 2023
#
# Analyses:
# 1. Weighted depression prevalence
# 2. Design effect and effective N
# 3. RQ1: Sleep and PHQ-9
# 4. RQ2: Sedentary time and depression
# 5. Interview vs MEC-weighted comparison
#
# Requires: survey, arrow
# ============================================================

# ------------------------------------------------------------
# 1. Packages
# ------------------------------------------------------------

required_packages <- c("survey", "arrow")

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(
      "Missing package: ", pkg,
      ". Install using install.packages()."
    )
  }
}

library(survey)
library(arrow)

options(survey.lonely.psu = "fail")

# ------------------------------------------------------------
# 2. Paths
# ------------------------------------------------------------

input_file <- "data/processed/nhanes_adults.parquet"
output_dir <- "reports/survey_design"

dir.create(
  output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

if (!file.exists(input_file)) {
  stop("Missing input file: ", input_file)
}

# ------------------------------------------------------------
# 3. Load processed NHANES dataset
# ------------------------------------------------------------

df <- as.data.frame(
  read_parquet(input_file)
)

cat("\n=== DATA LOADING ===\n")
cat("Records:", nrow(df), "\n")

required_columns <- c(
  "SEQN",
  "RIDAGEYR",
  "RIAGENDR",
  "RIDRETH3",
  "DMDEDUC2",
  "INDFMPIR",
  "SLD012",
  "sedentary_hours",
  "BMXBMI",
  "SMQ020",
  "phq9_total",
  "elevated_symptoms",
  "WTINT2YR",
  "WTMEC2YR",
  "SDMVSTRA",
  "SDMVPSU"
)

missing_columns <- setdiff(
  required_columns,
  names(df)
)

if (length(missing_columns) > 0) {
  stop(
    "Missing columns: ",
    paste(missing_columns, collapse = ", ")
  )
}

# ------------------------------------------------------------
# 4. Validate values and weights
# ------------------------------------------------------------

df <- subset(
  df,
  !is.na(RIDAGEYR) & RIDAGEYR >= 20
)

if (anyDuplicated(df$SEQN) > 0) {
  stop("Duplicate SEQN values detected")
}

if (any(
  !is.na(df$elevated_symptoms) &
  !(df$elevated_symptoms %in% c(0, 1))
)) {
  stop("Outcome must be binary: 0 or 1")
}

# Recheck PHQ-9 outcome consistency.
valid_phq <- !is.na(df$phq9_total)

if (any(
  df$phq9_total[valid_phq] < 0 |
  df$phq9_total[valid_phq] > 27
)) {
  stop("Invalid PHQ-9 scores detected")
}

expected_outcome <- ifelse(
  df$phq9_total >= 10, 1, 0
)

mismatch <- (
  !is.na(expected_outcome) &
  (
    is.na(df$elevated_symptoms) |
    df$elevated_symptoms != expected_outcome
  )
)

if (any(mismatch)) {
  stop("PHQ-9 outcome inconsistency detected")
}

# Survey weights: zero or missing indicates
# nonparticipation or ineligibility for that
# weighted analysis.

df$valid_interview_weight <- (
  is.finite(df$WTINT2YR) &
  df$WTINT2YR > 0
)

df$valid_mec_weight <- (
  is.finite(df$WTMEC2YR) &
  df$WTMEC2YR > 0
)

# Keep complete survey-design identifiers.
df <- df[
  !is.na(df$SDMVSTRA) &
  !is.na(df$SDMVPSU),
]

cat("Adults with design identifiers:", nrow(df), "\n")
cat(
  "Positive interview weights:",
  sum(df$valid_interview_weight), "\n"
)
cat(
  "Positive MEC weights:",
  sum(df$valid_mec_weight), "\n"
)

# ------------------------------------------------------------
# 5. Create interview and MEC survey designs
# ------------------------------------------------------------

interview_df <- df[
  df$valid_interview_weight,
]

mec_df <- df[
  df$valid_mec_weight,
]

interview_design <- svydesign(
  ids = ~SDMVPSU,
  strata = ~SDMVSTRA,
  weights = ~WTINT2YR,
  nest = TRUE,
  data = interview_df
)

mec_design <- svydesign(
  ids = ~SDMVPSU,
  strata = ~SDMVSTRA,
  weights = ~WTMEC2YR,
  nest = TRUE,
  data = mec_df
)

cat("\n=== SURVEY DESIGN ===\n")

cat(
  "Interview records:",
  nrow(interview_df), "\n"
)

cat(
  "MEC records:",
  nrow(mec_df), "\n"
)

cat(
  "Interview strata:",
  length(unique(interview_df$SDMVSTRA)),
  "\n"
)

cat(
  "Interview PSU combinations:",
  nrow(unique(
    interview_df[c("SDMVSTRA", "SDMVPSU")]
  )),
  "\n"
)

cat(
  "Interview design degrees of freedom:",
  degf(interview_design),
  "\n"
)

# ------------------------------------------------------------
# 6. Restrict to valid PHQ-9 subpopulation
# ------------------------------------------------------------

interview_phq <- subset(
  interview_design,
  !is.na(phq9_total) &
  !is.na(elevated_symptoms)
)

mec_phq <- subset(
  mec_design,
  !is.na(phq9_total) &
  !is.na(elevated_symptoms)
)

# ------------------------------------------------------------
# 7. Helper: prevalence and design effect
# ------------------------------------------------------------

analyze_prevalence <- function(design, label) {

  estimate <- svymean(
    ~elevated_symptoms,
    design = design,
    na.rm = TRUE,
    deff = TRUE
  )

  prevalence <- as.numeric(coef(estimate)[1])
  standard_error <- as.numeric(SE(estimate)[1])

  confidence_interval <- confint(
    estimate,
    level = 0.95
  )

  # Default survey deff compares with a
  # corresponding SRS without replacement.
  design_effect <- as.numeric(
    deff(estimate)[1]
  )

  sample_n <- nrow(design$variables)

  effective_n <- if (
    is.finite(design_effect) &&
    design_effect > 0
  ) {
    sample_n / design_effect
  } else {
    NA_real_
  }

  data.frame(
    population = label,
    sample_n = sample_n,
    prevalence = prevalence,
    prevalence_percent = prevalence * 100,
    standard_error = standard_error,
    ci_lower = confidence_interval[1, 1],
    ci_upper = confidence_interval[1, 2],
    design_effect = design_effect,
    effective_n_approx = effective_n
  )
}

# ------------------------------------------------------------
# 8. Weighted prevalence
# ------------------------------------------------------------

cat("\n=== WEIGHTED PREVALENCE ===\n")

interview_prevalence <- analyze_prevalence(
  interview_phq,
  "Interview-weighted PHQ sample"
)

mec_prevalence <- analyze_prevalence(
  mec_phq,
  "MEC-weighted PHQ sample"
)

prevalence_results <- rbind(
  interview_prevalence,
  mec_prevalence
)

print(prevalence_results)

write.csv(
  prevalence_results,
  file.path(output_dir, "prevalence.csv"),
  row.names = FALSE
)

# ------------------------------------------------------------
# 9. RQ1: Sleep duration vs PHQ-9 score
# ------------------------------------------------------------

cat("\n=== RQ1: SLEEP AND PHQ-9 ===\n")

# Primary descriptive association.
# Survey-adjusted linear regression.
#
# The sleep variable must be verified
# against its official codebook.

rq1_design <- subset(
  interview_phq,
  !is.na(SLD012) &
  !is.na(RIDAGEYR) &
  !is.na(RIAGENDR)
)

rq1_model <- svyglm(
  phq9_total ~
    SLD012 +
    RIDAGEYR +
    factor(RIAGENDR),
  design = rq1_design,
  family = gaussian()
)

print(summary(rq1_model))

rq1_ci <- confint(rq1_model)

rq1_results <- data.frame(
  variable = names(coef(rq1_model)),
  coefficient = as.numeric(coef(rq1_model)),
  standard_error = as.numeric(
    SE(rq1_model)
  ),
  ci_lower = rq1_ci[, 1],
  ci_upper = rq1_ci[, 2],
  row.names = NULL
)

write.csv(
  rq1_results,
  file.path(output_dir, "rq1_sleep_regression.csv"),
  row.names = FALSE
)

# ------------------------------------------------------------
# 10. RQ2: Sedentary time and elevated symptoms
# ------------------------------------------------------------

cat("\n=== RQ2: SEDENTARY TIME ===\n")

rq2_design <- subset(
  interview_phq,
  !is.na(sedentary_hours) &
  !is.na(RIDAGEYR) &
  !is.na(RIAGENDR)
)

rq2_model <- svyglm(
  elevated_symptoms ~
    sedentary_hours +
    RIDAGEYR +
    factor(RIAGENDR),
  design = rq2_design,
  family = quasibinomial()
)

print(summary(rq2_model))

rq2_coefficients <- coef(rq2_model)
rq2_ci <- confint(rq2_model)

rq2_results <- data.frame(
  variable = names(rq2_coefficients),
  log_odds = as.numeric(rq2_coefficients),
  odds_ratio = exp(
    as.numeric(rq2_coefficients)
  ),
  ci_lower_or = exp(rq2_ci[, 1]),
  ci_upper_or = exp(rq2_ci[, 2]),
  row.names = NULL
)

write.csv(
  rq2_results,
  file.path(output_dir, "rq2_sedentary_logistic.csv"),
  row.names = FALSE
)

# ------------------------------------------------------------
# 11. Interview vs MEC-weighted comparison
# ------------------------------------------------------------

cat("\n=== INTERVIEW VS MEC COMPARISON ===\n")

comparison <- prevalence_results[
  ,
  c(
    "population",
    "sample_n",
    "prevalence_percent",
    "ci_lower",
    "ci_upper",
    "design_effect",
    "effective_n_approx"
  )
]

print(comparison)

write.csv(
  comparison,
  file.path(output_dir, "weight_comparison.csv"),
  row.names = FALSE
)

# ------------------------------------------------------------
# 12. Save model summaries
# ------------------------------------------------------------

capture.output(
  {
    cat("NHANES SURVEY DESIGN VALIDATION\n\n")

    cat("SURVEY PREVALENCE\n")
    print(prevalence_results)

    cat("\nRQ1: SLEEP REGRESSION\n")
    print(summary(rq1_model))

    cat("\nRQ2: SEDENTARY LOGISTIC REGRESSION\n")
    print(summary(rq2_model))

    cat("\nINTERVIEW VS MEC\n")
    print(comparison)

    cat("\nINTERPRETATION NOTES\n")
    cat(
      "Survey estimates use strata, PSU and weights.\n"
    )
    cat(
      "Effective N is prevalence-estimator-specific.\n"
    )
    cat(
      "No universal design effect is assumed.\n"
    )
    cat(
      "NHANES cycle-specific weight guidance ",
      "must be verified.\n",
      sep = ""
    )
  },
  file = file.path(
    output_dir,
    "survey_design_summary.txt"
  )
)

cat("\n=== COMPLETED ===\n")

cat("Reports saved in:", output_dir, "\n")
