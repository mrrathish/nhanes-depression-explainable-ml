
# ============================================================
# QM 640 CAPSTONE
# RQ3: Prediction Model Sample Size Calculation
#
# Method: Riley et al. prediction model sample size framework
# Package: pmsampsize
#
# Outcome:
#   Elevated depressive symptoms (PHQ-9 >= 10)
#
# Dataset:
#   NHANES August 2021 - August 2023
# ============================================================


# ------------------------------------------------------------
# 1. Install and load package
# ------------------------------------------------------------

if (!requireNamespace("pmsampsize", quietly = TRUE)) {
  install.packages(
    "pmsampsize",
    repos = "https://cloud.r-project.org"
  )
}

library(pmsampsize)


# ------------------------------------------------------------
# 2. Create output directory
# ------------------------------------------------------------

dir.create(
  "reports",
  showWarnings = FALSE,
  recursive = TRUE
)


# ------------------------------------------------------------
# 3. NHANES sample information
# ------------------------------------------------------------

available_total_n <- 5210
available_total_positive <- 661

available_complete_n <- 4448
available_complete_positive <- 549

observed_prevalence <- (
  available_total_positive / available_total_n
)

complete_case_prevalence <- (
  available_complete_positive / available_complete_n
)


# ------------------------------------------------------------
# 4. Planning assumptions
# ------------------------------------------------------------

# Number of predictor coefficients, excluding intercept.
# Verify against the final model specification.

predictor_parameters <- 15

# Anticipated Cox-Snell R-squared.
# Illustrative assumption requiring justification.

anticipated_r2 <- 0.10

# Target global shrinkage.

target_shrinkage <- 0.90

# Model development fraction.

training_fraction <- 0.80


# ------------------------------------------------------------
# 5. Helper: extract required sample size
# ------------------------------------------------------------

# The pmsampsize return structure may vary by version.
# Search for recognized sample-size fields.
# Fail explicitly rather than silently using a wrong value.

extract_sample_size <- function(result) {

  candidate_names <- c(
    "sample_size",
    "samplesize",
    "sample.size",
    "n"
  )

  find_value <- function(object, depth = 0) {

    if (depth > 5 || !is.list(object)) {
      return(NULL)
    }

    field_names <- names(object)

    if (!is.null(field_names)) {

      for (candidate in candidate_names) {

        index <- which(
          tolower(field_names) == candidate
        )

        if (length(index) > 0) {

          value <- object[[index[1]]]

          if (
            is.numeric(value) &&
            length(value) == 1 &&
            is.finite(value) &&
            value > 0
          ) {
            return(as.numeric(value))
          }
        }
      }
    }

    # Inspect nested lists if required.
    for (item in object) {

      if (is.list(item)) {

        found <- find_value(
          item,
          depth + 1
        )

        if (!is.null(found)) {
          return(found)
        }
      }
    }

    return(NULL)
  }

  extracted <- find_value(result)

  if (is.null(extracted)) {

    cat("\nUnable to identify sample-size field.\n")
    cat("Inspect the returned object:\n")

    str(result)

    stop(
      "Sample-size extraction failed. ",
      "Check the pmsampsize result structure."
    )
  }

  return(ceiling(extracted))
}


# ------------------------------------------------------------
# 6. Helper: run pmsampsize
# ------------------------------------------------------------

run_sample_size <- function(
    r2,
    parameters,
    prevalence,
    shrinkage
) {

  pmsampsize(
    type = "b",
    csrsquared = r2,
    parameters = parameters,
    prevalence = prevalence,
    shrinkage = shrinkage
  )
}


# ------------------------------------------------------------
# 7. Main sample-size calculation
# ------------------------------------------------------------

cat("\n============================================\n")
cat("RQ3: PREDICTION MODEL SAMPLE SIZE\n")
cat("============================================\n")

cat("\nPlanning assumptions:\n")

cat(
  "Observed prevalence:",
  round(observed_prevalence, 4),
  "\n"
)

cat(
  "Complete-case prevalence:",
  round(complete_case_prevalence, 4),
  "\n"
)

cat(
  "Predictor parameters:",
  predictor_parameters,
  "\n"
)

cat(
  "Anticipated Cox-Snell R2:",
  anticipated_r2,
  "\n"
)

cat(
  "Target shrinkage:",
  target_shrinkage,
  "\n"
)

cat(
  "Training fraction:",
  training_fraction,
  "\n"
)


main_result <- run_sample_size(
  r2 = anticipated_r2,
  parameters = predictor_parameters,
  prevalence = observed_prevalence,
  shrinkage = target_shrinkage
)

cat("\n=== PMSAMPSIZE RESULT ===\n")

print(main_result)

cat("\n=== RESULT STRUCTURE ===\n")

str(main_result)


# ------------------------------------------------------------
# 8. Extract required sample size
# ------------------------------------------------------------

required_development_n <- extract_sample_size(
  main_result
)

required_total_n <- ceiling(
  required_development_n / training_fraction
)

available_training_n <- floor(
  available_complete_n * training_fraction
)

available_test_n <- (
  available_complete_n - available_training_n
)

development_feasible <- (
  available_training_n >= required_development_n
)


# ------------------------------------------------------------
# 9. NHANES feasibility comparison
# ------------------------------------------------------------

cat("\n============================================\n")
cat("NHANES FEASIBILITY\n")
cat("============================================\n")

cat(
  "\nAvailable outcome-valid N:",
  available_total_n,
  "\n"
)

cat(
  "Available complete-case N:",
  available_complete_n,
  "\n"
)

cat(
  "Available complete-case positive:",
  available_complete_positive,
  "\n"
)

cat(
  "Available training N:",
  available_training_n,
  "\n"
)

cat(
  "Available test N:",
  available_test_n,
  "\n"
)

cat(
  "Required development N:",
  required_development_n,
  "\n"
)

cat(
  "Required total N (80/20 split):",
  required_total_n,
  "\n"
)

cat(
  "Development sample feasible (unadjusted):",
  development_feasible,
  "\n"
)


# ------------------------------------------------------------
# 10. Sensitivity analysis
# ------------------------------------------------------------

cat("\n============================================\n")
cat("SENSITIVITY ANALYSIS\n")
cat("============================================\n")

# Alternative anticipated Cox-Snell R2 values.
r2_values <- c(0.05, 0.10, 0.15)

# Alternative numbers of predictor parameters.
parameter_values <- c(10, 15, 20)

sensitivity_results <- data.frame()

for (r2 in r2_values) {

  for (p in parameter_values) {

    cat(
      "\nRunning scenario:",
      "R2 =", r2,
      "Parameters =", p,
      "\n"
    )

    result <- tryCatch(
      run_sample_size(
        r2 = r2,
        parameters = p,
        prevalence = observed_prevalence,
        shrinkage = target_shrinkage
      ),
      error = function(e) {
        warning(
          paste(
            "Scenario failed:",
            "R2 =", r2,
            "Parameters =", p,
            conditionMessage(e)
          )
        )
        return(NULL)
      }
    )

    if (is.null(result)) {
      next
    }

    required_n <- extract_sample_size(result)

    required_total <- ceiling(
      required_n / training_fraction
    )

    row <- data.frame(
      anticipated_r2 = r2,
      parameters = p,
      prevalence = observed_prevalence,
      shrinkage = target_shrinkage,
      required_development_n = required_n,
      required_total_n = required_total,
      available_training_n = available_training_n,
      feasible_unadjusted = (
        available_training_n >= required_n
      )
    )

    sensitivity_results <- rbind(
      sensitivity_results,
      row
    )
  }
}

cat("\n=== SENSITIVITY RESULTS ===\n")

print(
  sensitivity_results,
  row.names = FALSE
)


# ------------------------------------------------------------
# 11. Save main report
# ------------------------------------------------------------

report_file <- "reports/rq3_sample_size.txt"

capture.output(
  {

    cat("QM 640 CAPSTONE\n")
    cat("RQ3 PREDICTION MODEL SAMPLE SIZE\n\n")

    cat("DATASET\n")
    cat("NHANES August 2021 - August 2023\n\n")

    cat("OUTCOME\n")
    cat("Elevated depressive symptoms: PHQ-9 >= 10\n\n")

    cat("PLANNING ASSUMPTIONS\n")

    cat(
      "Observed prevalence:",
      observed_prevalence,
      "\n"
    )

    cat(
      "Complete-case prevalence:",
      complete_case_prevalence,
      "\n"
    )

    cat(
      "Predictor parameters:",
      predictor_parameters,
      "\n"
    )

    cat(
      "Anticipated Cox-Snell R2:",
      anticipated_r2,
      "\n"
    )

    cat(
      "Target shrinkage:",
      target_shrinkage,
      "\n"
    )

    cat(
      "Training fraction:",
      training_fraction,
      "\n\n"
    )

    cat("MAIN PMSAMPSIZE RESULT\n")
    print(main_result)

    cat("\nNHANES FEASIBILITY\n")

    cat(
      "Available outcome-valid N:",
      available_total_n,
      "\n"
    )

    cat(
      "Available complete-case N:",
      available_complete_n,
      "\n"
    )

    cat(
      "Available complete-case positive:",
      available_complete_positive,
      "\n"
    )

    cat(
      "Available training N:",
      available_training_n,
      "\n"
    )

    cat(
      "Available test N:",
      available_test_n,
      "\n"
    )

    cat(
      "Required development N:",
      required_development_n,
      "\n"
    )

    cat(
      "Required total N:",
      required_total_n,
      "\n"
    )

    cat(
      "Development feasible (unadjusted):",
      development_feasible,
      "\n\n"
    )

    cat("SENSITIVITY ANALYSIS\n")

    print(
      sensitivity_results,
      row.names = FALSE
    )

    cat("\nLIMITATIONS\n")

    cat(
      paste(
        "- Anticipated R2 and parameter count",
        "require justification.\n"
      )
    )

    cat(
      paste(
        "- Calculations do not directly account",
        "for NHANES clustering or weighting.\n"
      )
    )

    cat(
      paste(
        "- Feasibility assumes an 80/20 split",
        "of the complete-case population.\n"
      )
    )

    cat(
      paste(
        "- Test-set adequacy must be assessed",
        "separately.\n"
      )
    )

  },
  file = report_file
)


# ------------------------------------------------------------
# 12. Save sensitivity CSV
# ------------------------------------------------------------

write.csv(
  sensitivity_results,
  file = "reports/rq3_sensitivity.csv",
  row.names = FALSE
)


# ------------------------------------------------------------
# 13. Save summary CSV
# ------------------------------------------------------------

summary_results <- data.frame(
  observed_prevalence = observed_prevalence,
  complete_case_prevalence = complete_case_prevalence,
  predictor_parameters = predictor_parameters,
  anticipated_r2 = anticipated_r2,
  target_shrinkage = target_shrinkage,
  required_development_n = required_development_n,
  required_total_n = required_total_n,
  available_complete_n = available_complete_n,
  available_training_n = available_training_n,
  available_test_n = available_test_n,
  feasible_unadjusted = development_feasible
)

write.csv(
  summary_results,
  file = "reports/rq3_summary.csv",
  row.names = FALSE
)


# ------------------------------------------------------------
# 14. Final summary
# ------------------------------------------------------------

cat("\n============================================\n")
cat("RQ3 ANALYSIS COMPLETED\n")
cat("============================================\n")

cat(
  "\nRequired development N:",
  required_development_n,
  "\n"
)

cat(
  "Available training N:",
  available_training_n,
  "\n"
)

cat(
  "Preliminary feasibility:",
  ifelse(
    development_feasible,
    "MEETS ASSUMPTIONS",
    "DOES NOT MEET ASSUMPTIONS"
  ),
  "\n"
)

cat("\nFiles generated:\n")

cat("reports/rq3_sample_size.txt\n")
cat("reports/rq3_sensitivity.csv\n")
cat("reports/rq3_summary.csv\n")

cat("\nNote: Survey-design adjustment remains pending.\n")
