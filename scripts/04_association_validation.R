# ============================================================
# QM 640 CAPSTONE - PHASE 4 (parsimonious + joint Wald tests)
# NHANES August 2021-August 2023
# Run from project root: Rscript scripts/04_association_validation.R
# ============================================================
options(warn = 1)
for (pkg in c('survey', 'arrow')) {
  if (!requireNamespace(pkg, quietly = TRUE)) stop('Install missing R package: ', pkg)
}
suppressPackageStartupMessages(library(survey))
suppressPackageStartupMessages(library(arrow))
options(survey.lonely.psu = 'fail')

input <- 'data/processed/nhanes_adults.parquet'
out <- 'reports/association_validation'
dir.create(out, recursive = TRUE, showWarnings = FALSE)
if (!file.exists(input)) stop('Input file not found: ', input)
d <- as.data.frame(arrow::read_parquet(input))
required <- c('SEQN','RIDAGEYR','RIAGENDR','RIDRETH3','DMDEDUC2',
              'INDFMPIR','SLD012','sedentary_hours','BMXBMI','SMQ020',
              'phq9_total','elevated_symptoms','WTINT2YR','WTMEC2YR',
              'SDMVSTRA','SDMVPSU')
missing <- setdiff(required, names(d))
if (length(missing)) stop('Missing columns: ', paste(missing, collapse=', '))
if (anyDuplicated(d$SEQN)) stop('Duplicate SEQN identifiers')
for (nm in setdiff(required, 'SEQN')) {
  if (is.factor(d[[nm]])) d[[nm]] <- as.character(d[[nm]])
  v <- suppressWarnings(as.numeric(d[[nm]]))
  v[is.finite(v) & abs(v) < 1e-50] <- 0
  d[[nm]] <- v
}
d <- d[is.finite(d$RIDAGEYR) & d$RIDAGEYR >= 20 &
       is.finite(d$SDMVSTRA) & is.finite(d$SDMVPSU), , drop=FALSE]

# NHANES nonresponse codes excluded. Verify codebooks and weights.
d$sex <- factor(ifelse(d$RIAGENDR %in% 1:2, d$RIAGENDR, NA_real_),
                levels=1:2, labels=c('Male','Female'))
d$race <- factor(ifelse(d$RIDRETH3 %in% c(1,2,3,4,6,7), d$RIDRETH3, NA_real_))
d$education3 <- factor(ifelse(d$DMDEDUC2 %in% 1:2, 'Less_than_HS',
                              ifelse(d$DMDEDUC2 == 3, 'HS_GED',
                              ifelse(d$DMDEDUC2 %in% 4:5, 'Some_college_or_more', NA_character_))),
                       levels=c('Less_than_HS','HS_GED','Some_college_or_more'))
d$pir <- ifelse(is.finite(d$INDFMPIR) & d$INDFMPIR >= 0 & d$INDFMPIR <= 5,
                d$INDFMPIR, NA_real_)
d$sleep_hours <- ifelse(is.finite(d$SLD012) & d$SLD012 >= 2 & d$SLD012 <= 12,
                        d$SLD012, NA_real_)
d$sed_hours <- ifelse(is.finite(d$sedentary_hours) & d$sedentary_hours >= 0 &
                        d$sedentary_hours <= 24, d$sedentary_hours, NA_real_)
d$bmi <- ifelse(is.finite(d$BMXBMI) & d$BMXBMI > 0 & d$BMXBMI < 100,
                d$BMXBMI, NA_real_)
d$ever_smoked <- factor(ifelse(d$SMQ020 %in% 1:2, d$SMQ020, NA_real_),
                        levels=1:2, labels=c('Yes','No'))
d$phq <- ifelse(is.finite(d$phq9_total) & d$phq9_total >= 0 &
                d$phq9_total <= 27, d$phq9_total, NA_real_)
d$elevated <- ifelse(is.na(d$phq), NA_integer_, as.integer(d$phq >= 10))
if (any(!is.na(d$elevated_symptoms) & !is.na(d$elevated) &
        d$elevated_symptoms != d$elevated)) stop('PHQ outcome inconsistency')

# Primary candidates: MEC weights (DPQ examination respondents).
# Interview weights are sensitivity only. Verify official cycle guidance.
make_design <- function(weight) {
  x <- d[is.finite(d[[weight]]) & d[[weight]] > 0, , drop=FALSE]
  survey::svydesign(ids=~SDMVPSU, strata=~SDMVSTRA,
                    weights=as.formula(paste0('~',weight)), nest=TRUE, data=x)
}
designs <- list(MEC=make_design('WTMEC2YR'), Interview=make_design('WTINT2YR'))

# Keep modest parameter counts given only 15 design degrees of freedom.
# These are candidate adjustment sets, not automatically sufficient
# causal confounder controls. BMI/smoking are exploratory sensitivity.
sets <- list(
  basic=c('RIDAGEYR','sex'),
  income=c('RIDAGEYR','sex','pir'),
  education=c('RIDAGEYR','sex','education3'),
  income_education=c('RIDAGEYR','sex','pir','education3'),
  bmi_smoking_sensitivity=c('RIDAGEYR','sex','pir','bmi','ever_smoked')
)
form <- function(y,x,covars,quad=FALSE) {
  reformulate(c(x, if (quad) paste0('I(',x,'^2)'), covars), response=y)
}
rows <- function(x) if (length(x)) do.call(rbind,x) else data.frame()
coefs <- list(); tests <- list(); counts <- list(); diagnostics <- list()
common <- list(); curves <- list()

# Survey Wald tests. A joint exposure test answers whether the
# linear AND squared terms are both zero. The curvature test asks
# whether the squared term is zero given the linear term.
wald_test <- function(fit, terms) {
  df <- fit$df.residual
  if (!is.finite(df) || df <= 0) {
    return(list(p=NA_real_, F=NA_real_, note='insufficient residual survey df'))
  }
  b <- coef(fit)
  if (!all(terms %in% names(b))) {
    return(list(p=NA_real_, F=NA_real_, note='terms missing / aliased'))
  }
  V <- vcov(fit)[terms,terms,drop=FALSE]
  if (any(!is.finite(V)) || qr(V)$rank < length(terms)) {
    return(list(p=NA_real_, F=NA_real_, note='singular coefficient covariance'))
  }
  bb <- matrix(b[terms], ncol=1)
  fstat <- as.numeric(t(bb) %*% solve(V,bb)) / length(terms)
  p <- pf(fstat, df1=length(terms), df2=df, lower.tail=FALSE)
  list(p=p,F=fstat,note='design-based Wald F; residual survey df')
}
coef_table <- function(fit, weight, rq, adj, shape, n) {
  b <- coef(fit); se <- sqrt(diag(vcov(fit)))
  df <- fit$df.residual
  valid <- is.finite(df) && df > 0
  pv <- if (valid) 2*pt(abs(b/se),df=df,lower.tail=FALSE) else rep(NA_real_,length(b))
  crit <- if (valid) qt(.975,df=df) else NA_real_
  lo <- b-crit*se; hi <- b+crit*se
  data.frame(weights=weight,rq=rq,adjustment=adj,shape=shape,n=n,
             residual_df=df,term=names(b),estimate=as.numeric(b),
             se=as.numeric(se),p_value=as.numeric(pv),
             ci_lower=as.numeric(lo),ci_upper=as.numeric(hi),
             odds_ratio=if(rq=='RQ2') exp(as.numeric(b)) else NA_real_,
             or_lower=if(rq=='RQ2') exp(as.numeric(lo)) else NA_real_,
             or_upper=if(rq=='RQ2') exp(as.numeric(hi)) else NA_real_)
}

# Predict conditional mean / probability for reference covariates.
# Point predictions only: not population-marginal curves or confidence bands.
make_curve <- function(fit, data, exposure, covars, rq, weight, adj) {
  vals <- data[[exposure]]
  rng <- quantile(vals,c(.05,.95),na.rm=TRUE,names=FALSE)
  if (!all(is.finite(rng)) || rng[1]>=rng[2]) return(NULL)
  grid <- data.frame(seq(rng[1],rng[2],length.out=81))
  names(grid) <- exposure
  for (v in covars) {
    if (is.factor(data[[v]])) {
      # Most frequent observed category, preserving original factor levels.
      tab <- table(data[[v]])
      chosen <- names(tab)[which.max(tab)]
      grid[[v]] <- factor(chosen,levels=levels(data[[v]]))
    } else {
      grid[[v]] <- median(data[[v]],na.rm=TRUE)
    }
  }
  pred <- tryCatch(as.numeric(predict(fit,newdata=grid,type='response')),
                   error=function(e) {warning('Curve prediction: ',conditionMessage(e)); NULL})
  if (is.null(pred) || length(pred)!=nrow(grid)) return(NULL)
  data.frame(weights=weight,rq=rq,adjustment=adj,
             exposure=exposure,exposure_value=grid[[exposure]],
             predicted_outcome=pred,
             outcome_scale=if(rq=='RQ1') 'PHQ-9 points' else 'probability',
             prediction_type='conditional reference-covariate point prediction')
}

for (w in names(designs)) {
  parent <- designs[[w]]
  for (rq in c('RQ1','RQ2')) {
    y <- if(rq=='RQ1') 'phq' else 'elevated'
    x <- if(rq=='RQ1') 'sleep_hours' else 'sed_hours'
    family <- if(rq=='RQ1') gaussian() else quasibinomial()
    for (adj in names(sets)) {
      covars <- sets[[adj]]
      needed <- unique(c(y,x,covars))
      keep <- complete.cases(parent$variables[,needed,drop=FALSE])
      domain <- subset(parent,keep)
      n <- nrow(domain$variables)
      events <- sum(domain$variables$elevated==1,na.rm=TRUE)
      psu <- nrow(unique(domain$variables[c('SDMVSTRA','SDMVPSU')]))
      strata <- length(unique(domain$variables$SDMVSTRA))
      key <- paste(w,rq,adj,sep='__')
      counts[[key]] <- data.frame(weights=w,rq=rq,adjustment=adj,n=n,
                                  events=events,psus=psu,strata=strata,
                                  design_df=survey::degf(domain))
      if(n<100 || (rq=='RQ2' && events<20)) {warning('Skipping ',key);next}
      fit <- function(q) tryCatch(
        survey::svyglm(form(y,x,covars,q),design=domain,family=family),
        error=function(e){warning(key,': ',conditionMessage(e));NULL})
      linear <- fit(FALSE); quad <- fit(TRUE)
      if(is.null(linear)||is.null(quad)) next
      coefs[[paste0(key,'__linear')]] <- coef_table(linear,w,rq,adj,'linear',n)
      coefs[[paste0(key,'__quadratic')]] <- coef_table(quad,w,rq,adj,'quadratic',n)
      curvature <- wald_test(quad,paste0('I(',x,'^2)'))
      overall <- wald_test(quad,c(x,paste0('I(',x,'^2)')))
      tests[[key]] <- data.frame(weights=w,rq=rq,adjustment=adj,n=n,
        curvature_F=curvature$F,curvature_p=curvature$p,
        overall_exposure_F=overall$F,overall_exposure_p=overall$p,
        curvature_note=curvature$note,overall_note=overall$note,
        residual_df=quad$df.residual)
      diagnostics[[key]] <- data.frame(weights=w,rq=rq,adjustment=adj,n=n,
        design_df=survey::degf(domain),linear_rank=linear$rank,
        quadratic_rank=quad$rank,linear_residual_df=linear$df.residual,
        quadratic_residual_df=quad$df.residual)
      cat(sprintf('%-40s N=%4d df=%3s curvature_p=%-10s overall_p=%s\n',
        key,n,format(quad$df.residual),format(curvature$p,digits=4),
        format(overall$p,digits=4)))
      curve <- make_curve(quad,domain$variables,x,covars,rq,w,adj)
      if(!is.null(curve)) curves[[key]] <- curve
      if(adj!='basic') {
        basic_same <- tryCatch(survey::svyglm(form(y,x,sets$basic,TRUE),
           design=domain,family=family),error=function(e) NULL)
        if(!is.null(basic_same)) {
          bcurv <- wald_test(basic_same,paste0('I(',x,'^2)'))
          bover <- wald_test(basic_same,c(x,paste0('I(',x,'^2)')))
          common[[key]] <- data.frame(weights=w,rq=rq,adjustment=adj,n=n,
            basic_curvature_p_same_sample=bcurv$p,
            adjusted_curvature_p_same_sample=curvature$p,
            basic_overall_p_same_sample=bover$p,
            adjusted_overall_p_same_sample=overall$p)
        }
      }
    }
  }
}

outputs <- list(
 'model_coefficients.csv'=rows(coefs),
 'association_tests.csv'=rows(tests),
 'sample_counts.csv'=rows(counts),
 'model_diagnostics.csv'=rows(diagnostics),
 'common_sample_comparisons.csv'=rows(common),
 'predicted_curves.csv'=rows(curves)
)
for (nm in names(outputs)) write.csv(outputs[[nm]],file.path(out,nm),row.names=FALSE)

# Backward-compatible nonlinearity output and headline associations.
write.csv(outputs[['association_tests.csv']],file.path(out,'nonlinearity_tests.csv'),row.names=FALSE)
ct <- outputs[['model_coefficients.csv']]
headline <- ct[ct$shape=='linear' &
 ((ct$rq=='RQ1' & ct$term=='sleep_hours') |
  (ct$rq=='RQ2' & ct$term=='sed_hours')), ,drop=FALSE]
write.csv(headline,file.path(out,'headline_associations.csv'),row.names=FALSE)

capture.output({
  cat('PHASE 4: PARSIMONIOUS SURVEY-ADJUSTED ASSOCIATION VALIDATION\n\n')
  cat('MODEL SAMPLE COUNTS\n');print(outputs[['sample_counts.csv']],row.names=FALSE)
  cat('\nMODEL DIAGNOSTICS\n');print(outputs[['model_diagnostics.csv']],row.names=FALSE)
  cat('\nCURVATURE AND JOINT EXPOSURE TESTS\n')
  print(outputs[['association_tests.csv']],row.names=FALSE)
  cat('\nCOMMON-SAMPLE COMPARISONS\n')
  print(outputs[['common_sample_comparisons.csv']],row.names=FALSE)
  cat('\nLINEAR EXPOSURE COEFFICIENTS\n');print(headline,row.names=FALSE)
  cat('\nCAUTIONS\n')
  cat('- These are cross-sectional associations, not causal effects.\n')
  cat('- Confirm NHANES 2021-2023 cycle-specific weights and codebooks.\n')
  cat('- MEC models are primary candidates; interview models sensitivity only.\n')
  cat('- Survey residual df constrain Wald-test reliability.\n')
  cat('- Curvature and overall association are different hypothesis tests.\n')
  cat('- Different adjustment sets may have different complete-case samples.\n')
  cat('- Curves are conditional predictions at reference covariates, not marginal.\n')
  cat('- Income/education adjustment is not automatically a causal adjustment set.\n')
  cat('- No automatic multiple-testing correction; interpret exploratory tests cautiously.\n')
},file=file.path(out,'summary.txt'))
cat('\nReports saved in: ',out,'\n',sep='')
cat(paste(names(outputs),collapse=', '),'\n')
cat('Also: nonlinearity_tests.csv, headline_associations.csv, summary.txt\n')
