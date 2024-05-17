Sys.setlocale("LC_ALL", "English_United States.UTF-8")
library(copula)
library(rmgarch)

fit_copula <- function(residuals_data, copula_type = "Normal") {
  copula_model <- switch(copula_type,
                         "Normal" = normalCopula(dim = 2),
                         "Student" = tCopula(dim = 2),
                         "Plackett" = plackettCopula(param = 2),
                         "Clayton" = claytonCopula(dim = 2),
                         "rotClayton" = rotCopula(claytonCopula(dim = 2), 180),
                         "Frank" = frankCopula(dim = 2),
                         "Gumbel" = gumbelCopula(dim = 2),
                         "rotGumbel" = rotCopula(gumbelCopula(dim = 2), 180),
                         normalCopula(dim = 2)) # Default to Normal Copula
  
  u <- pnorm(residuals_data[, "Residuals1"])
  v <- pnorm(residuals_data[, "Residuals2"])
  
  u_sorted <- sort(u)
  v_sorted <- sort(v)
  
  fit <- fitCopula(copula_model, cbind(u, v), method = "ml")
  
  simulated <- rCopula(10000, fit@copula)
  
  simulated_normal <- cbind(qnorm(simulated[,1]), qnorm(simulated[,2]))
  
  return(simulated_normal)
}


fit_copula_params <- function(residuals_data, copula_type = "Normal") {
  copula_model <- switch(copula_type,
                         "Normal" = normalCopula(dim = 2),
                         "Student" = tCopula(dim = 2),
                         "Plackett" = plackettCopula(param = 2),
                         "Clayton" = claytonCopula(dim = 2),
                         "rotClayton" = rotCopula(claytonCopula(dim = 2), flip = c(TRUE, FALSE)),
                         "Frank" = frankCopula(dim = 2),
                         "Gumbel" = gumbelCopula(dim = 2),
                         "rotGumbel" = rotCopula(gumbelCopula(dim = 2), flip = c(TRUE, FALSE)),
                         normalCopula(dim = 2)) # Default to Normal Copula
  
  u <- pnorm(residuals_data[, "Residuals1"])
  v <- pnorm(residuals_data[, "Residuals2"])
  
  fit <- fitCopula(copula_model, cbind(u, v), method = "ml")
  
  logLik <- logLik(fit)
  n_params <- length(fit@estimate)
  n_obs <- nrow(residuals_data)
  
  AIC <- -2 * logLik + 2 * n_params
  BIC <- -2 * logLik + log(n_obs) * n_params
  
  result <- list(
    summary = summary(fit),
    AIC = AIC,
    BIC = BIC
  )
  
  return(result)
}
