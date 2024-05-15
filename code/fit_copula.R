# Load necessary library
Sys.setlocale("LC_ALL", "English_United States.UTF-8")
#options(encoding = "UTF-8")
library(copula)
library(rmgarch)

# Define the function fit_copula
fit_copula <- function(residuals_data, copula_type = "Normal") {
  # Define the copula model based on user input
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
  
  # Load data from dataframe assuming the names match those in the dataframe
  u <- pnorm(residuals_data[, "Residuals1"])
  v <- pnorm(residuals_data[, "Residuals2"])
  
  u_sorted <- sort(u)
  v_sorted <- sort(v)
  
  # Fit the copula to the data
  fit <- fitCopula(copula_model, cbind(u, v), method = "ml")
  
  # Simulate new data
  simulated <- rCopula(10000, fit@copula)
  
  # Transform simulated data back using quantile function of normal distribution
  simulated_normal <- cbind(qnorm(simulated[,1]), qnorm(simulated[,2]))
  
  return(simulated_normal)
}

#copula_dens <- fit_copula(residuals, copula_type = "Normal")
