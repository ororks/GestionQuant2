if (!require("copula")) {
  install.packages("copula", dependencies=TRUE)
  library(copula)
}

tryCatch({
  residuals <- read.csv("residuals.csv")
}, error = function(e) {
  cat("Error reading the CSV file: ", e$message, "\n")
  quit("no")
})

if (!("Residuals1" %in% names(residuals)) || !("Residuals2" %in% names(residuals))) {
  cat("CSV file does not contain required columns.\n")
  quit("no")
}

u <- pnorm(residuals$Residuals1)
v <- pnorm(residuals$Residuals2)


copula_model <- switch(copula_type,
                       "Normal" = normalCopula(dim = 2),
                       "Student" = tCopula(dim = 2),
                       "Plackett" = plackettCopula(param = 2),
                       "Clayton" = claytonCopula(dim = 2),
                       "rotClayton" = rotCopula(claytonCopula(dim = 2)),
                       "Frank" = frankCopula(dim = 2),
                       "Gumbel" = gumbelCopula(dim = 2),
                       "rotGumbel" = rotCopula(gumbelCopula(dim = 2)),
                       normalCopula(dim = 2))  # Default case

suppressMessages(suppressWarnings({
  fit <- fitCopula(copula_model, data = cbind(u, v), method = "ml")
  fit_summary <- capture.output(print(summary(fit)))
  for (line in fit_summary) {
    print(enc2utf8(line))
  }
}))

fit_results <- summary(fit)
assign('fit_results', fit_results, envir = .GlobalEnv)
