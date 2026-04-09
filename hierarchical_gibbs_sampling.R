# =============================================================================
# Bayesian Hierarchical Normal Model with Gibbs Sampling
# Topic: U.S. Political Opinions on Military Aid to Ukraine
# Data: Nationally representative survey (n = 1,603)
# Groups: Democrats, Independents, Republicans
# Stances: Increase Aid, Maintain Aid, Not Sure, Decrease Aid
# =============================================================================

library(tidyverse)

# =============================================================================
# DATA SETUP
# =============================================================================

total_n <- 1603
group_n <- round(total_n / 3)  # ~534 per political group

# Survey percentage data (12 group-stance combinations)
percent_data <- tribble(
  ~Group,          ~Stance,        ~Percent,
  "Democrats",     "Increase Aid",  35,
  "Democrats",     "Maintain Aid",  39,
  "Democrats",     "Not Sure",      16,
  "Democrats",     "Decrease Aid",  10,
  "Independents",  "Increase Aid",  19,
  "Independents",  "Maintain Aid",  23,
  "Independents",  "Not Sure",      26,
  "Independents",  "Decrease Aid",  33,
  "Republicans",   "Increase Aid",  10,
  "Republicans",   "Maintain Aid",  24,
  "Republicans",   "Not Sure",      21,
  "Republicans",   "Decrease Aid",  45
)

# Convert percentages to counts and proportions
separate_data <- percent_data %>%
  mutate(
    Count      = round((Percent / 100) * group_n),
    Group_Size = group_n,
    Proportion = Count / Group_Size
  )

# =============================================================================
# BAYESIAN POSTERIOR FUNCTION (Conjugate Normal-Normal)
# Likelihood: y ~ N(theta, sigma2)
# Prior:      theta ~ N(mu0, tau2)
# Posterior:  theta | y ~ N(post_mean, post_var)
# =============================================================================

posterior_estimate_prop <- function(y_hat, n, sigma2 = 0.0025, mu0 = 0.5, tau2 = 1) {
  post_var  <- 1 / (n / sigma2 + 1 / tau2)
  post_mean <- post_var * (y_hat * n / sigma2 + mu0 / tau2)
  list(mean = post_mean, sd = sqrt(post_var))
}

# =============================================================================
# MODEL 1: SEPARATE MODEL
# Each of the 12 group-stance combinations treated independently
# Prior: theta ~ N(0, 100^2) [weakly informative]
# =============================================================================

separate_results <- separate_data %>%
  rowwise() %>%
  mutate(
    Est           = list(posterior_estimate_prop(Proportion, Group_Size)),
    Posterior_Mean = Est$mean,
    Posterior_SD   = Est$sd,
    CI_Lower       = Posterior_Mean - 1.96 * Posterior_SD,
    CI_Upper       = Posterior_Mean + 1.96 * Posterior_SD
  ) %>%
  ungroup() %>%
  select(Group, Stance, Proportion, Posterior_Mean, Posterior_SD, CI_Lower, CI_Upper)

cat("=== SEPARATE MODEL RESULTS ===\n")
print(separate_results)

# =============================================================================
# MODEL 2: POOLED MODEL
# Ignore political affiliation — estimate overall public opinion on each stance
# =============================================================================

combined_data <- percent_data %>%
  mutate(
    Group_Size = group_n,
    Count      = round((Percent / 100) * group_n)
  ) %>%
  group_by(Stance) %>%
  summarise(
    Total_Count = sum(Count),
    Total_Size  = sum(Group_Size),
    Proportion  = Total_Count / Total_Size,
    .groups = "drop"
  )

pooled_results <- combined_data %>%
  rowwise() %>%
  mutate(
    Est           = list(posterior_estimate_prop(Proportion, Total_Size)),
    Posterior_Mean = Est$mean,
    Posterior_SD   = Est$sd,
    CI_Lower       = Posterior_Mean - 1.96 * Posterior_SD,
    CI_Upper       = Posterior_Mean + 1.96 * Posterior_SD
  ) %>%
  ungroup() %>%
  select(Stance, Proportion, Posterior_Mean, Posterior_SD, CI_Lower, CI_Upper)

cat("\n=== POOLED MODEL RESULTS ===\n")
print(pooled_results)

# =============================================================================
# MODEL 3: HIERARCHICAL MODEL WITH GIBBS SAMPLING
# Focus: "Increase Aid" stance across 3 political groups
#
# Model structure:
#   y_j | theta_j  ~ N(theta_j, sigma2)    [likelihood]
#   theta_j | mu, tau2 ~ N(mu, tau2)       [group-level prior]
#   mu               ~ N(0.5, 1)           [hyperprior on mean]
#   tau2             ~ Inv-Gamma(shape, rate) [hyperprior on variance]
#
# Parameters:
#   sigma2 = 0.0025 (fixed known variance)
#   J = 3 groups (Democrats, Independents, Republicans)
# =============================================================================

increase_data <- separate_data %>%
  filter(Stance == "Increase Aid") %>%
  mutate(y = Proportion, n = Group_Size)

# Observed data
y      <- increase_data$y
n      <- increase_data$n
J      <- length(y)
sigma2 <- 0.0025  # fixed known variance

# Initialize parameters
mu    <- 0.5
tau2  <- 0.01
theta <- y  # start at observed proportions

# Gibbs sampler settings
n_iter        <- 2000
theta_samples <- matrix(NA, nrow = n_iter, ncol = J)
mu_samples    <- numeric(n_iter)
tau2_samples  <- numeric(n_iter)

set.seed(42)

for (t in 1:n_iter) {

  # --- Step 1: Sample theta_j (group-level means) ---
  for (j in 1:J) {
    V_theta  <- 1 / (n[j] / sigma2 + 1 / tau2)
    m_theta  <- V_theta * (y[j] * n[j] / sigma2 + mu / tau2)
    theta[j] <- rnorm(1, m_theta, sqrt(V_theta))
  }

  # --- Step 2: Sample mu (overall population mean) ---
  V_mu <- 1 / (J / tau2 + 1)
  m_mu <- V_mu * (sum(theta) / tau2 + 0.5)
  mu   <- rnorm(1, m_mu, sqrt(V_mu))

  # --- Step 3: Sample tau^2 (between-group variance) ---
  shape <- (J - 1) / 2
  scale <- sum((theta - mu)^2) / 2
  tau2  <- 1 / rgamma(1, shape = shape, rate = scale)

  # Store draws
  theta_samples[t, ] <- theta
  mu_samples[t]      <- mu
  tau2_samples[t]    <- tau2
}

colnames(theta_samples) <- increase_data$Group

# --- Posterior Summaries ---
theta_summary <- apply(theta_samples, 2, function(x) quantile(x, c(0.025, 0.5, 0.975)))
mu_summary    <- quantile(mu_samples,   c(0.025, 0.5, 0.975))
tau2_summary  <- quantile(tau2_samples, c(0.025, 0.5, 0.975))

cat("\n=== HIERARCHICAL MODEL RESULTS (Gibbs Sampler, n_iter=2000) ===\n")
cat("\nPosterior summaries for group-level support (theta):\n")
print(round(theta_summary, 4))

cat("\nPosterior summary for overall population mean (mu):\n")
print(round(mu_summary, 4))

cat("\nPosterior summary for between-group variance (tau^2):\n")
print(round(tau2_summary, 6))

# =============================================================================
# INTERPRETATION
# =============================================================================
cat("\n=== KEY FINDINGS ===\n")
cat("Democrats    — posterior median support for increasing aid:", round(theta_summary["50%", "Democrats"], 3), "\n")
cat("Independents — posterior median support for increasing aid:", round(theta_summary["50%", "Independents"], 3), "\n")
cat("Republicans  — posterior median support for increasing aid:", round(theta_summary["50%", "Republicans"], 3), "\n")
cat("\nThe hierarchical model shares information across groups via the hyperparameters\n")
cat("(mu, tau^2), providing shrinkage toward the overall mean while preserving\n")
cat("group-level differences — especially valuable with only J=3 groups.\n")
