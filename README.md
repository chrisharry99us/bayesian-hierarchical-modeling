# Bayesian Hierarchical Modeling

**Master of Data Science — Bayesian Statistics**

This repository contains two Bayesian analysis projects completed as part of an MDS program, demonstrating Bayesian inference techniques from foundational approximations through advanced hierarchical modeling with custom MCMC sampling.

---

## Projects

### 1. Hierarchical Normal Model with Gibbs Sampling
**`hierarchical_gibbs_sampling.R`**

Analysis of U.S. political opinions on military aid to Ukraine (n = 1,603), grouped by party affiliation (Democrat, Independent, Republican). Compares three Bayesian modeling strategies:

| Model | Approach | Groups |
|-------|----------|--------|
| **Separate** | Independent posteriors per group-stance pair | 12 (3 parties × 4 stances) |
| **Pooled** | Single shared posterior, ignores group structure | 4 (stances only) |
| **Hierarchical** | Group-specific means drawn from shared population distribution | 3 parties |

The hierarchical model uses a **custom Gibbs sampler** (2,000 iterations) to jointly estimate:
- `θⱼ` — true support level for each political group
- `μ` — overall population mean support
- `τ²` — between-group variance

**Key Findings:**
- Democrats: ~35.0% support for increasing aid (posterior median)
- Independents: ~18.9% support
- Republicans: ~9.9% support
- The hierarchical approach outperforms both separate and pooled models by sharing information across groups while preserving group-level variation

**Methods:** Conjugate normal-normal model, Gibbs sampling, credible intervals, posterior predictive inference

---

### 2. Bayesian Crime Analysis — Los Angeles
**`bayesian_la_crime.R`**

Bayesian analysis of crime victimization patterns in Los Angeles using open data from the LAPD (2020–present). Applies two techniques:

- **Normal approximation to posterior** — estimates crime proportions by gender and geographic area
- **Bayesian linear regression via `brms`** — models victim age and crime type as functions of time, area, and crime category

**Research Questions Answered:**
1. Are men or women more likely to be crime victims? → ~53% male / 47% female
2. What predicts victim age? → Crime type, area, and time of day all significant
3. Which areas have the highest crime rates? → Central (7.2%), 77th Street (6.1%), North Hollywood (5.5%)
4. Does time of day affect crime type? → Violent crimes peak at night; property crimes more common in daytime

**Methods:** Bayesian binomial approximation, `brms` package, Normal(0,10) weakly informative priors, MCMC (4 chains × 2,000 iterations), posterior summaries with 95% credible intervals

---

## Tech Stack

- **Language:** R
- **Key Packages:** `tidyverse`, `brms`, `rstan`
- **Sampling:** Custom Gibbs sampler (hierarchical model), Stan/MCMC (brms)
- **Inference:** Conjugate normal-normal model, posterior means, credible intervals

---

## Repository Structure

```
bayesian-hierarchical-modeling/
├── hierarchical_gibbs_sampling.R   # Hierarchical model: separate + pooled + hierarchical
├── bayesian_la_crime.R             # LA crime analysis: binomial approx + brms regression
└── README.md
```

---

## Author

**Chris Harry** — Master of Data Science
