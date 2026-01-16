# ==============================================================================
# 1. MATH & HELPER FUNCTIONS
# ==============================================================================

inv_logit <- function(x){ 1 / (1 + exp(-x)) }
deg2rad <- function(x) x * pi / 180 %% (2*pi)
rad2deg <- function(x) x * 180 / pi %% (360)

# Manual Von Mises PDF
dvm_manual <- function(x_rad, mu, kappa) {
  exp(kappa * cos(x_rad - mu)) / (2 * pi * besselI(kappa, 0))
}

estimate_shared_log_likelihood <- function(x1, x2, kappa){
  delta_x <- abs(x1 - x2)
  delta_x <- ifelse(delta_x > pi, 2 * pi - delta_x, delta_x)
  kappa_eff <- 2 * kappa * cos(delta_x / 2)
  log_num <- log(besselI(kappa_eff, 0))
  log_den <- 2 * log(2 * pi) + 2 * log(besselI(kappa, 0))
  return(log_num - log_den)
}

compute_p_stay <- function(x1, x2, x3, x4, kappa, p=0.5){
  log_lik_1 <- estimate_shared_log_likelihood(x1, x3, kappa) + 
    estimate_shared_log_likelihood(x2, x4, kappa)
  log_lik_2 <- estimate_shared_log_likelihood(x1, x4, kappa) + 
    estimate_shared_log_likelihood(x2, x3, kappa)
  log_p1 <- log(p) + log_lik_1
  log_p2 <- log(1 - p) + log_lik_2
  return(inv_logit(log_p1 - log_p2))
}

calc_mixture_circ_mean_and_conf <- function(theta, kappa_1, kappa_2, mu_1, mu_2){
  r1 <- besselI(kappa_1, 1) / besselI(kappa_1, 0)
  r2 <- besselI(kappa_2, 1) / besselI(kappa_2, 0)
  z1 <- r1 * exp(1i * mu_1)
  z2 <- r2 * exp(1i * mu_2)
  z_mix <- theta * z1 + (1 - theta) * z2
  return(list(angle = Arg(z_mix), length = Mod(z_mix)))
}

compute_circular_mean <- function(angle1, angle2){
  x <- cos(angle1) + cos(angle2)
  y <- sin(angle1) + sin(angle2)
  return(atan2(y, x))
}

# ==============================================================================
# 2. THE POLAR PLOTTING FUNCTION
# ==============================================================================

plot_polar_posterior <- function(mu_stay, kappa_stay, mu_swap, kappa_swap, 
                                 p_stay, est_angle, est_conf,
                                 a, b, c, d) {
  
  # 1. Generate Grid
  theta <- seq(0, 2*pi, length.out=720)
  
  # 2. Calculate Densities
  y_stay <- dvm_manual(theta, mu_stay, kappa_stay)
  y_swap <- dvm_manual(theta, mu_swap, kappa_swap)
  y_mix  <- p_stay * y_stay + (1 - p_stay) * y_swap
  
  # 3. Determine Geometry & Limits
  max_density <- max(y_mix)
  
  # Define the "Rim" where points sit (outside the density)
  rim_r <- max_density * 1.2 
  
  # Define where Text sits (outside the rim)
  text_r <- rim_r * 1.2
  
  # Define the Plot Limits (must include the text)
  plot_limit <- text_r * 1.1
  
  # 4. Convert Polar to Cartesian for Polygons
  xs_mix <- y_mix * cos(theta); ys_mix <- y_mix * sin(theta)
  xs_stay <- (p_stay * y_stay) * cos(theta); ys_stay <- (p_stay * y_stay) * sin(theta)
  xs_swap <- ((1-p_stay) * y_swap) * cos(theta); ys_swap <- ((1-p_stay) * y_swap) * sin(theta)
  
  # 5. DRAW PLOT
  # xpd=TRUE allows drawing outside the plot region (prevents clipping labels)
  par(pty="s", mar=c(1,1,2,1), xpd=TRUE) 
  
  plot(0, 0, type="n", 
       xlim=c(-plot_limit, plot_limit), 
       ylim=c(-plot_limit, plot_limit), 
       axes=FALSE, xlab="", ylab="", 
       main=paste0("Posterior Belief (P_stay = ", round(p_stay, 2), ")"))
  
  # Reference Rings (at 50% and 100% of density max)
  symbols(c(0,0), c(0,0), circles=c(max_density*0.5, max_density), 
          add=TRUE, fg="gray90", inches=FALSE)
  abline(h=0, v=0, col="gray90")
  
  # Draw Components
  polygon(xs_stay, ys_stay, border="blue", col=rgb(0,0,1,0.05), lty=2)
  polygon(xs_swap, ys_swap, border="red",  col=rgb(1,0,0,0.05), lty=2)
  
  # Draw Total Mixture
  polygon(xs_mix, ys_mix, border="black", col=rgb(0,0,0,0.1), lwd=2)
  
  # Draw Estimate Arrow (Scale relative to density so it looks right)
  # We make the arrow max length equal to the max density radius
  arrow_len <- est_conf * max_density 
  arrows(0, 0, arrow_len*cos(est_angle), arrow_len*sin(est_angle), 
         col="purple", lwd=3, length=0.1)
  
  # 6. PLOT INPUT POINTS (On the calculated Rim)
  
  # Helper to plot point and text
  plot_rim_point <- function(angle, label, col, pch) {
    points(rim_r*cos(angle), rim_r*sin(angle), pch=pch, col=col, cex=1.5)
    text(text_r*cos(angle), text_r*sin(angle), label, col=col, font=2, cex=0.9)
  }
  
  plot_rim_point(a, "A", "darkgreen", 17)
  plot_rim_point(b, "B", "brown", 17)
  plot_rim_point(c, "C", "blue", 19)
  plot_rim_point(d, "D", "red", 19)
  
  legend("topleft", legend=c("Posterior", "Est (Conf)", "Hyp: Stay", "Hyp: Swap"),
         col=c("black", "purple", "blue", "red"), lwd=c(2,3,1,1), lty=c(1,1,2,2), 
         bty="n", cex=0.8)
}

# ==============================================================================
# 3. UPDATED MAIN FUNCTION
# ==============================================================================

compute_ideal_observer_estimates_1d <- function(a, b, c, d, kappa_tilde, p_prior, plot_result=FALSE){
  
  # 1. P(Stay)
  p_stay <- compute_p_stay(a, b, c, d, kappa_tilde, p_prior)
  
  # 2. Means
  mu_stay <- compute_circular_mean(a, c)
  mu_swap <- compute_circular_mean(a, d)
  
  # 3. Kappas
  delta_x_1 <- abs(a - c); delta_x_1 <- ifelse(delta_x_1 > pi, 2*pi - delta_x_1, delta_x_1)
  delta_x_2 <- abs(a - d); delta_x_2 <- ifelse(delta_x_2 > pi, 2*pi - delta_x_2, delta_x_2)
  
  kappa_eff_1 <- 2 * kappa_tilde * cos(delta_x_1 / 2)
  kappa_eff_2 <- 2 * kappa_tilde * cos(delta_x_2 / 2)
  
  # 4. Estimate
  res <- calc_mixture_circ_mean_and_conf(p_stay, kappa_eff_1, kappa_eff_2, mu_stay, mu_swap)
  
  # 5. Plotting (Optional)
  if(plot_result){
    plot_polar_posterior(mu_stay, kappa_eff_1, mu_swap, kappa_eff_2, 
                         p_stay, res$angle, res$length,
                         a, b, c, d)
  }
  
  return(list(
    estimate = res$angle,
    confidence = res$length,
    p_stay = p_stay
  ))
}

# ==============================================================================
# 4. EXAMPLE RUN (Simulated Experiment with Error Check)
# ==============================================================================

# 1. Generate Ground Truth
# Randomly place two objects on the circle
x_1 <- as.numeric(rvonmises(1, circular(0), 0)) 
x_2 <- as.numeric(rvonmises(1, circular(0), 0)) 

# 2. Decide Association (Hidden State)
# 0 = Stay (Obj 1 -> Obs C), 1 = Swap (Obj 1 -> Obs D)
swap <- rbinom(1, 1, .5) 

# 3. Generate Noisy Measurements (Von Mises)
kappa_gen <- 5

# Time 1: A comes from x1, B comes from x2
a <- as.numeric(rvonmises(1, circular(x_1), kappa_gen))
b <- as.numeric(rvonmises(1, circular(x_2), kappa_gen))

# Time 2: C and D depend on the swap
if(swap == 0){
  c <- as.numeric(rvonmises(1, circular(x_1), kappa_gen))
  d <- as.numeric(rvonmises(1, circular(x_2), kappa_gen))
} else {
  c <- as.numeric(rvonmises(1, circular(x_2), kappa_gen))
  d <- as.numeric(rvonmises(1, circular(x_1), kappa_gen))
}

# 4. Run the Ideal Observer
est <- compute_ideal_observer_estimates_1d(
  a, b, c, d,
  kappa_tilde = 5,
  p_prior = 0.5,
  plot_result = TRUE
)

# 5. Calculate Circular Error vs Truth (x_1)
# Shortest distance around the circle
diff_rad <- abs(est$estimate - x_1)
diff_rad <- ifelse(diff_rad > pi, 2*pi - diff_rad, diff_rad)
diff_deg <- rad2deg(diff_rad)

# 6. Report Results
cat("------------------------------------------------\n")
cat(sprintf("TRUE STATE:  %s\n", ifelse(swap==0, "STAY", "SWAP")))
cat(sprintf("CALC PROB:   P(Stay) = %.4f\n", est$p_stay))
cat("------------------------------------------------\n")
cat(sprintf("TRUE X1:     %.2f deg\n", rad2deg(x_1)))
cat(sprintf("ESTIMATE:    %.2f deg\n", rad2deg(est$estimate)))
cat(sprintf("ERROR:       %.2f deg\n", diff_deg))
cat(sprintf("CONFIDENCE:  %.2f (R)\n", est$confidence))
cat("------------------------------------------------\n")

if(diff_deg < 20) {
  cat("VERDICT:     GOOD TRACKING\n")
} else {
  cat("VERDICT:     LOST TRACK (Likely Ambiguous or Wrong Swap)\n")
}