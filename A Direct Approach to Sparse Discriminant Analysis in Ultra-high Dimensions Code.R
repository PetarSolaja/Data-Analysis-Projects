install.packages("glmnet")
install.packages("MASS")
install.packages("pamr")
install.packages("penalizedLDA")

library(MASS)
library(glmnet)
library(pamr)
library(penalizedLDA)




set.seed(2606) 

############################################PARAMETERS########################################################
n_sims <- 100 #number of simulations
n <- 100 # sample size
p <- 400 # number of predictors
pi1 <- 0.5 # proportion of class 1
pi2 <- 0.5 # proportion of class 2
beta <- 0.556 * c(3, 1.5, 0, 0, 2, rep(0, p-5)) # coefficient vector

# covariance matrix for both classes
sigma <- matrix(0, nrow = p, ncol = p)
for (i in 1:p) {
  for (j in 1:p) {
    sigma[i,j] <- 0.5^(abs(i-j))
  }
}

# mean vectors for each class
mu1 <- rep(0, p)
mu2 <- sigma%*%beta


###############################################SIMULATIONS#####################################################

lasso_errors <- numeric(n_sims)
nsc_errors <- numeric(n_sims)
l1_errors <- numeric(n_sims)
ttest_errors <- numeric(n_sims)

for (s in 1:n_sims) {
  
  # generate class labels
  y <- sample(c(1,2), size = n, replace = TRUE, prob = c(pi1, pi2))
  
  # generate predictors for each class
  x <- matrix(0, nrow = n, ncol = p)
  for (i in 1:n) {
    if (y[i] == 1) {
      x[i,] <- MASS::mvrnorm(1, mu1, sigma)
    } else {
      x[i,] <- MASS::mvrnorm(1, mu2, sigma)
    }
  }
  
  # split data into training and testing sets
  n_train <- round(2 * n / 3)
  train_set <- sample(n, n_train)
  test_set <- setdiff(1:n, train_set)
  x_train <- x[train_set,]
  y_train <- y[train_set]
  x_test <- x[test_set,]
  y_test <- y[test_set]
  
  ##################Lassoed Discriminant Analysis###################################
  
  # perform cross-validation to choose lambda
  cv_fit <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1)
  lambda_min <- cv_fit$lambda.min
  
  # fit Lasso model using lambda_min
  fit <- glmnet(x_train, y_train, family = "binomial", alpha = 1, lambda = lambda_min)
  
  # predict on testing data
  y_pred_lasso <- predict(fit, newx = x_test, type = "class")
  
  # calculate error percentage
  lasso_errors[s] <- mean(y_pred_lasso != y_test) * 100
  
  #################Nearest Shrunken Centroids########################################
  
  nsc_train_data <- list(x = t(x_train),y = factor(y_train))
  nsc_test_data <- list(x= t(x_test),y=factor(y_test))
  
  nsc_fit <- pamr.train(nsc_train_data) #trains NSC model 
  
  pred_nsc <- pamr.predict(nsc_fit, nsc_test_data$x , threshold=1)
  
  nsc_errors[s] <- mean(pred_nsc != y_test) * 100
  
  #################L1-penalized Linear Discriminant#####################################
  
  cv_fit2 <- PenalizedLDA.cv(x_train,y_train,lambdas=c(1e-4,1e-3,1e-2,.1,1,10))
  fit_2 <- PenalizedLDA(x=x_train,y=y_train, xte=x_test,lambda=cv_fit2$bestlambda,K=cv_fit2$bestK)
  
  
  l1_errors[s] <- mean(fit_2$ypred != y_test) * 100
  
  ####################t-test Classifier#####################################
  
  # Perform Bonferroni-adjusted t-tests with size 0.05
  p_values <- sapply(1:p, function(i) t.test(x_train[,i]~y_train)$p.value)
  adjusted_p_values <- p.adjust(p_values, method="bonferroni")
  sign_vars <- which(adjusted_p_values < 0.05)
  
  # Perform linear discriminant analysis using only the significant variables
  lda_model <- lda(x_train[, sign_vars], y_train)
  
  t_pred <- predict(lda_model, x_test[, sign_vars])$class
  
  ttest_errors[s] <- mean(t_pred != y_test) * 100
  
  
}



###################Median Error Percentages with Standard Error####################

lasso_med_error <- median(lasso_errors) #Lasso error percentage
lasso_med_error_se <- sqrt(var(lasso_errors))

nsc_med_error <- median(nsc_errors)     # Nearest Shrunken error percentage
nsc_med_error_se <- sqrt(var(nsc_errors))

l1_med_error <- median(l1_errors)       #L1-penalized error percentage
l1_med_error_se <- sqrt(var(l1_errors))

t_med_error <- median(ttest_errors)     #t-test classifier error percentage
t_med_error_se <- sqrt(var(ttest_errors))

cat("Lassoed Discriminant Analysis Median error percentage:", lasso_med_error, " (",lasso_med_error_se,")")
cat("NSC Median error percentage:", nsc_med_error," (",nsc_med_error_se,")")
cat("L1 Median error percentage:", l1_med_error," (",l1_med_error_se,")")
cat("t-Test Classifier Median error percentage:", t_med_error," (",t_med_error_se,")")
