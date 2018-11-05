functions {
  row_vector get_log_p_s(row_vector state_prior, int L, real[] l, real f, real t) {
    row_vector[num_elements(state_prior)] log_p_s;

    log_p_s = log(state_prior);
    for (l_i in 1:L) {
      //study transitions
      //study_trans_mat = [[(1 - t), t],
                        //[0, 1]];
      //L state
      //p_s[2] = p_s[2] + (p_s[1] * t);
      log_p_s[2] = log_sum_exp(log_p_s[1] + log(t), 
                             log_p_s[2]); 
      //U state
      //p_s[1] = p_s[1] * (1 - t);
      log_p_s[1] = log_p_s[1] + log1m(t);
      
      //forget transitions
      //forget_trans_mat = [1, 0]
                          //[f, 1 - f]
      //U state
      //p_s[1] = p_s[1] + p_s[2] * (1 - ((1 - f) ^ l[l_i]));
      log_p_s[1] = log_sum_exp(log_p_s[1], log_p_s[2] + log1m_exp(l[l_i] * log1m(f)));
      //L state
      //p_s[2] = p_s[2] * ((1 - f) ^ l[l_i]);
      log_p_s[2] = log_p_s[2] + (l[l_i] * log1m(f)); //L -> L
    }
    return log_p_s;
  }
  
  vector get_log_lik_readable(vector phi, vector theta, real[] x_r, int[] x_i) {
    int J = x_i[1];
	  int L = x_i[2];
	  int S = x_i[3];
	  int r[J] = x_i[4:(J + 3)];
	  row_vector[S] state_prior = to_row_vector(x_r[1:S]);
	  vector[S] recall_prob = to_vector(x_r[(S + 1):(2 * S)]);
	  //matrix[J, L] l = to_matrix(x_r[((2 * S) + 1):], J, L);
	  vector[J] f = theta[1:J];
	  vector[J] t = theta[(J + 1):(2 * J)];
	  vector[J] lp_j;
	  real lp;
	  for(j in 1:J) {
	    row_vector[S] p_s;
	    real p;
	    p_s = exp(get_log_p_s(state_prior, L, 
              	    x_r[((2 * S) + ((j - 1) * L) + 1):((2 * S) + (j * L))], //l[(j - 1), ]
              	    f[j], t[j]));
	    p = p_s * recall_prob;
	    lp_j[j] = bernoulli_lpmf(r[j] | p);
	  }
	  lp = sum(lp_j);
	  return([lp]');
  }
}
data {
  int<lower=0> S; // number of states
  int<lower=0> L; // length of protocol
  int<lower=0> W; // number of words
  
  int<lower=0> N_t; // number of subjects
  int<lower=0> T_t; // number of "trials" (subjects X words)
  real<lower=0> l_t[T_t, L]; // protocol (length of time between study sessions)
  int<lower=0> r_t[T_t]; // recalled
  int<lower=0> sub_t[T_t]; // subject
  int<lower=0> w_t[T_t]; // word
  
  int<lower=0> N_h; // number of subjects
  int<lower=0> T_h; // number of "trials" (subjects X words)
  real<lower=0> l_h[T_h, L]; // protocol (length of time between study sessions)
  int<lower=0> r_h[T_h]; // recalled
  int<lower=0> sub_h[T_h]; // subject
  int<lower=0> w_h[T_h]; // word
  int shards;
  
  int<lower=0> heldout;
  
  row_vector<lower=0, upper=1>[S] state_prior; // prior prob of being in each state
  vector<lower=0, upper=1>[S] recall_prob; // probability of recall in each state
}
transformed data {
  int<lower=0> T_heldout;
  
  //vector[0] theta[shards];
  int<lower = 0> J = T_t / shards;
  real x_r_t[shards, (J * L) + (2 * S)];
  int x_i_t[shards, J + 3];
  {
    int pos = 1;
    for (k in 1:shards) {
      int end = pos + J - 1;
      x_r_t[k, 1:S] = to_array_1d(state_prior); 
      x_r_t[k, (S + 1):(2 * S)] = to_array_1d(recall_prob); 
      x_r_t[k, ((2 * S) + 1):((J * L) + (2 * S))] = to_array_1d(l_t[pos:end,]);
      
      x_i_t[k, 1] = J;
      x_i_t[k, 2] = L;
      x_i_t[k, 3] = S;
      x_i_t[k, 4:(J + 3)] = to_array_1d(r_t[pos:end]);
      pos += J;
    }
  }
  
  if (heldout == 1) {
    T_heldout = T_h;
  }
  else {
    T_heldout = 0;
  }
}
parameters {
  real f_mu;   // population mean of success log-odds
  real<lower=0> f_sigma;  // population sd of success log-odds
  vector[W] f_alpha_std;  // success log-odds
  
  real t_mu;   // population mean of success log-odds
  real<lower=0> t_sigma;  // population sd of success log-odds
  vector[W] t_alpha_std;  // success log-odds
}
transformed parameters {
  vector<lower=0, upper=1>[W] f; //forgetting transition from L(earned) to U(nknown)
  vector<lower=0, upper=1>[W] t; //study transition from U(nknown) to L(earned)
  
  vector[0] phi_t;
  vector[J * 2] theta_t[shards];
  
  for(w in 1:W) {
    f[w] = inv_logit(f_mu + f_sigma * f_alpha_std[w]);  // likelihood
    t[w] = inv_logit(t_mu + t_sigma * t_alpha_std[w]);  // likelihood
  }
  
  {
    int pos = 1;
    for (k in 1:shards) {
      vector[J * 2] theta_temp;
      int end = pos + J - 1;
      theta_temp[1:J] = f[w_t[pos:end]];
      theta_temp[(J + 1):(2 * J)] = t[w_t[pos:end]];
      theta_t[k] = theta_temp;
      pos += J;
    }
  }
}
model {
  f_mu ~ normal(0, 6);                             // hyperprior
  f_sigma ~ normal(0, 1);                           // hyperprior
  f_alpha_std ~ normal(0, 1);                       // prior
  
  t_mu ~ normal(0, 6);                             // hyperprior
  t_sigma ~ normal(0, 1);                           // hyperprior
  t_alpha_std ~ normal(0, 1);                       // prior

  target += sum(map_rect(get_log_lik_readable, phi_t, theta_t, x_r_t, x_i_t));
}
generated quantities {
  vector[T_t] log_lik_t;
  vector[T_heldout] log_lik_h;
  vector[T_heldout] p_h; //probability of recall at test
  matrix[T_heldout, S] p_s_h; //state probabilities at test
  vector[T_t] p_t; //probability of recall at test
  matrix[T_t, S] p_s_t; //state probabilities at test
  vector[T_h] r_true;
  
  //LOO-CV
  for (t_i in 1:T_t) {
    p_s_t[t_i, ] = exp(get_log_p_s(state_prior, L, l_t[t_i, ], f[w_t[t_i]], t[w_t[t_i]]));
  }
  r_true = to_vector(r_h);
  p_t = p_s_t * recall_prob;
  for(t_i in 1:T_t) {
    log_lik_t[t_i] = bernoulli_lpmf(r_t[t_i] | p_t[t_i]);
  }
  
  //K-fold CV
  if(heldout == 1) {
    for (t_i in 1:T_heldout) {
      p_s_h[t_i, ] = exp(get_log_p_s(state_prior, L, l_h[t_i, ], f[w_h[t_i]], t[w_h[t_i]]));
    }
    //recall probs
    p_h = p_s_h * recall_prob;
    for (t_i in 1:T_heldout) {
      log_lik_h[t_i] = bernoulli_lpmf(r_h[t_i] | p_h[t_i]);
    }
  }
}
