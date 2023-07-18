#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <gsl/gsl_integration.h>
#include <stdio.h>
#include <math.h>
#include <non_central_chi_squared.hpp>
#include <chi_squared.hpp>


struct param_type_1det{
  double g;
  int k;
  int N;
  double pv;
  };

double signal_1det(double l, void *params){
  // import all parameters for the integral
  struct param_type_1det *my_params = (struct param_type_1det *) params;
  double g = my_params->g;  // x value for pdf
  int k = my_params->k;     // number of degrees of freedom for chi2 distribution for individual sft
  int N = my_params->N;     // Number of summed sfts 
  double pv = my_params->pv;// width of the exponential prior on SNR
  double wid = 1.0/pv;      // 1/width on the exponential prior
  //double val = 0;
  boost::math::non_central_chi_squared_distribution<> ch(k*N, l);
  double val;
  val = pdf(ch,g);
  
  return val*wid*exp(-wid*l);
}

double ch2_noise_1det(double g, int k, int N){
  // Noise is just the chi2 distibution for any frequency
  boost::math::chi_squared_distribution<> ch1(k*N);
  double val = pdf(ch1,g);
  return val;
}


double integral_signal_1det(double g,double fraction,int k,int N, double pv)
{

  // create workspace and assign memory
  gsl_integration_workspace *work_ptr = gsl_integration_workspace_alloc(100000);

  double abs_error = 0;	/* to avoid round-off problems */
  double rel_error = 1.0e-8;	/* the result will usually be much better */
  double result;		/* the result from the integration */
  double error;                 /* the estimated error from the integration */

  // set parameters for integration
  param_type_1det param_val = {g,k,N,pv};

  gsl_function Function;
  struct param_type_1det *params_ptr = &param_val;

  // set the function to integrate over
  Function.function = &signal_1det;
  Function.params = params_ptr;

  // integrate using gsl
  gsl_integration_qagiu(&Function, 0.0, abs_error, rel_error, 100000, work_ptr, &result, &error);

  gsl_integration_workspace_free(work_ptr);

  return result;
}


/*two detector integrals for line aware statistic */


struct param_type_2det{
  double g1;
  double g2;
  double fraction;
  int k;
  int N;
  double pv;
  };

double signal_2det(double l, void *params){
  // set the parameters for integration
  struct param_type_2det *my_params = (struct param_type_2det *) params;
  double g1 = my_params->g1; // sft power val for detector 1
  double g2 = my_params->g2; // sft power val for detector 2
  double fraction = my_params->fraction; // fraction is the ratio between the two detectors snrs
  int k = my_params->k; // number of degrees of freedom for chi2 for each sft
  int N = my_params->N; // number of summed sfts
  double pv = my_params->pv; // width of prior on the snr or amplitude
  double wid = 1.0/pv;
  double frac = 0;
  boost::math::non_central_chi_squared_distribution<> ch1(k*N, l);
  // this changes the fraction ratio depending if detector 1 has a higher snr or detector 2 has a higher snr
  if(fraction >=1){
    frac = fraction;
  }
  else{
    frac = 1./fraction;
  }
  boost::math::non_central_chi_squared_distribution<> ch2(k*N, l*frac);
  double val;
  if(fraction >= 1){
    val = pdf(ch1,g1)*pdf(ch2,g2);
  }
  else{
    val = pdf(ch1,g2)*pdf(ch2,g1);
  }
  
  return val*wid*exp(-wid*l);
}

double line_2det(double l, void *params){
  struct param_type_2det *my_params = (struct param_type_2det *) params;
  double g1;
  double g2;
  double fraction = abs(my_params->fraction);
  if(fraction >= 1){
    g1 = my_params->g2;
    g2 = my_params->g1;
  }
  else{
    g1 = my_params->g1;
    g2 = my_params->g2;
  }
  int k = my_params->k;
  int N = my_params->N;
  double pv = my_params->pv;
  double wid = 1.0/pv;
  boost::math::chi_squared_distribution<> ch1(k*N);
  double val;
  boost::math::non_central_chi_squared_distribution<> ch2(k*N, l*fraction);
  boost::math::non_central_chi_squared_distribution<> ch3(k*N, l);
  val = pdf(ch1,g1)*pdf(ch3,g2) + pdf(ch2,g1)*pdf(ch1,g2);
  
  return 0.5*val*wid*exp(-wid*l);
}

double ch2_noise_2det(double g1, double g2, int k, int N){
  boost::math::chi_squared_distribution<> ch1(k*N);
  double val = pdf(ch1,g1)*pdf(ch1,g2);
  return val;
}


double integral_signal_2det(double g1,double g2,double fraction,int k,int N, double pv)
{
  gsl_integration_workspace *work_ptr = gsl_integration_workspace_alloc(100000);

  double abs_error = 0;	/* to avoid round-off problems */
  double rel_error = 1.0e-8;	/* the result will usually be much better */
  double result;		/* the result from the integration */
  double error;                 /* the estimated error from the integration */
  
  param_type_2det param_val = {g1,g2,fraction,k,N,pv};

  gsl_function Function;
  struct param_type_2det *params_ptr = &param_val;


  Function.function = &signal_2det;
  Function.params = params_ptr;


  gsl_integration_qagiu(&Function, 0.0, abs_error, rel_error, 100000, work_ptr, &result, &error);

  gsl_integration_workspace_free(work_ptr);

  return result;
}

double integral_line_2det(double g1,double g2,double fraction,int k,int N, double pv)
{
  gsl_integration_workspace *work_ptr = gsl_integration_workspace_alloc(100000);

  double abs_error = 0;	/* to avoid round-off problems */
  double rel_error = 1.0e-8;	/* the result will usually be much better */
  double result;		/* the result from the integration */
  double error;                 /* the estimated error from the integration */
  
  param_type_2det param_val = {g1,g2,fraction,k,N,pv};

  gsl_function Function;
  struct param_type_2det *params_ptr = &param_val;


  Function.function = &line_2det;
  Function.params = params_ptr;


  gsl_integration_qagiu(&Function, 0.0, abs_error, rel_error, 100000, work_ptr, &result, &error);

  gsl_integration_workspace_free(work_ptr);

  return result;
}


