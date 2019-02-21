#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <string>
#include <vector>
#include <cmath>
#include <iostream>
#include <numeric>
#include <iterator>

struct gaussian_t {
  double mean;
  double stdev;
};

// Gaussian Naive Bayes
class GNB {
public:
  /**
   * Constructor
   */
  GNB();

  /**
   * Destructor
   */
  virtual ~GNB() = default;

  /**
   * Train classifier
   */
  void train(const std::vector<std::vector<double>> &data,
             const std::vector<std::string> &labels);

  /**
   * Predict with trained classifier
   */
  std::string predict(const std::vector<double> &sample);

  std::vector<std::string> possible_labels = {"left","keep","right"};

private:
  /*
   * calculates mean and stdev of an array
   */
  gaussian_t calc_gaussian(std::vector<double>& v);

  /**
   * Calculates gaussian probability
   */
  double calc_prob(double x, double ux, double sigmax);

  // prior probability p(C_k) for each label
  std::vector<double> label_prob_;

  // conditional probability for s feature p(s | label)
  std::vector<gaussian_t> s_prob_;

  // conditional probability for d feature p(d | label)
  std::vector<gaussian_t> d_prob_;

  // conditional probability for d_dot feature p(d_dot | label)
  std::vector<gaussian_t> d_dot_prob_;
};

#endif  // CLASSIFIER_H
