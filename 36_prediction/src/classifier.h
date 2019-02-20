#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <string>
#include <vector>
#include <cmath>
#include "Eigen/Dense"

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
};

#endif  // CLASSIFIER_H
