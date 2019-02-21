  #include "classifier.h"

GNB::GNB() {}

void GNB::train(const std::vector<std::vector<double>> &data,
                const std::vector<std::string> &labels) {
  // split data between features
  std::vector<std::vector<std::vector<double>>> feature_data(possible_labels.size());
  for(int i=0; i<labels.size(); i++) {
    auto it = std::find(possible_labels.begin(), possible_labels.end(), labels[i]);
    if(it != possible_labels.end()) {
      int index = std::distance(possible_labels.begin(), it);
      feature_data[index].push_back(data[i]);
    } else {
      std::cerr << "Unknown label" << std::endl;
    }
  }

  // prior probability p(C_k) for each label
  for(int i=0; i<possible_labels.size(); ++i) {
    label_prob_.push_back(float(feature_data[i].size()) / labels.size());
    std::cout << "p(" << possible_labels[i] << ") = " << label_prob_[i] << std::endl;
  }

  // calculate conditional probability for each label/feature p(feature | label)
  // aka compute and store the mean and standard deviation of over all points of
  // feature with label x

  // feature: S
  for(int i=0; i<possible_labels.size(); ++i) {
    std::vector<double> v;
    for(auto const& x : feature_data[i]) {
      v.push_back(x[0]);
    }
    s_prob_.push_back(calc_gaussian(v));
  }
  for(int i=0; i<possible_labels.size(); ++i) {
    std::cout << "p(s | " << possible_labels[i] << ") = N(" << s_prob_[i].mean << "," << s_prob_[i].stdev << ")" << std::endl;
  }

  // feature: D
  for(int i=0; i<possible_labels.size(); ++i) {
    std::vector<double> v;
    for(auto const& x : feature_data[i]) {
      v.push_back(x[1]);
    }
    d_prob_.push_back(calc_gaussian(v));
  }
  for(int i=0; i<possible_labels.size(); ++i) {
    std::cout << "p(d | " << possible_labels[i] << ") = N(" << d_prob_[i].mean << "," << d_prob_[i].stdev << ")" << std::endl;
  }

  // features: D_dot
  for(int i=0; i<possible_labels.size(); ++i) {
    std::vector<double> v;
    for(auto const& x : feature_data[i]) {
      v.push_back(x[3]);
    }
    d_dot_prob_.push_back(calc_gaussian(v));
  }
  for(int i=0; i<possible_labels.size(); ++i) {
    std::cout << "p(d_dot | " << possible_labels[i] << ") = N(" << d_dot_prob_[i].mean << "," << d_dot_prob_[i].stdev << ")" << std::endl;
  }
}

std::string GNB::predict(const std::vector<double> &sample) {
  double max = -1;
  int index = 0;
  for(int i=0; i<possible_labels.size(); ++i) {
    float p = label_prob_[i] *
      calc_prob(sample[3], d_dot_prob_[i].mean, d_dot_prob_[i].stdev) *
      calc_prob(sample[1], d_prob_[i].mean, d_prob_[i].stdev) *
      calc_prob(sample[0], s_prob_[i].mean, s_prob_[i].stdev);
    if(p > max) {
      max = p;
      index = i;
    }
  }
  return possible_labels[index];
}

gaussian_t GNB::calc_gaussian(std::vector<double>& v) {
  double sum = std::accumulate(v.begin(), v.end(), 0.0);
  double mean = sum / v.size();

  std::vector<double> diff(v.size());
  std::transform(v.begin(), v.end(), diff.begin(), [mean](double x) { return x - mean; });
  double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
  double stdev = std::sqrt(sq_sum / v.size());
  return gaussian_t{mean,stdev};
}

double GNB::calc_prob(double x, double ux, double sigmax) {
  return std::exp(-(x - ux) * (x - ux) / (2 * sigmax * sigmax)) / std::sqrt(2 * M_PI * sigmax * sigmax);
}
