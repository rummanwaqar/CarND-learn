#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "classifier.h"

// helper functions to load data
std::vector<std::vector<double>> load_states(std::string file_name);
std::vector<std::string> load_labels(std::string file_name);

int main(int argc, char** argv) {
  if(argc < 2) {
    std::cerr << "Usage: ./naive_bayes data_folder" << std::endl;
    return 1;
  }
  const std::string path = argv[1];

  // load data
  std::vector<std::vector<double>> X_train = load_states(path + "/train_states.txt");
  std::vector<std::string> Y_train = load_labels(path + "/train_labels.txt");
  std::vector<std::vector<double>> X_test = load_states(path + "/test_states.txt");
  std::vector<std::string> Y_test = load_labels(path + "/test_labels.txt");

  std::cout << "X_train number of elements " << X_train.size() << "x"
    << X_train[0].size() << std::endl;
  std::cout << "Y_train number of elements " << Y_train.size() << std::endl;
  std::cout << "X_test number of elements " << X_test.size() << "x"
    << X_test[0].size() << std::endl;
  std::cout << "Y_test number of elements " << Y_test.size() << std::endl;

  // initialize and train classifier
  GNB classifier = GNB();
  classifier.train(X_train, Y_train);

  // calculate score for test set
  int score = 0;
  for(int i=0; i < X_test.size(); ++i) {
    std::vector<double> coords = X_test[i];
    std::string predicted = classifier.predict(coords);
    if(predicted.compare(Y_test[i]) == 0) {
      score += 1;
    }
  }
  float correct = float(score) / Y_test.size();
  std::cout << "You got " << (100 * correct) << "% correct" << std::endl;

  return 0;
}

/*
 * load state from txt file
 */
std::vector<std::vector<double>> load_states(std::string file_name) {
  std::ifstream in_state(file_name.c_str(), std::ifstream::in);
  std::vector<std::vector<double>> output;

  std::string line;
  while(getline(in_state, line)) {
    std::istringstream iss(line);
    std::vector<double> x_coord;

    std::string token;
    while(getline(iss, token, ',')) {
      x_coord.push_back(std::stod(token));
    }
    output.push_back(x_coord);
  }
  return output;
}

/*
 * load labels from txt file
 */
std::vector<std::string> load_labels(std::string file_name) {
  std::ifstream in_label(file_name.c_str(), std::ifstream::in);
  std::vector<std::string> output;

  std::string line;
  while (getline(in_label, line)) {
    std::istringstream iss(line);
    std::string label;
    iss >> label;
    output.push_back(label);
  }
  return output;
}
