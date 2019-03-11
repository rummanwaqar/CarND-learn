#ifndef MPC_H
#define MPC_H

#include <vector>
#include "Eigen/Core"

class MPC {
 public:
  MPC() {};

  virtual ~MPC() = default;

  // Solve the model given an initial state and poly trajactory
  // Return the next state and actuations as a vector.
  std::vector<double> solve(const Eigen::VectorXd &x0,
                            const Eigen::VectorXd &coeffs);
};

#endif
