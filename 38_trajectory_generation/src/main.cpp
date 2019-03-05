#include <iostream>
#include <vector>
#include "hybrid_breadth_first.h"

using std::vector;

// Sets up maze grid
int X = 1;
int _ = 0;
vector<vector<int>> GRID = {
  {_,X,X,_,_,_,_,_,_,_,X,X,_,_,_,_,},
  {_,X,X,_,_,_,_,_,_,X,X,_,_,_,_,_,},
  {_,X,X,_,_,_,_,_,X,X,_,_,_,_,_,_,},
  {_,X,X,_,_,_,_,X,X,_,_,_,X,X,X,_,},
  {_,X,X,_,_,_,X,X,_,_,_,X,X,X,_,_,},
  {_,X,X,_,_,X,X,_,_,_,X,X,X,_,_,_,},
  {_,X,X,_,X,X,_,_,_,X,X,X,_,_,_,_,},
  {_,X,X,X,X,_,_,_,X,X,X,_,_,_,_,_,},
  {_,X,X,X,_,_,_,X,X,X,_,_,_,_,_,_,},
  {_,X,X,_,_,_,X,X,X,_,_,X,X,X,X,X,},
  {_,X,_,_,_,X,X,X,_,_,X,X,X,X,X,X,},
  {_,_,_,_,X,X,X,_,_,X,X,X,X,X,X,X,},
  {_,_,_,X,X,X,_,_,X,X,X,X,X,X,X,X,},
  {_,_,X,X,X,_,_,X,X,X,X,X,X,X,X,X,},
  {_,X,X,X,_,_,_,_,_,_,_,_,_,_,_,_,},
  {X,X,X,_,_,_,_,_,_,_,_,_,_,_,_,_,}};
vector<double> START = {0.0,0.0,0.0};
vector<int> GOAL = {(int)GRID.size()-1, (int)GRID[0].size()-1};

int main() {
  std::cout << "Finding path through grid:" << std::endl;

  // Creates an Empty Maze and for testing the number of expansions with it
  for(int i = 0; i < GRID.size(); ++i) {
    std::cout << GRID[i][0];
    for(int j = 1; j < GRID[0].size(); ++j) {
      std::cout << "," << GRID[i][j];
    }
    std::cout << std::endl;
  }

  HBF hbf = HBF();

  HBF::maze_path get_path = hbf.search(GRID,START,GOAL);

  vector<HBF::maze_s> show_path = hbf.reconstruct_path(get_path.came_from,
                                                       START, get_path.final);

  std::cout << "show path from start to finish" << std::endl;
  for(int i = show_path.size()-1; i >= 0; --i) {
      HBF::maze_s step = show_path[i];
      std::cout << "##### step " << step.g << " #####" << std::endl;
      std::cout << "x " << step.x << std::endl;
      std::cout << "y " << step.y << std::endl;
      std::cout << "theta " << step.theta << std::endl;
  }

  return 0;
}
