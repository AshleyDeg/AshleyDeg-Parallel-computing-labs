/* Problem 4 - Assignment 1
 * Gauss-Seidel iterative solver for heat spread.
 * Shared-memory parallelism
 */

#ifdef _OPENMP
#include <omp.h> // for OpenMP library functions
#endif

#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

struct HeatSource {
    double x, y, range, temp;
};

void apply_hs(std::vector<std::vector<double>>& m, const HeatSource& hs, const std::string& border);
std::vector<std::vector<double>> init(const std::vector<HeatSource>& heatSources, int n);
double gauss_seidel_step_red(std::vector<std::vector<double>>& m, int s_row, int e_row);
double gauss_seidel_step_black(std::vector<std::vector<double>>& m, int s_row, int e_row);
std::pair<double, int> grid_solver(std::vector<std::vector<double>>& grid, int max_iter, double tol);
std::pair<double, int> grid_solver_par(std::vector<std::vector<double>>& grid, int max_iter, double tol);
double mean_value(double arr[], int len);
void print_grid(std::vector<std::vector<double>> grid, int size);


int main() {
#ifdef _OPENMP
    std::cout << "\n _OPENMP defined" << std::endl;
    std::cout << "Num processors: " << omp_get_num_procs() << std::endl << "\n";
#endif

    std::vector<HeatSource> heatSources = {
            {0.0, 0.0, 1.0, 2.5},
            {0.5, 1.0, 1.0, 2.5}
    };


    double tol = 0.00001;
    int max_iter=250000;

    int n_avg = 10;
    double grid[n_avg], grid_par[n_avg], grid_speed_up[n_avg];
    double avg_seq, avg_par, avg_speedup;
    double end_time, start_time;
    double start_time_par, end_time_par;

    for (int n = 96; n < 417; n += 16) {
        // Iterate over the grid size
        std::cout << "Grid size: " << n << std::endl;
        // Write results to file
        std::ofstream results_file("/Users/ashleydeglinnocenti/Desktop/Parallel Computing LNU/Assignment 1/Problem 4/results_correct.csv", std::ios::app);
        for (int i = 0; i<n_avg; i++) {
            // Perform multiple runs with the same grid size, then take the average of computation times and speeed-up
            std::cout << "Iteration: " << i << std::endl;

            auto m = init(heatSources, n);
            start_time = omp_get_wtime();
            auto result = grid_solver(m, max_iter, tol);
            end_time = omp_get_wtime();
            grid[i] = end_time - start_time;

            auto m_par = init(heatSources, n);
            start_time_par = omp_get_wtime();
            auto result_par = grid_solver_par(m_par, max_iter, tol);
            end_time_par = omp_get_wtime();
            grid_par[i] = end_time_par - start_time_par;
            grid_speed_up[i] = grid[i]/grid_par[i];


            //Print results
            std::cout << "Sequential.  " << std::endl;
            std::cout << "Result:  " << std::endl;
            std::cout << "Residual  = " << result.first << " after " << result.second << " iterations\n";
            std::cout << "Computation time (sequential):  " << grid[i] << "s." << std::endl;
            std::cout << "\nParallel.  " << std::endl;
            std::cout << "Residual  = " << result_par.first << " after " << result_par.second << " iterations\n";
            std::cout << "Computation time:  " << grid_par[i] << "s." << std::endl;
            std::cout << "Speed-up:  " << grid_speed_up[i] << std::endl;

        }

        // Compute the averages
        avg_seq = mean_value(grid, n_avg);
        avg_par = mean_value(grid_par, n_avg);
        avg_speedup = mean_value(grid_speed_up, n_avg);

        // Writing to the results file
        results_file << n << "," << avg_seq << "," <<
                         avg_par << "," << avg_speedup << std::endl;

        results_file.close();

    }

    return 0;
}


void print_grid(std::vector<std::vector<double>> grid, int size){
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++)
            std::cout << grid[i][j] << "\t";
        std::cout << std::endl;
    }
}

void apply_hs(std::vector<std::vector<double>>& m, const HeatSource& hs, const std::string& border) {
    int numRows = m.size();
    int numCols = m[0].size();

    std::pair<int, int> r;
    if (border == "top" || border == "bottom") {
        r = std::make_pair(0, numRows);
    } else if (border == "left" || border == "right") {
        r = std::make_pair(1, numRows - 1);
    } else {
        return;
    }

    for (int i = r.first; i < r.second; ++i) {
        double dist;
        if (border == "top") {
            dist = sqrt(pow(static_cast<double>(i) / (numRows - 1) - hs.x, 2) + pow(hs.y, 2));
        } else if (border == "bottom") {
            dist = sqrt(pow(static_cast<double>(i) / (numRows - 1) - hs.x, 2) + pow(1 - hs.y, 2));
        } else if (border == "left") {
            dist = sqrt(pow(hs.x, 2) + pow(static_cast<double>(i) / (numRows - 1) - hs.y, 2));
        } else if (border == "right") {
            dist = sqrt(pow(1 - hs.x, 2) + pow(static_cast<double>(i) / (numRows - 1) - hs.y, 2));
        }

        if (dist <= hs.range) {
            if (border == "top") {
                m[0][i] += (hs.range - dist) / hs.range * hs.temp;
            } else if (border == "bottom") {
                m[numRows - 1][i] += (hs.range - dist) / hs.range * hs.temp;
            } else if (border == "left") {
                m[i][0] += (hs.range - dist) / hs.range * hs.temp;
            } else if (border == "right") {
                m[i][numCols - 1] += (hs.range - dist) / hs.range * hs.temp;
            }
        }
    }
}

std::vector<std::vector<double>> init(const std::vector<HeatSource>& heatSources, int n) {
    int num_p = n + 2;

    std::vector<std::vector<double>> m(num_p, std::vector<double>(num_p, 0.0));
    for (const auto& hs : heatSources) {
        apply_hs(m, hs, "top");
        apply_hs(m, hs, "bottom");
        apply_hs(m, hs, "left");
        apply_hs(m, hs, "right");
    }

    return m;
}


double gauss_seidel_step_red(std::vector<std::vector<double>>& m, int s_row, int e_row) {
    // Update step over red dots in the grid
    double s = 0.0;
    int numRows = m.size();
    int numCols = m[0].size();
    int n_cols = numCols-2;
    double diff;

    // Red dots. Alternating Positions (excluding border)
    for (int i = s_row; i <= e_row; ++i) {
        for (int j = i % 2; j <= n_cols; j += 2) {
            if (j<1)
                continue;
            double tmp = m[i][j];
            m[i][j] = (m[i][j - 1] + m[i][j + 1] + m[i - 1][j] + m[i + 1][j]) / 4.0;
            diff = tmp - m[i][j];
            s += diff * diff;
        }
    }

    return s;
}

double gauss_seidel_step_black(std::vector<std::vector<double>>& m, int s_row, int e_row) {
    // Update step over black dots in the grid
    double s = 0.0;
    int numRows = m.size();
    int numCols = m[0].size();
    int n_cols = numCols-2;
    double diff;


    // Black dots. Alternating Positions (excluding border)
    for (int i = s_row; i <= e_row; ++i) {
        for (int j = 1 + i % 2; j <= n_cols; j += 2) {
            double tmp = m[i][j];
            m[i][j] = (m[i][j - 1] + m[i][j + 1] + m[i - 1][j] + m[i + 1][j]) / 4.0;
            diff = tmp - m[i][j];
            s += diff * diff;
        }
    }

    return s;
}

std::pair<double, int> grid_solver(std::vector<std::vector<double>>& grid, int max_iter, double tol) {
    // Grid solver. Alternates between red and black passes. Then checks if the current residual is below the threshold
    bool done = false;
    int iter = 0;
    double my_res_red, my_res_black, my_res;

    while (!done) {
        my_res_red = gauss_seidel_step_red(grid, 1, grid.size()-2);
        my_res_black = gauss_seidel_step_black(grid, 1, grid.size()-2);
        my_res = my_res_red + my_res_black;
        iter++;
        if ((my_res < tol)||(iter > max_iter)) {
            done = true;
        }
    }
    return std::make_pair(my_res, iter);
}

std::pair<double, int> grid_solver_par(std::vector<std::vector<double>>& grid, int max_iter, double tol) {
    // Parallel version of grid solver
    int iter = 0;
    bool done = false;
    double res=0;

#pragma omp parallel default(none) shared(iter, res, done, grid, max_iter, tol) num_threads(omp_get_num_procs())
    {
        double my_res;
        int n_rows = grid.size() - 2;
        int num_threads = omp_get_num_procs();

        // Get the thread ID
        const int thread_ID = omp_get_thread_num();
        //Split the grid
        // Calculate the row range for each thread
        const int rows_per_thread = n_rows / num_threads;
        const int s_row = 1 + thread_ID * rows_per_thread;
        const int e_row = (thread_ID == num_threads - 1) ? n_rows: s_row + rows_per_thread - 1;

        while (!done) {
            // Barrier to make sure everyone is ready to start
#pragma omp barrier
            if (thread_ID == 0) {
                res = 0;
                iter ++;
            }
#pragma omp barrier
            // Red dots. Alternating Positions (excluding border)
            my_res = gauss_seidel_step_red(grid, s_row, e_row);
#pragma omp barrier
            // Black dots. Alternating Positions (excluding border)
            my_res = my_res + gauss_seidel_step_black(grid, s_row, e_row);
            // Critical section
#pragma omp critical
            {
                res += my_res;
            }
#pragma omp barrier
            if ((res < tol) || (iter > max_iter)) {
                done = true;
            }
        }
    }
    return std::make_pair(res, iter);
}

double mean_value(double arr[], int len) {
    // Computes the average over an array of doubles of length len
    double mean, sum = 0;
    for (int i = 0; i < len; i++) {
        sum += arr[i];
    }
    mean = sum/len;

    return mean;
}