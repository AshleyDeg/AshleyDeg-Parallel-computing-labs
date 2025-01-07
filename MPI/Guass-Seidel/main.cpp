#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <random>
#include <cmath>
#include <mpi.h>

using namespace std;


// Structure to store the different computations time
struct Time {
    unsigned int vec_size;
    double seq_time;
    double par_time;
};


vector<Time> quicksort_mpi(long n, int rank, int size);
vector<int> generateRandomIntegers(int n);
vector<int> quick_sort_par(vector<int> data, int length, MPI_Comm comm);
void quicksort(std::vector<int>& vec);
void aux(vector<int>& vec, int f, int t);


int main(int argc, char* argv[]) {
    int n = 20000000;
    
    // Initalize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    vector<Time> res = quicksort_mpi(n, rank, size);

    if (rank == 0) {
        for (unsigned int i = 0; i < res.size(); i++) {
            ofstream results_file("results2.csv", ios::app);

            results_file << (res[i]).vec_size << "," << (res[i]).seq_time << "," << (res[i]).par_time << endl;

            results_file.close();
        }
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}


vector<Time> quicksort_mpi(long n, int rank, int size) {
    vector<Time> results;

    int precision = 20;

    vector<int> data;
    vector<int> data_sub;
    vector<int> data_sorted;
    vector<int> data_sorted_seq;
    vector<int> recieve_counts;
    vector<int> receive_displacements;

    vector<chrono::duration<double>> seq_iteration_results;
    vector<chrono::duration<double>> mpi_iteration_results;
    chrono::duration<double> seq_elapsed;
    chrono::duration<double> mpi_elapsed;
    double seq_mean;
    double mpi_mean;

    for (unsigned int i = 8; i < n + 1; i *= 2) {
        if (rank == 0) {
            seq_iteration_results.clear();
            mpi_iteration_results.clear();
        }

        for (unsigned int j = 0; j < 10; j++) {
            if (rank == 0) {
                cout << i << ", " << j << endl;

                // Definition of the size of the different vectors used
                data.resize(i);
                data_sorted.resize(i);
                data_sorted_seq.resize(i);
                recieve_counts.resize(size);
                receive_displacements.resize(size);

                // Generate the random vector
                data = generateRandomIntegers(i);
                data_sorted_seq = data;

                // Start the time
                auto seq_start = chrono::high_resolution_clock::now();

                quicksort(data_sorted_seq);
                
                // End the time
                auto seq_end = chrono::high_resolution_clock::now();

                seq_elapsed = seq_end - seq_start;
                cout << "Sequential: is sorted ? " << is_sorted(data_sorted_seq.begin(), data_sorted_seq.end()) << " and time = " << seq_elapsed.count() << endl;

                if (is_sorted(data_sorted_seq.begin(), data_sorted_seq.end())) {
                    seq_iteration_results.push_back(seq_elapsed);
                }
            }

            // Start the time
            MPI_Barrier(MPI_COMM_WORLD);
            auto mpi_start = chrono::high_resolution_clock::now();

            // Definition of the size of the vector scatter to each processor
            int sub_size = i / size;

            // Scatter the vector in smaller parts send to each processor
            data_sub.resize(sub_size);
            MPI_Scatter(data.data(), sub_size, MPI_INT, data_sub.data(),
                sub_size, MPI_INT, 0, MPI_COMM_WORLD);

            // Quicksort on the partial vector in parallel by each processor
            data_sub = quick_sort_par(data_sub, sub_size, MPI_COMM_WORLD);
            sub_size = data_sub.size();

            // Gather of the different partial vector
            MPI_Gather(&sub_size, 1, MPI_INT, recieve_counts.data(), 1,
                MPI_INT, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                int index = 0;
                receive_displacements[0] = index;

                for (unsigned int k = 1; k < size; k++) {
                    index = index + recieve_counts[k - 1];
                    receive_displacements[k] = index;
                }
            }

            MPI_Gatherv(data_sub.data(), sub_size, MPI_INT, data_sorted.data(),
                recieve_counts.data(), receive_displacements.data(), MPI_INT, 0, MPI_COMM_WORLD);

            // End the time
            MPI_Barrier(MPI_COMM_WORLD);
            auto mpi_end = chrono::high_resolution_clock::now();

            if (rank == 0) {
                mpi_elapsed = mpi_end - mpi_start;
                cout << "Parallelism: is sorted ? " << is_sorted(data_sorted.begin(), data_sorted.end()) << " and time = " << mpi_elapsed.count() << endl;

                if (is_sorted(data_sorted.begin(), data_sorted.end())) {
                    mpi_iteration_results.push_back(mpi_elapsed);
                }
            }
        }

        // Calculation of the means for 10 iterations of sorting a vector of size n
        if (rank == 0) {
            seq_mean = 0.0;
            for (chrono::duration<double> seq_val : seq_iteration_results) {
                seq_mean += seq_val.count();
            }
            seq_mean /= seq_iteration_results.size();

            mpi_mean = 0.0;
            for (chrono::duration<double> mpi_val : mpi_iteration_results) {
                mpi_mean += mpi_val.count();
            }
            mpi_mean /= mpi_iteration_results.size();

            results.emplace_back(Time{ i, seq_mean, mpi_mean });
        }
    }

    return results;
}


// Function to generate a vector with n random positive integers
vector<int> generateRandomIntegers(int n) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<int> dis(0, n);

    std::vector<int> randomNumbers;
    randomNumbers.reserve(n);

    for (int i = 0; i < n; ++i) {
        randomNumbers.push_back(dis(gen));
    }

    return randomNumbers;
}


// Function to quicksort executed by each processor
vector<int> quick_sort_par(vector<int> data, int length, MPI_Comm comm) {
    int size, rank, number_amount;

    double pivot;
    double mean_local[2] = { 0.0, static_cast<double>(length)};
    double mean_global[2] = { 0.0, 0.0 };
    
    vector<int> left;
    vector<int> right;
    vector<int> data_recieve;
    vector<int> data_keep;

    // Definition of the new communication environment
    MPI_Status status;
    MPI_Comm new_comm;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    // If there's only one processor in the communication then quickort sequentially
    if (size == 1) {
        quicksort(data);
        return data;
    }

    // Otherwise, calculate the mean of the local vector
    for (unsigned int i = 0; i < length; i++) {
        mean_local[0] = mean_local[0] + data[i];
    }

    // Reduce the mean to calculate the global mean of the communication environment
    MPI_Reduce(&mean_local, &mean_global, 2, MPI_DOUBLE, MPI_SUM, 0, comm);
    if (rank == 0) {
        // Definition of the pivot value
        pivot = mean_global[0] / mean_global[1];
    }
    // Broadcast of the pivot value to the communication environment
    MPI_Bcast(&pivot, 1, MPI_DOUBLE, 0, comm);

    // Split the vector in two parts smaller and bigger than the pivot value
    for (const auto& val : data) {
        if (val < pivot) {
            left.push_back(val);
        }
        else {
            right.push_back(val);
        }
    }

    if (rank < size / 2) {
        // If the processor is in the lower part of the environment
        // Send the bigger value to the associated processor
        MPI_Send(right.data(), right.size(), MPI_INT,
            rank + size / 2, 00, comm);

        // And receive the smaller value from the associated processor
        MPI_Probe(rank + size / 2, 11, comm, &status);
        MPI_Get_count(&status, MPI_INT, &number_amount);
        data_recieve.resize(number_amount);

        MPI_Recv(data_recieve.data(), number_amount, MPI_INT, rank + size / 2,
            11, comm, MPI_STATUS_IGNORE);
    }
    else {
        // If the processor is in the higher part of the environment
        // Receive the bigger value from the associated processor
        MPI_Probe(rank - size / 2, 00, comm, &status);
        MPI_Get_count(&status, MPI_INT, &number_amount);
        data_recieve.resize(number_amount);

        MPI_Recv(data_recieve.data(), number_amount, MPI_INT,
            rank - size / 2, 00, comm, MPI_STATUS_IGNORE);

        // Send the smaller value to the associated processor
        MPI_Send(left.data(), left.size(), MPI_INT, rank - size / 2, 11, comm);
    }

    // Define the new communication environment for the next recursion
    int color = rank / (size / 2);
    MPI_Comm_split(comm, color, rank, &new_comm);

    // Concatenate the kept values and the one receives
    // And recursive quicksort of these values
    if (rank < size / 2) {
        data_keep = left;
        data_keep.insert(data_keep.end(), data_recieve.begin(), data_recieve.end());

        return quick_sort_par(data_keep, data_keep.size(), new_comm);
    }
    else {
        data_keep = right;
        data_recieve.insert(data_recieve.end(), data_keep.begin(), data_keep.end());

        return quick_sort_par(data_recieve, data_recieve.size(), new_comm);
    }
}


// Sequential quicksort
void quicksort(std::vector<int>& vec) { aux(vec, 0, vec.size() - 1); };


void aux(vector<int>& vec, int f, int t) {
    int i = f;
    int j = t;

    if (j - i >= 1)
    {
        // Definition of the pivot
        int p = vec[i];

        // Sorting loop : every number smaller than the pivot goes on its left and every number higher on its right
        while (i <= j)
        {
            if (vec[i] > vec[j])
            {
                swap(vec[i], vec[j]); // Wrong order so we swap them
            }
            if (p == vec[i])
            {
                j--;
            }
            else
            {
                i++;
            }

        }

        // Recursive quicksort on the left and right part
        aux(vec, f, i - 1);
        aux(vec, i + 1, t);
    }
}
