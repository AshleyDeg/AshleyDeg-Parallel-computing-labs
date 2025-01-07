/* Problem 2 - Assignment 2
 * MPI message passing interface
 * Implementation of two image kernels: Blur and Sobel filters.
 * Blur is implemented in two ways: Box blur and Gaussian blur
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <mpi.h>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <chrono>

using namespace cv;
using namespace std;

// Struct to store computation times and speedup
struct TripleValues {
    double seq_time, par_time, speed_up;
};

TripleValues filter(const char c, int filter_size, int rank, int size);
TripleValues sobel_filter(int rank, int size, int img_num);

void create_gaussian_filter(const int filter_size, std::vector<std::vector<double>> &filter);
void create_box_filter(const int filter_size, std::vector<std::vector<double>> &filter);
void print_filter(std::vector<std::vector<double>> filter, int size);
double mean_value(double arr[], int len);
Mat convert_to_gray(const Mat& input);
Mat apply_2d_convolution(const Mat& img, const int filter_size, std::vector<std::vector<double>> &filter, int s_row, int e_row);
Mat sobel_tot(const Mat& img, int s_row, int e_row);



int main() {
    const char box = 'b';
    const char gaussian = 'g';
    int n_avg = 30;
    double box_seq[n_avg], box_par[n_avg], speed_up_box[n_avg];
    double gauss[n_avg], gauss_par[n_avg], speed_up_gauss[n_avg];
    double sobel[n_avg], sobel_par[n_avg], speed_up_sobel[n_avg];
    double avg_box, avg_box_par, avg_speedup_box, avg_gauss, avg_gauss_par, avg_speedup_gauss, avg_sob, avg_sob_par, avg_speedup_sob;

    MPI_Init(nullptr, nullptr);

    // Get the number of processes
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //MPI_Barrier(MPI_COMM_WORLD);
    //TripleValues box_val = filter(box, filter_size, rank, size);
    for (int filter_size = 3; filter_size < 110; filter_size +=2 ){
        std::ofstream results_file("/Users/ashleydeglinnocenti/Desktop/Parallel Computing LNU/Assignment 2/Problem2/results_correct.csv", std::ios::app);
        for (int i = 0; i <n_avg; i++){
            TripleValues box_val = filter(box, filter_size, rank, size);
            MPI_Barrier(MPI_COMM_WORLD);
            TripleValues gauss_val = filter(gaussian, filter_size, rank, size);
            MPI_Barrier(MPI_COMM_WORLD);

            if (rank == 0 ){
                std::cout << "Filter size: " << filter_size << ", iteration: " << i << std::endl;
                box_seq[i] = box_val.seq_time;
                box_par[i] = box_val.par_time;
                speed_up_box[i] = box_val.speed_up;

                gauss[i] = gauss_val.seq_time;
                gauss_par[i] = gauss_val.par_time;
                speed_up_gauss[i] = gauss_val.speed_up;
            }
        }
        if (rank == 0){
            avg_box = mean_value(box_seq, n_avg);
            avg_box_par = mean_value(box_par, n_avg);
            avg_speedup_box = mean_value(speed_up_box, n_avg);

            avg_gauss = mean_value(gauss, n_avg);
            avg_gauss_par = mean_value(gauss_par, n_avg);
            avg_speedup_gauss = mean_value(speed_up_gauss, n_avg);

            // Print average results
            cout << "\nAverage computation time for box filter (sequential) " << avg_box << endl;
            cout << "Average computation time for box filter (parallel) " << avg_box_par << endl;
            cout << "Average speed-up for box filter " << avg_speedup_box << endl;

            cout << "\nAverage computation time for Gaussian filter (sequential) " << avg_gauss << endl;
            cout << "Average computation time for Gaussian filter (parallel) " << avg_gauss_par << endl;
            cout << "Average speed-up for Gaussian filter " << avg_speedup_gauss << endl;

            // Writing to the results file
            results_file << filter_size << "," << avg_box << "," << avg_box_par << "," << avg_speedup_box <<
                         "," << avg_gauss << "," << avg_gauss_par << "," << avg_speedup_gauss << std::endl;

            results_file.close();

        }
    }
    /*
    for (int img_num = 1; img_num<=3; img_num++){
        std::ofstream results_file("/Users/ashleydeglinnocenti/Desktop/Parallel Computing LNU/Assignment 2/Problem2/results_sobel.csv", std::ios::app);
        for (int i = 0; i <n_avg; i++) {
            TripleValues sobel_val = sobel_filter(rank, size, img_num);
            if (rank == 0 ){
                sobel[i] = sobel_val.seq_time;
                sobel_par[i] = sobel_val.par_time;
                speed_up_sobel[i] = sobel_val.speed_up;
            }
        }
        if (rank == 0){
            avg_sob = mean_value(sobel, n_avg);
            avg_sob_par = mean_value(sobel_par, n_avg);
            avg_speedup_sob = mean_value(speed_up_sobel, n_avg);

            // Print average results
            cout << "\nAverage computation time for Sobel filter (sequential) " << avg_sob << endl;
            cout << "Average computation time for Sobel filter (parallel) " << avg_sob_par << endl;
            cout << "Average speed-up for Sobel filter " << avg_speedup_sob << endl;

            // Writing to the results file
            results_file << img_num << "," << avg_sob << "," << avg_sob_par << "," << avg_speedup_sob << std::endl;

            results_file.close();

        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    TripleValues gauss_val = filter(gaussian, filter_size, rank, size);

    //MPI_Barrier(MPI_COMM_WORLD);
    //TripleValues sobel_val = sobel_filter(rank, size); */

    // Terminates MPI execution environment
    MPI_Finalize();

    return 0;
}

TripleValues filter(const char c, int filter_size, int rank, int size){
    double start_time, end_time, elapsed_time;

    // Load image
    Mat img = imread("/Users/ashleydeglinnocenti/Desktop/Parallel Computing LNU/Assignment 2/Problem2/Images/2.jpg"); //medium image
    if (img.empty()) {
        std::cerr << "Error: Could not read the image." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rows_per_thread = img.rows / size;
    const int s_row = rank * rows_per_thread;
    const int e_row = (rank == size - 1) ? img.rows : ( s_row +1 ) + rows_per_thread;

    int img_rows = img.rows, img_cols = img.cols;
    // Create and compute filter
    std::vector<std::vector<double>> filter(filter_size, std::vector<double>(filter_size, 0.0));
    std::string win_name;
    switch (c) {
        case 'b':
            create_box_filter(filter_size, filter);
            if (rank==0){
                win_name = "Box filter";
                std::cout << win_name << std::endl;
            }
            break;
        case 'g':
            create_gaussian_filter(filter_size, filter);
            if (rank==0){
                win_name = "Gaussian filter";
                std::cout << win_name << std::endl;
            }
            break;
        default:
            MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Apply box filter
    // Local result
    Mat local_img(img_rows, img_cols, CV_8UC3, Scalar(0,0,0));
    // Final result
    Mat result(img_rows, img_cols, CV_8UC3, Scalar(0,0,0));

    MPI_Barrier ( MPI_COMM_WORLD );
    start_time = MPI_Wtime();
    // Apply convolution locally
    local_img = apply_2d_convolution(img, filter_size, filter, s_row, e_row);

    //Gather results back to rank 0
    MPI_Gather(local_img.data + s_row * img_cols * 3, // Data sent by each process
               (e_row-s_row) * local_img.cols * 3, // Number of elements that are sent
               MPI_UNSIGNED_CHAR, // Type of the data that is sent
               result.data + s_row * img_cols * 3, // Vector in which the data is collected
               (e_row-s_row) * local_img.cols * 3, // Number of data that are expected to be received by each process
               MPI_UNSIGNED_CHAR, // process that will receive the data
               0, MPI_COMM_WORLD); // Communication channel (Global Communicator)
    MPI_Barrier ( MPI_COMM_WORLD );
    end_time = MPI_Wtime();
    TripleValues time_result;
    double start_time_seq, end_time_seq, elapsed_time_seq, speed_up;
    if (rank == 0) {
        //Sequential part
        start_time_seq = MPI_Wtime();
        Mat out_box = apply_2d_convolution(img, filter_size, filter, 0, img_rows); // computation
        end_time_seq = MPI_Wtime();
        time_result.seq_time = end_time_seq - start_time_seq;
        time_result.par_time = end_time - start_time;
        time_result.speed_up = time_result.seq_time / time_result.par_time;
        std::cout << "Computation time (sequential):" << time_result.seq_time << " s." << std::endl;
        std::cout << "Computation time (parallel):" << time_result.par_time << " s." << std::endl;
        std::cout << "Speedup: " << time_result.speed_up << std::endl;
        /*
        // Display the final convolved image
        imshow("Input image.jpg", img);
        imshow(win_name, out_box);
        imshow(win_name + " PARALLEL", result);
        waitKey(0);*/
    }
    return time_result;
}

void create_box_filter(const int filter_size, std::vector<std::vector<double>> &filter){
    // Generate the box filter
    int radius = filter_size / 2;
    double sum = 0.0; // sum is for normalization
    double value;

    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            value = 1;
            filter[i + radius][j + radius] = value;
            sum += value;
        }
    }

    // Normalizing the kernel
    for (int i = 0; i < filter_size; i++) {
        for (int j = 0; j < filter_size; j++) {
            filter[i][j] /= sum;
        }
    }
}

void create_gaussian_filter(const int filter_size, std::vector<std::vector<double>> &filter){
    // Generate the Gaussian filter
    int radius = filter_size / 2;
    double sum = 0.0; // sum is for normalization
    double sigma = 1.0; // initialising standard deviation to 1.0
    double value, r, s = 2 * sigma * sigma;

    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            r = sqrt(i * i + j * j);
            value = std::exp(-(r) / s) / (M_PI * s);
            filter[i + radius][j + radius] = value;
            sum += value;
        }
    }

    // Normalizing the kernel
    for (int i = 0; i < filter_size; i++) {
        for (int j = 0; j < filter_size; j++) {
            filter[i][j] /= sum;
        }
    }
}

void print_filter(std::vector<std::vector<double>> filter, int size){
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++)
            std::cout << filter[i][j] << "\t";
        std::cout << std::endl;
    }
}

double mean_value(double arr[], int len) {
    double mean, sum = 0;
    for (int i = 0; i < len; i++) {
        sum += arr[i];
    }
    mean = sum/len;

    return mean;
}

Mat convert_to_gray(const Mat& input) {
    Mat output_gray(input.rows, input.cols, CV_8U, Scalar(0));
    if (input.channels() == 3) {
        for (int i = 0; i < input.rows; ++i) {
            for (int j = 0; j < input.cols; ++j) {
                int blue = input.at<cv::Vec3b>(i, j)[0];
                int green = input.at<cv::Vec3b>(i, j)[1];
                int red = input.at<cv::Vec3b>(i, j)[2];

                int grayscaleValue = 0.11 * blue + 0.59 * green + 0.3 * red;

                output_gray.at<uchar>(i, j) = static_cast<uchar>(grayscaleValue);
            }
        }
    } else {
        output_gray = input.clone();
    }
    return output_gray;
}

Mat apply_2d_convolution(const Mat& img, const int filter_size, std::vector<std::vector<double>> &filter, int s_row, int e_row) {
    int radius = filter_size / 2;
    int x, y;
    double tmp;
    cv::Mat out(img.rows, img.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    // 2D convolution
    for (int i = s_row; i < e_row; i++) {
        for (int j = 0; j < img.cols; j++) {
            for (int c=0; c < img.channels(); c++) {
                tmp = 0.0f;
                for (int k = 0; k < filter_size; k++) {
                    for (int l = 0; l < filter_size; l++) {
                        x = j - radius + l;
                        y = i - radius + k;
                        if (x >= 0 && y < img.rows && y >= 0 && x < img.cols) {
                            // c = 0 blue channel, c = 1 green channel, c = 2 red channel
                            tmp += img.data[(y * img.cols + x) * 3 + c] * filter[k][l];
                        }
                    }
                }
                out.at<cv::Vec3b>(i, j)[c] = saturate_cast<uchar>(tmp);
            }
        }
    }

    return out;
}

Mat sobel_tot(const Mat& img, int s_row, int e_row) {
    Mat gray_img = convert_to_gray(img);
    int filter_size = 3, radius = filter_size / 2;
    std::vector<std::vector<double>> sobelX = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}}; // x direction
    std::vector<std::vector<double>> sobelY = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}}; // y direction
    Mat out(img.rows, img.cols, CV_8U, Scalar(0));
    int x, y;
    double tmpX, tmpY;
    // 2D convolution
    for (int i = s_row; i < e_row; i++) {
        for (int j = 0; j < gray_img.cols; j++) {
            tmpX = 0.0f, tmpY = 0.0f;
            for (int k = 0; k < filter_size; k++) {
                for (int l = 0; l < filter_size; l++) {
                    x = j - radius + l;
                    y = i - radius + k;
                    if (x >= 0 && y < gray_img.rows && y >= 0 && x < gray_img.cols) {
                        // c = 0 blue channel, c = 1 green channel, c = 2 red channel
                        tmpX += gray_img.data[(y * gray_img.cols + x)] * sobelX[k][l];
                        tmpY += gray_img.data[(y * gray_img.cols + x)] * sobelY[k][l];
                    }
                }
            }
            out.at<uchar>(i, j) = saturate_cast<uchar>(sqrt(tmpX*tmpX + tmpY*tmpY));
        }
    }
    return out;
}

TripleValues sobel_filter(int rank, int size, int img_num){
    double start_time, end_time, elapsed_time;

    // Load image
    Mat img = imread("/Users/ashleydeglinnocenti/Desktop/Parallel Computing LNU/Assignment 2/Problem2/Images/" + std::to_string(img_num) + ".jpg");
    if (img.empty()) {
        std::cerr << "Error: Could not read the image." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rows_per_thread = img.rows / size;
    const int s_row = rank * rows_per_thread;
    const int e_row = (rank == size - 1) ? img.rows : ( s_row +1 ) + rows_per_thread;

    int img_rows = img.rows, img_cols = img.cols;

    // Apply sobel filter
    Mat local_img(img_rows, img_cols, CV_8U, Scalar(0));
    Mat result(img_rows, img_cols, CV_8U, Scalar(0));

    MPI_Barrier ( MPI_COMM_WORLD );
    start_time = MPI_Wtime();
    // Apply convolution locally
    local_img = sobel_tot(img, s_row, e_row);

    //Gather results back to rank 0
    MPI_Gather(local_img.data + s_row * img_cols, // Data sent by each process
               (e_row-s_row) * local_img.cols, // Number of elements that are sent
               MPI_UNSIGNED_CHAR, // Type of the data that is sent
               result.data + s_row * img_cols, // Vector in which the data is collected
               (e_row-s_row) * local_img.cols, // Number of data that are expected to be received by each process
               MPI_UNSIGNED_CHAR, // process that will receive the data
               0, MPI_COMM_WORLD); // Communication channel (Global Communicator)
    MPI_Barrier ( MPI_COMM_WORLD );
    end_time = MPI_Wtime();

    double start_time_seq, end_time_seq, elapsed_time_seq, speed_up;
    TripleValues time_result;
    if (rank == 0) {
        //Sequential part
        start_time_seq = MPI_Wtime();
        Mat out_box = sobel_tot(img, 0, img_rows);  // computation
        end_time_seq = MPI_Wtime();
        time_result.seq_time = end_time_seq - start_time_seq;
        time_result.par_time = end_time - start_time;
        time_result.speed_up = time_result.seq_time / time_result.par_time;
        std::cout << "Sobel filter" << std::endl;
        std::cout << "Computation time (sequential):" << time_result.seq_time << " s." << std::endl;

        std::cout << "Computation time (parallel):" << time_result.par_time << " s." << std::endl;
        std::cout << "Speedup: " << time_result.speed_up << std::endl;

        /*
        // Display the final convolved image
        imshow("Input image.jpg", img);
        imshow("Sobel", out_box);
        imshow("Sobel PARALLEL", result);
        waitKey(0);*/

    }
    return time_result;
}