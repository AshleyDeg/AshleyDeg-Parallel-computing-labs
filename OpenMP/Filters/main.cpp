/* Problem 2 - Assignment 1
 * Implementation of two image kernels: Blur and Sobel filters.
 * Blur is implemented in two ways: Box blur and Gaussian blur
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <fstream>

#ifdef _OPENMP
#include <omp.h> // for OpenMP library functions
#endif

using namespace cv;
using namespace std;

void create_gaussian_filter(const int filter_size, std::vector<std::vector<double>> &filter);
void create_box_filter(const int filter_size, std::vector<std::vector<double>> &filter);
void print_filter(std::vector<std::vector<double>> filter, int size);
double mean_value(double arr[], int len);
Mat convert_to_gray(const Mat& input);
Mat apply_2d_convolution(const Mat& img, const int filter_size, std::vector<std::vector<double>> &filter);
Mat sobel_tot(const Mat& img);

//Parallel functions
Mat apply_2d_convolution_par(const Mat& img, const int filter_size, std::vector<std::vector<double>> &filter);
Mat sobel_tot_par(const Mat& img);

int main() {

#ifdef _OPENMP
    std::cout << "\n _OPENMP defined" << std::endl;
    std::cout << "Num processors (Phys+HT): " << omp_get_num_procs() << std::endl << "\n";
#endif

    // Read the image file
    Mat img = imread("/Users/ashleydeglinnocenti/Desktop/Parallel Computing LNU/Assignment 1/P2/2.jpg"); //medium image
    // Check for failure
    if (img.empty())
    {
        cout << "Could not open or find the image" << endl;
        cin.get(); //wait for any key press
        return -1;
    }

    // Image properties
    cout << "image width: " << img.cols << '\n' <<
         "image height: " << img.rows << '\n'  <<
         "image size: " << img.size().width << '*' << img.size().height << '\n' <<
         "image depth: " << img.depth() << '\n' <<
         "image channels " << img.channels() << '\n' << endl;

    // Arrays for average values
    int n_avg = 10;
    double box[n_avg], box_par[n_avg], speed_up_box[n_avg];
    double gauss[n_avg], gauss_par[n_avg], speed_up_gauss[n_avg];
    double sobel[n_avg], sobel_par[n_avg], speed_up_sobel[n_avg];
    double avg_box, avg_box_par, avg_speedup_box, avg_gauss, avg_gauss_par, avg_speedup_gauss, avg_sob, avg_sob_par, avg_speedup_sob;


    // For loop over filter size - Box and Gaussian filters
    for (unsigned int filter_size = 3; filter_size < 102; filter_size += 2){
        cout << "\n\nFilter size: " << filter_size << endl;
        // Write average result to file
        std::ofstream results_file("/Users/ashleydeglinnocenti/Desktop/Parallel Computing LNU/Assignment 1/P2/results_a1_p2_correct.csv", std::ios::app);

        // Create and compute box filter
        std::vector<std::vector<double>> box_filter(filter_size, std::vector<double>(filter_size, 0.0));
        create_box_filter(filter_size, box_filter);
        //create Gaussian filter
        std::vector<std::vector<double>> gaussian_filter(filter_size, std::vector<double>(filter_size, 0.0));
        create_gaussian_filter(filter_size, gaussian_filter);

        /*
        // Print filters
        // Print box filter
        cout << "Box filter: " << endl;
        print_filter(box_filter, filter_size);

        // Print Gaussian filter
        cout << "\nGaussian filter: " << endl;
        print_filter(gaussian_filter, filter_size);
         */
        // Compute the average computation time over multiple runs
        for (int i = 0; i<n_avg; i++){
            cout << "Iteration: " << i << endl;

            // Apply box filter
            cout << "Box filter" << endl;
            // Sequential
            double elapsed_time, start_time = omp_get_wtime(); // get start time
            Mat out_box = apply_2d_convolution(img, filter_size, box_filter); // computation
            double end_time = omp_get_wtime(); // get end time
            elapsed_time = end_time - start_time; // elapsed time
            cout << "Computation time (sequential): " << elapsed_time << "s." << endl;
            box[i] = elapsed_time;

            // Parallel
            double elapsed_time_par, start_time_par = omp_get_wtime(); // get start time
            Mat out_box_par = apply_2d_convolution_par(img, filter_size, box_filter); // computation
            double end_time_par = omp_get_wtime(); // get end time
            elapsed_time_par = end_time_par - start_time_par; // elapsed time
            cout << "Computation time (parallel): " << elapsed_time_par << "s." << endl;
            double speed_up = elapsed_time/elapsed_time_par;
            cout << "Speed-up: " << speed_up<< endl;
            box_par[i] = elapsed_time_par;
            speed_up_box[i] = speed_up;


            //Gaussian filter
            cout << "Gaussian filter" << endl;
            // Sequential
            start_time = omp_get_wtime(); // get start time
            Mat out_gauss = apply_2d_convolution(img, filter_size, gaussian_filter); // computation
            end_time = omp_get_wtime(); // get end time
            elapsed_time = end_time - start_time; // elapsed time
            cout << "Computation time (sequential): " << elapsed_time << "s." << endl;
            gauss[i] = elapsed_time;

            // Parallel
            start_time_par = omp_get_wtime(); // get start time
            Mat out_gauss_par = apply_2d_convolution_par(img, filter_size, gaussian_filter); // computation
            end_time_par = omp_get_wtime(); // get end time
            elapsed_time_par = end_time_par - start_time_par; // elapsed time
            cout << "Computation time (parallel): " << elapsed_time_par << "s." << endl;
            speed_up = elapsed_time/elapsed_time_par;
            cout << "Speed-up: " << speed_up<< endl;
            gauss_par[i] = elapsed_time_par;
            speed_up_gauss[i] = speed_up;

            /*
            // Show convolution results on image
            if ((filter_size == 7)&&(i == 0)){
                imshow("Original image", img);
                imshow("Box filter, filter size = 7 ", out_box);
                imshow("Box filter PARALLEL", out_box_par);
                imshow("Gaussian filter, filter size = 7 ", out_gauss);
                imshow("Gaussian filter PARALLEL", out_gauss_par);
                waitKey(0);
            }
             */

        }

        // Compute average values
        avg_box = mean_value(box, n_avg);
        avg_box_par = mean_value(box_par, n_avg);
        avg_speedup_box = mean_value(speed_up_box, n_avg);

        avg_gauss = mean_value(gauss, n_avg);
        avg_gauss_par = mean_value(gauss_par, n_avg);
        avg_speedup_gauss = mean_value(speed_up_gauss, n_avg);

        /*
        // Print average results
        cout << "\nAverage computation time for box filter (sequential) " << avg_box << endl;
        cout << "Average computation time for box filter (parallel) " << avg_box_par << endl;
        cout << "Average speed-up for box filter " << avg_speedup_box << endl;

        cout << "\nAverage computation time for Gaussian filter (sequential) " << avg_gauss << endl;
        cout << "Average computation time for Gaussian filter (parallel) " << avg_gauss_par << endl;
        cout << "Average speed-up for Gaussian filter " << avg_speedup_gauss << endl;

        */


        // Writing the results to file
        results_file << filter_size << "," << avg_box << "," << avg_box_par << "," << avg_speedup_box <<
        "," << avg_gauss << "," << avg_gauss_par << "," << avg_speedup_gauss << std::endl;

        results_file.close();

    }

    n_avg = 15;
    // For loop over different images (with different sizes) - Sobel filter
    for (int img_number = 1; img_number<=3; img_number++) {
        // For loop over different images - Sobel filter
        std::ofstream results_file_sob("/Users/ashleydeglinnocenti/Desktop/Parallel Computing LNU/Assignment 1/P2/results_a1_p2_sobel.csv", std::ios::app);

        // Read the image file
        img = imread("/Users/ashleydeglinnocenti/Desktop/Parallel Computing LNU/Assignment 1/P2/" + std::to_string(img_number) + ".jpg"); //medium image
        // Check for failure
        if (img.empty())
        {
            cout << "Could not open or find the image" << endl;
            cin.get(); //wait for any key press
            return -1;
        }
        std::string img_size = std::to_string(img.size().width) + "*" + std::to_string(img.size().height);

        for (int i = 0; i < n_avg; i++) {
            cout << "Iteration: " << i << endl;
            double elapsed_time, start_time, end_time;
            double elapsed_time_par, start_time_par, end_time_par;
            double speed_up;


            //Sobel filter, applied in X and Y directions
            cout << "Sobel filter" << endl;
            // Sequential
            start_time = omp_get_wtime(); // get start time
            Mat out_sobel = sobel_tot(img);  // computation
            end_time = omp_get_wtime(); // get end time
            elapsed_time = end_time - start_time; // elapsed time
            cout << "Computation time (sequential): " << elapsed_time << "s." << endl;
            sobel[i] = elapsed_time;

            // Parallel
            start_time_par = omp_get_wtime(); // get start time
            Mat out_sobel_par = sobel_tot_par(img); // computation
            end_time_par = omp_get_wtime(); // get end time
            elapsed_time_par = end_time_par - start_time_par; // elapsed time
            cout << "Computation time (parallel): " << elapsed_time_par << "s." << endl;
            speed_up = elapsed_time / elapsed_time_par;
            cout << "Speed-up: " << speed_up << endl;
            sobel_par[i] = elapsed_time_par;
            speed_up_sobel[i] = speed_up;

            /*
            if  (i == 0){
                //Show Sobel convolution results on image
                imshow("Original image", img);
                imshow("Sobel filter ", out_sobel);
                imshow("Sobel filter PARALLEL", out_sobel_par);
                waitKey(0);
            }
             */

        }

        // Compute average values
        avg_sob = mean_value(sobel, n_avg);
        avg_sob_par = mean_value(sobel_par, n_avg);
        avg_speedup_sob = mean_value(speed_up_sobel, n_avg);

        // Writing the results to file
        results_file_sob << img_size << "," << avg_sob << "," <<
                     avg_sob_par << "," << avg_speedup_sob << std::endl;

        results_file_sob.close();
    }

    return 0;
}

void create_box_filter(const int filter_size, std::vector<std::vector<double>> &filter){
    // Function to generate the box filter
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
    // Function to generate the Gaussian filter
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
    // Function that takes the mean value of an array of doubles
    double mean, sum = 0;
    for (int i = 0; i < len; i++) {
        sum += arr[i];
    }
    mean = sum/len;

    return mean;
}

Mat convert_to_gray(const Mat& input) {
    // Function that converts to gray a color image
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

Mat apply_2d_convolution(const Mat& img, const int filter_size, std::vector<std::vector<double>> &filter) {
    // 2D Convolution: applies a filter to a color image
    int radius = filter_size / 2;
    int x, y;
    double tmp;
    cv::Mat out(img.rows, img.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    // 2D convolution, iterate over the three channels
    for (int i = 0; i < img.rows; i++) {
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
                out.at<cv::Vec3b>(i,j)[c] = saturate_cast<uchar>(tmp);
            }
        }
    }

    return out;
}

Mat sobel_tot(const Mat& img) {
    /* Function that applies the Sobel filter to an image.
     * The image is first converted to grayscale.
     * The filter is applied both in horizontal and vertical directions, then the results are combined. */
    Mat gray_img = convert_to_gray(img);
    int filter_size = 3, radius = filter_size / 2;
    std::vector<std::vector<double>> sobelX = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}}; // x direction
    std::vector<std::vector<double>> sobelY = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}}; // y direction
    Mat out(img.rows, img.cols, CV_8U, Scalar(0));
    int x, y;
    double tmpX, tmpY;
    // 2D convolution
    for (int i = 0; i < gray_img.rows; i++) {
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
            out.at<uchar>(i,j) = saturate_cast<uchar>(sqrt(tmpX*tmpX + tmpY*tmpY));
        }
    }
    return out;
}

Mat apply_2d_convolution_par(const Mat& img, const int filter_size, std::vector<std::vector<double>> &filter) {
    /* 2D Convolution: applies a filter to a color image
     * Parallel version */
    int radius = filter_size / 2;
    cv::Mat out(img.rows, img.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    // 2D convolution
    int x, y;
    double tmp;
#pragma omp parallel for default(none) shared(img, out, radius, filter_size, filter) private(x, y, tmp) collapse(2)
    for (int i = 0; i < img.rows; i++) {
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
                out.at<cv::Vec3b>(i,j)[c] = saturate_cast<uchar>(tmp);
            }
        }
    }

    return out;
}

Mat sobel_tot_par(const Mat& img) {
    /* Function that applies the Sobel filter to an image.
     * The image is first converted to grayscale.
     * The filter is applied both in horizontal and vertical directions, then the results are combined.
     * Parallel version. */
    Mat gray_img = convert_to_gray(img);
    int filter_size = 3, radius = filter_size / 2;
    std::vector<std::vector<double>> sobelX = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}}; // x direction
    std::vector<std::vector<double>> sobelY = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}}; // y direction
    Mat out(gray_img.rows, gray_img.cols, CV_8U, Scalar(0));
    // 2D convolution
    int x, y;
    double tmpX, tmpY;
#pragma omp parallel for default(none) shared(gray_img, out, sobelX, sobelY, radius, filter_size) private(x, y, tmpX, tmpY) collapse(2)
    for (int i = 0; i < gray_img.rows; i++) {
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
            out.at<uchar>(i,j) = saturate_cast<uchar>(sqrt(tmpX*tmpX + tmpY*tmpY));
        }
    }
    return out;
}

