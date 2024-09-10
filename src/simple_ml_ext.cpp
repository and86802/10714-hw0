#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    std::vector<float> logits(batch * k, 0.0f);
    std::vector<float> Z(batch * k, 0.0f);
    std::vector<float> y_one_hot(batch * k, 0.0f);
    std::vector<float> grad(n * k, 0.0f);

    for (size_t i = 0; i < m; i += batch){
        size_t batch_size = std::min(batch, m-i);

        for (size_t j = 0; j < batch_size; j++){
            for (size_t c = 0; c < k; c++){
                logits[j * k + c] = 0;
                for (size_t p = 0; p < n; p++){
                    logits[j * k + c] += X[(i + j) * n + p] * theta[p * k + c];
                }
            }
        }

        for (size_t j = 0; j < batch_size; j++){
            float sum_exp = 0;
            for (size_t c = 0; c < k; c++){
                Z[j * k + c] = exp(logits[j * k + c]);
                sum_exp += Z[j * k + c];
            }

            for (size_t c = 0; c < k; c++){
                Z[j * k + c] /= sum_exp;
            }
        }

        std::fill(y_one_hot.begin(), y_one_hot.end(), 0.0f);
        for (size_t j = 0; j < batch_size; j++){
            y_one_hot[j * k + y[i + j]] = 1.0f;
        }

        std::fill(grad.begin(), grad.end(), 0.0f);
        for (size_t p = 0; p < n; p++){
            for (size_t c = 0; c < k; c++){
                for (size_t j = 0; j < batch_size; j++){
                    grad[p * k + c] += X[(i + j) * n + p] * (Z[j * k + c] - y_one_hot[j * k + c]);
                }
            }
        }

        for (size_t p = 0; p < n; p++){
            for (size_t c = 0; c < k; c++){
                theta[p * k + c] -= lr * grad[p * k + c] / batch_size;
            }
        }

    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
