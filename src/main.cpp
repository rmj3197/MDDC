#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <tuple>
#include <limits>
#include <algorithm>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

Eigen::MatrixXd getEijMat(const Eigen::MatrixXd &continTable)
{
    // This calculates the expected count matrix Eij

    int nRow = continTable.rows();
    int nCol = continTable.cols();
    Eigen::VectorXd niDot = continTable.rowwise().sum();
    Eigen::VectorXd nDotj = continTable.colwise().sum();
    double nDotDot = continTable.sum();

    Eigen::VectorXd piDot = niDot / nDotDot;
    Eigen::VectorXd pDotj = nDotj / nDotDot;

    Eigen::MatrixXd EijMat = (niDot * nDotj.transpose()) / nDotDot;

    return EijMat;
}

std::tuple<Eigen::MatrixXd, double, Eigen::VectorXd, Eigen::VectorXd> getZijMat(const Eigen::MatrixXd &continTable, bool na = true)
{
    // This calculates the standardized Pearson residuals Zij
    int nRow = continTable.rows();
    int nCol = continTable.cols();
    Eigen::VectorXd niDot = continTable.rowwise().sum();
    Eigen::VectorXd nDotj = continTable.colwise().sum();
    double nDotDot = continTable.sum();

    Eigen::VectorXd piDot = niDot / nDotDot;
    Eigen::VectorXd pDotj = nDotj / nDotDot;

    Eigen::MatrixXd EijMat = (niDot * nDotj.transpose()) / nDotDot;

    Eigen::MatrixXd temp = (1 - piDot.array()).matrix() * (1 - pDotj.array()).transpose().matrix();
    Eigen::MatrixXd sqrtEijMat = EijMat.cwiseProduct(temp).array().sqrt();

    Eigen::MatrixXd ZijMat = (continTable - EijMat).array() / sqrtEijMat.array();

    if (na)
    {
        ZijMat = (continTable.array() < 6).select(std::numeric_limits<double>::quiet_NaN(), ZijMat);
    }

    return std::make_tuple(ZijMat, nDotDot, piDot, pDotj);
}
Eigen::VectorXd getPVal(const Eigen::VectorXd &obs, const Eigen::VectorXd &dist)
{
    // This calculates the p values

    // Create a sorted copy of dist
    Eigen::VectorXd sortedDist = dist;
    std::sort(sortedDist.data(), sortedDist.data() + sortedDist.size());

    // Initialize the p-values vector
    Eigen::VectorXd pVals = Eigen::VectorXd::Constant(obs.size(), std::numeric_limits<double>::quiet_NaN());

    // Convert Eigen vectors to std::vectors for use with STL algorithms
    std::vector<double> obsVec(obs.data(), obs.data() + obs.size());
    std::vector<double> sortedDistVec(sortedDist.data(), sortedDist.data() + sortedDist.size());

    // Compute p-values for each observation
    std::transform(obsVec.begin(), obsVec.end(), pVals.data(),
                   [&sortedDistVec, distSize = dist.size()](double obs_i)
                   {
                       if (std::isnan(obs_i))
                           return std::numeric_limits<double>::quiet_NaN();
                       auto it = std::upper_bound(sortedDistVec.begin(), sortedDistVec.end(), obs_i);
                       int count = std::distance(it, sortedDistVec.end());
                       return static_cast<double>(1 + count) / (1 + distSize);
                   });

    return pVals;
}

Eigen::MatrixXd getFisherExactTestTable(Eigen::MatrixXd continTable, int rowIdx, int colIdx, bool excludeSameDrugClass)
{
    // This generates the tables used for Fisher Exact Test
    Eigen::MatrixXd tabl(2, 2);

    // Set the values of tabl
    tabl(0, 0) = continTable(rowIdx, colIdx);
    tabl(1, 0) = continTable.col(colIdx).sum() - continTable(rowIdx, colIdx);

    if (excludeSameDrugClass)
    {
        int n_col = continTable.cols() - 1;
        tabl(0, 1) = continTable(rowIdx, n_col);
        tabl(1, 1) = continTable.col(n_col).sum() - continTable(rowIdx, n_col);
    }
    else
    {
        tabl(0, 1) = continTable.row(rowIdx).sum() - continTable(rowIdx, colIdx);
        tabl(1, 1) = continTable.sum() - continTable.row(rowIdx).sum() - continTable.col(colIdx).sum() + continTable(rowIdx, colIdx);
    }

    return tabl;
}

Eigen::MatrixXd pearsonCorWithNA(const Eigen::MatrixXd &mat, bool ifColCorr = true)
{
    // Computes the Pearson Correlation in the presence of NA values
    Eigen::MatrixXd matrix = ifColCorr ? mat : mat.transpose();
    int nCol = matrix.cols();
    Eigen::MatrixXd corMat(nCol, nCol);
    corMat.fill(std::numeric_limits<double>::quiet_NaN());

    for (int i = 0; i < nCol; ++i)
    {
        for (int j = i + 1; j < nCol; ++j)
        {
            std::vector<double> x, y;
            for (int k = 0; k < matrix.rows(); ++k)
            {
                if (!std::isnan(matrix(k, i)) && !std::isnan(matrix(k, j)))
                {
                    x.push_back(matrix(k, i));
                    y.push_back(matrix(k, j));
                }
            }
            if (x.size() >= 3)
            {
                double meanX = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
                double meanY = std::accumulate(y.begin(), y.end(), 0.0) / y.size();

                double numerator = 0.0;
                double denomX = 0.0;
                double denomY = 0.0;

                for (size_t k = 0; k < x.size(); ++k)
                {
                    double diffX = x[k] - meanX;
                    double diffY = y[k] - meanY;
                    numerator += diffX * diffY;
                    denomX += diffX * diffX;
                    denomY += diffY * diffY;
                }

                double denominator = std::sqrt(denomX * denomY);
                double correlation = numerator / denominator;

                {
                    corMat(i, j) = correlation;
                    corMat(j, i) = correlation;
                }
            }
        }
    }

    return corMat;
}

namespace py = pybind11;

PYBIND11_MODULE(mddc_cpp_helper, m)
{
    m.def("getEijMat", &getEijMat, "Calculate the expected count matrix Eij",
          py::arg("continTable"));

    m.def("getZijMat", &getZijMat, "Calculate the standardized Pearson residuals Zij",
          py::arg("continTable"), py::arg("na") = true);

    m.def("getPVal", &getPVal, "Calculate p vals",
          py::arg("obs"), py::arg("dist"));

    m.def("getFisherExactTestTable", &getFisherExactTestTable,
          py::call_guard<py::gil_scoped_release>(),
          "Generate table used for Fisher Exact Test",
          py::arg("continTable"),
          py::arg("rowIdx"),
          py::arg("colIdx"),
          py::arg("excludeSameDrugClass"));

    m.def("pearsonCorWithNA", &pearsonCorWithNA, "Pearson Correlation with NA values",
          py::arg("mat"), py::arg("ifColCorr") = true);
}
