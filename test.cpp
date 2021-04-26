#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>

using namespace std;


int main(){
//     Eigen::MatrixXd A = Eigen::MatrixXd::Random(2,2);
//     for (int i=0;i<2;i++){
//         for (int j=0;j<2;j++){
//             A(i,j) = i*j;
//         }
//     }
//     Eigen::MatrixXd B = Eigen::MatrixXd::Random(2,2);
//     for (int i=0;i<2;i++){
//         for (int j=0;j<2;j++){
//             A(i,j) = i*j;
//         }
//     }
//     Eigen::MatrixXd res3;
//     res3 = (A.cwiseProduct(B)).rowwise().sum();
//     cout << res3 << endl;
    vector<vector<vector<double>>>(3, vector<vector<double>>(3, vector<double>(3, 0)));
}