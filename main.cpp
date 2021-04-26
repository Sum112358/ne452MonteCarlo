using namespace std;

#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>


//need to install the eigen library libeigen3-dev
#include <eigen3/Eigen/Dense>

#define PI 3.14159265

struct param {
    bool     PIGS;
    double   T;
    double   beta;
    int      m_max;
    int      Ngrid;
    int      P;
    double   tau;
    double   B;
    double   V0;
    double   g;
    int      N_accept;
    int      N_total;
    int      MC_steps;
    int      Nskip;
    int      N;
    double   delta_phi;

    // in future version of this code, this should support auto population of this struct
    //from the JSON. But for proof of concept, only the hardcodes struct values are used
    param() {
        PIGS = false;
        T = 1.0;
        beta = 1.0/T;
        m_max = 10;
        Ngrid = 2 * m_max + 1;
        P = 9;
        tau = beta/P;
        B = 1.0;
        V0 = 1.0;
        g = 1.0;
        N_accept = 0;
        N_total = 0;
        MC_steps = 10000;
        Nskip = 100;
        N = 2;
        delta_phi = 2 * PI/Ngrid;
    }
};

//initialize randomness 
random_device rd;
mt19937 gen(rd());

class aliasMethod{
    vector<double> distribution;
    int k;
    vector<double> q;
    vector<double> J;
    vector<double> S;
    vector<double> L;
    uniform_int_distribution<> dis_int;
    uniform_real_distribution<> dis_real;
    
    void buildTable(){
        for (int i=0;i<k;i++){
            q[i] = k*distribution[i];
            if (q[i] < 1.0){
                S.push_back(i);
            } else {
                L.push_back(i);
            }
        }
        while (not S.empty() && not L.empty()){
            double s = S.back();
            S.pop_back();
            double l = L.back();
            L.pop_back();
            J[s] = l;
            q[l] = (q[l] + q[s]) - 1; //this ordering to minimize round-off error
            if (q[l] < 1.0){
                S.push_back(l);
            } else {
                L.push_back(l);
            }
        }
    };
    public:
        aliasMethod(vector<double> _distribution) : 
            dis_int(0,int(_distribution.size())),
            dis_real(0.0, 1.0),
            q(int(_distribution.size())),
            J(int(_distribution.size())),
            S(0),
            L(0)
        {
        k = int(_distribution.size());
        distribution = _distribution;
        buildTable();
    };

        double draw(){
            int index = dis_int(gen);
            if (dis_real(gen) < q[index]){
                return index;
            } else {
                return J[index];
            }
        };
};

int main(){
    param P; //get parameters

    // Solve the 1 Body Hamiltonian
    Eigen::MatrixXd V_mat(P.Ngrid, P.Ngrid);
    for (int i=0;i<P.Ngrid;i++){
        for (int j=0;j<P.Ngrid;j++){
            if (i == j+1 || i == j -1 ){
                V_mat(i,j) = 0.5;
            }
        }
    }

    Eigen::MatrixXd V = P.V0 * V_mat;
    //Copy V
    Eigen::MatrixXd H(P.Ngrid, P.Ngrid);
    for (int i=0;i<P.Ngrid;i++){
        for (int j=0;j<P.Ngrid;j++){
            H(i,j) = V(i,j);
        }
    }

    for (int m=0;m<P.Ngrid;m++){
        double m_value = -P.m_max + m;
        H(m,m) = P.B * pow(m_value,2) + P.V0;
    }

    //Get the eigenValues and EigenVectors
    Eigen::EigenSolver<Eigen::MatrixXd> es;
    es.compute(H);
    Eigen::VectorXd evals = es.eigenvalues().real();
    Eigen::MatrixXd evecs = es.eigenvectors().real();

    Eigen::MatrixXd rho_mmp(P.Ngrid, P.Ngrid);
    //Sum over state method
    double Z_exact = 0;
    for (int m=0;m<P.Ngrid;m++){
        Z_exact += exp(-P.beta*evals[m]);
        for (int mp=0;mp<P.Ngrid;mp++){
            for (int n=0;n<P.Ngrid;n++){
                rho_mmp(m,mp) += exp(-P.beta*evals[n])*evecs(m,n)*evecs(mp,n);
            }
        }
    }
    double Z_exact_pigs = rho_mmp(P.m_max, P.m_max);
    Eigen::MatrixXd rho_dot_V_mmp = rho_mmp * H;
    double E0_pigs_sos = rho_dot_V_mmp(P.m_max, P.m_max);

    // find E0 (sos)
    double minE0 = 9999999999999999;
    for (int i=0;i<evals.size();i++){
        if (minE0 > evals[i]){
            minE0 = evals[i];
        }
    }
    cout << "Z (sos) = " << Z_exact << endl;
    cout << "A (sos) = " << -(1.0/P.beta)*log(Z_exact) << endl;
    cout << "E0 (sos) = " << minE0 << endl;
    cout << "E0 (pigs sos) = " << E0_pigs_sos/Z_exact_pigs << endl;

    //<phi|m><m|n> exp(-beta E n) <n|m'><m'|phi>
    //build Basis
    Eigen::MatrixXd psi_m_phi(P.Ngrid, P.Ngrid);
    for (int i=0;i<P.Ngrid;i++){
        for (int m=0;m<P.Ngrid;m++){
            double m_value = -P.m_max + m;
            psi_m_phi(i,m) = cos(i*P.delta_phi*m_value) / sqrt(2.0*PI);
        }
    }

    Eigen::MatrixXd psi_phi(P.Ngrid, P.Ngrid);
    for (int i=0;i<P.Ngrid;i++){
        for (int n=0;n<P.Ngrid;n++){
            for (int m=0;m<P.Ngrid;m++){
                psi_phi(i,n) += evecs(m,n) * psi_m_phi(i,m);
            }
        }
    }

    ofstream myfile;
    myfile.open("rho_sos");
    for (int i=0;i<P.Ngrid;i++){
        double rho_exact = 0;
        for (int n=0;n<P.Ngrid;n++){
            rho_exact += exp(-P.beta*evals[n])*pow(psi_phi(i,n), 2);
        }
        rho_exact /= Z_exact;
        myfile << i*P.delta_phi << "\t" << rho_exact << "\t" << pow(psi_phi(i,0), 2) << endl;
    }
    myfile.close();

    //free rotor density matrices below
    ofstream myfile2;
    myfile.open("rhofree_sos");
    myfile2.open("rhofree_pqc");
    Eigen::VectorXd rho_phi(P.Ngrid);
    for (int i=0;i<P.Ngrid;i++){
        double dphi = i*P.delta_phi;
        double integral = 0;
        for (int m=1;m<P.m_max;m++){
            integral += (2*cos(m*dphi))*exp(-P.tau*P.B*pow(m,2));
        }
        integral = integral/(2*PI);
        integral = integral + 1/(2*PI);
        rho_phi(i) = abs(integral);
        myfile << dphi << " " << rho_phi[i] << endl;
    }
    myfile.close();
    //PQC rho
    Eigen::VectorXd rho_phi_pqc(P.Ngrid);
    for (int i=0;i<P.Ngrid;i++){
        double dphi = i * P.delta_phi;
        rho_phi_pqc(i) = sqrt(i/(4*PI*P.B*P.tau))*exp(-1./(2.*P.tau*P.B)*(1. - cos(dphi)));
        myfile2 << dphi << " " << rho_phi_pqc(i) << endl;
    }
    myfile2.close();

    // marx Method
    myfile.open("rhofree_marx");
    for (int i=0;i<P.Ngrid;i++){
        double dphi = i*P.delta_phi;
        double integral = 0;
        for (int m=0;m<P.m_max;m++){
            integral += exp(-1./(4.*P.tau*P.B)*pow(dphi+2*PI*m, 2));
        }
        for (int m=1;m<P.m_max;m++){
            integral += exp(-1./ (4.*P.tau*P.B)*pow(dphi+2.*PI*-m, 2));\
        }
        integral *= sqrt(1./(4.*PI*P.B*P.tau));
        rho_phi(i) = integral;
        myfile << dphi << " " << integral << endl;
    }
    myfile.close();

    //potential rho
    Eigen::VectorXd rhoV(P.Ngrid);
    for (int i=0;i<P.Ngrid;i++){
        rhoV(i) = exp(-P.tau*(P.V0*(1+cos(i*P.delta_phi))));
    }
    //rho pair
    Eigen::MatrixXd rhoVij(P.Ngrid, P.Ngrid);
    for (int i=0;i<P.Ngrid;i++){
        for (int j=0;j<P.Ngrid;j++){
            rhoVij(i,j) = exp(-P.tau) * (P.g * (cos(i*P.delta_phi - j*P.delta_phi)) - 3*cos(j*P.delta_phi) );
        }
    }

    //NMM results
    Eigen::MatrixXd rho_free(P.Ngrid, P.Ngrid);
    Eigen::VectorXd rho_potential(P.Ngrid);
    Eigen::VectorXd potential(P.Ngrid);
    for (int i=0;i<P.Ngrid;i++){
        potential(i) = P.V0*(1+cos(i*P.delta_phi));
        rho_potential(i) = exp(-(P.tau/2)*potential(i));
        for (int ip=0;ip<P.Ngrid;ip++){
            double integral = 0;
            double dphi = (i - ip) * P.delta_phi;
            for (int m=0;m<P.m_max;m++){
                integral += exp(-1./(4.*P.tau*P.B)*pow(dphi+2.*PI*m, 2));
            }
            for (int m=1;m<P.m_max;m++){
                integral += exp(-1./(4.*P.tau*P.B)*pow(dphi+2.*PI*-m, 2));
            }
            integral *= sqrt(1./(4.*PI*P.B*P.tau));
            rho_free(i,ip) = integral;
        }
    }

    //output potential to a file
    myfile.open("V");
    for (int i=0;i<P.Ngrid;i++){
        myfile << i*P.delta_phi << " " << potential(i) << endl;
    }
    myfile.close();

    //construct the high temperature density matrix
    Eigen::MatrixXd rho_tau(P.Ngrid, P.Ngrid);
    Eigen::MatrixXd rho_beta(P.Ngrid, P.Ngrid); //copy
    for (int i=0;i<P.Ngrid;i++){
        for (int j=0;j<P.Ngrid;j++){
            rho_tau(i,j) = rho_potential(i) * rho_free(i,j)* rho_potential(j);
            rho_beta(i,j) = rho_potential(i) * rho_free(i,j)* rho_potential(j);
        }
    }
    
    //for the density matrix via matrix multiplication
    for (int k=0;k<P.P-1;k++){
        rho_beta = P.delta_phi*rho_beta*rho_tau;
    }

    double E0_nmm = 0;
    Eigen::MatrixXd rho_dot_V(P.Ngrid, P.Ngrid);
    rho_dot_V = rho_beta * potential;
    double Z0 = 0; //PIGS pseudo Z
    myfile.open("rho_nmm");
    double Z_nmm = rho_beta.trace()*P.delta_phi; //thermal Z

    for (int i=0;i<P.Ngrid;i++){
        E0_nmm += rho_dot_V(i);
        for (int ip=0;ip<P.Ngrid;ip++){
            Z0 += rho_beta(i,ip);
            myfile << i*P.delta_phi << " " << rho_beta(i,i) / Z_nmm << endl;
        }
    }
    myfile.close();
    E0_nmm /= Z0;

    cout << "Z (tau) = " << Z_nmm << endl;
    cout << "E0 (tau) = " << E0_nmm << endl;

    myfile.open("Evst");
    myfile << P.tau << " " << E0_nmm << endl;
    myfile.close();
    cout << endl;

    vector<double> histo_L(P.Ngrid, 0);
    vector<double> histo_R(P.Ngrid, 0);
    vector<double> histo_middle(P.Ngrid, 0);
    vector<double> histo_pimc(P.Ngrid, 0);

    Eigen::VectorXd p_test(P.Ngrid);

    // for (int i=0;i<P.Ngrid;i++){
    //     cout << rho_phi(i) << endl;
    // }
    // Gen Prob Dist Old Method
    vector<vector<vector<double>>> p_dist(P.Ngrid, vector<vector<double>>(P.Ngrid, vector<double>(P.Ngrid, 0)));
    vector<vector<double>> p_norm(P.Ngrid, vector<double>(P.Ngrid, 0));
    for (int i0=0;i0<P.Ngrid;i0++){
        for (int i1=0;i1<P.Ngrid;i1++){
            double di01 = (((i0 -i1) % P.Ngrid) + P.Ngrid) % P.Ngrid;
            for (int i2=0;i2<P.Ngrid;i2++){
                double di12 = (((i1 - i2) % P.Ngrid) + P.Ngrid) % P.Ngrid;
                p_dist[i0][i1][i2] = rho_phi(di01)*rho_phi(di12);
                p_norm[i0][i2] += p_dist[i0][i1][i2];
            }
        }
    }
    for (int i0=0;i0<P.Ngrid;i0++){
        for (int i1=0;i1<P.Ngrid;i1++){
            for (int i2=0;i2<P.Ngrid;i2++){
                p_dist[i0][i1][i2] = p_dist[i0][i1][i2] / p_norm[i0][i2];
            }
        }
    }
    p_norm.clear(); //free the p_norm memory

    //Gen prob Dist End Old Method
    vector<vector<double>> p_dist_end(P.Ngrid, vector<double>(P.Ngrid, 0));
    vector<double> p_norm_end(P.Ngrid, 0);
    for (int i0=0;i0<P.Ngrid;i0++){
        for (int i1=0;i1<P.Ngrid;i1++){
            double di01 = (((i0 -i1) % P.Ngrid) + P.Ngrid) % P.Ngrid;
            p_dist_end[i0][i1] = rho_phi(di01);
            p_norm_end[i0] += p_dist_end[i0][i1];
        }
    }
    for (int i0=0;i0<P.Ngrid;i0++){
        for (int i1=0;i1<P.Ngrid;i1++){
            p_dist_end[i0][i1] = p_dist_end[i0][i1] / p_norm_end[i0];
        }
    }
    p_norm_end.clear(); //free the p_norm_end memory


    // path phi
    vector<vector<int>> path_phi(P.N, vector<int>(P.P, 0));
    // random generator
    uniform_int_distribution<> distrib(0,P.Ngrid-1);
    for (int i=0;i<P.N;i++){
        for (int p=0;p<P.P;p++){
            path_phi[i][p] = P.Ngrid/2;
            path_phi[i][p] = 0;
            path_phi[i][p] = distrib(gen); //randomly sample the distribution
        }
    }

    myfile.open("traj_A.dat");
    cout << "Start MC" << endl;
    //begin MC
    double V_average = 0;
    double E_average = 0;
    vector<double> prob_full(P.Ngrid);
    for (int n=0;n<P.MC_steps;n++){
        double V_total = 0;
        double E_total = 0;
        for (int i=0;i<P.N;i++){
            for (int j=0;j<P.P;j++){
                int p_minus = (j-1) % P.P;
                int p_plus = (j+1) % P.P;
                if (P.PIGS){
                    //kinetic action
                    if (j == 0){
                        for (int ip=0;ip<P.Ngrid;ip++){
                            prob_full[ip] = p_dist_end[ip][path_phi[i][p_plus]];
                        }
                    }
                    if (j == P.P-1){
                        for (int ip=0;ip<P.Ngrid;ip++){
                            prob_full[ip] = p_dist_end[path_phi[i][p_minus]][ip];
                        }
                    if (j!=0 && j != (P.P-1)){
                        for (int ip=0;ip<P.Ngrid;ip++){
                            prob_full[ip] = p_dist[path_phi[i][p_minus]][ip][path_phi[i][p_plus]];
                        }
                    }

                    }
                } else {
                    for (int ip=0;ip<P.Ngrid;ip++){
                        prob_full[ip] = p_dist[path_phi[i][p_minus]][ip][path_phi[i][p_plus]];
                    }
                }
                
                // NN interations and PBC (periodic boundary conditions)
                if (i < (P.N-1)){
                    for (int ir=0;ir<int(prob_full.size());ir++){
                        prob_full[ir] *= rhoVij(ir,path_phi[i+1][j]);
                    }
                }
                if (i>0){
                    for (int ir=0;ir<int(prob_full.size());ir++){
                        prob_full[ir] *= rhoVij(ir,path_phi[i-1][j]);
                    }
                }
                if (i == 0){
                    for (int ir=0;ir<int(prob_full.size());ir++){
                        prob_full[ir] *= rhoVij(ir,path_phi[P.N-1][j]);
                    }
                }
                if (i == (P.N-1)){
                    for (int ir=0;ir<int(prob_full.size());ir++){
                        prob_full[ir] *= rhoVij(ir,path_phi[0][j]);
                    }
                }
                // Normalize
                double norm_pro = 0;
                for (int ir=0;ir<int(prob_full.size());ir++){
                    norm_pro += prob_full[ir];
                }
                for (int ir=0;ir<int(prob_full.size());ir++){
                    prob_full[ir] /= norm_pro;
                }
                // pick an index using the alias method

                //This alias method potential has a bug in the implemenetation. 
                //This is resulting in much higher <V> and <E> with less variance then 
                aliasMethod alias(prob_full);
                int index = alias.draw();

                path_phi[i][j] = index;

                P.N_total += 1;

                histo_pimc[path_phi[i][j]] += 1;
            }
            
            if (n % P.Nskip == 0){
                myfile << path_phi[i][0]*P.delta_phi << endl;
                myfile << path_phi[i][P.P-1]*P.delta_phi << endl;
                myfile << path_phi[i][int((P.P-1)/2)]*P.delta_phi << endl;
                myfile << endl;
            }

            histo_L[path_phi[i][0]] += 1;
            histo_R[path_phi[i][P.P-1]] += 1;
            histo_middle[path_phi[i][int((P.P-1)/2)]] += 1;

            
            V_total += P.V0 * (1 + cos((path_phi[i][int((P.P-1)/2)])*P.delta_phi));
            E_total += P.V0 * (1 + cos((path_phi[i][0])*P.delta_phi));
            E_total += P.V0 * (1 + cos((path_phi[i][P.P-1])*P.delta_phi));
        }
        V_average += V_total;
        E_average += E_total;
    }

    myfile.close();
    cout << "<V> = " << V_average / P.MC_steps/P.N << endl;
    cout << "<E> = " << E_average / P.MC_steps/2/P.N << endl;
    myfile.open("histo_A_P");
    for (int i=0;i<P.Ngrid;i++){
        myfile << i*P.delta_phi << " " << histo_pimc[i]/(P.MC_steps*P.N*P.P)/P.delta_phi << " " << histo_middle[i]/(P.MC_steps*P.N)/P.delta_phi;
        myfile << " " << histo_L[i]/(P.MC_steps*P.N)/P.delta_phi;
        myfile << " " << histo_R[i]/(P.MC_steps*P.N)/P.delta_phi << endl;
    } 
    myfile.close();
}

