//
//  NNclass.h
//  Cpp RStarz NN
//
//  Created by Stephan Boettcher on 11/29/12.
//  Copyright (c) 2012 Stephan Boettcher. All rights reserved.
//

#ifndef __Cpp_RStarz_NN__NNclass__
#define __Cpp_RStarz_NN__NNclass__

#include <iostream>
#include <string.h>
#include <Eigen/Core>
#include <mysql.h>

using namespace std;
typedef Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;
typedef Eigen::Matrix<long double, Eigen::Dynamic, 1> VectorXd;

struct selfNN{
    VectorXd nodeHidden;
    VectorXd NodeIn;
    VectorXd nodeOut;
    VectorXd errors;
    VectorXd trainer;
    MatrixXd WeightIN;
    MatrixXd WeightOut;
    VectorXd ActHid;
    VectorXd ActOut;
    VectorXd ActIn;
    MatrixXd momentIn;
    MatrixXd momentOut;
    long double eta;
    long double mo;    
};

class NeuNet {
public:
    selfNN NN;
    NeuNet();
    //default constuctor. creates a neural net with random values
    
    VectorXd Upd(selfNN &NN, VectorXd &Upvals);
    // Modifies: none
    // effect: propigates the data thru the neural net
    
    long double BP(selfNN &NN, double TargetVals);
    //back propigates the error thru the NN.
    
    long double NNout(selfNN &NN,string fname);
    
    int loadNN(selfNN &NN, string Inname);
    
    selfNN getNN(int nodeIn, int NodeOut, int NodeHidden, long double initWmin, long double initWmax, long double eta, long double mo) ;

};

long double training(NeuNet &NN, long double dec1, long double dec2, long double ra1, long double ra2, long double limit,selfNN &meh);

long double tester(selfNN &NN, int star);

long double rand( long double a, long double b);

long double sig(long double x);

long double dsig(long double x);



#endif /* defined(__Cpp_RStarz_NN__NNclass__) */
