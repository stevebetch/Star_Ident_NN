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
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <string>
#include <fstream>
#include <Eigen/Core>
#include <iostream>

#include <Eigen/Core>
#include <mysql.h>
//#include "timer_cs4225.h"
#include <ctime>
#include <cilk.h>
#include <common.h>
#include <mysql.h>
//#include <OpenCL/opencl.h>
#include <reducer_opadd.h>




using namespace std;
typedef Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;
typedef Eigen::Matrix<long double, Eigen::Dynamic, 1> VectorXd;

struct selfNN{
    VectorXd nodeHidden;
    VectorXd NodeIn;
    VectorXd nodeOut;
    VectorXd errors;
    VectorXd dist;
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
/* =================================================== */
/*
 * Timing functions
 */
#if !defined(HAVE_TIMER)
#  define TIMER_DESC "gettimeofday"

#define USE_STD_CREATE
#define USE_STD_DESTROY

#include <sys/time.h>

struct stopwatch_t * stopwatch_create (void);
void stopwatch_destroy (struct stopwatch_t* T);
void stopwatch_init (void);

void stopwatch_start (struct stopwatch_t* T);

long double stopwatch_stop (struct stopwatch_t* T);

#endif


#endif /* defined(__Cpp_RStarz_NN__NNclass__) */
