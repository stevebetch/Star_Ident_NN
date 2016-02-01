//
//  main.c
//  Cpp RStarz NN
//
//  Created by Stephan Boettcher on 11/29/12.
//  Copyright (c) 2012 Stephan Boettcher. All rights reserved.
//
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <stdio.h>
#include <mysql.h>
#include <string.h>
#include <vector>
#include "NNclass.h"
#include "timer_cs4225.h"

using namespace std;


int main(int argc, const char * argv[])
{
srand((unsigned)time(NULL));

// timer 
    struct stopwatch_t* timer = NULL;
    long double  t_cpu;
   
//int nodeIn, int NodeOut, int NodeHidden, double initWmin, double initWmax, double eta, double mo)
    int dec1=-90;
    int dec2=90;
    int ra1=0;
    int ra2=360;
//Start up the neural network classes.
    int nodein=200;
    int   nodeout=1;
    int nodehidden=200;
  double  min=-.5;
  double  max=.5;
    double execlimit=50; // the number of iterations to train over
    double eta=.5;
    double mo=.5;

// initalize neural network
    double err;
    NeuNet starz;
    
    
// initialize timer
    stopwatch_init ();
    timer = stopwatch_create ();
    
//setup neural net
    selfNN meh= starz.getNN(nodein,nodeout,nodehidden,min,max,eta,mo);
    
//begin training
    stopwatch_start (timer); //start timer
    err=training(starz,dec1,dec2,ra1,ra2,execlimit,meh);
    
//record time
    t_cpu = stopwatch_stop (timer);
    fprintf (stderr, "Time to execute sequential training: %Lg secs\n",t_cpu);
 
    
// save to ouput
    double foo=starz.NNout(meh, "try2");
    if (foo==1){
        cout<<"good out";
    }
    return 0;
}

