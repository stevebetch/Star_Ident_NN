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
//#include "timer_cs4225.h"

using namespace std;

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




struct stopwatch_t
{
    struct timeval t_start_;
    struct timeval t_stop_;
    int is_running_;
};

static
long double
elapsed (struct timeval start, struct timeval stop)
{
    return (long double)(stop.tv_sec - start.tv_sec)
    + (long double)(stop.tv_usec - start.tv_usec)*1e-6;
}

long double
stopwatch_elapsed (struct stopwatch_t* T)
{
    long double dt = 0;
    if (T) {
        if (T->is_running_) {
            struct timeval stop;
            gettimeofday (&stop, 0);
            dt = elapsed (T->t_start_, stop);
        } else {
            dt = elapsed (T->t_start_, T->t_stop_);
        }
    }
    return dt;
}

void
stopwatch_init (void)
{
    fprintf (stderr, "Timer: %s\n", TIMER_DESC);
    fprintf (stderr, "Timer resolution: ~ 1 us (?)\n");
    fflush (stderr);
}

void
stopwatch_start (struct stopwatch_t* T)
{
    assert (T);
    T->is_running_ = 1;
    gettimeofday (&(T->t_start_), 0);
}

long double
stopwatch_stop (struct stopwatch_t* T)
{
    long double dt = 0;
    if (T) {
        if (T->is_running_) {
            gettimeofday (&(T->t_stop_), 0);
            T->is_running_ = 0;
        }
        dt = stopwatch_elapsed (T);
    }
    return dt;
}

#  define HAVE_TIMER 1
#endif

#if defined(USE_STD_CREATE)
struct stopwatch_t *
stopwatch_create (void)
{
    struct stopwatch_t* new_timer =
    (struct stopwatch_t *)malloc (sizeof (struct stopwatch_t));
    if (new_timer)
        memset (new_timer, 0, sizeof (struct stopwatch_t));
    return new_timer;
}
#endif

#if defined(USE_STD_DESTROY)
void
stopwatch_destroy (struct stopwatch_t* T)
{
    if (T) {
        stopwatch_stop (T);
        free (T);
    }
}
#endif
/* =================================================== */






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
    int nodein=250;
    int   nodeout=1;
    int nodehidden=nodein;
    double  min=-.5;
    double  max=.5;
    double execlimit=1; // the number of iterations to train over
    double eta=.5 ;
    double mo=.25;
    
    
    /*
     int dec1=60;
     int dec2=65;
     int ra1=10;
     int ra2=36;
     //Start up the neural network classes.
     int nodein=20;
     int   nodeout=1;
     int nodehidden=10;
     double  min=-.5;
     double  max=.5;
     double execlimit=10000; // the number of iterations to train over
     double eta=.5 ;
     double mo=.5;
     */
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
    fprintf (stderr, "Time to execute parallel training: %Lg secs\n",t_cpu);
    
    
    // save to ouput
    double foo=starz.NNout(meh, "try2");
    if (foo==1){
        cout<<"good out";
    }
    return 0;
}





