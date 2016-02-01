//
//  NNclass.cpp
//  Cpp RStarz NN
//
//  Created by Stephan Boettcher on 11/29/12.
//  Copyright (c) 2012 Stephan Boettcher. All rights reserved.
//


#include "NNclass.h"


using namespace std;





NeuNet::NeuNet()
{
    selfNN NN;
}

selfNN NeuNet::getNN(int nodeIn, int NodeOut, int NodeHidden, long double initWmin, long double initWmax, long double eta, long double mo)
{
    
    NN.nodeHidden.resize(NodeHidden);
    NN.NodeIn.resize(nodeIn);
    NN.nodeOut.resize(NodeOut);
    NN.eta=eta;
    NN.mo=mo;
    NN.WeightIN.resize(nodeIn,NodeHidden);
    NN.WeightOut.resize(NodeHidden,NodeOut);
    NN.ActHid.resize(NodeHidden);
    NN.ActIn.resize(nodeIn);
    NN.ActOut.resize(NodeOut);
    NN.momentIn.resize(nodeIn,NodeHidden);
    NN.momentOut.resize(NodeHidden,NodeOut);
    
    
    cilk_for (int i=0; i<NN.NodeIn.size(); i++) {
        cilk_for (int j=0; j<NN.nodeHidden.size(); j++) {
            
            NN.WeightIN(i,j)=rand(initWmin,initWmax);
            NN.momentIn(i,j)=rand(initWmin,initWmax);
            //   printf("%f ",(NN.WeightIN(i,j)));
        }
        NN.ActIn(i)=1;
        NN.NodeIn(i)=1;
        
        //     cout<<"\n";
    }
    
    cout<<"\n\n";
    cilk_for (int p=0; p<NN.nodeHidden.size(); p++) {
        cilk_for (int q=0; q<NN.nodeOut.size(); q++) {
            NN.WeightOut(p,q)=rand(initWmin,initWmax);
            NN.momentOut(p,q)=rand(initWmin,initWmax);
            //       printf("%f ",(NN.WeightOut(p,q)));
        }
        NN.ActHid(p)=1;
        NN.nodeHidden(p)=1;
        //  cout<<"\n";
    }
    cilk_for (int i=0; i<NN.nodeOut.size(); i++) {
        NN.ActOut(i)=1;
        NN.nodeOut(i)=1;
    }
    
    
    
    return NN;
    
    
    
}

VectorXd NeuNet::Upd(selfNN &NN, VectorXd &Upvals)
{
    
    if(Upvals.size()!=NN.NodeIn.size())
    { throw 2;}
    
    
    
    cilk_for (int i=0; i<NN.NodeIn.size(); i++ ) {
        NN.ActIn(i)=Upvals(i);
        //cout<<NN.ActIn(i)<<endl;
    }
    
    
    //    cout<<"Hidden nodes"<<endl;
    cilk_for (int i=0; i<NN.nodeHidden.size();i++){
        //long double meh=0;
        long double moo=0;
        for (int j=0; j<NN.NodeIn.size(); j++) {
            moo=moo+NN.ActIn(j)*NN.WeightIN(j,i);
        }
        NN.ActHid(i)=sig(moo);
        //     cout<<NN.ActHid(i)<<endl;
    }
    // cout<<"out\n";
    cilk_for (int i=0; i<NN.nodeOut.size();i++){
        long double moo=0;
        for (int j=0; j<NN.nodeHidden.size();j++){
            moo=moo+NN.ActHid(j)*NN.WeightOut(j,i);
        }
        //  cout<<"moo: "<<moo<<endl;
        NN.ActOut(i)=sig(moo);
        //    cout<<NN.ActOut(i)<<endl;
    }
    return NN.ActOut;
    
    /////////////////////////////////////////////////////////////////////////
    
    
};

long double NeuNet::BP(selfNN &NN, double TargetVals){
    
    VectorXd deltaOut(NN.nodeOut.size());
    long double moo=0;
    long double AcO=0;
    
    for (int i=0; i<NN.nodeOut.size(); i++) {
        AcO=NN.ActOut(i);
        deltaOut(i)=(dsig(AcO))*(TargetVals-AcO);
        
        moo=moo+.5*pow((TargetVals-AcO),2);
        //     cout<<deltaOut(i)<<endl;
    }
    // cout<<"\nsum";
    VectorXd deltahid(NN.nodeHidden.size());
    
    
    cilk_for (int i=0; i<NN.nodeHidden.size(); i++) {
        
        long double sum=0;
        for(int j=0; j<NN.nodeOut.size();j++){
            sum=sum+deltaOut(j)*NN.WeightOut(i,j);
            
        }
        deltahid(i)=NN.ActHid(i)*(1-NN.ActHid(i))*sum;
        
    }
    
    
    cilk_for (int i=0; i<NN.nodeHidden.size(); i++) {
        
        cilk_for(int j=0; j<NN.nodeOut.size();j++){
            
            long double moo1= NN.eta*deltaOut(j)*NN.ActHid(i);
            
            long  moo2=NN.mo*NN.momentOut(i,j);
            
            
            NN.WeightOut(i,j)=NN.WeightOut(i,j)+moo1+moo2;
            NN.momentOut(i,j)=deltaOut(j)*NN.ActHid(i);
            
        }
    }
    
    
    cilk_for (int i=0; i<NN.NodeIn.size(); i++) {
        cilk_for(int j=0; j<NN.nodeHidden.size();j++){
            
            long double moo1=NN.eta*deltahid(j)*NN.ActIn(i);
            
            NN.WeightIN(i,j)=NN.WeightIN(i,j)+moo1+NN.mo*NN.momentIn(i,j);
            NN.momentIn(i,j)=deltahid(j)*NN.ActIn(i);
        }
    }
    
    
    
    return moo;
};

long double NeuNet::NNout(selfNN &NN,string fname)
{
    int greatsucess=0;
    ofstream moo;
    string fname11("_weightin");
    string fname1;
    fname1=fname+fname11;
    moo.open(fname1.c_str()); //weight in matrix.
    
    if (moo.is_open())
    {
        for(int i=0; i<NN.NodeIn.size();i++){
            for(int j=0; j<NN.nodeHidden.size();j++){
                moo<<NN.WeightIN(i,j)<<" ";
            }
            
        }
        moo.close();
        
    }
    
    string fname12("weightout");
    fname1=fname+fname12;
    moo.open(fname1.c_str()); //weight out matrix.
    
    if (moo.is_open())
    {
        for(int i=0; i<NN.nodeHidden.size();i++){
            for(int j=0; j<NN.nodeOut.size();j++){
                moo<<NN.WeightOut(i,j)<<" ";
            }
            
        }
        moo.close();
        
    }
    string fname13("errors");
    fname1=fname+fname13;
    moo.open(fname1.c_str()); //weight out matrix.
    if (moo.is_open())
    {
        for (int i=0; i<NN.errors.size(); i++) {
            moo<<NN.errors(i)<<" ";
        }
        greatsucess=1;
    }
    
    return greatsucess;
};

int NeuNet::loadNN(selfNN &NN, string fname)
{
    int greatsucess=0;
    ifstream moo;
    string fname11("weightin");
    string fname1;
    fname1=fname+fname11;
    moo.open(fname1.c_str()); //weight in matrix.
    string vals;
    
    
    
    if (moo.is_open())
    {
        
        for(int i=0; i<NN.NodeIn.size();i++){
            for(int j=0; j<NN.nodeHidden.size();j++){
                getline( moo,vals,' ' );
                NN.WeightIN(i,j)=atof(vals.c_str());
            }
            
        }
        moo.close();
        
    }
    
    string fname12("weightout");
    fname1=fname+fname12;
    moo.open(fname1.c_str()); //weight out matrix.
    
    if (moo.is_open())
    {
        for(int i=0; i<NN.nodeHidden.size();i++){
            for(int j=0; j<NN.nodeOut.size();j++){
                getline( moo,vals,' ' );
                
                NN.WeightOut(i,j)=atof(vals.c_str());
            }
            
        }
        moo.close();
        
    }
    string fname13("errors");
    fname1=fname+fname13;
    moo.open(fname1.c_str()); //weight out matrix.
    if (moo.is_open())
    {
        for (int i=0; i<NN.errors.size(); i++) {
            getline( moo,vals,' ' );
            NN.errors(i)=atof(vals.c_str());
        }
        greatsucess=1;
    }
    
    return greatsucess;
};

///////////////////////////////////////////////////////////////////////

long double rand( long double a, long double b)
{
    
    long double debugger=(rand()*fabs(b-a))/RAND_MAX +a;
    return debugger;
};

long double sig(long double x)
{
    if(x<(-600)){
        return 0;}
    
    return (1.0/(1.0+exp(-1*x)));
};

long double dsig(long double x){
    return (sig(x)*(1.0-sig(x)));
};


///////////////////////////////////////////////////////////////////////





long double training(NeuNet &starz, long double dec1, long double dec2, long double ra1, long double ra2, long double limit, selfNN &NN ){
    
    //   MYSQL mysql;
    // Convert digits to strings
    ostringstream convert1;
    stringstream convert2;
    stringstream convert3;
    stringstream convert4;
    convert1<<dec1;
    string sdec1=convert1.str();
    convert2<<dec2;
    string sdec2=convert2.str();
    convert3<<ra1;
    string sra1=convert3.str();
    convert4<<ra2;
    string sra2=convert4.str();
    
    // begin MYsql connections
    MYSQL *connect1;
    MYSQL *connect2;
    connect1=mysql_init(NULL);
    connect2=mysql_init(NULL);
    
    
    if(!connect1)    /* If instance didn't initialize say so and exit with fault.*/
    {
        fprintf(stderr,"MySQL Initialization Failed");
        return 1;
    }
    connect1=mysql_real_connect(connect1, "localhost", "NeuNet", "celnav", "RNNStarz", 0, NULL, 0);
    connect2=mysql_real_connect(connect2, "localhost", "NeuNet", "celnav", "RNNStarz", 0, NULL, 0);
    
    if(connect1){
        printf("Connection1 Succeeded\n");
    }
    else{
        printf("Connection1 Failed!\n");
    }
    
    
    if(connect2){
        printf("Connection2 Succeeded\n");
    }
    else{
        printf("Connection2 Failed!\n");
    }
    
    
    MYSQL_RES *res_set1;
    MYSQL_RES *res_set2;
    MYSQL_ROW row;
    MYSQL_ROW row2;
    // mysql_query(connect1,"SET SESSION wait_timeout = 1728000"); //dont let the session die.....
    // mysql_query(connect2,"SET SESSION wait_timeout = 1728000");
    
    
    // set up mysql tables
    string sqlp1("SELECT * FROM stardata WHERE (decl BETWEEN ");
    string sqlp2(" AND " );
    string sqlp3(") AND ( ra BETWEEN ");
    string sqlp4(")");
    string sql=sqlp1+sdec1+sqlp2+sdec2+sqlp3+sra1+sqlp2+sra2+sqlp4;
    //string sql=sqlp1+"60"+sqlp2+"65"+sqlp3+"0"+sqlp2+"5"+sqlp4;
    
    mysql_query(connect1,sql.c_str());
    //  unsigned int i = 0;
    res_set1 = mysql_store_result(connect1);
    // unsigned int numrows=mysql_num_rows(res_set1);
    
    
    //Init variables for training
    int moo=0;
    long double err;
    int tt;
    long double correct;
    long double ender;
    //  long double testnum;
    //int starid2;
    int starid1;
    
    
    MatrixXd traningFiles;// note to self: NN.WeightIN.resize(nodeIn,NodeHidden);
    VectorXd starIDs;
    VectorXd res(NN.nodeOut.size());
    VectorXd flag;
    
    long double result2;
    VectorXd NNID(1);
    long double er;
    
    long double retval; //
    
    
    retval = mysql_num_rows(res_set1);// number of training items
    traningFiles.resize(retval, NN.NodeIn.size());
    starIDs.resize(retval,1); //keep track of star ids
    
    flag.resize(retval, 1);
    int intters=0;
    
    
    
    // timer
    struct stopwatch_t* t2 = NULL;
    long double  t_sql;
    // initialize timer
    stopwatch_init ();
    t2 = stopwatch_create ();
    stopwatch_start (t2); //start timer
    
    
    
    /////////////// load up the training matrix. should save time...
    cilk_for(int moo=0; moo<(int)retval; moo++)
    {
        
        
        
        row = mysql_fetch_row(res_set1);
        if(row==NULL)
        {}
        else
        {
            starid1=atoi(row[0]); //atof(vals.c_str())
            starIDs(moo)=starid1;
            string sqlp1("SELECT * FROM starrels2 WHERE starID1 = ");
            string sqlp2(" and starID2 !=");
            string sqlp3(" ORDER BY distance");
            string sql1= sqlp1+row[0]+sqlp2+row[0]+sqlp3;
            
            MYSQL *connectn;
            connectn=mysql_init(NULL);
            connectn=mysql_real_connect(connectn, "localhost", "NeuNet", "celnav", "RNNStarz", 0, NULL, 0);
            MYSQL_RES *res_setn;
            MYSQL_ROW rown;
            mysql_query(connectn,sql1.c_str());
            res_setn = mysql_store_result(connectn);
            flag(moo)=1;
            
            
            for (int i=0; i<NN.NodeIn.size(); i++){
                
                rown = mysql_fetch_row(res_setn);
                //char *val=row2[0];
                long unsigned int numrows=mysql_num_rows(res_setn);
                if(numrows!=0){
                    
                    string s = rown[2];
                    long double meh= atof(s.c_str());
                    traningFiles(moo,i)=meh;
                    
                    //cout<<"distance:"<<dist(i)<<endl;
                }
                else
                {
                    flag(intters)=0;
                }
            }
            
            
            mysql_close(connectn);
        }
        
    }
    
    
    /*
     while ((row = mysql_fetch_row(res_set1)) != NULL){
     starid1=atoi(row[0]); //atof(vals.c_str())
     starIDs(intters)=starid1;
     
     string sqlp1("SELECT * FROM starrels2 WHERE starID1 = ");
     string sqlp2(" and starID2 !=");
     string sqlp3(" ORDER BY distance");
     string sql1= sqlp1+row[0]+sqlp2+row[0]+sqlp3;
     
     
     mysql_query(connect2,sql1.c_str());
     res_set2 = mysql_store_result(connect2);
     flag(intters)=1;
     
     
     cilk_for (int i=0; i<NN.NodeIn.size(); i++){
     
     row2 = mysql_fetch_row(res_set2);
     //char *val=row2[0];
     long unsigned int numrows=mysql_num_rows(res_set2);
     if(numrows!=0){
     
     string s = row2[2];
     long double meh= atof(s.c_str());
     traningFiles(intters,i)=meh;
     
     //cout<<"distance:"<<dist(i)<<endl;
     }
     else
     {
     flag(intters)=0;
     }
     }
     intters++;
     mysql_free_result(res_set2);
     }*/
    
    t_sql = stopwatch_stop (t2);
    fprintf (stderr, "Time to execute mysql load: %Lg secs\n",t_sql);
    
    
    
    ////////////////////////////////////////////////////////////////////////////////////////////
    moo=0; // double check.
    VectorXd dist(NN.NodeIn.size());
    
    while(moo<limit)
    {
        moo+=1;
        // printf("\nitteration:%d \n",moo);
        err=1.0;
        tt=0;
        correct=0;
        ender=0;
        
        // testnum=rand(1,NN.NodeIn.size());
        int rownum=0;
        while (rownum <retval){
            
            
            
            //      mysql_free_result(res_set2);
            
            
            // cout<<"sum:"<<sum<<endl;
            if(flag(rownum)!=0){
                for(int poo=0;poo<NN.NodeIn.size();poo++){
                    dist(poo)=traningFiles(rownum,poo);
                    // double meh=dist(poo);
                    // printf("dist:%f\n",meh);
                }
                dist= traningFiles.row(rownum);
                res= starz.Upd(NN, dist);
                double res2=res(0);
                //printf("res:%f\n",res(0));
                result2=8077+res(0)*8075;
                
                //  cout<<"Iteration:"<<moo<<" Post-filter:"<<result2<<" || Actual:"<<starIDs(rownum)<<" || difference: "<<fabs(starIDs(rownum)-result2)<<endl;
                NNID(0)=(starIDs(rownum)-8077.0)/8075.0;
                //  long double debu= NNID(0);
                ender+=fabs(starIDs(rownum)-result2);
                er=starz.BP(NN,NNID(0));
                tt+=1;
                if((result2>=starIDs(rownum)-1)&&(result2<=starIDs(rownum)+1)){
                    correct+=1;
                }
            }
            
            
            
            rownum++;
        }
        // mysql_free_result(res_set1);
        // mysql_query(connect1,sql.c_str());
        //res_set1 = mysql_store_result(connect1);
        long int errsize=starz.NN.errors.size();
        long int distsize=starz.NN.dist.size();
        starz.NN.errors.resize(errsize+1);
        starz.NN.dist.resize(distsize+1);
        starz.NN.errors(errsize)=(1-(correct/tt));
        starz.NN.dist(distsize)=ender/tt;
        if((ender/tt)<.3 && (correct/tt)==1){
            return err;
        }
        
        if(moo%10==0){
          //  printf("itteration:%d Stars: %d Average distance: %Lf Percent correct %Lf\n",moo, tt,(ender/tt),(correct/tt));
        }
    }
  //  printf("Stars: %d Average distance: %Lf\n", tt, starz.NN.dist(starz.NN.dist.size()-2));
    mysql_close(connect1);
    mysql_close(connect2);
    return err;
};


