SMC_ROOT = /opt/intel/composer_xe_2013

CC = icpc
CFLAGS = -std=c99 -I$(SMC_ROOT)/include/cilk -I.
COPTFLAGS = -O3 -g -no_pie

LDFLAGS = -Wl,-rpath -Wl,$(SMC_ROOT)/lib -L$(SMC_ROOT)/lib -I $(SMC_ROOT)/include/cilk  -I/usr/local/include/eigen3/ -I/usr/local/mysql/include  -Os -g -fno-common -fno-strict-aliasing -arch x86_64 -L/usr/local/mysql/lib -lmysqlclient   -lpthread
CLEANFILES =

#=======================================================================

all: celnav$(EXEEXT)

#=======================================================================

celnav_HDRS = NNclass.h 
celnav_SRCS = main.cpp NNclass.cpp 
celnav_OBJS = $(celnav_SRCS:.c=.o)

celnav$(EXEEXT): $(celnav_OBJS)
	$(CC) $(COPTFLAGS) -o $@ $(celnav_OBJS) $(LDFLAGS)

CLEANFILES += celnav$(EXEEXT) $(celnav_OBJS)
CLEANFILES += rhist__sequential.out
CLEANFILES += rhist__parfor.out
CLEANFILES += rhist__recspawn.out
CLEANFILES += rhist__segscan.out

#=======================================================================

%.o: %.c
	$(CC) $(CFLAGS) $(COPTFLAGS) -o $@ -c $<

#=======================================================================

clean:
	rm -rf core *~ $(CLEANFILES)

# eof
