# use nvcc compiler
CC = nvcc

# conpiler flags:
# -g          add debugging information to the executable file
# -Wall	      turn on most, but not all, compiler warnings
# -arch=sm_35	set architechture	to computatbility 35
# -rdc=true   set rdc to true
CFLAGS = -g -Wall -arch=sm_35 -rdc=true

# libraries to link into executable
LIBS = -lcudadevrt -lcurand

# CUDA source files
SRCS = RandomTest.cu graphMatching.cu hypergraphMatching.cu nearestDSmax_RE.cu maxColSumP.cu exactTotalSum.cu matlib.cu 

# CUDA object files
#
# This uses Suffix Replacement within a macro:
#   $(name:string1=string2)
#         For each word in 'name' replace 'string1' with 'string2'
# Below we are replacing the suffix .c of all words in the macro SRCS
# with the .o suffix
OBJS = $(SRCS:.c=.o)

# the build target executable
TARGET = RandomTest

.PHONY: clean

# default target
default: $(TARGET)
	@echo RandomTest has been compiled

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LIBS)

# this is a suffix replacement rule for building .o's from .cu's
# it uses automatic variables $<: the name of the prerequisite of
# the rule(a .cu file) and $@: the name of the target of the rule (a .o file) 
# (see the gnu make manual section about automatic variables)
.cu.o:
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) *.o *~ $(TARGET)


