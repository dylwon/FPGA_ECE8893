AUTOPILOT_ROOT :=/tools/software/xilinx/latest/Vitis

IFLAG += -I "${AUTOPILOT_ROOT}/include"
IFLAG += -D__SIM_FPO__ -D__SIM_OPENCV__ -D__SIM_FFT__ 
IFLAG += -D__SIM_FIR__ -D__SIM_DDS__ -D__DSP48E1__ -DHLS_NO_XIL_FPO_LIB
IFLAG += -g 

CFLAG += -fPIC -O0
CFLAG += -lm
CFLAG += -std=c++11 -Wno-unused-result 

# Default make command
all: baseline optimized

# Make Baseline Design
baseline:
	g++ baseline.cpp host.cpp -o result $(CFLAG) $(IFLAG)

# Make Optimized Design
optimized:
	g++ top.cpp host.cpp -o result $(CFLAG) $(IFLAG)
    
clean:
	rm -f *.o result
