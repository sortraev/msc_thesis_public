NVCC_FLAGS=
NVCC_FLAGS+=-Werror all-warnings
NVCC_FLAGS+=-diag-suppress=177
NVCC_FLAGS+=--std=c++17
NVCC_FLAGS+=-arch=native
NVCC_FLAGS+=-O3

MAIN=main
SRC=main.cu kernels.cu util.h

# defaults
sd1_7_tiles=-DTA=16 -DTB=2 -DTC=1 -DTI=2 -DTJ=2 -DTK=8 -DQQ=32 -DRA=1 -DRB=2 -DRC=2 -DRI=2 -DRJ=2 -DRK=1
sd2_3_tiles=-DTA=1 -DTB=1 -DTC=8 -DTI=1 -DTJ=1 -DTK=32 -DQQ=4 -DRA=2 -DRB=1 -DRC=4 -DRI=1 -DRJ=2 -DRK=1 

USE_EPILOGUE=-DUSE_EPILOGUE=1
COMPILE_KERNEL=-DCOMPILE_KERNEL=0
TILE_DEFINES=$(sd1_7_tiles)


$(MAIN): $(SRC)
	nvcc $(NVCC_FLAGS) $(COMPILE_KERNEL) $(USE_EPILOGUE) $(TILE_DEFINES) -o $@ $<

compile: sd1_7_EPILOGUE sd1_7_NO_EPILOGUE sd2_3_EPILOGUE sd2_3_NO_EPILOGUE


sd1_7_EPILOGUE: $(SRC)
	nvcc $(NVCC_FLAGS) -DCOMPILE_KERNEL=0 -DUSE_EPILOGUE=1 $(sd1_7_tiles) -o $@ $<
sd1_7_NO_EPILOGUE: $(SRC)
	nvcc $(NVCC_FLAGS) -DCOMPILE_KERNEL=0 -DUSE_EPILOGUE=0 $(sd1_7_tiles) -o $@ $<

sd2_3_EPILOGUE: $(SRC)
	nvcc $(NVCC_FLAGS) -DCOMPILE_KERNEL=1 -DUSE_EPILOGUE=1 $(sd2_3_tiles) -o $@ $<
sd2_3_NO_EPILOGUE: $(SRC)                                      
	nvcc $(NVCC_FLAGS) -DCOMPILE_KERNEL=1 -DUSE_EPILOGUE=0 $(sd2_3_tiles) -o $@ $<


clean:
	rm -f $(MAIN) vgcore*
