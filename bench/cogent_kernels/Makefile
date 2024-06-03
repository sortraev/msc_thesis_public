NVCC=nvcc --std=c++17 -O3 -Werror all-warnings -diag-suppress=177

MAIN=main
SRC=$(MAIN).cu util.h \
	cogent_icaq_qbjk_abcijk_32x32x32x32x32x32x32.cu \
	cogent_kiaq_bcjq_abcijk_32x32x32x32x32x32x32.cu \
    cogent_icaq_qbjk_abcijk_16x16x16x16x16x16x2048.cu \
    cogent_kiaq_bcjq_abcijk_16x16x16x16x16x16x2048.cu

default: compile

compile: $(MAIN)

$(MAIN): $(SRC)
	$(NVCC) -o $@ $<

run: $(MAIN)
	./$< 16 16 16 16 16 16 2048 icaq_qbjk_abcijk_16x16x16x16x16x16x2048
	@# ./$< 16 16 16 16 16 16 2048 kiaq_bcjq_abcijk_16x16x16x16x16x16x2048

clean:
	rm -f $(MAIN) vgcore*
