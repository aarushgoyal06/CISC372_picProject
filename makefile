# Linux/GCC: -fopenmp. Darwin + Apple Clang: OpenMP via Homebrew libomp.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  LIBOMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
  ifneq ($(LIBOMP_PREFIX),)
    OMPFLAGS = -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include
    OMPLIBS = -L$(LIBOMP_PREFIX)/lib -lomp
  else
    OMPFLAGS = -fopenmp
    OMPLIBS = -fopenmp
  endif
else
  OMPFLAGS = -fopenmp
  OMPLIBS = -fopenmp
endif

imageOpen:imageOpen.c image.h
	gcc -g $(OMPFLAGS) imageOpen.c -o imageOpen -lm $(OMPLIBS)
imageThread:imageThread.c image.h
	gcc -g imageThread.c -o imageThread -lm -pthread
clean:
	rm -f imageOpen imageThread output.png
	rm -rf imageOpen.dSYM imageThread.dSYM
