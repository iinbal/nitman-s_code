CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors -DBUILD_STANDALONE
LFLAGS = -lm
my_app: symnmf.o symnmf.h
	$(CC) -o symnmf symnmf.o $(LFLAGS)

symnmf.o: symnmf.c symnmf.h
	$(CC) -c symnmf.c $(CFLAGS)
		
clean:
	rm -f *.o symnmf *.so