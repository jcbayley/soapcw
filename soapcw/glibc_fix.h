
#if defined(__linux__) && defined(__x86_64__)
__asm__(".symver memcpy,memcpy@GLIBC_2.2.5");
#endif
