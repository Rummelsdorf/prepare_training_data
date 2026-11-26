/* Minimal benchmark kernel for prepare_training_data.
 *
 * Modes:
 *   bench_ops cpu <threads> <seconds>
 *   bench_ops ram <bytes>
 *
 * Both print a single JSON object to stdout. Errors go to stderr.
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

typedef struct {
    double seconds;
    uint64_t iterations;
    uint64_t acc;
    int flops_per_iter;
} worker_t;

static void *cpu_worker(void *arg) {
    worker_t *w = (worker_t *)arg;
    uint64_t acc = 0x9e3779b97f4a7c15ULL;
    uint64_t n = 0;
    const double start = now_seconds();
    for (;;) {
        /* One 64-bit integer addition per iteration. */
        acc += 1;
        n++;
        if ((n & 0x3ffffULL) == 0 && now_seconds() - start >= w->seconds) {
            break;
        }
    }
    w->iterations = n;
    w->acc = acc;
    return NULL;
}

static int simd_bits(void) {
#if defined(__AVX512F__)
    return 512;
#elif defined(__AVX2__)
    return 256;
#elif defined(__SSE2__)
    return 128;
#else
    return 0;
#endif
}

static int flops_per_iteration(void) {
#if defined(__AVX2__) && defined(__FMA__)
    return 16; /* 8 AVX2-lanes * 2 FLOP (fused multiply-add) */
#elif defined(__FMA__)
    return 2;  /* one scalar FMA */
#else
    return 2;  /* one scalar multiply + one add */
#endif
}

static void *flops_worker(void *arg) {
    worker_t *w = (worker_t *)arg;
    uint64_t n = 0;
    volatile double sink = 0.0;
#if defined(__AVX2__) && defined(__FMA__)
    static const float va_data[8] __attribute__((aligned(32))) = {
        1.000001f, 1.000002f, 1.000003f, 1.000004f,
        1.000005f, 1.000006f, 1.000007f, 1.000008f
    };
    static const float vb_data[8] __attribute__((aligned(32))) = {
        0.999991f, 0.999992f, 0.999993f, 0.999994f,
        0.999995f, 0.999996f, 0.999997f, 0.999998f
    };
    const __m256 va = _mm256_load_ps(va_data);
    const __m256 vb = _mm256_load_ps(vb_data);
    __m256 vacc = _mm256_setzero_ps();
    const double start = now_seconds();
    for (;;) {
        vacc = _mm256_fmadd_ps(va, vb, vacc);
        n++;
        if ((n & 0x3ffffULL) == 0 && now_seconds() - start >= w->seconds) {
            break;
        }
    }
    float out[8] __attribute__((aligned(32)));
    _mm256_store_ps(out, vacc);
    for (int i = 0; i < 8; i++) {
        sink += out[i];
    }
#else
    const float a = 1.000001f;
    const float b = 0.999999f;
    float acc = 0.0f;
    const double start = now_seconds();
    for (;;) {
        acc = fmaf(a, b, acc);
        n++;
        if ((n & 0x3ffffULL) == 0 && now_seconds() - start >= w->seconds) {
            break;
        }
    }
    sink = acc;
#endif
    (void)sink;
    w->iterations = n;
    w->acc = (uint64_t)n;
    w->flops_per_iter = flops_per_iteration();
    return NULL;
}

static int print_header_fields(int threads, double elapsed, uint64_t ops) {
    printf("{\"threads\":%d,\"seconds\":%.6f,\"ops\":%llu,"
           "\"ops_per_sec\":%.3f,\"operand_bits\":%d,"
           "\"word_bits\":%zu,\"long_bits\":%zu,\"uint64_bits\":%d,"
           "\"simd_bits\":%d,\"compiler\":\"%s\"",
           threads, elapsed, (unsigned long long)ops,
           elapsed > 0 ? (double)ops / elapsed : 0.0,
           64, sizeof(void *) * 8, sizeof(long) * 8, 64,
           simd_bits(), __VERSION__);
    return 0;
}

static int run_cpu(int threads, double seconds) {
    if (threads < 1 || seconds <= 0) {
        fprintf(stderr, "cpu: threads>=1 and seconds>0 required\n");
        return 2;
    }
    worker_t *workers = (worker_t *)calloc((size_t)threads, sizeof(worker_t));
    pthread_t *ids = (pthread_t *)calloc((size_t)threads, sizeof(pthread_t));
    if (!workers || !ids) {
        fprintf(stderr, "calloc failed\n");
        return 2;
    }
    for (int i = 0; i < threads; i++) {
        workers[i].seconds = seconds;
        if (pthread_create(&ids[i], NULL, cpu_worker, &workers[i]) != 0) {
            fprintf(stderr, "pthread_create failed\n");
            return 2;
        }
    }
    const double start = now_seconds();
    for (int i = 0; i < threads; i++) {
        pthread_join(ids[i], NULL);
    }
    const double elapsed = now_seconds() - start;

    uint64_t total = 0;
    volatile uint64_t sink = 0;
    for (int i = 0; i < threads; i++) {
        total += workers[i].iterations;
        sink += workers[i].acc;
    }
    (void)sink;
    print_header_fields(threads, elapsed, total);
    printf("}\n");
    free(workers);
    free(ids);
    return 0;
}

static int run_flops(int threads, double seconds) {
    if (threads < 1 || seconds <= 0) {
        fprintf(stderr, "flops: threads>=1 and seconds>0 required\n");
        return 2;
    }
    worker_t *workers = (worker_t *)calloc((size_t)threads, sizeof(worker_t));
    pthread_t *ids = (pthread_t *)calloc((size_t)threads, sizeof(pthread_t));
    if (!workers || !ids) {
        fprintf(stderr, "calloc failed\n");
        return 2;
    }
    for (int i = 0; i < threads; i++) {
        workers[i].seconds = seconds;
        if (pthread_create(&ids[i], NULL, flops_worker, &workers[i]) != 0) {
            fprintf(stderr, "pthread_create failed\n");
            return 2;
        }
    }
    const double start = now_seconds();
    for (int i = 0; i < threads; i++) {
        pthread_join(ids[i], NULL);
    }
    const double elapsed = now_seconds() - start;

    uint64_t ops = 0;
    int per_iter = 0;
    for (int i = 0; i < threads; i++) {
        ops += workers[i].iterations * (uint64_t)workers[i].flops_per_iter;
        per_iter = workers[i].flops_per_iter;
    }
    printf("{\"threads\":%d,\"seconds\":%.6f,\"flops\":%llu,"
           "\"flops_per_sec\":%.3f,\"flops_per_iter\":%d,"
           "\"simd_bits\":%d,\"word_bits\":%zu}\n",
           threads, elapsed, (unsigned long long)ops,
           elapsed > 0 ? (double)ops / elapsed : 0.0,
           per_iter, simd_bits(), sizeof(void *) * 8);
    free(workers);
    free(ids);
    return 0;
}

static int run_ram(size_t bytes) {
    unsigned char *buf = (unsigned char *)malloc(bytes);
    if (!buf) {
        fprintf(stderr, "malloc(%zu) failed\n", bytes);
        return 2;
    }
    /* Force physical pages to be resident before timing. */
    memset(buf, 0xa5, bytes);

    double t0 = now_seconds();
    memset(buf, 0x5a, bytes);
    double t1 = now_seconds();

    uint64_t sum = 0;
    double t2 = now_seconds();
    for (size_t i = 0; i < bytes; i++) {
        sum += buf[i];
    }
    double t3 = now_seconds();
    volatile uint64_t sink = sum;
    (void)sink;

    const double write_s = t1 - t0;
    const double read_s = t3 - t2;
    printf("{\"bytes\":%zu,\"write_seconds\":%.6f,\"read_seconds\":%.6f,"
           "\"write_bytes_per_sec\":%.3f,\"read_bytes_per_sec\":%.3f,"
           "\"word_bits\":%zu,\"long_bits\":%zu,\"simd_bits\":%d}\n",
           bytes, write_s, read_s,
           write_s > 0 ? (double)bytes / write_s : 0.0,
           read_s > 0 ? (double)bytes / read_s : 0.0,
           sizeof(void *) * 8, sizeof(long) * 8, simd_bits());
    free(buf);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: bench_ops cpu|flops <threads> <seconds> | ram <bytes>\n");
        return 2;
    }
    if (strcmp(argv[1], "cpu") == 0) {
        int threads = (argc >= 3) ? atoi(argv[2]) : (int)sysconf(_SC_NPROCESSORS_ONLN);
        double seconds = (argc >= 4) ? atof(argv[3]) : 2.0;
        return run_cpu(threads, seconds);
    }
    if (strcmp(argv[1], "flops") == 0) {
        int threads = (argc >= 3) ? atoi(argv[2]) : (int)sysconf(_SC_NPROCESSORS_ONLN);
        double seconds = (argc >= 4) ? atof(argv[3]) : 2.0;
        return run_flops(threads, seconds);
    }
    if (strcmp(argv[1], "ram") == 0) {
        if (argc < 3) {
            fprintf(stderr, "ram: byte count required\n");
            return 2;
        }
        return run_ram(strtoull(argv[2], NULL, 10));
    }
    fprintf(stderr, "unknown mode: %s\n", argv[1]);
    return 2;
}
