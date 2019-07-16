#include <stdlib.h>
#define LEN 300
#define TOTAL_LEN 300*300*300
static double sq(double x) { return x*x; }
double* make_input_data() {
  double* data = (double*)malloc(sizeof(double)*LEN*LEN*LEN);
  for (int i = 0; i < LEN; ++i) {
    for (int j = 0; j < LEN; ++j) {
      for (int k = 0; k < LEN; ++k) {
        data[i*(LEN*LEN) + j*(LEN) + k] = sq(i-150)/20.0 + sq(j-150)/30.0 + sq(k-150)/40.0;
      }
    }
  }
  return data;
}
