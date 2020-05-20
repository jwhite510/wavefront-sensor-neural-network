

#ifdef __cplusplus
extern "C"{
#endif

void* MeasuredImageFormatter_new(int i,
    double* dif_in, int dif_in_s0, int dif_in_s1,
    double* dif_out, int dif_out_s0, int dif_out_s1);

void MeasuredImageFormatter_PrintInt(void*p);
void MeasuredImageFormatter_Format(void*p);

#ifdef __cplusplus
}
#endif
