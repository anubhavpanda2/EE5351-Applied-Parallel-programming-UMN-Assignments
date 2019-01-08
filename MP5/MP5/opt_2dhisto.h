#ifndef OPT_KERNEL
#define OPT_KERNEL
#define BLOCK_SIZE 1024
void opt_2dhisto(uint32_t *input_data,int inputheight,int inputwidth ,uint32_t *input_bins);


/* Include below the function headers of any other functions that you implement */
uint32_t * AllocateDataInDevice(int height,int width,int size);
void FreeDeviceData(uint32_t * data);
void CopyFromHostToDevice(uint32_t *device_data,uint32_t **host_data,int inputrow,int inputcol,int size);
void CopyFromDeviceToHost(uint32_t *host,uint32_t *device,int size,int elementsize);
#endif
