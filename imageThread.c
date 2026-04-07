#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

//An array of kernel matrices to be used for image convolution.  
//The indexes of these match the enumeration from the header file. ie. algorithms[BLUR] returns the kernel corresponding to a box blur.
Matrix algorithms[]={
    {{0,-1,0},{-1,4,-1},{0,-1,0}},
    {{0,-1,0},{-1,5,-1},{0,-1,0}},
    {{1/9.0,1/9.0,1/9.0},{1/9.0,1/9.0,1/9.0},{1/9.0,1/9.0,1/9.0}},
    {{1.0/16,1.0/8,1.0/16},{1.0/8,1.0/4,1.0/8},{1.0/16,1.0/8,1.0/16}},
    {{-2,-1,0},{-1,1,1},{0,1,2}},
    {{0,0,0},{0,1,0},{0,0,0}}
};


//getPixelValue - Computes the value of a specific pixel on a specific channel using the selected convolution kernel
//Paramters: srcImage:  An Image struct populated with the image being convoluted
//           x: The x coordinate of the pixel
//          y: The y coordinate of the pixel
//          bit: The color channel being manipulated
//          algorithm: The 3x3 kernel matrix to use for the convolution
//Returns: The new value for this x,y pixel and bit channel
uint8_t getPixelValue(Image* srcImage,int x,int y,int bit,Matrix algorithm){
    int px,mx,py,my,i,span;
    span=srcImage->width*srcImage->bpp;
    // for the edge pixes, just reuse the edge pixel
    px=x+1; py=y+1; mx=x-1; my=y-1;
    if (mx<0) mx=0;
    if (my<0) my=0;
    if (px>=srcImage->width) px=srcImage->width-1;
    if (py>=srcImage->height) py=srcImage->height-1;
    uint8_t result=
        algorithm[0][0]*srcImage->data[Index(mx,my,srcImage->width,bit,srcImage->bpp)]+
        algorithm[0][1]*srcImage->data[Index(x,my,srcImage->width,bit,srcImage->bpp)]+
        algorithm[0][2]*srcImage->data[Index(px,my,srcImage->width,bit,srcImage->bpp)]+
        algorithm[1][0]*srcImage->data[Index(mx,y,srcImage->width,bit,srcImage->bpp)]+
        algorithm[1][1]*srcImage->data[Index(x,y,srcImage->width,bit,srcImage->bpp)]+
        algorithm[1][2]*srcImage->data[Index(px,y,srcImage->width,bit,srcImage->bpp)]+
        algorithm[2][0]*srcImage->data[Index(mx,py,srcImage->width,bit,srcImage->bpp)]+
        algorithm[2][1]*srcImage->data[Index(x,py,srcImage->width,bit,srcImage->bpp)]+
        algorithm[2][2]*srcImage->data[Index(px,py,srcImage->width,bit,srcImage->bpp)];
    return result;
}

/* Arguments passed to pthread_create's start_routine (one struct per thread). */
typedef struct {
    Image* srcImage;
    Image* destImage;
    Matrix algorithm;
    long idxStart; /* linear index in [0, height*width): row = idx/width, col = idx%width */
    long idxEnd;
} ConvArgs;

/* Row-major slice of the pixel grid: each (row,col) written once — no races. */
static void* convolute_thread_fn(void* arg) {
    ConvArgs* a = (ConvArgs*)arg;
    int w = a->srcImage->width;
    long idx;
    int row, pix, bit;
    for (idx = a->idxStart; idx < a->idxEnd; idx++) {
        row = (int)(idx / w);
        pix = (int)(idx % w);
        for (bit = 0; bit < a->srcImage->bpp; bit++) {
            a->destImage->data[Index(pix, row, w, bit, a->srcImage->bpp)] =
                getPixelValue(a->srcImage, pix, row, bit, a->algorithm);
        }
    }
    return NULL;
}

//convolute:  Applies a kernel matrix to an image (parallel: pthread_create / pthread_join)
//Parameters: srcImage: The image being convoluted
//            destImage: A pointer to a  pre-allocated (including space for the pixel array) structure to receive the convoluted image.  It should be the same size as srcImage
//            algorithm: The kernel matrix to use for the convolution
//Returns: Nothing
void convolute(Image* srcImage, Image* destImage, Matrix algorithm) {
    int h = srcImage->height;
    int w = srcImage->width;
    if (h <= 0 || w <= 0) {
        return;
    }

    long total_pixels = (long)h * (long)w;

    long n = sysconf(_SC_NPROCESSORS_ONLN);
    int num_threads = (n < 1) ? 1 : (int)n;
    if ((long)num_threads > total_pixels) {
        num_threads = (int)total_pixels; /* never spawn more threads than pixels */
    }
    if (num_threads < 1) {
        num_threads = 1;
    }

    pthread_t* threads = malloc((size_t)num_threads * sizeof(pthread_t));
    ConvArgs* args = malloc((size_t)num_threads * sizeof(ConvArgs));

    long base = total_pixels / (long)num_threads;
    long rem = total_pixels % (long)num_threads;
    long start = 0;
    int t;
    for (t = 0; t < num_threads; t++) {
        long chunk = base + ((long)t < rem ? 1 : 0);
        args[t].srcImage = srcImage;
        args[t].destImage = destImage;
        memcpy(args[t].algorithm, algorithm, sizeof(Matrix));
        args[t].idxStart = start;
        args[t].idxEnd = start + chunk;
        start += chunk;
    }

    for (t = 0; t < num_threads; t++) {
        if (pthread_create(&threads[t], NULL, convolute_thread_fn, &args[t]) != 0) {
            perror("pthread_create");
            for (int k = 0; k < t; k++) {
                pthread_join(threads[k], NULL);
            }
            free(threads);
            free(args);
            exit(EXIT_FAILURE);
        }
    }
    for (t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
    free(threads);
    free(args);
}

//Usage: Prints usage information for the program
//Returns: -1
int Usage(){
    printf("Usage: image <filename> <type>\n\twhere type is one of (edge,sharpen,blur,gauss,emboss,identity)\n");
    return -1;
}

//GetKernelType: Converts the string name of a convolution into a value from the KernelTypes enumeration
//Parameters: type: A string representation of the type
//Returns: an appropriate entry from the KernelTypes enumeration, defaults to IDENTITY, which does nothing but copy the image.
enum KernelTypes GetKernelType(char* type){
    if (!strcmp(type,"edge")) return EDGE;
    else if (!strcmp(type,"sharpen")) return SHARPEN;
    else if (!strcmp(type,"blur")) return BLUR;
    else if (!strcmp(type,"gauss")) return GAUSE_BLUR;
    else if (!strcmp(type,"emboss")) return EMBOSS;
    else return IDENTITY;
}

//main:
//argv is expected to take 2 arguments.  First is the source file name (can be jpg, png, bmp, tga).  Second is the lower case name of the algorithm.
int main(int argc,char** argv){
    long t1,t2;
    t1=time(NULL);

    stbi_set_flip_vertically_on_load(0); 
    if (argc!=3) return Usage();
    char* fileName=argv[1];
    if (!strcmp(argv[1],"pic4.jpg")&&!strcmp(argv[2],"gauss")){
        printf("You have applied a gaussian filter to Gauss which has caused a tear in the time-space continum.\n");
    }
    enum KernelTypes type=GetKernelType(argv[2]);

    Image srcImage,destImage,bwImage;   
    srcImage.data=stbi_load(fileName,&srcImage.width,&srcImage.height,&srcImage.bpp,0);
    if (!srcImage.data){
        printf("Error loading file %s.\n",fileName);
        return -1;
    }
    destImage.bpp=srcImage.bpp;
    destImage.height=srcImage.height;
    destImage.width=srcImage.width;
    destImage.data=malloc(sizeof(uint8_t)*destImage.width*destImage.bpp*destImage.height);
    convolute(&srcImage,&destImage,algorithms[type]);
    stbi_write_png("output.png",destImage.width,destImage.height,destImage.bpp,destImage.data,destImage.bpp*destImage.width);
    stbi_image_free(srcImage.data);
    
    free(destImage.data);
    t2=time(NULL);
    printf("Took %ld seconds\n",t2-t1);
   return 0;
}