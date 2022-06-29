/*
 * Empty C++ Application
 */
#include <stdio.h>
#include "platform.h"
#include "xil_printf.h"
#include "xaxidma.h"
#include "xtop.h"

#include "AxiTimer.h"
#include "math.h"

#include <iostream>


using namespace std;

XAxiDma do_axi_dma[4];
XAxiDma_Config *do_axi_dma_cfg[4];


XTop do_top_simple;
XTop_Config * do_top_simple_cfg;

//
//AxiTimer timer;
//double timeInterval;


//define data structure

typedef float FPGA_DATA;
//typedef int ufix16;


struct DMA_DATA{
	FPGA_DATA data;

};

struct DOUBLE{
	FPGA_DATA data1;
	FPGA_DATA data2;
};

struct DMA_DATA_64{
	DOUBLE data;
};

//same as HLS
const int Tm=16;//2;//16;
const int Tn=16;//2;//16;
const int Tr=32;//4;//8;
const int Tc=32;//4;//8;

const int TiTo=64;
const int stride=1;
const int K=3;
const int K2=9;
const int Tr_bound=(Tr-1)*stride+K;
const int Tc_bound=(Tc-1)*stride+K;
const int M1=6;
const int R1=9;
const int C1=9;
const int M2=12;
const int R2=4;
const int C2=4;
const int pool_k=2;
const float learn_rate =0.1;
const float minimal=-3.40282e+038 ;
const float eps=1.0e-05 ;
//
//const int ifm_length1=1*Tn*2*((8-1)*1+5)
//		*2*((8-1)*1+5);
//const	int weights_length1=1*Tm*1*8*8*Tn*5*5;
//const	int output_length1=1*Tm*2*8*2*8;
//need to be modified, L5
const int bs=2;
//ddr_memory
static DMA_DATA input_x_buffer[bs*4*224*224];
static DMA_DATA weights_conv1_buffer[64*3*3*3];
static DMA_DATA gama1[64];
static DMA_DATA beta1[64];
static DMA_DATA namuda1[64];
static DMA_DATA weights_conv2_buffer[64*64*3*3];
static DMA_DATA gama2[64];
static DMA_DATA beta2[64];
static DMA_DATA namuda2[64];
static DMA_DATA weights_conv3_buffer[64*128*3*3];
static DMA_DATA gama3[128];
static DMA_DATA beta3[128];
static DMA_DATA namuda3[128];
static DMA_DATA weights_conv4_buffer[128*128*3*3];
static DMA_DATA gama4[128];
static DMA_DATA beta4[128];
static DMA_DATA namuda4[128];
static DMA_DATA weights_conv5_buffer[128*256*3*3];
static DMA_DATA gama5[256];
static DMA_DATA beta5[256];
static DMA_DATA namuda5[256];
static DMA_DATA weights_conv6_buffer[256*256*3*3];
static DMA_DATA gama6[256];
static DMA_DATA beta6[256];
static DMA_DATA namuda6[256];
static DMA_DATA weights_conv7_buffer[256*256*3*3];
static DMA_DATA gama7[256];
static DMA_DATA beta7[256];
static DMA_DATA namuda7[256];
static DMA_DATA weights_conv8_buffer[256*512*3*3];
static DMA_DATA gama8[512];
static DMA_DATA beta8[512];
static DMA_DATA namuda8[512];
static DMA_DATA weights_conv9_buffer[512*512*3*3];
static DMA_DATA gama9[512];
static DMA_DATA beta9[512];
static DMA_DATA namuda9[512];
static DMA_DATA weights_conv10_buffer[512*512*3*3];
static DMA_DATA gama10[512];
static DMA_DATA beta10[512];
static DMA_DATA namuda10[512];
static DMA_DATA weights_conv11_buffer[512*512*3*3];
static DMA_DATA gama11[512];
static DMA_DATA beta11[512];
static DMA_DATA namuda11[512];
static DMA_DATA weights_conv12_buffer[512*512*3*3];
static DMA_DATA gama12[512];
static DMA_DATA beta12[512];
static DMA_DATA namuda12[512];
static DMA_DATA weights_conv13_buffer[512*512*3*3];
static DMA_DATA gama13[512];
static DMA_DATA beta13[512];
static DMA_DATA namuda13[512];
static DMA_DATA weights_fc1_buffer[1000*512*1*1];

static DMA_DATA output1[bs*224*224*64]; //[r][c][m]
static DMA_DATA output1hat[bs*224*224*64]; //[r][c][m]
static DMA_DATA output2[bs*224*224*64];//[r][c][m]
static DMA_DATA output2hat[bs*224*224*64];//[r][c][m]
static DMA_DATA output3[bs*112*112*64];//[r][c][m]
static DMA_DATA pool1index[bs*112*112*8];//r*c*[m/8]

static DMA_DATA output4[bs*112*112*128];//[r][c][m]
static DMA_DATA output4hat[bs*112*112*128];//[r][c][m]
static DMA_DATA output5[bs*112*112*128];//[r][c][m]
static DMA_DATA output5hat[bs*112*112*128];//[r][c][m]
static DMA_DATA output6[bs*56*56*128];//[r][c][m]
static DMA_DATA pool2index[bs*56*56*16];//r*c*[m/8]

static DMA_DATA output7[bs*56*56*256];//[r][c][m]
static DMA_DATA output7hat[bs*56*56*256];//[r][c][m]
static DMA_DATA output8[bs*56*56*256];//[r][c][m]
static DMA_DATA output8hat[bs*56*56*256];//[r][c][m]
static DMA_DATA output9[bs*56*56*256];//[r][c][m]
static DMA_DATA output9hat[bs*56*56*256];//[r][c][m]
static DMA_DATA output10[bs*28*28*256];//[r][c][m]
static DMA_DATA pool3index[bs*28*28*32];//r*c*[m/8]

static DMA_DATA output11[bs*28*28*512];//[r][c][m]
static DMA_DATA output11hat[bs*28*28*512];//[r][c][m]
static DMA_DATA output12[bs*28*28*512];//[r][c][m]
static DMA_DATA output12hat[bs*28*28*512];//[r][c][m]
static DMA_DATA output13[bs*28*28*512];//[r][c][m]
static DMA_DATA output13hat[bs*28*28*512];//[r][c][m]
static DMA_DATA output14[bs*14*14*512];//[r][c][m]
static DMA_DATA pool4index[bs*14*14*64];//r*c*[m/8]

static DMA_DATA output15[bs*14*14*512];//[r][c][m]
static DMA_DATA output15hat[bs*14*14*512];//[r][c][m]
static DMA_DATA output16[bs*14*14*512];//[r][c][m]
static DMA_DATA output16hat[bs*14*14*512];//[r][c][m]
static DMA_DATA output17[bs*14*14*512];//[r][c][m]
static DMA_DATA output17hat[bs*14*14*512];//[r][c][m]
static DMA_DATA output18[bs*7*7*512];//[r][c][m]
static DMA_DATA pool5index[bs*7*7*64];//r*c*[m/8]

static DMA_DATA output19[bs*1*1*512];//[r][c][m]

static DMA_DATA output20[bs*1000];//[r][c][m]


static DMA_DATA loss1[bs*1000];
static DMA_DATA loss2[bs*1*1*512];
static DMA_DATA loss3[bs*7*7*512];
static DMA_DATA loss4[bs*14*14*512];
static DMA_DATA loss5[bs*14*14*512];//[r][c][m]
static DMA_DATA loss6[bs*14*14*512];//[r][c][m]
static DMA_DATA loss7[bs*14*14*512];//[r][c][m]
static DMA_DATA loss8[bs*28*28*512];//[r][c][m]
static DMA_DATA loss9[bs*28*28*512];//[r][c][m]
static DMA_DATA loss10[bs*28*28*512];//[r][c][m]
static DMA_DATA loss11[bs*28*28*256];//[r][c][m]
static DMA_DATA loss12[bs*56*56*256];//[r][c][m]
static DMA_DATA loss13[bs*56*56*256];//[r][c][m]
static DMA_DATA loss14[bs*56*56*256];//[r][c][m]
static DMA_DATA loss15[bs*56*56*128];//[r][c][m]
static DMA_DATA loss16[bs*112*112*128];//[r][c][m]
static DMA_DATA loss17[bs*112*112*128];//[r][c][m]
static DMA_DATA loss18[bs*112*112*64];//[r][c][m]
static DMA_DATA loss19[bs*224*224*64];//[r][c][m]
static DMA_DATA loss20[bs*224*224*64];//[r][c][m]


//	static FPGA_DATA loss4_print[bs][8][8][50]; //[r][c][m]
//	static FPGA_DATA loss5_print[bs][12][12][20]; //[r][c][m]
//	static FPGA_DATA loss6_print[bs][24][24][20];//[r][c][m]
static DMA_DATA weights_conv1[64][3][3][3];//[m][n][r][c]
static DMA_DATA weights_conv2[64][64][3][3];
static DMA_DATA weights_conv3[128][64][3][3];
static DMA_DATA weights_conv4[128][128][3][3];
static DMA_DATA weights_conv5[256][128][3][3];
static DMA_DATA weights_conv6[256][256][3][3];
static DMA_DATA weights_conv7[256][256][3][3];
static DMA_DATA weights_conv8[512][256][3][3];
static DMA_DATA weights_conv9[512][512][3][3];
static DMA_DATA weights_conv10[512][512][3][3];
static DMA_DATA weights_conv11[512][512][3][3];
static DMA_DATA weights_conv12[512][512][3][3];
static DMA_DATA weights_conv13[512][512][3][3];


//	static DMA_DATA loss4_print[bs][8][8][50]; //[r][c][m]
//	static DMA_DATA loss5_print[bs][12][12][20]; //[r][c][m]
//	static DMA_DATA loss6_print[bs][24][24][20];//[r][c][m]
static DMA_DATA outbuf[bs*224*224*64]; //[r][c][m]
DMA_DATA * outbuf_addr = outbuf;

static int target0[bs];
static DMA_DATA target[bs][1000];


////ddr_dma_buffer
//static DMA_DATA ifm_buffer[16*5*12*5*12];//max(M_loop)*Tm*max(R_loop)*Tr_in*max(C_loop)*Tc_in
//static DMA_DATA weights_buffer[3*3*16*16*5*5];//max(R_loop)*max(C_loop)*max(M_loop)*Tm*max(N_loop)*Tn*max(k)2
//static DMA_DATA ofm_buffer[16*16*16];//max(M_loop)*Tm*max(R_loop)*Tr*max(C_loop)*Tc
//static DMA_DATA output_buffer[16*16*25];//max(M_loop)*Tm*max(max(R_loop)*Tr*max(C_loop)*Tc,max(N_loop)*Tn*max(k)2)

DMA_DATA * input_x_addr = input_x_buffer;
DMA_DATA * weights_conv1_addr = weights_conv1_buffer;
DMA_DATA * gama1_addr = gama1;
DMA_DATA * beta1_addr = beta1;
DMA_DATA * namuda1_addr = namuda1;
DMA_DATA * weights_conv2_addr = weights_conv2_buffer;
DMA_DATA * gama2_addr = gama2;
DMA_DATA * beta2_addr = beta2;
DMA_DATA * namuda2_addr = namuda2;
DMA_DATA * weights_conv3_addr = weights_conv3_buffer;
DMA_DATA * gama3_addr = gama3;
DMA_DATA * beta3_addr = beta3;
DMA_DATA * namuda3_addr = namuda3;
DMA_DATA * weights_conv4_addr = weights_conv4_buffer;
DMA_DATA * gama4_addr = gama4;
DMA_DATA * beta4_addr = beta4;
DMA_DATA * namuda4_addr = namuda4;
DMA_DATA * weights_conv5_addr = weights_conv5_buffer;
DMA_DATA * gama5_addr = gama5;
DMA_DATA * beta5_addr = beta5;
DMA_DATA * namuda5_addr = namuda5;
DMA_DATA * weights_conv6_addr = weights_conv6_buffer;
DMA_DATA * gama6_addr = gama6;
DMA_DATA * beta6_addr = beta6;
DMA_DATA * namuda6_addr = namuda6;
DMA_DATA * weights_conv7_addr = weights_conv7_buffer;
DMA_DATA * gama7_addr = gama7;
DMA_DATA * beta7_addr = beta7;
DMA_DATA * namuda7_addr = namuda7;
DMA_DATA * weights_conv8_addr = weights_conv8_buffer;
DMA_DATA * gama8_addr = gama8;
DMA_DATA * beta8_addr = beta8;
DMA_DATA * namuda8_addr = namuda8;
DMA_DATA * weights_conv9_addr = weights_conv9_buffer;
DMA_DATA * gama9_addr = gama9;
DMA_DATA * beta9_addr = beta9;
DMA_DATA * namuda9_addr = namuda9;
DMA_DATA * weights_conv10_addr = weights_conv10_buffer;
DMA_DATA * gama10_addr = gama10;
DMA_DATA * beta10_addr = beta10;
DMA_DATA * namuda10_addr = namuda10;
DMA_DATA * weights_conv11_addr = weights_conv11_buffer;
DMA_DATA * gama11_addr = gama11;
DMA_DATA * beta11_addr = beta11;
DMA_DATA * namuda11_addr = namuda11;
DMA_DATA * weights_conv12_addr = weights_conv12_buffer;
DMA_DATA * gama12_addr = gama12;
DMA_DATA * beta12_addr = beta12;
DMA_DATA * namuda12_addr = namuda12;
DMA_DATA * weights_conv13_addr = weights_conv13_buffer;
DMA_DATA * gama13_addr = gama13;
DMA_DATA * beta13_addr = beta13;
DMA_DATA * namuda13_addr = namuda13;
DMA_DATA * weights_fc1_addr = weights_fc1_buffer;


DMA_DATA * output1_addr = output1;
DMA_DATA * output1hat_addr = output1hat;
DMA_DATA * output2_addr = output2;
DMA_DATA * output2hat_addr = output2hat;
DMA_DATA * output3_addr = output3;
DMA_DATA * pool1index_addr = pool1index;
DMA_DATA * output4_addr = output4;
DMA_DATA * output4hat_addr = output4hat;
DMA_DATA * output5_addr = output5;
DMA_DATA * output5hat_addr = output5hat;
DMA_DATA * output6_addr = output6;
DMA_DATA * pool2index_addr = pool2index;
DMA_DATA * output7_addr = output7;
DMA_DATA * output7hat_addr = output7hat;
DMA_DATA * output8_addr = output8;
DMA_DATA * output8hat_addr = output8hat;
DMA_DATA * output9_addr = output9;
DMA_DATA * output9hat_addr = output9hat;
DMA_DATA * output10_addr = output10;
DMA_DATA * pool3index_addr = pool3index;
DMA_DATA * output11_addr = output11;
DMA_DATA * output11hat_addr = output11hat;
DMA_DATA * output12_addr = output12;
DMA_DATA * output12hat_addr = output12hat;
DMA_DATA * output13_addr = output13;
DMA_DATA * output13hat_addr = output13hat;
DMA_DATA * output14_addr = output14;
DMA_DATA * pool4index_addr = pool4index;
DMA_DATA * output15_addr = output15;
DMA_DATA * output15hat_addr = output15hat;
DMA_DATA * output16_addr = output16;
DMA_DATA * output16hat_addr = output16hat;
DMA_DATA * output17_addr = output17;
DMA_DATA * output17hat_addr = output17hat;
DMA_DATA * output18_addr = output18;
DMA_DATA * pool5index_addr = pool5index;
DMA_DATA * output19_addr = output19;
DMA_DATA * output20_addr = output20;


DMA_DATA * loss1_addr = loss1;
DMA_DATA * loss2_addr = loss2;
DMA_DATA * loss3_addr = loss3;
DMA_DATA * loss4_addr = loss4;
DMA_DATA * loss5_addr = loss5;
DMA_DATA * loss6_addr = loss6;
DMA_DATA * loss7_addr = loss7;
DMA_DATA * loss8_addr = loss8;
DMA_DATA * loss9_addr = loss9;
DMA_DATA * loss10_addr = loss10;
DMA_DATA * loss11_addr = loss11;
DMA_DATA * loss12_addr = loss12;
DMA_DATA * loss13_addr = loss13;
DMA_DATA * loss14_addr = loss14;
DMA_DATA * loss15_addr = loss15;
DMA_DATA * loss16_addr = loss16;
DMA_DATA * loss17_addr = loss17;
DMA_DATA * loss18_addr = loss18;
DMA_DATA * loss19_addr = loss19;
DMA_DATA * loss20_addr = loss20;



int ifm_length;
int weights_length;
int wout_length;
int output_length;
int ofm_length;
DMA_DATA * ifm_addr;
DMA_DATA * weights_addr;
DMA_DATA * wout_addr;
DMA_DATA * output_addr;
DMA_DATA * ofm_addr;

int tb_statec;
int tb_statep;
int tb_stateb;
//	int	tb_Relulayerin;
//	int	tb_Relulayerout;
//	int	tb_Poollayerin;
//	int	tb_Poollayerout;
int tb_custom_batch;
int tb_batch_size;
int tb_M_in;
int tb_custom_Tib;
int	tb_N;
int	tb_M;
int	tb_R;
int	tb_C;
int	tb_custom_stride;
int	tb_padding;
int	tb_custom_k;
int tb_custom_kb;
int	tb_custom_Tr;
int	tb_custom_Tc;
int tb_R_in;
int tb_C_in;
int bf_index;



void init(){

	for(int i=0;i<4;i++){
		switch(i){
		case 0:
			do_axi_dma_cfg[i] = XAxiDma_LookupConfig(XPAR_AXIDMA_0_DEVICE_ID); //AXI_DMA_IFM
			break;
		case 1:
			do_axi_dma_cfg[i] = XAxiDma_LookupConfig(XPAR_AXIDMA_1_DEVICE_ID); //AXI_DMA_OFM
			break;
		case 2:
			do_axi_dma_cfg[i] = XAxiDma_LookupConfig(XPAR_AXIDMA_2_DEVICE_ID); //AXI_DMA_Output
			break;
		case 3:
			do_axi_dma_cfg[i] = XAxiDma_LookupConfig(XPAR_AXIDMA_3_DEVICE_ID); //AXI_DMA_WEIGHTS
			break;
		default:
			break;
		}

		if(do_axi_dma_cfg[i]){
			int status = XAxiDma_CfgInitialize(&do_axi_dma[i],do_axi_dma_cfg[i]);
			if (status != XST_SUCCESS){
				cout<<"Error initializing AxiDMA, ID is "<<i<<endl;
				}
			XAxiDma_IntrDisable(&do_axi_dma[i],XAXIDMA_IRQ_ALL_MASK,XAXIDMA_DEVICE_TO_DMA);
			XAxiDma_IntrDisable(&do_axi_dma[i],XAXIDMA_IRQ_ALL_MASK,XAXIDMA_DMA_TO_DEVICE);
			}
		}


do_top_simple_cfg=XTop_LookupConfig(XPAR_TOP_0_DEVICE_ID);
if(do_top_simple_cfg){
	int status = XTop_CfgInitialize(&do_top_simple,do_top_simple_cfg);
			if (status != XST_SUCCESS){
				cout<<"Error initializing IP"<<endl;
				}
			}

cout<<"All init done!"<<endl;

}


void convfp_3_64_224( int outnum,
	DMA_DATA * ifm,
	DMA_DATA * weights,
	DMA_DATA * outhat,
	DMA_DATA * gama,
	DMA_DATA * beta,
	DMA_DATA * namuda,
	DMA_DATA * out

){
//forward conv1 N,M,R,C=3,64,224,224, stride=1,padding=1,kernel=3
	//AxiTimer timer;
	//double timeInterval;
	//timer.startTimer();
	tb_statec=6;
	tb_statep=0;
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=1;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=64;
	tb_custom_Tib=1;//[N/2Tn]=1
	//tb_custom_k2=25;
	tb_N=3;
	tb_M=64;
	tb_R=224;
	tb_C=224;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=9;//24;
	tb_custom_Tc=224;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=224;
	tb_C_in=224;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls


	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ifm_addr=ifm;
	weights_addr=weights;
	weights_length=1728;//M*N*tb_custom_k*tb_custom_k=3*64*3*3
	output_addr=out;
	output_length=bs*3211264;//min(Tm,M-Tm*to)*min(custom_Tr,R-row*custom_Tr)*C=64*224*224;

	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);

	for(int b=0;b<bs;b++){
		for (int to=0;to<tb_M;to+=Tm){//repeate [M/Tm]
			ifm_length=8960;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
									//*((C-1)*tb_custom_stride+tb_custom_k)=4*224*10//64bits N is multiple of 2
			ifm_addr=ifm+200704*b;//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*4*224*224+row*4*224*1
			XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
			for(int row=tb_custom_Tr;row<tb_R-8;row+=tb_custom_Tr){
				ifm_length=9856;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
										//*((C-1)*tb_custom_stride+tb_custom_k)=4*224*11//64bits N is multiple of 2
				ifm_addr=ifm+200704*b+896*(row-1);//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*4*224*224+row*4*224*1
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//check if dma is working
				//cout<<"IFM in Done!"<<endl;
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
			}
			//for(int row=tb_R-1;row<tb_R;row+=tb_custom_Tr){
			ifm_length=8064;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
									//*((C-1)*tb_custom_stride+tb_custom_k)=4*224*9;//64bits N is multiple of 2
			ifm_addr=ifm+200704*b+192640;//896*(row-1)=896*(216-1);//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*4*224*224+row*4*224*1
			//for(int ti=0;ti<1;ti++){
				//ifm_length=448;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
										//*((C-1)*tb_custom_stride+tb_custom_k)=1*16*28;
			while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//check if dma is working
			//cout<<"IFM in Done!"<<endl;
			XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
			while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//check if dma is working
			//cout<<"IFM in Done!"<<endl;
		}
	}


	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	//while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"output in Done!"<<endl;

	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

//	for(int b=0;b<bs;b++){
//		cout << "batch"<<b<<"for out"<<outnum<<endl;
//		//for(int to=0;to<tb_M_in;to+=Tm){
//		cout <<" 1st channel tile  " <<endl;
//		cout<<"1st 3rows 1st 3cols"<<":";
//		for (int r=0;r<3;r++){
//			for (int c=0;c<3;c++){
//				for (int too=0;too<3;too++){
//					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
//					cout <<out[bf_index].data<< ";";
//
//				}
//			}
//		}
//		cout<<"1st 3rows last 3cols"<<":";
//		for (int r=0;r<3;r++){
//			for (int c=tb_C-3;c<tb_C;c++){
//				for (int too=0;too<3;too++){
//					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
//					cout <<out[bf_index].data<< ";";
//
//				}
//			}
//		}
//		cout<<"last 3rows 1st 3cols"<<":";
//		for (int r=tb_R-3;r<tb_R;r++){
//			for (int c=0;c<3;c++){
//				for (int too=0;too<3;too++){
//					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
//					cout <<out[bf_index].data<< ";";
//
//				}
//			}
//		}
//		cout<<"last 3rows last 3cols"<<":";
//		for (int r=tb_R-3;r<tb_R;r++){
//			for (int c=tb_C-3;c<tb_C;c++){
//				for (int too=0;too<3;too++){
//					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
//					cout <<out[bf_index].data<< ";";
//
//				}
//			}
//		}
//		cout <<" last channel tile  " <<endl;
//		cout<<"1st 3rows 1st 3cols"<<":";
//		for (int r=0;r<3;r++){
//			for (int c=0;c<3;c++){
//				for (int too=Tm-3;too<Tm;too++){
//					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
//					cout <<out[bf_index].data<< ";";
//
//				}
//			}
//		}
//		cout<<"1st 3rows last 3cols"<<":";
//		for (int r=0;r<3;r++){
//			for (int c=tb_C-3;c<tb_C;c++){
//				for (int too=Tm-3;too<Tm;too++){
//					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
//					cout <<out[bf_index].data<< ";";
//
//				}
//			}
//		}
//		cout<<"last 3rows 1st 3cols"<<":";
//		for (int r=tb_R-3;r<tb_R;r++){
//			for (int c=0;c<3;c++){
//				for (int too=Tm-3;too<Tm;too++){
//					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
//					cout <<out[bf_index].data<< ";";
//
//				}
//			}
//		}
//		cout<<"last 3rows last 3cols"<<":";
//		for (int r=tb_R-3;r<tb_R;r++){
//			for (int c=tb_C-3;c<tb_C;c++){
//				for (int too=Tm-3;too<Tm;too++){
//					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
//					cout <<out[bf_index].data<< ";";
//
//				}
//			}
//		}
//		cout<<""<<endl;
//	}
//	cout<<"finish"<<endl;
	cout<<"out"<<endl;
	cout<<out[0].data<<endl;

//
//
//forward bn1+relu N,M,R,C=64,64,224,224, stride=1,padding=1,kernel=3
	tb_statec=0;
	tb_statep=0;
	tb_stateb=7;
//	tb_Relulayerin=0;
//	tb_Relulayerout=1;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=64;
	tb_custom_Tib=1;//[N/2Tn]=1
	//tb_custom_k2=25;
	tb_N=64;
	tb_M=64;
	tb_R=224;
	tb_C=224;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=9;//24;
	tb_custom_Tc=224;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=224;
	tb_C_in=224;


	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	cout<<"gama"<<endl;
	cout<<gama[0].data<<endl;
	cout<<"beta"<<endl;
	cout<<beta[0].data<<endl;

	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ofm_addr=beta;
	weights_addr=gama;
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;


	//ifm_addr=0;
	//ofm_addr=0;
	ifm_addr=out;
	ifm_length=802816;//Tm*R*C
	for (int to=0;to<4;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=out+3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32

			XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
			//ifm_addr+=ifm_length;
			while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//check if dma is working
		}
	}

	ofm_addr=out;//+3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	wout_addr=outhat;//+3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	output_addr=out;//+3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	ofm_length=3211264;//802816;//M*R*C
	output_length=3211264;//802816;//M*R*C
	wout_length=3211264;//802816;//M*R*C
	for(int b=0;b<bs;b++){

		XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);

		XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)wout_addr,wout_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);

		ofm_addr+=ofm_length;
		wout_addr+=wout_length;
		output_addr+=output_length;

		while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
		//cout<<"OFM in Done!"<<endl;
		while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"Weights out Done!"<<endl;
		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"output in Done!"<<endl;
	}
									//=6*3*5*5



	output_addr=namuda;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	output_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"output in Done!"<<endl;
	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
		//cout<<"IP done"<<endl;

	//timer.stopTimer();
	cout<<"print out"<<outnum<<endl;
	//	//timeInterval = timer.getElapsedTimerInSeconds();
	//	//cout<<timeInterval<<" ";

	cout<<"namuda"<<endl;
	for(int i =0;i<tb_M;i++){
		cout<<namuda[i].data<<";";
	}
	cout<<"outhat"<<endl;
	cout<<outhat[0].data<<endl;

	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for out"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void convfp_64_64_224( int outnum,
	DMA_DATA * ifm,
	DMA_DATA * weights,
	DMA_DATA * outhat,
	DMA_DATA * gama,
	DMA_DATA * beta,
	DMA_DATA * namuda,
	DMA_DATA * out

){
	//forward conv2 N,M,R,C=64,64,224,224, stride=1,padding=1,kernel=3
	//AxiTimer timer;
	//double timeInterval;
	//timer.startTimer();
	tb_statec=6;
	tb_statep=0;
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=1;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=64;
	tb_custom_Tib=2;//[N/2Tn]=2
	//tb_custom_k2=25;
	tb_N=64;
	tb_M=64;
	tb_R=224;
	tb_C=224;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=8;//24;
	tb_custom_Tc=224;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=224;
	tb_C_in=224;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	ifm_addr=ifm;
	weights_addr=weights;
	weights_length=36864;//M*N*tb_custom_k*tb_custom_k=64*64*3*3
	output_addr=out;
	output_length=bs*3211264;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=224*224*64;
	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);

	for(int b=0;b<bs;b++){
		for (int to=0;to<tb_M;to+=Tm){//repeate [M/Tm]
			ifm_length=32256;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
									//*((C-1)*tb_custom_stride+tb_custom_k)=16*9*224;
			ifm_addr=ifm+3211264*b;//+3584*row;//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*64*224*224+row*16*224*1
			for(int ti=0;ti<tb_N;ti+=Tn){
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				ifm_addr+=802816;//Tn*R_in*C_in=16*224*224
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//check if dma is working
			}
			for(int row=tb_custom_Tr;row<tb_R-tb_custom_Tr;row+=tb_custom_Tr){
				ifm_length=35840;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
														//*((C-1)*tb_custom_stride+tb_custom_k)=16*10*224;
				ifm_addr=ifm+3211264*b+(row-1)*3584;//+3584*row;//b*N*C_in*R_in+C_in*(row-padding)*stride*min(Tn,N)=b*96*27*27+(row-2)*1*27*16
				for(int ti=0;ti<tb_N;ti+=Tn){
					XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
					ifm_addr+=802816;//Tn*R_in*C_in=16*224*224
					while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//check if dma is working
				}
			}
			ifm_length=32256;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
									//*((C-1)*tb_custom_stride+tb_custom_k)=16*9*224;
			ifm_addr=ifm+3211264*b+770560;//+3584*(row-1);//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*64*224*224+row*16*224*1
			for(int ti=0;ti<tb_N;ti+=Tn){
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				ifm_addr+=802816;//Tn*R_in*C_in=16*224*224
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//check if dma is working
			}
		}
	}
//}
	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	//while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"output in Done!"<<endl;

	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

//forward bn1+relu N,M,R,C=64,64,224,224, stride=1,padding=1,kernel=3
	tb_statec=0;
	tb_statep=0;
	tb_stateb=7;
//	tb_Relulayerin=0;
//	tb_Relulayerout=1;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=64;
	tb_custom_Tib=1;//[N/2Tn]=1
	//tb_custom_k2=25;
	tb_N=64;
	tb_M=64;
	tb_R=224;
	tb_C=224;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=9;//24;
	tb_custom_Tc=224;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=224;
	tb_C_in=224;


	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	cout<<"gama"<<endl;
	cout<<gama[0].data<<endl;
	cout<<"beta"<<endl;
	cout<<beta[0].data<<endl;

	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ofm_addr=beta;
	weights_addr=gama;
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;


	//ifm_addr=0;
	//ofm_addr=0;
	ifm_addr=out;
	ifm_length=802816;//Tm*R*C
	for (int to=0;to<4;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=out+3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32

			XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
			//ifm_addr+=ifm_length;
			while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//check if dma is working
			//cout<<"IFM in Done!"<<endl;
		}
	}

	ofm_addr=out;//+3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	wout_addr=outhat;//+3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	output_addr=out;//+3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	ofm_length=3211264;//802816;//M*R*C
	output_length=3211264;//802816;//M*R*C
	wout_length=3211264;//802816;//M*R*C
	for(int b=0;b<bs;b++){

		XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);

		XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)wout_addr,wout_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);

		ofm_addr+=ofm_length;
		wout_addr+=wout_length;
		output_addr+=output_length;

		while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
		//cout<<"OFM in Done!"<<endl;
		while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"Weights out Done!"<<endl;
		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"output in Done!"<<endl;
	}
										//=6*3*5*5



	output_addr=namuda;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	output_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"output in Done!"<<endl;
	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
		//cout<<"IP done"<<endl;
	cout<<"namuda"<<endl;
	cout<<namuda[0].data<<endl;
	cout<<"outhat"<<endl;
	cout<<outhat[0].data<<endl;



	//timer.stopTimer();
	cout<<"print out"<<outnum<<endl;
	//timeInterval = timer.getElapsedTimerInSeconds();

	//cout<<timeInterval<<" ";

	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for out"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
	//cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void maxpoolfp_64_112( int outnum,
	DMA_DATA * ifm,
	DMA_DATA * weights,
	DMA_DATA * out

){
	//forward pool N,M,R,C=64,64,112,112, stride=2,padding=0,kernel=2
	//timer.startTimer();
	tb_statec=0;
	tb_statep=6;
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=0;
	//tb_Poollayerin=0;
	//tb_Poollayerout=2;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=64;
	tb_custom_Tib=1;//[N/2Tn]=3
	//tb_custom_k2=4;
	tb_N=64;
	tb_M=64;
	tb_R=112;
	tb_C=112;
	tb_custom_stride=2;
	tb_padding=0;
	tb_custom_kb=1;
	tb_custom_k=2;
	tb_custom_Tr=1;//12;
	tb_custom_Tc=112;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=224;
	tb_C_in=224;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	ifm_addr=ifm;
	ifm_length=bs*3211264;//M*R_in*C_in=224*224*64;
	output_addr=out;
	weights_addr=weights;
	output_length=802816;//M*min(custom_Tr,R-row*custom_Tr)*C=64*112*112;
	weights_length=100352;//r*c*[m/8]=112*112*8

	XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
	//XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	//XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	for(int b=0;b<bs;b++){
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
		output_addr+=output_length;
		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"Output in Done!"<<endl;
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
		weights_addr+=weights_length;
		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"Output in Done!"<<endl;
	}

	while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
	//cout<<"IFM in Done!"<<endl;
	//while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"Output in Done!"<<endl;

	//weights_addr+=weights_length;
	//while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"OFM in Done!"<<endl;

	//while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;



	//while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;



	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"print out"<<outnum<<endl;
	//cout<<timeInterval<<" ";



	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for out"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void convfp_64_128_112( int outnum,
	DMA_DATA * ifm,
	DMA_DATA * weights,
	DMA_DATA * outhat,
	DMA_DATA * gama,
	DMA_DATA * beta,
	DMA_DATA * namuda,
	DMA_DATA * out

){
	//forward conv3 N,M,R,C=64,128,112,112, stride=1,padding=1,kernel=3
	//timer.startTimer();
	tb_statec=6;
	tb_statep=0;
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=1;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=128;
	tb_custom_Tib=2;//[N/2Tn]=2
	//tb_custom_k2=25;
	tb_N=64;
	tb_M=128;
	tb_R=112;
	tb_C=112;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=16;//24;
	tb_custom_Tc=112;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=112;
	tb_C_in=112;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	ifm_addr=ifm;
	weights_addr=weights;
	weights_length=73728;//M*N*tb_custom_k*tb_custom_k=64*128*3*3
	output_addr=out;
	output_length=bs*1605632;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=112*112*128;

	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);

	for(int b=0;b<bs;b++){
		for (int to=0;to<tb_M;to+=Tm){//repeate [M/Tm]
			ifm_length=30464;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
									//*((C-1)*tb_custom_stride+tb_custom_k)=16*17*112;
			ifm_addr=ifm+802816*b;//+3584*row;//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*64*112*112+row*16*224*1
			for(int ti=0;ti<tb_N;ti+=Tn){
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				ifm_addr+=200704;//Tn*R_in*C_in=16*112*112
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//check if dma is working
			}
			for(int row=tb_custom_Tr;row<tb_R-tb_custom_Tr;row+=tb_custom_Tr){
				ifm_length=32256;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
														//*((C-1)*tb_custom_stride+tb_custom_k)=16*18*112;
				ifm_addr=ifm+802816*b+(row-1)*1792;//+1792*row;//b*N*C_in*R_in+C_in*(row-padding)*stride*min(Tn,N)=b*64*112*112+(row-1)*1*112*16
				for(int ti=0;ti<tb_N;ti+=Tn){
					XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
					ifm_addr+=200704;//Tn*R_in*C_in=16*112*112
					while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//check if dma is working
				}
			}
			ifm_length=30464;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
									//*((C-1)*tb_custom_stride+tb_custom_k)=16*17*112;
			ifm_addr=ifm+802816*b+170240;//+1792*row;//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*64*112*112+row*16*224*1
			for(int ti=0;ti<tb_N;ti+=Tn){
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				ifm_addr+=200704;//Tn*R_in*C_in=16*112*112
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//check if dma is working
			}
		}
	}



	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	//while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"output in Done!"<<endl;

	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

//forward bn3+relu N,M,R,C=128,128,112,112, stride=1,padding=1,kernel=3
	tb_statec=0;
	tb_statep=0;
	tb_stateb=7;
//	tb_Relulayerin=0;
//	tb_Relulayerout=1;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=128;
	tb_custom_Tib=1;//[N/2Tn]=1
	//tb_custom_k2=25;
	tb_N=128;
	tb_M=128;
	tb_R=112;
	tb_C=112;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=9;//24;
	tb_custom_Tc=224;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=112;
	tb_C_in=112;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ofm_addr=beta;
	weights_addr=gama;
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;
	//ifm_addr=0;
	//ofm_addr=0;
	ifm_addr=out;
	ifm_length=200704;//Tm*R*C
	for (int to=0;to<8;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=out+1605632*b+200704*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32

			XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
			//ifm_addr+=ifm_length;
			while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//check if dma is working
		}
	}

	ofm_addr=out;//+3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	wout_addr=outhat;//+3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	output_addr=out;//+3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	ofm_length=1605632;//802816;//M*R*C
	output_length=1605632;//802816;//M*R*C
	wout_length=1605632;//802816;//M*R*C
	for(int b=0;b<bs;b++){

		XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);

		XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)wout_addr,wout_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);

		ofm_addr+=ofm_length;
		wout_addr+=wout_length;
		output_addr+=output_length;

		while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
		//cout<<"OFM in Done!"<<endl;
		while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"Weights out Done!"<<endl;
		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"output in Done!"<<endl;
	}


	output_addr=namuda;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	output_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"output in Done!"<<endl;
	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
		//cout<<"IP done"<<endl;
	cout<<"namuda"<<endl;
	cout<<namuda[0].data<<endl;
	cout<<"outhat"<<endl;
	cout<<outhat[0].data<<endl;



	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"print out"<<outnum<<endl;
	//cout<<timeInterval<<" ";

	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for out"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void convfp_128_128_112( int outnum,
	DMA_DATA * ifm,
	DMA_DATA * weights,
	DMA_DATA * outhat,
	DMA_DATA * gama,
	DMA_DATA * beta,
	DMA_DATA * namuda,
	DMA_DATA * out

){
	//forward conv4 N,M,R,C=128,128,112,112, stride=1,padding=1,kernel=3
	//timer.startTimer();
	tb_statec=6;
	tb_statep=0;
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=1;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=128;
	tb_custom_Tib=4;//[N/2Tn]=2
	//tb_custom_k2=25;
	tb_N=128;
	tb_M=128;
	tb_R=112;
	tb_C=112;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=16;//24;
	tb_custom_Tc=112;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=112;
	tb_C_in=112;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	ifm_addr=ifm;
	weights_addr=weights;
	weights_length=147456;//M*N*tb_custom_k*tb_custom_k=128*128*3*3
	output_addr=out;
	output_length=bs*1605632;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=112*112*128;

	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	for(int b=0;b<bs;b++){
		for (int to=0;to<tb_M;to+=Tm){//repeate [M/Tm]
			ifm_length=30464;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
									//*((C-1)*tb_custom_stride+tb_custom_k)=16*17*112;
			ifm_addr=ifm+1605632*b;//+3584*row;//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*128*112*112+row*16*224*1
			for(int ti=0;ti<tb_N;ti+=Tn){
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				ifm_addr+=200704;//Tn*R_in*C_in=16*112*112
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//check if dma is working
			}
			for(int row=tb_custom_Tr;row<tb_R-tb_custom_Tr;row+=tb_custom_Tr){
				ifm_length=32256;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
														//*((C-1)*tb_custom_stride+tb_custom_k)=16*18*112;
				ifm_addr=ifm+1605632*b+(row-1)*1792;//+1792*row;//b*N*C_in*R_in+C_in*(row-padding)*stride*min(Tn,N)=b*128*112*112+(row-1)*1*112*16
				for(int ti=0;ti<tb_N;ti+=Tn){
					XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
					ifm_addr+=200704;//Tn*R_in*C_in=16*112*112
					while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//check if dma is working
				}
			}
			ifm_length=30464;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
									//*((C-1)*tb_custom_stride+tb_custom_k)=16*17*112;
			ifm_addr=ifm+1605632*b+170240;//+1792*row;//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*128*112*112+row*16*224*1
			for(int ti=0;ti<tb_N;ti+=Tn){
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				ifm_addr+=200704;//Tn*R_in*C_in=16*112*112
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//check if dma is working
			}
		}
	}


	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	//while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"output in Done!"<<endl;

	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

//forward bn4+relu N,M,R,C=128,128,112,112, stride=1,padding=1,kernel=3
	tb_statec=0;
	tb_statep=0;
	tb_stateb=7;
//	tb_Relulayerin=0;
//	tb_Relulayerout=1;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=128;
	tb_custom_Tib=1;//[N/2Tn]=1
	//tb_custom_k2=25;
	tb_N=128;
	tb_M=128;
	tb_R=112;
	tb_C=112;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=9;//24;
	tb_custom_Tc=224;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=112;
	tb_C_in=112;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls



	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ofm_addr=beta;
	weights_addr=gama;
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;

	//ifm_addr=0;
	//ofm_addr=0;
	ifm_addr=out;
	ifm_length=200704;//Tm*R*C
	for (int to=0;to<8;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=out+1605632*b+200704*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32

			XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
			//ifm_addr+=ifm_length;
			while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//check if dma is working
		}
	}

	ofm_addr=out;//+3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	wout_addr=outhat;//+3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	output_addr=out;//+3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	ofm_length=1605632;//802816;//M*R*C
	output_length=1605632;//802816;//M*R*C
	wout_length=1605632;//802816;//M*R*C
	for(int b=0;b<bs;b++){

		XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);

		XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)wout_addr,wout_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);

		ofm_addr+=ofm_length;
		wout_addr+=wout_length;
		output_addr+=output_length;

		while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
		//cout<<"OFM in Done!"<<endl;
		while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"Weights out Done!"<<endl;
		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"output in Done!"<<endl;
	}


	output_addr=namuda;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	output_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"output in Done!"<<endl;
	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
		//cout<<"IP done"<<endl;
	cout<<"namuda"<<endl;
	cout<<namuda[0].data<<endl;
	cout<<"outhat"<<endl;
	cout<<outhat[0].data<<endl;


	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"print out"<<outnum<<endl;
	//cout<<timeInterval<<" ";

	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for out"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void maxpoolfp_128_56( int outnum,
	DMA_DATA * ifm,
	DMA_DATA * weights,
	DMA_DATA * out

){
	//forward pool N,M,R,C=128,128,56,56, stride=2,padding=0,kernel=2
	//timer.startTimer();
	tb_statec=0;
	tb_statep=6;
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=0;
	//tb_Poollayerin=0;
	//tb_Poollayerout=2;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=128;
	tb_custom_Tib=1;//[N/2Tn]=3
	//tb_custom_k2=4;
	tb_N=128;
	tb_M=128;
	tb_R=56;
	tb_C=56;
	tb_custom_stride=2;
	tb_padding=0;
	tb_custom_kb=1;
	tb_custom_k=2;
	tb_custom_Tr=1;//12;
	tb_custom_Tc=56;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=112;
	tb_C_in=112;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls


	ifm_addr=ifm;
	ifm_length=bs*1605632;//M*R_in*C_in=112*112*128;
	output_addr=out;
	weights_addr=weights;
	output_length=401408;//M*min(custom_Tr,R-row*custom_Tr)*C=128*56*56;
	weights_length=50176;//r*c*[m/8]=56*56*16

	XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
	//XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	//XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	for(int b=0;b<bs;b++){
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
		output_addr+=output_length;
		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"Output in Done!"<<endl;
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
		weights_addr+=weights_length;
		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"Output in Done!"<<endl;
	}

	while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
	//cout<<"IFM in Done!"<<endl;
	//while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"Output in Done!"<<endl;

	//weights_addr+=weights_length;
	//while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"OFM in Done!"<<endl;

	//while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;



	//while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;



	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"print out"<<outnum<<endl;
	//cout<<timeInterval<<" ";

	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for out"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void convfp_128_256_56( int outnum,
	DMA_DATA * ifm,
	DMA_DATA * weights,
	DMA_DATA * outhat,
	DMA_DATA * gama,
	DMA_DATA * beta,
	DMA_DATA * namuda,
	DMA_DATA * out

){
	//forward conv5 N,M,R,C=128,256,56,56, stride=1,padding=1,kernel=3
	//timer.startTimer();
	tb_statec=6;
	tb_statep=0;
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=1;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=256;
	tb_custom_Tib=4;//[N/2Tn]=2
	//tb_custom_k2=25;
	tb_N=128;
	tb_M=256;
	tb_R=56;
	tb_C=56;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=28;//24;
	tb_custom_Tc=56;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=56;
	tb_C_in=56;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	ifm_addr=ifm;
	weights_addr=weights;
	ifm_length=25984;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
											//*((C-1)*tb_custom_stride+tb_custom_k)=16*29*56;
	weights_length=294912;//M*N*tb_custom_k*tb_custom_k=128*256*3*3
	output_addr=out;
	output_length=bs*802816;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=56*56*256;

	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);

	for(int b=0;b<bs;b++){
		for (int to=0;to<tb_M;to+=Tm){//repeate [M/Tm]
			//ifm_length=25984;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
									//*((C-1)*tb_custom_stride+tb_custom_k)=16*29*56;
			ifm_addr=ifm+401408*b;//+3584*row;//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*128*56*56+row*16*224*1
			for(int ti=0;ti<tb_N;ti+=Tn){
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				ifm_addr+=50176;//Tn*R_in*C_in=16*56*56
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));
			}
//				for(int row=tb_custom_Tr;row<tb_R-tb_custom_Tr;row+=tb_custom_Tr){
//					ifm_length=8960;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
//															//*((C-1)*tb_custom_stride+tb_custom_k)=16*10*56;
//					ifm_addr=401408*b+(row-1)*896;//+1792*row;//b*N*C_in*R_in+C_in*(row-padding)*stride*min(Tn,N)=b*128*56*56+(row-1)*1*56*16
//					for(int ti=0;ti<tb_N;ti+=Tn){
//						for(int i=0;i<ifm_length;i+=4){
//							input_ifm.data.data1 = ifm[i+ifm_addr];
//							input_ifm.data.data2 = ifm[i+ifm_addr+1];
//							input_ifm.data.data3 = ifm[i+ifm_addr+2];
//							input_ifm.data.data4 = ifm[i+ifm_addr+3];
//							tb_input_dma_ifm.write(input_ifm);
//						}
//						ifm_addr+=50176;//Tn*R_in*C_in=16*56*56
//					}
//				}
			//ifm_length=25984;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
								//*((C-1)*tb_custom_stride+tb_custom_k)=16*29*56;
			ifm_addr=ifm+401408*b+24192;//+1792*row;//b*N*C_in*R_in+C_in*(row-padding)*stride*min(Tn,N)=b*128*56*56+(row-1)*1*56*16
			for(int ti=0;ti<tb_N;ti+=Tn){
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				ifm_addr+=50176;//Tn*R_in*C_in=16*56*56
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));
			}
		}
	}

	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	//while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"output in Done!"<<endl;

	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

//forward bn5+relu N,M,R,C=256,256,56,56, stride=1,padding=1,kernel=3
	tb_statec=0;
	tb_statep=0;
	tb_stateb=7;
//	tb_Relulayerin=0;
//	tb_Relulayerout=1;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=256;
	tb_custom_Tib=1;//[N/2Tn]=1
	//tb_custom_k2=25;
	tb_N=256;
	tb_M=256;
	tb_R=56;
	tb_C=56;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=9;//24;
	tb_custom_Tc=224;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=56;
	tb_C_in=56;


	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls



	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ofm_addr=beta;
	weights_addr=gama;
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;

	ifm_addr=out;
	ifm_length=50176;//Tm*R*C
	for (int to=0;to<16;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=out+802816*b+50176*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32

			XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
			//ifm_addr+=ifm_length;
			while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//check if dma is working
		}
	}

	ofm_addr=out;//+3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	wout_addr=outhat;//+3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	output_addr=out;//+3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	ofm_length=802816;//802816;//M*R*C
	output_length=802816;//802816;//M*R*C
	wout_length=802816;//802816;//M*R*C
	for(int b=0;b<bs;b++){

		XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);

		XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)wout_addr,wout_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);

		ofm_addr+=ofm_length;
		wout_addr+=wout_length;
		output_addr+=output_length;

		while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
		//cout<<"OFM in Done!"<<endl;
		while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"Weights out Done!"<<endl;
		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"output in Done!"<<endl;
	}


	output_addr=namuda;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	output_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"output in Done!"<<endl;
	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
		//cout<<"IP done"<<endl;
	cout<<"namuda"<<endl;
	cout<<namuda[0].data<<endl;
	cout<<"outhat"<<endl;
	cout<<outhat[0].data<<endl;

	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"print out"<<outnum<<endl;
	//cout<<timeInterval<<" ";


	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for out"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void convfp_256_256_56( int outnum,
	DMA_DATA * ifm,
	DMA_DATA * weights,
	DMA_DATA * outhat,
	DMA_DATA * gama,
	DMA_DATA * beta,
	DMA_DATA * namuda,
	DMA_DATA * out

){
	//forward conv6,7 N,M,R,C=128,256,56,56, stride=1,padding=1,kernel=3
	//timer.startTimer();
	tb_statec=6;
	tb_statep=0;
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=1;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=256;
	tb_custom_Tib=8;//[N/2Tn]=2
	//tb_custom_k2=25;
	tb_N=256;
	tb_M=128;
	tb_R=56;
	tb_C=56;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=28;//24;
	tb_custom_Tc=56;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=56;
	tb_C_in=56;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	ifm_addr=ifm;
	weights_addr=weights;
	ifm_length=25984;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
											//*((C-1)*tb_custom_stride+tb_custom_k)=16*29*56;
	weights_length=589824;//M*N*tb_custom_k*tb_custom_k=256*256*3*3
	output_addr=out;

	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);

	for (int toM=0;toM<2;toM++){
		ifm_addr=ifm;
		output_length=401408;//tb_M*R*C=56*56*128;
		output_addr=out+toM*401408;
		for(int b=0;b<bs;b++){
			XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
			for (int to=0;to<tb_M;to+=Tm){//repeate [M/Tm]
				//ifm_length=25984;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
										//*((C-1)*tb_custom_stride+tb_custom_k)=16*29*56;
				ifm_addr=ifm+802816*b;//+3584*row;//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*256*56*56+row*16*224*1
				for(int ti=0;ti<tb_N;ti+=Tn){
					XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
					ifm_addr+=50176;//Tn*R_in*C_in=16*56*56
					while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));
				}
//				for(int row=tb_custom_Tr;row<tb_R-tb_custom_Tr;row+=tb_custom_Tr){
//					ifm_length=8960;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
//															//*((C-1)*tb_custom_stride+tb_custom_k)=16*10*56;
//					ifm_addr=802816*b+(row-1)*896;//+1792*row;//b*N*C_in*R_in+C_in*(row-padding)*stride*min(Tn,N)=b*256*56*56+(row-1)*1*56*16
//					for(int ti=0;ti<tb_N;ti+=Tn){
//						for(int i=0;i<ifm_length;i+=4){
//							input_ifm.data.data1 = ifm[i+ifm_addr];
//							input_ifm.data.data2 = ifm[i+ifm_addr+1];
//							input_ifm.data.data3 = ifm[i+ifm_addr+2];
//							input_ifm.data.data4 = ifm[i+ifm_addr+3];
//							tb_input_dma_ifm.write(input_ifm);
//						}
//						ifm_addr+=50176;//Tn*R_in*C_in=16*56*56
//					}
//				}
				//ifm_length=25984;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
								//*((C-1)*tb_custom_stride+tb_custom_k)=16*9*28
				ifm_addr=ifm+802816*b+24192;//+896*row;//b*N*C_in*R_in+C_in*(row-padding)*stride*min(Tn,N)=b*256*56*56+(row-1)*1*56*16
				for(int ti=0;ti<tb_N;ti+=Tn){
					XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
					ifm_addr+=50176;//Tn*R_in*C_in=16*56*56
					while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));
				}
			}
			output_addr+=802816;
			while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
			//cout<<"output in Done!"<<endl;
		}
	}

	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	//while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;

//	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
//	//cout<<"output in Done!"<<endl;

	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

//forward bn6,7+relu N,M,R,C=256,256,56,56, stride=1,padding=1,kernel=3
	tb_statec=0;
	tb_statep=0;
	tb_stateb=7;
//	tb_Relulayerin=0;
//	tb_Relulayerout=1;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=256;
	tb_custom_Tib=1;//[N/2Tn]=1
	//tb_custom_k2=25;
	tb_N=256;
	tb_M=256;
	tb_R=56;
	tb_C=56;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=9;//24;
	tb_custom_Tc=224;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=56;
	tb_C_in=56;


	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls



	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ofm_addr=beta;
	weights_addr=gama;
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;
	//ifm_addr=0;
	//ofm_addr=0;
	ifm_addr=out;
	ifm_length=50176;//Tm*R*C
	for (int to=0;to<16;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=out+802816*b+50176*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32

			XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
			//ifm_addr+=ifm_length;
			while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//check if dma is working
		}
	}

	ofm_addr=out;//+3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	wout_addr=outhat;//+3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	output_addr=out;//+3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	ofm_length=802816;//802816;//M*R*C
	output_length=802816;//802816;//M*R*C
	wout_length=802816;//802816;//M*R*C
	for(int b=0;b<bs;b++){

		XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);

		XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)wout_addr,wout_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);

		ofm_addr+=ofm_length;
		wout_addr+=wout_length;
		output_addr+=output_length;

		while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
		//cout<<"OFM in Done!"<<endl;
		while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"Weights out Done!"<<endl;
		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"output in Done!"<<endl;
	}

	output_addr=namuda;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	output_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"output in Done!"<<endl;
	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
		//cout<<"IP done"<<endl;
	cout<<"namuda"<<endl;
	cout<<namuda[0].data<<endl;
	cout<<"outhat"<<endl;
	cout<<outhat[0].data<<endl;

	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"print out"<<outnum<<endl;
	//cout<<timeInterval<<" ";


	//FPGA_DATA output1_print[bs][24][24][20]; //[r][c][m]//only print half and last 10 channels

	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for out"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void maxpoolfp_256_28( int outnum,
	DMA_DATA * ifm,
	DMA_DATA * weights,
	DMA_DATA * out

){
	//forward pool N,M,R,C=256,256,28,28, stride=2,padding=0,kernel=2
	//timer.startTimer();
	tb_statec=0;
	tb_statep=6;
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=0;
	//tb_Poollayerin=0;
	//tb_Poollayerout=2;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=256;
	tb_custom_Tib=1;//[N/2Tn]=3
	//tb_custom_k2=4;
	tb_N=256;
	tb_M=256;
	tb_R=28;
	tb_C=28;
	tb_custom_stride=2;
	tb_padding=0;
	tb_custom_kb=1;
	tb_custom_k=2;
	tb_custom_Tr=1;//12;
	tb_custom_Tc=28;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=56;
	tb_C_in=56;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls


	ifm_addr=ifm;
	ifm_length=bs*802816;//M*R_in*C_in=56*56*256;
	output_addr=out;
	weights_addr=weights;
	output_length=200704;//M*min(custom_Tr,R-row*custom_Tr)*C=256*28*28;
	weights_length=25088;//r*c*[m/8]=28*28*32

	XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
	//XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	//XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	for(int b=0;b<bs;b++){
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
		output_addr+=output_length;
		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"Output in Done!"<<endl;
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
		weights_addr+=weights_length;
		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"Output in Done!"<<endl;
	}

	while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
	//cout<<"IFM in Done!"<<endl;
	//while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"Output in Done!"<<endl;

	//weights_addr+=weights_length;
	//while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"OFM in Done!"<<endl;

	//while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;



	//while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;



	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;
	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"print out"<<outnum<<endl;
	//cout<<timeInterval<<" ";


	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for out"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void convfp_256_512_28( int outnum,
	DMA_DATA * ifm,
	DMA_DATA * weights,
	DMA_DATA * outhat,
	DMA_DATA * gama,
	DMA_DATA * beta,
	DMA_DATA * namuda,
	DMA_DATA * out

){
	//forward conv8 N,M,R,C=256,512,28,28, stride=1,padding=1,kernel=3
	//timer.startTimer();
	tb_statec=6;
	tb_statep=0;
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=1;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=512;
	tb_custom_Tib=8;//[N/2Tn]=2
	//tb_custom_k2=25;
	tb_N=256;
	tb_M=128;
	tb_R=28;
	tb_C=28;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=28;//24;
	tb_custom_Tc=28;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=28;
	tb_C_in=28;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	weights_addr=weights;
	ifm_length=200704;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
											//*((C-1)*tb_custom_stride+tb_custom_k)=256*28*28;
	weights_length=1179648;//M*N*tb_custom_k*tb_custom_k=512*256*3*3
	output_addr=out;

	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	for (int toM=0;toM<4;toM++){
		ifm_addr=ifm;
		output_length=100352;//tb_M*R*C=28*28*128;
		output_addr=out+toM*100352;
		for(int b=0;b<bs;b++){
			XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
			for (int to=0;to<tb_M;to+=Tm){//repeate [M/Tm]
				//ifm_length=200704;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
										//*((C-1)*tb_custom_stride+tb_custom_k)=256*28*28;
				//ifm_addr=200704*b;//+3584*row;//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*256*28*28+row*16*28*1
				//for(int ti=0;ti<tb_N;ti+=Tn){
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));
			}
			ifm_addr+=200704;//N*R_in*C_in=256*28*28
			output_addr+=401408;
			while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
			//cout<<"output in Done!"<<endl;
		}
	}

	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	//while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;

	//	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//	//cout<<"output in Done!"<<endl;

	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

//forward bn8+relu N,M,R,C=512,512,28,28, stride=1,padding=1,kernel=3
	tb_statec=0;
	tb_statep=0;
	tb_stateb=7;
//	tb_Relulayerin=0;
//	tb_Relulayerout=1;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=512;
	tb_custom_Tib=1;//[N/2Tn]=1
	//tb_custom_k2=25;
	tb_N=512;
	tb_M=512;
	tb_R=28;
	tb_C=28;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=9;//24;
	tb_custom_Tc=224;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=28;
	tb_C_in=28;


	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls



	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ofm_addr=beta;
	weights_addr=gama;
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;
	//ifm_addr=0;
	//ofm_addr=0;
	ifm_addr=out;
	ifm_length=12544;//Tm*R*C
	for (int to=0;to<32;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=out+401408*b+12544*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32

			XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
			//ifm_addr+=ifm_length;
			while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//check if dma is working
		}
	}

	ofm_addr=out;//+3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	wout_addr=outhat;//+3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	output_addr=out;//+3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	ofm_length=401408;//802816;//M*R*C
	output_length=401408;//802816;//M*R*C
	wout_length=401408;//802816;//M*R*C
	for(int b=0;b<bs;b++){

		XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);

		XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)wout_addr,wout_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);

		ofm_addr+=ofm_length;
		wout_addr+=wout_length;
		output_addr+=output_length;

		while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
		//cout<<"OFM in Done!"<<endl;
		while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"Weights out Done!"<<endl;
		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"output in Done!"<<endl;
	}


	output_addr=namuda;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	output_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"output in Done!"<<endl;
	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
		//cout<<"IP done"<<endl;

	cout<<"namuda"<<endl;
	cout<<namuda[0].data<<endl;
	cout<<"outhat"<<endl;
	cout<<outhat[0].data<<endl;

	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"print out"<<outnum<<endl;
	//cout<<timeInterval<<" ";


	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for out"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void convfp_512_512_28( int outnum,
	DMA_DATA * ifm,
	DMA_DATA * weights,
	DMA_DATA * outhat,
	DMA_DATA * gama,
	DMA_DATA * beta,
	DMA_DATA * namuda,
	DMA_DATA * out

){
	//forward conv9,10 N,M,R,C=512,512,28,28, stride=1,padding=1,kernel=3
	//timer.startTimer();
	tb_statec=6;
	tb_statep=0;
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=1;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=512;
	tb_custom_Tib=16;//[N/2Tn]=2
	//tb_custom_k2=25;
	tb_N=512;
	tb_M=64;
	tb_R=28;
	tb_C=28;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=28;//24;
	tb_custom_Tc=28;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=28;
	tb_C_in=28;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	weights_addr=weights;
	ifm_length=401408;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
											//*((C-1)*tb_custom_stride+tb_custom_k)=512*28*28;
	weights_length=2359296;//M*N*tb_custom_k*tb_custom_k=512*512*3*3


	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);

	for (int toM=0;toM<8;toM++){
		ifm_addr=ifm;
		output_length=50176;//tb_M*R*C=28*28*64;
		output_addr=out+toM*50176;
		for(int b=0;b<bs;b++){
			XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
			for (int to=0;to<tb_M;to+=Tm){//repeate [M/Tm]
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
				//cout<<"IFM in Done!"<<endl;
			}
			ifm_addr+=401408;//N*R_in*C_in=512*28*28
			output_addr+=401408;
			while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
			//cout<<"output in Done!"<<endl;
		}
	}


	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	//while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;

	//	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//	//cout<<"output in Done!"<<endl;

	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

//forward bn9,10+relu N,M,R,C=512,512,28,28, stride=1,padding=1,kernel=3
	tb_statec=0;
	tb_statep=0;
	tb_stateb=7;
//	tb_Relulayerin=0;
//	tb_Relulayerout=1;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=512;
	tb_custom_Tib=1;//[N/2Tn]=1
	//tb_custom_k2=25;
	tb_N=512;
	tb_M=512;
	tb_R=28;
	tb_C=28;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=9;//24;
	tb_custom_Tc=224;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=28;
	tb_C_in=28;


	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls



	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ofm_addr=beta;
	weights_addr=gama;
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;


	ifm_addr=out;
	ifm_length=12544;//Tm*R*C
	for (int to=0;to<32;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=out+401408*b+12544*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32

			XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
			//ifm_addr+=ifm_length;
			while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//check if dma is working
		}
	}

	ofm_addr=out;//+3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	wout_addr=outhat;//+3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	output_addr=out;//+3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	ofm_length=401408;//802816;//M*R*C
	output_length=401408;//802816;//M*R*C
	wout_length=401408;//802816;//M*R*C
	for(int b=0;b<bs;b++){

		XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);

		XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)wout_addr,wout_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);

		ofm_addr+=ofm_length;
		wout_addr+=wout_length;
		output_addr+=output_length;

		while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
		//cout<<"OFM in Done!"<<endl;
		while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"Weights out Done!"<<endl;
		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"output in Done!"<<endl;
	}

	output_addr=namuda;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	output_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"output in Done!"<<endl;
	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
		//cout<<"IP done"<<endl;
	cout<<"namuda"<<endl;
	cout<<namuda[0].data<<endl;
	cout<<"outhat"<<endl;
	cout<<outhat[0].data<<endl;


	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"print out"<<outnum<<endl;
	//cout<<timeInterval<<" ";

	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for out"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void maxpoolfp_512_14( int outnum,
	DMA_DATA * ifm,
	DMA_DATA * weights,
	DMA_DATA * out


){
	//forward pool N,M,R,C=512,512,14,14, stride=2,padding=0,kernel=2
	//timer.startTimer();
	tb_statec=0;
	tb_statep=6;
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=0;
	//tb_Poollayerin=0;
	//tb_Poollayerout=2;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=512;
	tb_custom_Tib=1;//[N/2Tn]=3
	//tb_custom_k2=4;
	tb_N=512;
	tb_M=512;
	tb_R=14;
	tb_C=14;
	tb_custom_stride=2;
	tb_padding=0;
	tb_custom_kb=1;
	tb_custom_k=2;
	tb_custom_Tr=1;//12;
	tb_custom_Tc=14;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=28;
	tb_C_in=28;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	ifm_addr=ifm;
	ifm_length=bs*401408;//M*R_in*C_in=28*28*512;
	output_addr=out;
	weights_addr=weights;
	output_length=100352;//M*min(custom_Tr,R-row*custom_Tr)*C=512*14*14;
	weights_length=12544;//r*c*[m/8]=14*14*64

	XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
	//XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	//XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	for(int b=0;b<bs;b++){
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
		output_addr+=output_length;
		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"Output in Done!"<<endl;
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
		weights_addr+=weights_length;
		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"Output in Done!"<<endl;
	}


	//while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
	//cout<<"IFM in Done!"<<endl;

	//while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;

	//while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"Output in Done!"<<endl;


	//while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"OFM in Done!"<<endl;


	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"print out"<<outnum<<endl;
	//cout<<timeInterval<<" ";

	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for out"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void convfp_512_512_14( int outnum,
	DMA_DATA * ifm,
	DMA_DATA * weights,
	DMA_DATA * outhat,
	DMA_DATA * gama,
	DMA_DATA * beta,
	DMA_DATA * namuda,
	DMA_DATA * out

){
	//forward conv11,12,13 N,M,R,C=512,512,14,14, stride=1,padding=1,kernel=3
	//timer.startTimer();
	tb_statec=6;
	tb_statep=0;
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=1;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=512;
	tb_custom_Tib=16;//[N/2Tn]=2
	//tb_custom_k2=25;
	tb_N=512;
	tb_M=64;
	tb_R=14;
	tb_C=14;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=14;//24;
	tb_custom_Tc=14;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=14;
	tb_C_in=14;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	weights_addr=weights;
	weights_length=2359296;//M*N*tb_custom_k*tb_custom_k=512*512*3*3

	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);

	for (int toM=0;toM<8;toM++){
		ifm_addr=ifm;
		output_length=12544;//tb_M*R*C=14*14*64;
		output_addr=out+toM*12544;
		for(int b=0;b<bs;b++){
			XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
			for (int to=0;to<tb_M;to+=Tm){//repeate [M/Tm]
				ifm_length=100352;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
										//*((C-1)*tb_custom_stride+tb_custom_k)=512*14*14;
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));
				//cout<<"IFM in Done!"<<endl;
			}
			ifm_addr+=100352;//N*R_in*C_in=512*14*14
			output_addr+=100352;
			while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
			//cout<<"output in Done!"<<endl;
		}
	}

	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	//while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;

	//	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//	//cout<<"output in Done!"<<endl;

	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

//forward bn11,12,13+relu N,M,R,C=512,512,14,14, stride=1,padding=1,kernel=3
	tb_statec=0;
	tb_statep=0;
	tb_stateb=7;
//	tb_Relulayerin=0;
//	tb_Relulayerout=1;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=512;
	tb_custom_Tib=1;//[N/2Tn]=1
	//tb_custom_k2=25;
	tb_N=512;
	tb_M=512;
	tb_R=14;
	tb_C=14;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=9;//24;
	tb_custom_Tc=224;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=14;
	tb_C_in=14;


	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls



	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ofm_addr=beta;
	weights_addr=gama;
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;
	//ifm_addr=0;
	//ofm_addr=0;
	ifm_addr=out;
	ifm_length=3136;//Tm*R*C
	for (int to=0;to<32;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=out+100352*b+3136*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32

			XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
			//ifm_addr+=ifm_length;
			while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//check if dma is working
		}
	}

	ofm_addr=out;//+3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	wout_addr=outhat;//+3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	output_addr=out;//+3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	ofm_length=100352;//802816;//M*R*C
	output_length=100352;//802816;//M*R*C
	wout_length=100352;//802816;//M*R*C
	for(int b=0;b<bs;b++){

		XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);

		XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)wout_addr,wout_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);

		ofm_addr+=ofm_length;
		wout_addr+=wout_length;
		output_addr+=output_length;

		while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
		//cout<<"OFM in Done!"<<endl;
		while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"Weights out Done!"<<endl;
		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"output in Done!"<<endl;
	}


	output_addr=namuda;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
	output_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"output in Done!"<<endl;
	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
		//cout<<"IP done"<<endl;

	cout<<"namuda"<<endl;
	cout<<namuda[0].data<<endl;
	cout<<"outhat"<<endl;
	cout<<outhat[0].data<<endl;

	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"print out"<<outnum<<endl;
	//cout<<timeInterval<<" ";

	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for out"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void maxpoolfp_512_7( int outnum,
	DMA_DATA * ifm,
	DMA_DATA * weights,
	DMA_DATA * out

){
	//forward pool N,M,R,C=512,512,7,7, stride=2,padding=0,kernel=2
	//timer.startTimer();
	tb_statec=0;
	tb_statep=6;
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=0;
	//tb_Poollayerin=0;
	//tb_Poollayerout=2;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=512;
	tb_custom_Tib=1;//[N/2Tn]=3
	//tb_custom_k2=4;
	tb_N=512;
	tb_M=512;
	tb_R=7;
	tb_C=7;
	tb_custom_stride=2;
	tb_padding=0;
	tb_custom_kb=1;
	tb_custom_k=2;
	tb_custom_Tr=1;//12;
	tb_custom_Tc=7;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=14;
	tb_C_in=14;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	ifm_addr=ifm;
	ifm_length=bs*100352;//M*R_in*C_in=14*14*512;
	output_addr=out;
	weights_addr=weights;
	output_length=25088;//M*min(custom_Tr,R-row*custom_Tr)*C=512*7*7;
	weights_length=3136;//r*c*[m/8]=7*7*64

	XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
	//XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	//XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	for(int b=0;b<bs;b++){
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
		output_addr+=output_length;
		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"Output in Done!"<<endl;
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
		weights_addr+=weights_length;
		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"Output in Done!"<<endl;
	}


	//while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"IFM in Done!"<<endl;

	//while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;

	//while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"Output in Done!"<<endl;


	//while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"OFM in Done!"<<endl;


	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"print out"<<outnum<<endl;
	//cout<<timeInterval<<" ";

	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for out"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void avgpoolfp_512_1( int outnum,
		DMA_DATA * ifm,
		DMA_DATA * out

){
	//forward avgpool N,M,R,C=512,512,1,1, stride=1,padding=0,kernel=6
	//timer.startTimer();
	tb_statec=0;
	tb_statep=14;//view=8+6
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=0;
	//tb_Poollayerin=0;
	//tb_Poollayerout=2;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=512;
	tb_custom_Tib=1;//[N/2Tn]=6
	//tb_custom_k2=4;
	tb_N=512;
	tb_M=512;
	tb_R=1;
	tb_C=1;
	tb_custom_stride=1;
	tb_padding=0;
	tb_custom_kb=1;
	tb_custom_k=7;
	tb_custom_Tr=1;//12;
	tb_custom_Tc=6;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=7;
	tb_C_in=7;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	ifm_addr=ifm;
	ifm_length=bs*25088;//M*R_in*C_in=512*7*7;
	output_addr=out;
	output_length=512;//bs*512;//M*min(custom_Tr,R-row*custom_Tr)*C=512*1*1;

	XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip

	for(int b=0;b<bs;b++){
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
		output_addr+=output_length;
		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"Output in Done!"<<endl;
	}
	while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"IFM in Done!"<<endl;


	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"print out"<<outnum<<endl;
	//cout<<timeInterval<<" ";

	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for out"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void convfp_512_1000_1( int outnum,
	DMA_DATA * ifm,
	DMA_DATA * weights,
	DMA_DATA * out

){
	//forward fc1 N,M,R,C=512,1000,1,1, stride=1,padding=0,kernel=1
	//timer.startTimer();
	tb_statec=6;
	tb_statep=0;
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=6;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=1000;
	tb_custom_Tib=2;//for fc:[N/2Tn/K2]=1*1*16/9=2
	//tb_custom_k2=1;
	tb_N=512;
	//tb_M=12;
	tb_M=512;
	tb_R=1;
	tb_C=1;
	tb_custom_stride=1;
	tb_padding=0;
	tb_custom_kb=3;
	tb_custom_k=1;
	tb_custom_Tr=1;
	tb_custom_Tc=1;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=1;
	tb_C_in=1;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	weights_addr=weights;
	ifm_length=512;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
						//*((C-1)*tb_custom_stride+tb_custom_k)=9216;
	weights_length=512000;//M*N*tb_custom_k*tb_custom_k=512*1000*1*1

	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);

	for (int toM=0;toM<1;toM++){
		ifm_addr=ifm;
		output_length=512;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=48;
		output_addr=out+toM*512;
		for(int b=0;b<bs;b++){
			XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
			for (int to=0;to<tb_M;to+=Tm){//repeate[M/Tm]
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));
				//cout<<"IFM in Done!"<<endl;
			}
			ifm_addr+=ifm_length;
			output_addr+=1000;
			while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
			//cout<<"output in Done!"<<endl;
		}
	}
	for (int toM=1;toM<2;toM++){
		ifm_addr=ifm;
		output_length=488;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=40;
		output_addr=out+toM*512;
		for(int b=0;b<bs;b++){
			XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
			for (int to=0;to<488;to+=Tm){//repeate[(tb_M_in-toM*M)/Tm]
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));
				//cout<<"IFM in Done!"<<endl;
			}
			ifm_addr+=ifm_length;
			output_addr+=1000;
			while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
			//cout<<"output in Done!"<<endl;
		}
	}


	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	//while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;

	//	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//	//cout<<"output in Done!"<<endl;

	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;
	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"print out"<<outnum<<endl;
	//cout<<timeInterval<<" ";

	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for out"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		//cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<1;r++){
			for (int c=0;c<1;c++){
				for (int too=0;too<Tm;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}

		cout <<" last channel tile  " <<endl;
		//cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<1;r++){
			for (int c=0;c<1;c++){
				for (int too=0;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}

		cout<<""<<endl;
	}
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void convbp_512_1000_1( int outnum,
	DMA_DATA * ifm,
	DMA_DATA * weights,
	DMA_DATA * out


){
	//backward fc1 N,M,R,C=1000,512,1,1, stride=1,padding=0,kernel=1
	//timer.startTimer();
	tb_statec=4;
	tb_statep=0;
	tb_stateb=0;
	//tb_Relulayerin=5;
	//tb_Relulayerout=0;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=512;
	tb_custom_Tib=4;////for fc:[N/2Tn/K2]=1*1*32/9=4
	//tb_custom_k2=1;
	tb_N=1000;
	//tb_M=12;
	tb_M=256;
	tb_R=1;
	tb_C=1;
	tb_custom_stride=1;
	tb_padding=0;
	tb_custom_kb=3;
	tb_custom_k=1;
	tb_custom_Tr=1;
	tb_custom_Tc=1;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=1;
	tb_C_in=1;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	//ifm_addr=0;
	ifm_length=1000;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
						//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=1000*1*1;
	weights_addr=weights;

	for (int toM=0;toM<2;toM++){
		ifm_addr=ifm;
		output_length=256;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=256*1*1;
		output_addr=out+toM*256;
		weights_addr=weights+4096*toM;//
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
		for(int to=0;to<tb_M;to+=Tm){
			XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
			if(to==0){
				for (int ti=0; ti<62; ti++){
					weights_length=4096;//16*Tm*min(Tn,N-Tn*ti)*tb_custom_k*tb_custom_k
										//=256*16*1*1
					XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
					weights_addr+=8192;//Tm*N'(M)*tb_custom_k*tb_custom_k=16*512*1*1;
					while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
					//cout<<"Weights in Done!"<<endl;
				}
				for (int ti=62; ti<63; ti++){
					weights_length=2048;//16*Tm*min(Tn,N-Tn*ti)*tb_custom_k*tb_custom_k
													//=256*8*1*1
					weights_addr=weights+507904+2048*toM;//ti*Tn*M_in+toM*16*Tm*8=62*16*512+toM*16*16*8
					XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
					while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
					//cout<<"Weights in Done!"<<endl;
				}
			}
			while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
					//cout<<"IFM in Done!"<<endl;
		}
		ifm_addr+=ifm_length;
		output_addr+=512;
		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
			//cout<<"Output in Done!"<<endl;
		for(int b=1;b<bs;b++){
			XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
			for(int to=0;to<tb_M;to+=Tm){
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
				//cout<<"IFM in Done!"<<endl;
			}
			ifm_addr+=ifm_length;
			output_addr+=512;
			while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		}
	}

	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"print loss"<<outnum<<endl;
	//cout<<timeInterval<<" ";

	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for loss"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		//cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<1;r++){
			for (int c=0;c<1;c++){
				for (int too=0;too<Tm;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}

		cout <<" last channel tile  " <<endl;
		//cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<1;r++){
			for (int c=0;c<1;c++){
				for (int too=0;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}

		cout<<""<<endl;
	}
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void convwu_512_1000_1( int outnum,
	DMA_DATA * ifm,
	DMA_DATA * weights,
	DMA_DATA * ofm

){

	//update fc1 weights N,M,R,C=512,1000,1,1, stride=1,padding=0,kernel=1
	//timer.startTimer();
	tb_statec=2;
	tb_statep=0;
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=0;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=1000;
	tb_custom_Tib=2;//for fc:[N/2Tn/K2]=1*1*16/9=2
	//tb_custom_k2=1;
	tb_N=512;
	//tb_M=12;
	tb_M=512;
	tb_R=1;
	tb_C=1;
	tb_custom_stride=1;
	tb_padding=0;
	tb_custom_kb=3;
	tb_custom_k=1;
	tb_custom_Tr=1;
	tb_custom_Tc=1;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=1;
	tb_C_in=1;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls


	ifm_length=512;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
				//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=512*1*1;
	ofm_addr=ofm;
	weights_addr=weights;
	weights_length=512000;//M*N*tb_custom_k*tb_custom_k
			//=1000*512*1*1
	//print output
	output_addr=weights;
	output_length=512000;//M*N*tb_custom_k*tb_custom_k
	//=1000*9216*1*1

	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);

	for (int toM=0;toM<1;toM++){
		ifm_addr=ifm;
		ofm_length=tb_M;
		ofm_addr=ofm+toM*tb_M;
		for(int b=0;b<bs;b++){
			XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
			for (int to=0;to<tb_M;to+=Tm){//repeate[M/Tm]
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
				//cout<<"IFM in Done!"<<endl;
			}
			ifm_addr+=ifm_length;
			ofm_addr+=tb_M_in;
			while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
			//cout<<"OFM in Done!"<<endl;
		}
	}
	for (int toM=1;toM<2;toM++){
		ifm_addr=ifm;
		ofm_length=488;
		ofm_addr=ofm+toM*tb_M;
		for(int b=0;b<bs;b++){
			XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
			for (int to=0;to<488;to+=Tm){//repeate[M/Tm]
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
				//cout<<"IFM in Done!"<<endl;
			}
			ifm_addr+=ifm_length;
			ofm_addr+=tb_M_in;
			while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
			//cout<<"OFM in Done!"<<endl;
		}
	}

	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"Output in Done!"<<endl;

	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	//cout<<"fc1 tim"<<endl;
	//cout<<timeInterval<<" ";
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
	cout<<"fc1:"<<endl;


	for (int n=0; n<16;n++){
		cout << weights[n].data << ";";
	}
	cout << "last 16 channels" <<endl;
	for (int n=tb_M_in*tb_N*tb_custom_k*tb_custom_k-16; n<tb_M_in*tb_N*tb_custom_k*tb_custom_k;n++){
		cout << weights[n].data << ";";
	}
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);


}

void avgpoolbp_512_1( int outnum,
	DMA_DATA * ifm,
	DMA_DATA * out

){
	//backward avgpool N,M,R,C=512,512,1,1, stride=1,padding=0,kernel=7
	//timer.startTimer();
	tb_statec=0;
	tb_statep=12;//view=8+4
	tb_stateb=0;
//	tb_Relulayerin=3;
//	tb_Relulayerout=0;
	//tb_Poollayerin=4;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=512;
	tb_custom_Tib=1;//[N/2Tn]=6
	//tb_custom_k2=25;
	tb_N=512;
	tb_M=512;
	tb_R=1;
	tb_C=1;
	tb_custom_stride=1;
	tb_padding=0;
	tb_custom_kb=1;
	tb_custom_k=7;
	tb_custom_Tr=1;
	tb_custom_Tc=6;
//	tb_viewTrc=1;
//	tb_viewTm=1;
	tb_R_in=7;
	tb_C_in=7;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	ifm_addr=ifm;
	ifm_length=bs*512;//M*tb_custom_Tr*tb_custom_Tc=512*1*1;
	output_addr=out;
	output_length=bs*25088;//M*R_in*C_in=512*7*7;

	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip

	//while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
	//cout<<"IFM in Done!"<<endl;

//	while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
//	//cout<<"OFM in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"Output in Done!"<<endl;

	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"print loss"<<outnum<<endl;
	//cout<<timeInterval<<" ";

	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for loss"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void maxpoolbp_512_7( int outnum,
	DMA_DATA * ifm,
	DMA_DATA * weights,
	DMA_DATA * ofm,
	DMA_DATA * out

){
	//backward pool+relu N,M,R,C=512,512,7,7, stride=2,padding=0,kernel=2
	//timer.startTimer();
	tb_statec=0;
	tb_statep=5;//view=8+4
	tb_stateb=0;//
//	tb_Relulayerin=3;
//	tb_Relulayerout=0;
	//tb_Poollayerin=4;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=512;
	tb_custom_Tib=1;//[N/2Tn]=6
	//tb_custom_k2=25;
	tb_N=512;
	tb_M=512;
	tb_R=7;
	tb_C=7;
	tb_custom_stride=2;
	tb_padding=0;
	tb_custom_kb=1;
	tb_custom_k=2;
	tb_custom_Tr=1;
	tb_custom_Tc=7;
//	tb_viewTrc=1;
//	tb_viewTm=1;
	tb_R_in=14;
	tb_C_in=14;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	ifm_addr=ifm;
	ifm_length=bs*25088;//M*tb_custom_Tr*tb_custom_Tc=512*7*7;
	weights_addr=weights;
	weights_length=bs*3136;//r*c*[m/4]=7*7*64
	ofm_addr=ofm;
	ofm_length=bs*100352;//M*R_in*C_in=512*14*14;
	output_addr=out;
	output_length=bs*100352;//M*R_in*C_in=512*14*14;

	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip

	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
	//cout<<"IFM in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"Output in Done!"<<endl;

	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"print loss"<<outnum<<endl;
	//cout<<timeInterval<<" ";

	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for loss"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void convbp_512_512_14relu( int outnum,
	//int	reluornot,//conv11=4;conv12,13=5
	DMA_DATA * ifm,
	DMA_DATA * weights,
	DMA_DATA * outhat,
	DMA_DATA * gama,
	DMA_DATA * beta,
	DMA_DATA * namuda,
	DMA_DATA * ofm,
	DMA_DATA * out

){

//backward bn12,13 N,M,R,C=512,512,14,14, stride=1,padding=1,kernel=3
	tb_statec=0;
	tb_statep=0;
	tb_stateb=2;
//	tb_Relulayerin=0;
//	tb_Relulayerout=1;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=512;
	tb_custom_Tib=1;//[N/2Tn]=1
	//tb_custom_k2=25;
	tb_N=512;
	tb_M=512;
	tb_R=14;
	tb_C=14;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=9;//24;
	tb_custom_Tc=224;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=14;
	tb_C_in=14;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ifm_addr=namuda;
	ofm_addr=beta;
	weights_addr=gama;
	ifm_length=tb_M;//M
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip

	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
	//cout<<"IFM in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;
	//ifm_addr=0;
	//ofm_addr=0;
	ofm_length=3136;//Tm*R*C
	ifm_length=3136;//Tm*R*C
	for (int to=0;to<32;to++){//repeate [M/Tm]for (int to=0;to<32;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=ifm+100352*b+3136*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			ofm_addr=outhat+100352*b+3136*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
			XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
			while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
			//cout<<"IFM in Done!"<<endl;

			while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
			//cout<<"OFM in Done!"<<endl;
		}
	}//=6*3*5*5
	ifm_addr=ifm;
	ofm_addr=outhat;
	output_addr=ifm;
	ofm_length=100352;//M*R*C
	ifm_length=100352;//M*R*C
	output_length=100352;//M*R*C
	for(int b=0;b<bs;b++){
		//ifm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		//ofm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
		XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
		XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip

		ifm_addr+=ifm_length;
		ofm_addr+=ofm_length;
		output_addr+=output_length;
		while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
		//cout<<"IFM in Done!"<<endl;

		while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
		//cout<<"OFM in Done!"<<endl;

		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"Output in Done!"<<endl;
	}


	wout_addr=gama;
	wout_length=tb_M;//M
	output_addr=beta;
	output_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)wout_addr,wout_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"Weights out Done!"<<endl;
	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"output in Done!"<<endl;
//	output_addr=gama;
//	output_length=tb_M;//M
//	//XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)wout_addr,wout_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
//	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
//	//while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DEVICE_TO_DMA));
//	//cout<<"Weights out Done!"<<endl;
//	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
//	//cout<<"output in Done!"<<endl;
	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;


	cout<<"gama first:"<<endl;
	for(int i=0;i<10;i++ ){
		cout<<gama[i].data<< ";";
	}
	cout<<"gama last:"<<endl;
	for(int i=tb_M-10;i<tb_M;i++ ){
		cout<<gama[i].data<< ";";
	}

	cout<<"beta first:"<<endl;
	cout<<beta[0].data<< ";";
	cout<<beta[1].data<< ";";
	cout<<beta[2].data<< ";";
	cout<<"beta last:"<<endl;
	cout<<beta[tb_M-3].data<< ";";
	cout<<beta[tb_M-2].data<< ";";
	cout<<beta[tb_M-1].data<<endl;



	//backward 12,13+relu N,M,R,C=512,512,14,14, stride=1,padding=1,kernel=3
	//timer.startTimer();
	tb_statec=5;
	tb_statep=0;
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=0;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=512;
	tb_custom_Tib=16;//[N/2Tn]=6
	//tb_custom_k2=25;
	tb_N=512;
	tb_M=64;
	tb_R=14;
	tb_C=14;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=14;
	tb_custom_Tc=14;
	tb_R_in=14;
	tb_C_in=14;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_statep);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	ifm_length=100352;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
				//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=512*14*14;
	ofm_length=12544;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=64*14*14;
	output_length=12544;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=64*14*14;
	for (int toM=0;toM<8;toM++){
		ifm_addr=ifm;
		weights_addr=weights+9216*toM;//
		ofm_addr=ofm+toM*12544;
		output_addr=out+toM*12544;
		XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);

		for(int to=0;to<tb_M;to+=Tm){
			XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
			if(to==0){
				for (int ti=0; ti<tb_N; ti+=Tn){
					weights_length=9216;//4*Tm*min(Tn,N-Tn*ti)*tb_custom_k*tb_custom_k
													//=64*16*3*3
					XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
					weights_addr+=73728;//Tm*N'(M)*tb_custom_k*tb_custom_k=16*512*3*3;
					while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
					//cout<<"Weights in Done!"<<endl;
				}
			}
			while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
					//cout<<"IFM in Done!"<<endl;
		}
		ifm_addr+=ifm_length;
		ofm_addr+=100352;
		output_addr+=100352;
		while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
		//cout<<"OFM in Done!"<<endl;
		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"Output in Done!"<<endl;
		for(int b=1;b<bs;b++){
			XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
			XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
			for(int to=0;to<tb_M;to+=Tm){
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
				//cout<<"IFM in Done!"<<endl;
			}
			ifm_addr+=ifm_length;
			ofm_addr+=100352;
			output_addr+=100352;
			while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
			//cout<<"OFM in Done!"<<endl;
			while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
			//cout<<"Output in Done!"<<endl;

		}
	}

	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"print loss"<<outnum<<endl;
	//cout<<timeInterval<<" ";

	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for loss"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}

		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}


void convbp_512_512_14( int outnum,
	//int	reluornot,//conv11=4;conv12,13=5
	DMA_DATA * ifm,
	DMA_DATA * weights,
	DMA_DATA * outhat,
	DMA_DATA * gama,
	DMA_DATA * beta,
	DMA_DATA * namuda,
	//DMA_DATA * ofm,
	DMA_DATA * out

){

//backward bn11 N,M,R,C=512,512,14,14, stride=1,padding=1,kernel=3
	tb_statec=0;
	tb_statep=0;
	tb_stateb=2;
//	tb_Relulayerin=0;
//	tb_Relulayerout=1;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=512;
	tb_custom_Tib=1;//[N/2Tn]=1
	//tb_custom_k2=25;
	tb_N=512;
	tb_M=512;
	tb_R=14;
	tb_C=14;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=9;//24;
	tb_custom_Tc=224;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=14;
	tb_C_in=14;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ifm_addr=namuda;
	ofm_addr=beta;
	weights_addr=gama;
	ifm_length=tb_M;//M
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip

	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
	//cout<<"IFM in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;
	//ifm_addr=0;
	//ofm_addr=0;
	ofm_length=3136;//Tm*R*C
	ifm_length=3136;//Tm*R*C
	for (int to=0;to<32;to++){//repeate [M/Tm]for (int to=0;to<32;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=ifm+100352*b+3136*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			ofm_addr=outhat+100352*b+3136*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
			XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
			while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
			//cout<<"IFM in Done!"<<endl;

			while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
			//cout<<"OFM in Done!"<<endl;
		}
	}//=6*3*5*5
	ifm_addr=ifm;
	ofm_addr=outhat;
	output_addr=ifm;
	ofm_length=100352;//M*R*C
	ifm_length=100352;//M*R*C
	output_length=100352;//M*R*C
	for(int b=0;b<bs;b++){
		//ifm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		//ofm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
		XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
		ifm_addr+=ifm_length;
		ofm_addr+=ofm_length;
		output_addr+=output_length;
		while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
		//cout<<"IFM in Done!"<<endl;

		while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
		//cout<<"OFM in Done!"<<endl;

		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"Output in Done!"<<endl;
	}


	wout_addr=gama;
	wout_length=tb_M;//M
	output_addr=beta;
	output_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)wout_addr,wout_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"Weights out Done!"<<endl;
	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"output in Done!"<<endl;
//	output_addr=gama;
//	output_length=tb_M;//M
//	//XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)wout_addr,wout_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
//	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
//	//while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DEVICE_TO_DMA));
//	//cout<<"Weights out Done!"<<endl;
//	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
//	//cout<<"output in Done!"<<endl;
	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;


	cout<<"gama first:"<<endl;
	for(int i=0;i<10;i++ ){
		cout<<gama[i].data<< ";";
	}
	cout<<"gama last:"<<endl;
	for(int i=tb_M-10;i<tb_M;i++ ){
		cout<<gama[i].data<< ";";
	}

	cout<<"beta first:"<<endl;
	cout<<beta[0].data<< ";";
	cout<<beta[1].data<< ";";
	cout<<beta[2].data<< ";";
	cout<<"beta last:"<<endl;
	cout<<beta[tb_M-3].data<< ";";
	cout<<beta[tb_M-2].data<< ";";
	cout<<beta[tb_M-1].data<<endl;



//backward 11 N,M,R,C=512,512,14,14, stride=1,padding=1,kernel=3
	//timer.startTimer();
	tb_statec=4;
	tb_statep=0;
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=0;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=512;
	tb_custom_Tib=16;//[N/2Tn]=6
	//tb_custom_k2=25;
	tb_N=512;
	tb_M=64;
	tb_R=14;
	tb_C=14;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=14;
	tb_custom_Tc=14;
	tb_R_in=14;
	tb_C_in=14;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	ifm_length=100352;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
				//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=512*14*14;
	//ofm_length=12544;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=64*14*14;
	output_length=12544;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=64*14*14;
	for (int toM=0;toM<8;toM++){
		ifm_addr=ifm;
		weights_addr=weights+9216*toM;//
		//ofm_addr=ofm+toM*12544;
		output_addr=out+toM*12544;
		//XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);

		for(int to=0;to<tb_M;to+=Tm){
			XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
			if(to==0){
				for (int ti=0; ti<tb_N; ti+=Tn){
					weights_length=9216;//4*Tm*min(Tn,N-Tn*ti)*tb_custom_k*tb_custom_k
													//=64*16*3*3
					XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
					weights_addr+=73728;//Tm*N'(M)*tb_custom_k*tb_custom_k=16*512*3*3;
					while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
					//cout<<"Weights in Done!"<<endl;
				}
			}
			while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
					//cout<<"IFM in Done!"<<endl;
		}
		ifm_addr+=ifm_length;
		//ofm_addr+=100352;
		output_addr+=100352;
		//while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
		//cout<<"OFM in Done!"<<endl;
		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"Output in Done!"<<endl;
		for(int b=1;b<bs;b++){
			//XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
			XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
			for(int to=0;to<tb_M;to+=Tm){
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
				//cout<<"IFM in Done!"<<endl;
			}
			ifm_addr+=ifm_length;
			output_addr+=100352;
			//while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
			//cout<<"OFM in Done!"<<endl;
			while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
			//cout<<"Output in Done!"<<endl;

		}
	}

	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"print loss"<<outnum<<endl;
	//cout<<timeInterval<<" ";

	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for loss"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}

		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}


void convwu_512_512_14( int outnum,
		DMA_DATA * ifm,
		DMA_DATA * weights,
		DMA_DATA * ofm,
		DMA_DATA weights_print[512][512][3][3]

){
	//update conv11,12,13 weights N,M,R,C=512,512,14,14, stride=1,padding=1,kernel=3
	//timer.startTimer();
	tb_statec=2;
	tb_statep=0;
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=0;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=512;
	tb_custom_Tib=16;//[N/2Tn]=6
	//tb_custom_k2=25;
	tb_N=512;
	tb_M=64;
	tb_R=14;
	tb_C=14;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=14;
	tb_custom_Tc=14;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=14;
	tb_C_in=14;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	ifm_length=100352;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
					//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=512*14*14;
	weights_addr=weights;
	weights_length=2359296;//M*N*tb_custom_k*tb_custom_k
				//=512*512*3*3
	output_addr=weights;
	output_length=2359296;//M*N*tb_custom_k*tb_custom_k
	//=512*512*3*3

	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);

	for (int toM=0;toM<8;toM++){
		ifm_addr=ifm;
		ofm_length=12544;//tb_M*tb_R*tb_C=64*14*14;
		ofm_addr=ofm+toM*12544;
		for(int b=0;b<bs;b++){
			XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
			for (int to=0;to<tb_M;to+=Tm){//repeate[M/Tm]
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
				//cout<<"IFM in Done!"<<endl;
			}
			ifm_addr+=ifm_length;
			ofm_addr+=100352;//tb_M_in*tb_R*tb_C=512*14*14;
			while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
			//cout<<"OFM in Done!"<<endl;
		}
	}

	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"Output in Done!"<<endl;

	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"conv"<<outnum<<endl;
	//cout<<timeInterval<<" ";
	//printf("Elapsed Time (seconds) %f\n",timeInterval);

	bf_index=0;
	for(int to=0;to<tb_M_in;to+=Tm){
		for(int ti=0;ti<tb_N;ti+=Tn){
			for (int too=0;too< Tm;too+=4){
				for(int tii=0;tii<Tn;tii++){
					if((too+to)<tb_M_in && (tii+ti)<tb_N){
						for (int r=0;r<tb_custom_k;r++){
							for (int c=0;c<tb_custom_k;c++){
								weights_print[too+to][tii+ti][r][c] = weights[bf_index] ;//output.data;
								weights_print[too+to+1][tii+ti][r][c] = weights[bf_index+1] ;//output.data;
								weights_print[too+to+2][tii+ti][r][c] = weights[bf_index+2] ;//output.data;
								weights_print[too+to+3][tii+ti][r][c] = weights[bf_index+3] ;//output.data;
								bf_index+=4;
							}
						}
					}
				}
			}
		}
	}
	cout<<"1st 3OCs 1st 3ICs" <<endl;
	for (int m=0; m<3;m++){
		for (int n =0; n<3;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"1st 3OCs last 3ICs" <<endl;
	for (int m=0; m<3;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"" <<endl;
	cout<<"last 3OCs 1st 3ICs" <<endl;
	for (int m=tb_M_in-3; m<tb_M_in;m++){
		for (int n =0; n<3;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"last 3OCs last 3ICs" <<endl;
	for (int m=tb_M_in-3; m<tb_M_in;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<""<<endl;
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void maxpoolbp_512_14( int outnum,
		DMA_DATA * ifm,
		DMA_DATA * weights,
		DMA_DATA * ofm,
		DMA_DATA * out

){
	//backward pool+relu N,M,R,C=512,512,14,14, stride=2,padding=0,kernel=2
	//timer.startTimer();
	tb_statec=0;
	tb_statep=5;//view=8+4
	tb_stateb=0;//view=8+4
//	tb_Relulayerin=3;
//	tb_Relulayerout=0;
	//tb_Poollayerin=4;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=512;
	tb_custom_Tib=1;//[N/2Tn]=6
	//tb_custom_k2=25;
	tb_N=512;
	tb_M=512;
	tb_R=14;
	tb_C=14;
	tb_custom_stride=2;
	tb_padding=0;
	tb_custom_kb=1;
	tb_custom_k=2;
	tb_custom_Tr=1;
	tb_custom_Tc=14;
//	tb_viewTrc=1;
//	tb_viewTm=1;
	tb_R_in=28;
	tb_C_in=28;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	ifm_addr=ifm;
	ifm_length=bs*100352;//M*tb_custom_Tr*tb_custom_Tc=512*14*14;
	weights_addr=weights;
	weights_length=bs*12544;//r*c*[m/8]=14*14*64
	ofm_addr=ofm;
	ofm_length=bs*401408;//M*R_in*C_in=512*28*28;
	output_addr=out;
	output_length=bs*401408;//M*R_in*C_in=512*28*28;

	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip

	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
	//cout<<"IFM in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"Output in Done!"<<endl;

	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"print loss"<<outnum<<endl;
	//cout<<timeInterval<<" ";

	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for loss"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void convbp_512_512_28( int outnum,
	DMA_DATA * ifm,
	DMA_DATA * weights,
	DMA_DATA * outhat,
	DMA_DATA * gama,
	DMA_DATA * beta,
	DMA_DATA * namuda,
	DMA_DATA * ofm,
	DMA_DATA * out

){
//backward bn9,10 N,M,R,C=512,512,28,28, stride=1,padding=1,kernel=3
	tb_statec=0;
	tb_statep=0;
	tb_stateb=2;
//	tb_Relulayerin=0;
//	tb_Relulayerout=1;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=512;
	tb_custom_Tib=1;//[N/2Tn]=1
	//tb_custom_k2=25;
	tb_N=512;
	tb_M=512;
	tb_R=28;
	tb_C=28;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=9;//24;
	tb_custom_Tc=224;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=28;
	tb_C_in=28;


	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ifm_addr=namuda;
	ofm_addr=beta;
	weights_addr=gama;
	ifm_length=tb_M;//M
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip

	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
	//cout<<"IFM in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;

	ofm_length=12544;//Tm*R*C
	ifm_length=12544;//Tm*R*C
	output_length=12544;//Tm*R*C

	for (int to=0;to<32;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=ifm+401408*b+12544*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			ofm_addr=outhat+401408*b+12544*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
			XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
			while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
			//cout<<"IFM in Done!"<<endl;

			while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
			//cout<<"OFM in Done!"<<endl;
		}
	}//=6*3*5*5
	ifm_addr=ifm;
	ofm_addr=outhat;
	output_addr=ifm;
	ofm_length=401408;//M*R*C
	ifm_length=401408;//M*R*C
	output_length=401408;//M*R*C
	for(int b=0;b<bs;b++){
		//ifm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		//ofm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
		XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
		ifm_addr+=ifm_length;
		ofm_addr+=ofm_length;
		output_addr+=output_length;
		while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
		//cout<<"IFM in Done!"<<endl;

		while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
		//cout<<"OFM in Done!"<<endl;

		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"Output in Done!"<<endl;

	}

	wout_addr=gama;
	wout_length=tb_M;//M
	output_addr=beta;
	output_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)wout_addr,wout_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"Weights out Done!"<<endl;
	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"output in Done!"<<endl;
//	output_addr=gama;
//	output_length=tb_M;//M
//	//XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)wout_addr,wout_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
//	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
//	//while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DEVICE_TO_DMA));
//	//cout<<"Weights out Done!"<<endl;
//	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
//	//cout<<"output in Done!"<<endl;
	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;


	cout<<"gama first:"<<endl;
	for(int i=0;i<10;i++ ){
		cout<<gama[i].data<< ";";
	}
	cout<<"gama last:"<<endl;
	for(int i=tb_M-10;i<tb_M;i++ ){
		cout<<gama[i].data<< ";";
	}

	cout<<"beta first:"<<endl;
	cout<<beta[0].data<< ";";
	cout<<beta[1].data<< ";";
	cout<<beta[2].data<< ";";
	cout<<"beta last:"<<endl;
	cout<<beta[tb_M-3].data<< ";";
	cout<<beta[tb_M-2].data<< ";";
	cout<<beta[tb_M-1].data<<endl;

//backward conv9,10+relu N,M,R,C=512,512,28,28, stride=1,padding=1,kernel=3
	//timer.startTimer();
	tb_statec=5;
	tb_statep=0;
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=0;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=512;
	tb_custom_Tib=16;//[N/2Tn]=6
	//tb_custom_k2=25;
	tb_N=512;
	tb_M=64;
	tb_R=28;
	tb_C=28;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=28;
	tb_custom_Tc=28;
	tb_R_in=28;
	tb_C_in=28;
	//tb_flag=0;


	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	ifm_length=401408;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
				//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=512*28*28;
	ofm_length=50176;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=64*28*28;
	output_length=50176;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=64*28*28;
	//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=512*28*28;
	for (int toM=0;toM<8;toM++){
		ifm_addr=ifm;
		weights_addr=weights+9216*toM;//
		ofm_addr=ofm+toM*50176;

		output_addr=out+toM*50176;
		XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
		for(int to=0;to<tb_M;to+=Tm){
			XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
			if(to==0){
				for (int ti=0; ti<tb_N; ti+=Tn){
					weights_length=9216;//4*Tm*min(Tn,N-Tn*ti)*tb_custom_k*tb_custom_k
													//=64*16*3*3
					XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
					weights_addr+=73728;//Tm*N'(M)*tb_custom_k*tb_custom_k=16*512*3*3;
					while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
					//cout<<"Weights in Done!"<<endl;
				}
			}
			while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
			//cout<<"IFM in Done!"<<endl;
		}
		ifm_addr+=ifm_length;
		ofm_addr+=401408;
		output_addr+=401408;
		while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
		//cout<<"OFM in Done!"<<endl;
		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"Output in Done!"<<endl;
		for(int b=1;b<bs;b++){
			XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
			XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
			for(int to=0;to<tb_M;to+=Tm){
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
				//cout<<"IFM in Done!"<<endl;
			}
			ifm_addr+=ifm_length;
			ofm_addr+=401408;
			output_addr+=401408;
			while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
			//cout<<"OFM in Done!"<<endl;
			while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
			//cout<<"Output in Done!"<<endl;
		}
	}

	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"print loss"<<outnum<<endl;
	//cout<<timeInterval<<" ";

	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for loss"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void convwu_512_512_28( int outnum,
	DMA_DATA * ifm,
	DMA_DATA * weights,
	DMA_DATA * ofm,
	DMA_DATA weights_print[512][512][3][3]

){
	//update conv9,10 weights N,M,R,C=512,512,28,28, stride=1,padding=1,kernel=3
	//timer.startTimer();
	tb_statec=2;
	tb_statep=0;
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=0;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=512;
	tb_custom_Tib=16;//[N/2Tn]=6
	//tb_custom_k2=25;
	tb_N=512;
	tb_M=64;
	tb_R=28;
	tb_C=28;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=28;
	tb_custom_Tc=28;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=28;
	tb_C_in=28;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	ifm_length=401408;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
					//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=512*28*28;
	weights_addr=weights;
	weights_length=2359296;//M*N*tb_custom_k*tb_custom_k
				//=512*512*3*3
	output_addr=weights;
	output_length=2359296;//M*N*tb_custom_k*tb_custom_k
		//=512*512*3*3
	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);

	for (int toM=0;toM<8;toM++){
		ifm_addr=ifm;
		ofm_length=50176;//tb_M*tb_R*tb_C=64*28*28;
		ofm_addr=ofm+toM*50176;
		for(int b=0;b<bs;b++){
			XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
			for (int to=0;to<tb_M;to+=Tm){//repeate[M/Tm]
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
				//cout<<"IFM in Done!"<<endl;
			}
			ifm_addr+=ifm_length;
			ofm_addr+=401408;//tb_M_in*tb_R*tb_C=512*28*28;
			while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
			//cout<<"OFM in Done!"<<endl;
		}
	}

	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;
	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"Output in Done!"<<endl;
	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"conv"<<outnum<<endl;
	//cout<<timeInterval<<" ";
	//printf("Elapsed Time (seconds) %f\n",timeInterval);

	bf_index=0;
	for(int to=0;to<tb_M_in;to+=Tm){
		for(int ti=0;ti<tb_N;ti+=Tn){
			for (int too=0;too< Tm;too+=4){
				for(int tii=0;tii<Tn;tii++){
					if((too+to)<tb_M_in && (tii+ti)<tb_N){
						for (int r=0;r<tb_custom_k;r++){
							for (int c=0;c<tb_custom_k;c++){
								weights_print[too+to][tii+ti][r][c] = weights[bf_index] ;//output.data;
								weights_print[too+to+1][tii+ti][r][c] = weights[bf_index+1] ;//output.data;
								weights_print[too+to+2][tii+ti][r][c] = weights[bf_index+2] ;//output.data;
								weights_print[too+to+3][tii+ti][r][c] = weights[bf_index+3] ;//output.data;
								bf_index+=4;
							}
						}
					}
				}
			}
		}
	}
	cout<<"1st 3OCs 1st 3ICs" <<endl;
	for (int m=0; m<3;m++){
		for (int n =0; n<3;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"1st 3OCs last 3ICs" <<endl;
	for (int m=0; m<3;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"" <<endl;
	cout<<"last 3OCs 1st 3ICs" <<endl;
	for (int m=tb_M_in-3; m<tb_M_in;m++){
		for (int n =0; n<3;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"last 3OCs last 3ICs" <<endl;
	for (int m=tb_M_in-3; m<tb_M_in;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"" <<endl;
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void convbp_256_512_28( int outnum,
	DMA_DATA * ifm,
	DMA_DATA * weights,
	DMA_DATA * outhat,
	DMA_DATA * gama,
	DMA_DATA * beta,
	DMA_DATA * namuda,
	DMA_DATA * out

){
//backward bn8 N,M,R,C=512,512,28,28, stride=1,padding=1,kernel=3
	tb_statec=0;
	tb_statep=0;
	tb_stateb=2;
//	tb_Relulayerin=0;
//	tb_Relulayerout=1;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=512;
	tb_custom_Tib=1;//[N/2Tn]=1
	//tb_custom_k2=25;
	tb_N=512;
	tb_M=512;
	tb_R=28;
	tb_C=28;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=9;//24;
	tb_custom_Tc=224;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=28;
	tb_C_in=28;


	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ifm_addr=namuda;
	ofm_addr=beta;
	weights_addr=gama;
	ifm_length=tb_M;//M
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip

	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
	//cout<<"IFM in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;
	//ifm_addr=0;
	//ofm_addr=0;
	ofm_length=12544;//Tm*R*C
	ifm_length=12544;//Tm*R*C
	for (int to=0;to<32;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=ifm+401408*b+12544*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			ofm_addr=outhat+401408*b+12544*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
			XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
			while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
			//cout<<"IFM in Done!"<<endl;

			while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
			//cout<<"OFM in Done!"<<endl;
		}
	}//=6*3*5*5
	ifm_addr=ifm;
	ofm_addr=outhat;
	output_addr=ifm;
	ofm_length=401408;//M*R*C
	ifm_length=401408;//M*R*C
	output_length=401408;//M*R*C
	for(int b=0;b<bs;b++){
		//ifm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		//ofm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
		XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
		ifm_addr+=ifm_length;
		ofm_addr+=ofm_length;
		output_addr+=output_length;
		while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
		//cout<<"IFM in Done!"<<endl;

		while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
		//cout<<"OFM in Done!"<<endl;

		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"Output in Done!"<<endl;
	}


	wout_addr=gama;
	wout_length=tb_M;//M
	output_addr=beta;
	output_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)wout_addr,wout_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"Weights out Done!"<<endl;
	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"output in Done!"<<endl;
//	output_addr=gama;
//	output_length=tb_M;//M
//	//XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)wout_addr,wout_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
//	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
//	//while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DEVICE_TO_DMA));
//	//cout<<"Weights out Done!"<<endl;
//	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
//	//cout<<"output in Done!"<<endl;
	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;


	cout<<"gama first:"<<endl;
	for(int i=0;i<10;i++ ){
		cout<<gama[i].data<< ";";
	}
	cout<<"gama last:"<<endl;
	for(int i=tb_M-10;i<tb_M;i++ ){
		cout<<gama[i].data<< ";";
	}

	cout<<"beta first:"<<endl;
	cout<<beta[0].data<< ";";
	cout<<beta[1].data<< ";";
	cout<<beta[2].data<< ";";
	cout<<"beta last:"<<endl;
	cout<<beta[tb_M-3].data<< ";";
	cout<<beta[tb_M-2].data<< ";";
	cout<<beta[tb_M-1].data<<endl;


	//backward conv8 N,M,R,C=512,256,28,28, stride=1,padding=1,kernel=3
	//timer.startTimer();
	tb_statec=4;
	tb_statep=0;
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=0;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=256;
	tb_custom_Tib=16;//[N/2Tn]=6
	//tb_custom_k2=25;
	tb_N=512;
	tb_M=64;
	tb_R=28;
	tb_C=28;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=28;
	tb_custom_Tc=28;
	tb_R_in=28;
	tb_C_in=28;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	ifm_length=401408;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
				//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=512*28*28;

	for (int toM=0;toM<4;toM++){
		ifm_addr=ifm;
		weights_addr=weights+9216*toM;//
		output_length=50176;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=64*28*28;
		output_addr=out+toM*50176;
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
		for(int to=0;to<tb_M;to+=Tm){
			XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
			if(to==0){
				for (int ti=0; ti<tb_N; ti+=Tn){
					weights_length=9216;//4*Tm*min(Tn,N-Tn*ti)*tb_custom_k*tb_custom_k
										//=64*16*3*3
					XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
					weights_addr+=36864;//Tm*N'(M)*tb_custom_k*tb_custom_k=16*256*3*3;
					while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
					//cout<<"Weights in Done!"<<endl;
				}
			}
			while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
			//cout<<"IFM in Done!"<<endl;
		}
		ifm_addr+=ifm_length;
		output_addr+=200704;//M*R*C=256*28*28
		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
			//cout<<"Output in Done!"<<endl;
		for(int b=1;b<bs;b++){
			XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
			for(int to=0;to<tb_M;to+=Tm){
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
				//cout<<"IFM in Done!"<<endl;
			}
			ifm_addr+=ifm_length;
			output_addr+=200704;//M*R*C=256*28*28
			while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		}
	}

	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"print loss"<<outnum<<endl;
	//cout<<timeInterval<<" ";

	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for loss"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}

		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void convwu_256_512_28( int outnum,
	DMA_DATA * ifm,
	DMA_DATA * weights,
	DMA_DATA * ofm,
	DMA_DATA weights_print[512][256][3][3]

){
	//update conv8 weights N,M,R,C=256,512,28,28, stride=1,padding=1,kernel=3
	//timer.startTimer();
	tb_statec=2;
	tb_statep=0;
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=0;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=512;
	tb_custom_Tib=8;//[N/2Tn]=6
	//tb_custom_k2=25;
	tb_N=256;
	tb_M=128;
	tb_R=28;
	tb_C=28;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=28;
	tb_custom_Tc=28;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=28;
	tb_C_in=28;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	ifm_length=200704;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
					//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=256*28*28;
	weights_addr=weights;
	weights_length=1179648;//M*N*tb_custom_k*tb_custom_k
				//=256*512*3*3
	output_addr=weights;
	output_length=1179648;//M*N*tb_custom_k*tb_custom_k
	//=256*512*3*3
	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	for (int toM=0;toM<4;toM++){
		ifm_addr=ifm;
		ofm_length=100352;//tb_M*tb_R*tb_C=128*28*28;
		ofm_addr=ofm+toM*100352;
		for(int b=0;b<bs;b++){
			XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
			for (int to=0;to<tb_M;to+=Tm){//repeate[M/Tm]
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
				//cout<<"IFM in Done!"<<endl;
			}
			ifm_addr+=ifm_length;
			ofm_addr+=401408;//tb_M_in*tb_R*tb_C=512*28*28;
		}
	}


	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
//	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"Output in Done!"<<endl;

	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"conv"<<outnum<<endl;
	//cout<<timeInterval<<" ";
	//printf("Elapsed Time (seconds) %f\n",timeInterval);

	bf_index=0;
	for(int to=0;to<tb_M_in;to+=Tm){
		for(int ti=0;ti<tb_N;ti+=Tn){
			for (int too=0;too< Tm;too+=4){
				for(int tii=0;tii<Tn;tii++){
					if((too+to)<tb_M_in && (tii+ti)<tb_N){
						for (int r=0;r<tb_custom_k;r++){
							for (int c=0;c<tb_custom_k;c++){
								weights_print[too+to][tii+ti][r][c] = weights[bf_index] ;//output.data;
								weights_print[too+to+1][tii+ti][r][c] = weights[bf_index+1] ;//output.data;
								weights_print[too+to+2][tii+ti][r][c] = weights[bf_index+2] ;//output.data;
								weights_print[too+to+3][tii+ti][r][c] = weights[bf_index+3] ;//output.data;
								bf_index+=4;
							}
						}
					}
				}
			}
		}
	}
	cout<<"1st 3OCs 1st 3ICs" <<endl;
	for (int m=0; m<3;m++){
		for (int n =0; n<3;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"1st 3OCs last 3ICs" <<endl;
	for (int m=0; m<3;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"" <<endl;
	cout<<"last 3OCs 1st 3ICs" <<endl;
	for (int m=tb_M_in-3; m<tb_M_in;m++){
		for (int n =0; n<3;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"last 3OCs last 3ICs" <<endl;
	for (int m=tb_M_in-3; m<tb_M_in;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"" <<endl;
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void maxpoolbp_256_28( int outnum,
	DMA_DATA * ifm,
	DMA_DATA * weights,
	DMA_DATA * ofm,
	DMA_DATA * out

){
	//backward pool+relu N,M,R,C=256,256,28,28, stride=2,padding=0,kernel=2
	//timer.startTimer();
	tb_statec=0;
	tb_statep=5;//view=8+4
	tb_stateb=0;
//	tb_Relulayerin=3;
//	tb_Relulayerout=0;
	//tb_Poollayerin=4;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=256;
	tb_custom_Tib=1;//[N/2Tn]=6
	//tb_custom_k2=25;
	tb_N=256;
	tb_M=256;
	tb_R=28;
	tb_C=28;
	tb_custom_stride=2;
	tb_padding=0;
	tb_custom_kb=1;
	tb_custom_k=2;
	tb_custom_Tr=1;
	tb_custom_Tc=28;
//	tb_viewTrc=1;
//	tb_viewTm=1;
	tb_R_in=56;
	tb_C_in=56;
	//tb_flag=0;


	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	ifm_addr=ifm;
	ifm_length=bs*200704;//M*tb_custom_Tr*tb_custom_Tc=256*28*28;
	weights_addr=weights;
	weights_length=bs*25088;//r*c*[m/8]=28*28*32
	ofm_addr=ofm;
	ofm_length=bs*802816;//M*R_in*C_in=256*56*56;
	output_addr=out;
	output_length=bs*802816;//M*R_in*C_in=256*56*56;

	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip

	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
	//cout<<"IFM in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"Output in Done!"<<endl;

	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"print loss"<<outnum<<endl;
	//cout<<timeInterval<<" ";

	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for loss"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void convbp_256_256_56( int outnum,
		DMA_DATA * ifm,
		DMA_DATA * weights,
		DMA_DATA * outhat,
		DMA_DATA * gama,
		DMA_DATA * beta,
		DMA_DATA * namuda,
		DMA_DATA * ofm,
		DMA_DATA * out

){
//backward bn6,7 N,M,R,C=256,256,56,56, stride=1,padding=1,kernel=3
	tb_statec=0;
	tb_statep=0;
	tb_stateb=2;
//	tb_Relulayerin=0;
//	tb_Relulayerout=1;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=256;
	tb_custom_Tib=1;//[N/2Tn]=1
	//tb_custom_k2=25;
	tb_N=256;
	tb_M=256;
	tb_R=56;
	tb_C=56;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=9;//24;
	tb_custom_Tc=224;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=56;
	tb_C_in=56;


	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ifm_addr=namuda;
	ofm_addr=beta;
	weights_addr=gama;
	ifm_length=tb_M;//M
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip

	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
	//cout<<"IFM in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;
	//ifm_addr=0;
	//ofm_addr=0;
	ofm_length=50176;//Tm*R*C
	ifm_length=50176;//Tm*R*C
	for (int to=0;to<16;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=ifm+802816*b+50176*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			ofm_addr=outhat+802816*b+50176*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32

			XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
			XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
			while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
			//cout<<"IFM in Done!"<<endl;

			while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
			//cout<<"OFM in Done!"<<endl;
		}
	}//=6*3*5*5
	ifm_addr=ifm;
	ofm_addr=outhat;
	output_addr=ifm;
	output_length=802816;//M*R*C
	ofm_length=802816;//M*R*C
	ifm_length=802816;//M*R*C
	for(int b=0;b<bs;b++){
		//ifm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		//ofm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
		XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
		ifm_addr+=ifm_length;
		ofm_addr+=ofm_length;
		output_addr+=output_length;
		while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
		//cout<<"IFM in Done!"<<endl;

		while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
		//cout<<"OFM in Done!"<<endl;

		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"Output in Done!"<<endl;
	}

	wout_addr=gama;
	wout_length=tb_M;//M
	output_addr=beta;
	output_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)wout_addr,wout_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"Weights out Done!"<<endl;
	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"output in Done!"<<endl;
//	output_addr=gama;
//	output_length=tb_M;//M
//	//XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)wout_addr,wout_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
//	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
//	//while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DEVICE_TO_DMA));
//	//cout<<"Weights out Done!"<<endl;
//	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
//	//cout<<"output in Done!"<<endl;
	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;


	cout<<"gama first:"<<endl;
	for(int i=0;i<10;i++ ){
		cout<<gama[i].data<< ";";
	}
	cout<<"gama last:"<<endl;
	for(int i=tb_M-10;i<tb_M;i++ ){
		cout<<gama[i].data<< ";";
	}

	cout<<"beta first:"<<endl;
	cout<<beta[0].data<< ";";
	cout<<beta[1].data<< ";";
	cout<<beta[2].data<< ";";
	cout<<"beta last:"<<endl;
	cout<<beta[tb_M-3].data<< ";";
	cout<<beta[tb_M-2].data<< ";";
	cout<<beta[tb_M-1].data<<endl;


	//backward conv6,7+relu N,M,R,C=256,256,56,56, stride=1,padding=1,kernel=3
	//timer.startTimer();
	tb_statec=5;
	tb_statep=0;
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=0;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=256;
	tb_custom_Tib=8;//[N/2Tn]=6
	//tb_custom_k2=25;
	tb_N=256;
	tb_M=128;
	tb_R=56;
	tb_C=56;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=28;
	tb_custom_Tc=56;
	tb_R_in=56;
	tb_C_in=56;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	ifm_length=25984;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
					//*((C-1)*tb_custom_stride+tb_custom_k)=16*29*56;
	ofm_length=401408;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=128*56*56;
	output_length=401408;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=128*56*56;
	for (int toM=0;toM<2;toM++){
		weights_addr=weights+18432*toM;

		ofm_addr=ofm+toM*401408;
		output_addr=out+toM*401408;
		XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
		for (int to=0;to<tb_M;to+=Tm){//repeate[M/Tm]
			ifm_addr=ifm;
			//ifm_length=25984;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
									//*((C-1)*tb_custom_stride+tb_custom_k)=16*29*56;
			for(int ti=0;ti<tb_N;ti+=Tn){
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				ifm_addr+=50176;//Tn*R_in*C_in=16*56*56
				if(to==0){
					weights_length=18432;//8*Tm*min(Tn,N-Tn*ti)*tb_custom_k*tb_custom_k
										//=128*16*3*3
					XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
					weights_addr+=36864;//Tm*N'(M)*tb_custom_k*tb_custom_k=16*256*3*3;
					while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
					//cout<<"Weights in Done!"<<endl;
				}
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
				//cout<<"IFM in Done!"<<endl;
			}
			ifm_addr=ifm+24192;//+896*(row-1);//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*256*56*56+row*1*56*16
			for(int ti=0;ti<tb_N;ti+=Tn){
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				ifm_addr+=50176;//Tn*R_in*C_in=16*56*56
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
				//cout<<"IFM in Done!"<<endl;
			}
		}
		ofm_addr+=802816;
		output_addr+=802816;
		while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
		//cout<<"OFM in Done!"<<endl;
		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"Output in Done!"<<endl;
		for(int b=1;b<bs;b++){
			XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
			XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
			for (int to=0;to<tb_M;to+=Tm){//repeate[M/Tm]
				//ifm_length=25984;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
													//*((C-1)*tb_custom_stride+tb_custom_k)=16*29*56;
				ifm_addr=ifm+802816*b;//+3632*row;//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*256*56*56+row*1*56*16
				for(int ti=0;ti<tb_N;ti+=Tn){
					XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
					ifm_addr+=50176;//Tn*R_in*C_in=16*56*56
					while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
					//cout<<"IFM in Done!"<<endl;
				}
				ifm_addr=ifm+802816*b+24192;//+896*(row-1);//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*256*56*56+row*1*56*16
				for(int ti=0;ti<tb_N;ti+=Tn){
					XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
					ifm_addr+=50176;//Tn*R_in*C_in=16*56*56
					while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
					//cout<<"IFM in Done!"<<endl;
				}
			}
			output_addr+=802816;
			ofm_addr+=802816;
			while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
			//cout<<"OFM in Done!"<<endl;
			while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
			//cout<<"Output in Done!"<<endl;
		}
	}

	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"print loss"<<outnum<<endl;
	//cout<<timeInterval<<" ";

	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for loss"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}

		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void convwu_256_256_56( int outnum,
	DMA_DATA *  ifm,
	DMA_DATA *  weights,
	DMA_DATA *  ofm,
	DMA_DATA weights_print[256][256][3][3]

){
	//update conv6,7 weights N,M,R,C=256,256,56,56, stride=1,padding=1,kernel=3
	//timer.startTimer();
	tb_statec=2;
	tb_statep=0;
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=0;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=256;
	tb_custom_Tib=8;//[N/2Tn]=6
	//tb_custom_k2=25;
	tb_N=256;
	tb_M=128;
	tb_R=56;
	tb_C=56;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=4;
	tb_custom_Tc=56;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=56;
	tb_C_in=56;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	weights_addr=weights;
	weights_length=589824;//M*N*tb_custom_k*tb_custom_k
			//=256*256*3*3
	output_addr=weights;
	output_length=589824;//M*N*tb_custom_k*tb_custom_k
	//=256*256*3*3

	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);

	for (int toM=0;toM<2;toM++){
		ofm_length=50176;// Tm*R*C=16*56*56
		for(int b=0;b<bs;b++){
			for (int to=0;to<8;to++){//repeate[M/Tm]
				ofm_addr=ofm+b*802816+toM*401408+to*50176;//M_in*R*C*b+toM*M*R*C+ti*Tm*R*C=256*56*56*b+toM*128*56*56+to*16*56*56
				for(int ti=0;ti<16;ti++){
					XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
					ifm_length=4480;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
						//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=16*56*5;
					ifm_addr=ifm+b*802816+ti*50176;//b*R*N*C+ti*Tn*R*C
					XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
					ifm_length=5376;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
									//16*56*6
					for(int row=tb_custom_Tr;row<tb_R-tb_custom_Tr;row+=tb_custom_Tr){
						while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
						//cout<<"IFM in Done!"<<endl;
						ifm_addr=ifm+b*802816+ti*50176+(row-1)*896;//+3632*row;//b*R*N*C+ti*Tn*R*C+C_in*row*stride*min(Tn,N)+(row-1)*1*16*56
						XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
					}
					ifm_length=4480;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
									//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=16*56*5;
					ifm_addr=ifm+b*802816+ti*50176+45696;//(row-1)*896;//+3632*row;//b*R*N*C+ti*Tn*R*C+C_in*row*stride*min(Tn,N)+(row-1)*1*16*56
					while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
					//cout<<"IFM in Done!"<<endl;
					XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
					while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
					//cout<<"IFM in Done!"<<endl;
					while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
					//cout<<"OFM in Done!"<<endl;
				}
			}
		}
	}

	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"Output in Done!"<<endl;
	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"conv"<<outnum<<endl;
	//cout<<timeInterval<<" ";
	//printf("Elapsed Time (seconds) %f\n",timeInterval);


	bf_index=0;
	for(int to=0;to<tb_M_in;to+=Tm){
		for(int ti=0;ti<tb_N;ti+=Tn){
			for (int too=0;too< Tm;too+=4){
				for(int tii=0;tii<Tn;tii++){
					if((too+to)<tb_M_in && (tii+ti)<tb_N){
						for (int r=0;r<tb_custom_k;r++){
							for (int c=0;c<tb_custom_k;c++){
								weights_print[too+to][tii+ti][r][c] = weights[bf_index] ;//output.data;
								weights_print[too+to+1][tii+ti][r][c] = weights[bf_index+1] ;//output.data;
								weights_print[too+to+2][tii+ti][r][c] = weights[bf_index+2] ;//output.data;
								weights_print[too+to+3][tii+ti][r][c] = weights[bf_index+3] ;//output.data;
								bf_index+=4;
							}
						}
					}
				}
			}
		}
	}
	cout<<"1st 3OCs 1st 3ICs" <<endl;
	for (int m=0; m<3;m++){
		for (int n =0; n<3;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"1st 3OCs last 3ICs" <<endl;
	for (int m=0; m<3;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"" <<endl;
	cout<<"last 3OCs 1st 3ICs" <<endl;
	for (int m=tb_M_in-3; m<tb_M_in;m++){
		for (int n =0; n<3;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"last 3OCs last 3ICs" <<endl;
	for (int m=tb_M_in-3; m<tb_M_in;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"" <<endl;
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void convbp_128_256_56( int outnum,
	DMA_DATA *  ifm,
	DMA_DATA *  weights,
	DMA_DATA * outhat,
	DMA_DATA * gama,
	DMA_DATA * beta,
	DMA_DATA * namuda,
	DMA_DATA *  out

){

//backward bn5 N,M,R,C=256,256,56,56, stride=1,padding=1,kernel=3
	tb_statec=0;
	tb_statep=0;
	tb_stateb=2;
//	tb_Relulayerin=0;
//	tb_Relulayerout=1;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=256;
	tb_custom_Tib=1;//[N/2Tn]=1
	//tb_custom_k2=25;
	tb_N=256;
	tb_M=256;
	tb_R=56;
	tb_C=56;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=9;//24;
	tb_custom_Tc=224;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=56;
	tb_C_in=56;


	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ifm_addr=namuda;
	ofm_addr=beta;
	weights_addr=gama;
	ifm_length=tb_M;//M
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip

	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
	//cout<<"IFM in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;
	//ifm_addr=0;
	//ofm_addr=0;
	ofm_length=50176;//Tm*R*C
	ifm_length=50176;//Tm*R*C
	for (int to=0;to<16;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){

			ifm_addr=ifm+802816*b+50176*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			ofm_addr=outhat+802816*b+50176*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
			XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
			while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
			//cout<<"IFM in Done!"<<endl;

			while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
			//cout<<"OFM in Done!"<<endl;
		}
	}//=6*3*5*5
	ifm_addr=ifm;
	ofm_addr=outhat;
	output_addr=ifm;
	ofm_length=802816;//M*R*C
	ifm_length=802816;//M*R*C
	output_length=802816;//M*R*C
	for(int b=0;b<bs;b++){
		//ifm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		//ofm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
		XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
		ifm_addr+=ifm_length;
		ofm_addr+=ofm_length;
		output_addr+=output_length;
		while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
		//cout<<"IFM in Done!"<<endl;

		while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
		//cout<<"OFM in Done!"<<endl;

		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"Output in Done!"<<endl;
	}

	wout_addr=gama;
	wout_length=tb_M;//M
	output_addr=beta;
	output_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)wout_addr,wout_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"Weights out Done!"<<endl;
	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"output in Done!"<<endl;
//	output_addr=gama;
//	output_length=tb_M;//M
//	//XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)wout_addr,wout_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
//	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
//	//while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DEVICE_TO_DMA));
//	//cout<<"Weights out Done!"<<endl;
//	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
//	//cout<<"output in Done!"<<endl;
	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;


	cout<<"gama first:"<<endl;
	for(int i=0;i<10;i++ ){
		cout<<gama[i].data<< ";";
	}
	cout<<"gama last:"<<endl;
	for(int i=tb_M-10;i<tb_M;i++ ){
		cout<<gama[i].data<< ";";
	}

	cout<<"beta first:"<<endl;
	cout<<beta[0].data<< ";";
	cout<<beta[1].data<< ";";
	cout<<beta[2].data<< ";";
	cout<<"beta last:"<<endl;
	cout<<beta[tb_M-3].data<< ";";
	cout<<beta[tb_M-2].data<< ";";
	cout<<beta[tb_M-1].data<<endl;


	//backward conv5 N,M,R,C=256,128,56,56, stride=1,padding=1,kernel=3
	//timer.startTimer();
	tb_statec=4;
	tb_statep=0;
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=0;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=128;
	tb_custom_Tib=8;//[N/2Tn]=6
	//tb_custom_k2=25;
	tb_N=256;
	tb_M=128;
	tb_R=56;
	tb_C=56;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=28;
	tb_custom_Tc=56;
	tb_R_in=56;
	tb_C_in=56;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	ifm_addr=ifm;
	weights_addr=weights;
	ifm_length=25984;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
				//*((C-1)*tb_custom_stride+tb_custom_k)=16*29*56;
	weights_length=294912;//M*N*tb_custom_k*tb_custom_k=`128*256*3*3
	output_addr=out;
	output_length=bs*401408;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=128*56*56;

	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);

	for(int b=0;b<bs;b++){
		for (int to=0;to<tb_M;to+=Tm){//repeate [M/Tm]
			//for(int row=0;row<tb_R;row+=tb_custom_Tr){
			ifm_addr=ifm+802816*b;//+256*row;//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*256*56*56+row*1*56*16
			for(int ti=0;ti<tb_N;ti+=Tn){
				//ifm_length=448;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
										//*((C-1)*tb_custom_stride+tb_custom_k)=1*16*28;
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				ifm_addr+=50176;//Tn*R_in*C_in=16*56*56
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
				//cout<<"IFM in Done!"<<endl;
			}
			ifm_addr=ifm+802816*b+24192;//+896*(row-1);//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*256*56*56+row*1*56*16
			for(int ti=0;ti<tb_N;ti+=Tn){
				//ifm_length=448;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
										//*((C-1)*tb_custom_stride+tb_custom_k)=1*16*28;
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				ifm_addr+=50176;//Tn*R_in*C_in=16*56*56
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
				//cout<<"IFM in Done!"<<endl;
			}
		}
	}

	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	//while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
	//cout<<"IFM in Done!"<<endl;

//	while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
//	//cout<<"OFM in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"Output in Done!"<<endl;

	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"print loss"<<outnum<<endl;
	//cout<<timeInterval<<" ";

	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for loss"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}

		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void convwu_128_256_56( int outnum,
	DMA_DATA * ifm,
	DMA_DATA * weights,
	DMA_DATA * ofm,
	DMA_DATA weights_print[256][128][3][3]

){
	//update conv5 weights N,M,R,C=128,256,56,56, stride=1,padding=1,kernel=3
	//timer.startTimer();
	tb_statec=2;
	tb_statep=0;
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=0;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=256;
	tb_custom_Tib=4;//[N/2Tn]=6
	//tb_custom_k2=25;
	tb_N=128;
	tb_M=256;
	tb_R=56;
	tb_C=56;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=4;
	tb_custom_Tc=56;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=56;
	tb_C_in=56;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	ofm_length=50176;// Tm*R*C=16*56*56
	weights_addr=weights;
	weights_length=294912;//M*N*tb_custom_k*tb_custom_k
				//=128*256*3*3
	output_addr=weights;
	output_length=294912;//M*N*tb_custom_k*tb_custom_k
					//=128*256*3*3

	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);

	for(int b=0;b<bs;b++){
		for (int to=0;to<16;to++){//repeate[M/Tm]
			ofm_addr=ofm+b*802816+to*50176;//M_in*R*C*b+toM*M*R*C+to*Tm*R*C=256*56*56*b+to*16*56*56
			for(int ti=0;ti<8;ti++){
				XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
				ifm_length=4480;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
					//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=16*56*5;
				ifm_addr=ifm+b*401408+ti*50176;//b*R*N*C+ti*Tn*R*C
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				ifm_length=5376;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
								//16*56*6
				for(int row=tb_custom_Tr;row<tb_R-tb_custom_Tr;row+=tb_custom_Tr){
					ifm_addr=ifm+b*401408+ti*50176+(row-1)*896;//+3632*row;//b*R*N*C+ti*Tn*R*C+C_in*row*stride*min(Tn,N)+(row-1)*1*16*56
					while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
					//cout<<"IFM in Done!"<<endl;
					XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				}
				ifm_length=4480;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
								//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=16*56*5;
				ifm_addr=ifm+b*401408+ti*50176+45696;//(row-1)*896;//+3632*row;//b*R*N*C+ti*Tn*R*C+C_in*row*stride*min(Tn,N)+(row-1)*1*16*56
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
				//cout<<"IFM in Done!"<<endl;
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
				//cout<<"IFM in Done!"<<endl;
				while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
				//cout<<"OFM in Done!"<<endl;
			}
		}
	}

	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"Output in Done!"<<endl;

	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"conv"<<outnum<<endl;
	//cout<<timeInterval<<" ";
	//printf("Elapsed Time (seconds) %f\n",timeInterval);

	bf_index=0;
	for(int to=0;to<tb_M_in;to+=Tm){
		for(int ti=0;ti<tb_N;ti+=Tn){
			for (int too=0;too< Tm;too+=4){
				for(int tii=0;tii<Tn;tii++){
					if((too+to)<tb_M_in && (tii+ti)<tb_N){
						for (int r=0;r<tb_custom_k;r++){
							for (int c=0;c<tb_custom_k;c++){
								weights_print[too+to][tii+ti][r][c] = weights[bf_index] ;//output.data;
								weights_print[too+to+1][tii+ti][r][c] = weights[bf_index+1] ;//output.data;
								weights_print[too+to+2][tii+ti][r][c] = weights[bf_index+2] ;//output.data;
								weights_print[too+to+3][tii+ti][r][c] = weights[bf_index+3] ;//output.data;
								bf_index+=4;
							}
						}
					}
				}
			}
		}
	}
	cout<<"1st 3OCs 1st 3ICs" <<endl;
	for (int m=0; m<3;m++){
		for (int n =0; n<3;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"1st 3OCs last 3ICs" <<endl;
	for (int m=0; m<3;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"" <<endl;
	cout<<"last 3OCs 1st 3ICs" <<endl;
	for (int m=tb_M_in-3; m<tb_M_in;m++){
		for (int n =0; n<3;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"last 3OCs last 3ICs" <<endl;
	for (int m=tb_M_in-3; m<tb_M_in;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"" <<endl;
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void maxpoolbp_128_56( int outnum,
	DMA_DATA * ifm,
	DMA_DATA * weights,
	DMA_DATA * ofm,
	DMA_DATA * out

){
	//backward pool N,M,R,C=128,128,56,56, stride=2,padding=0,kernel=2
	//timer.startTimer();
	tb_statec=0;
	tb_statep=5;//view=8+4
	tb_stateb=0;
//	tb_Relulayerin=3;
//	tb_Relulayerout=0;
	//tb_Poollayerin=4;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=128;
	tb_custom_Tib=1;//[N/2Tn]=6
	//tb_custom_k2=25;
	tb_N=128;
	tb_M=128;
	tb_R=56;
	tb_C=56;
	tb_custom_stride=2;
	tb_padding=0;
	tb_custom_kb=1;
	tb_custom_k=2;
	tb_custom_Tr=1;
	tb_custom_Tc=56;
//	tb_viewTrc=1;
//	tb_viewTm=1;
	tb_R_in=112;
	tb_C_in=112;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	//FPGA_DATA loss5[bs*10*10*16];//[r][c][m]
	ifm_addr=ifm;
	ifm_length=bs*401408;//M*tb_custom_Tr*tb_custom_Tc=128*56*56;
	weights_addr=weights;
	weights_length=bs*50176;//r*c*[m/8]=56*56*16
	ofm_addr=ofm;
	ofm_length=bs*1605632;//M*R_in*C_in=128*112*112;
	output_addr=out;
	output_length=bs*1605632;//M*R_in*C_in=128*112*112;

	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip

	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
	//cout<<"IFM in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"Output in Done!"<<endl;

	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;


	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"print loss"<<outnum<<endl;
	//cout<<timeInterval<<" ";

	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for loss"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void convbp_128_128_112( int outnum,
		DMA_DATA * ifm,
		DMA_DATA * weights,
		DMA_DATA * outhat,
		DMA_DATA * gama,
		DMA_DATA * beta,
		DMA_DATA * namuda,
		DMA_DATA * ofm,
		DMA_DATA * out
){
	//backward bn4 N,M,R,C=128,128,112,112, stride=1,padding=1,kernel=3
	tb_statec=0;
	tb_statep=0;
	tb_stateb=2;
//	tb_Relulayerin=0;
//	tb_Relulayerout=1;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=128;
	tb_custom_Tib=1;//[N/2Tn]=1
	//tb_custom_k2=25;
	tb_N=128;
	tb_M=128;
	tb_R=112;
	tb_C=112;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=9;//24;
	tb_custom_Tc=224;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=112;
	tb_C_in=112;


	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ifm_addr=namuda;
	ofm_addr=beta;
	weights_addr=gama;
	ifm_length=tb_M;//M
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip

	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
	//cout<<"IFM in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;
	ofm_length=200704;//Tm*R*C
	ifm_length=200704;//Tm*R*C
	for (int to=0;to<8;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=ifm+1605632*b+200704*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			ofm_addr=outhat+1605632*b+200704*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
			XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
			while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
			//cout<<"IFM in Done!"<<endl;

			while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
			//cout<<"OFM in Done!"<<endl;
		}
	}//=6*3*5*5
	ifm_addr=ifm;
	ofm_addr=outhat;
	output_addr=ifm;
	ofm_length=1605632;//M*R*C
	ifm_length=1605632;//M*R*C
	output_length=1605632;//M*R*C
	for(int b=0;b<bs;b++){
		//ifm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		//ofm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
		XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
		ifm_addr+=ifm_length;
		ofm_addr+=ofm_length;
		output_addr+=output_length;
		while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
		//cout<<"IFM in Done!"<<endl;

		while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
		//cout<<"OFM in Done!"<<endl;

		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"Output in Done!"<<endl;
	}

	wout_addr=gama;
	wout_length=tb_M;//M
	output_addr=beta;
	output_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)wout_addr,wout_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"Weights out Done!"<<endl;
	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"output in Done!"<<endl;
//	output_addr=gama;
//	output_length=tb_M;//M
//	//XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)wout_addr,wout_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
//	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
//	//while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DEVICE_TO_DMA));
//	//cout<<"Weights out Done!"<<endl;
//	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
//	//cout<<"output in Done!"<<endl;
	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;


	cout<<"gama first:"<<endl;
	for(int i=0;i<10;i++ ){
		cout<<gama[i].data<< ";";
	}
	cout<<"gama last:"<<endl;
	for(int i=tb_M-10;i<tb_M;i++ ){
		cout<<gama[i].data<< ";";
	}

	cout<<"beta first:"<<endl;
	cout<<beta[0].data<< ";";
	cout<<beta[1].data<< ";";
	cout<<beta[2].data<< ";";
	cout<<"beta last:"<<endl;
	cout<<beta[tb_M-3].data<< ";";
	cout<<beta[tb_M-2].data<< ";";
	cout<<beta[tb_M-1].data<<endl;


	//backward conv4+relu N,M,R,C=128,128,112,112, stride=1,padding=1,kernel=3
	tb_statec=5;
	tb_statep=0;
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=0;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=128;
	tb_custom_Tib=4;//[N/2Tn]=6
	//tb_custom_k2=25;
	tb_N=128;
	tb_M=128;
	tb_R=112;
	tb_C=112;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=16;
	tb_custom_Tc=112;
	tb_R_in=112;
	tb_C_in=112;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	ifm_addr=ifm;
	weights_addr=weights;
	//ifm_length=30464;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
				//*((C-1)*tb_custom_stride+tb_custom_k)=16*17*112;
	weights_length=147456;//M*N*tb_custom_k*tb_custom_k=128*128*3*3
	ofm_addr=ofm;
	ofm_length=bs*1605632;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=128*112*112;
	output_addr=out;
	output_length=bs*1605632;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=128*112*112;
	XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);

	for(int b=0;b<bs;b++){
		for (int to=0;to<tb_M;to+=Tm){//repeate [M/Tm]
			ifm_length=30464;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
							//*((C-1)*tb_custom_stride+tb_custom_k)=16*17*112;
			ifm_addr=ifm+1605632*b;//+1792*(row-1);//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*128*112*112+row*1*112*16
			for(int ti=0;ti<tb_N;ti+=Tn){
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				ifm_addr+=200704;//Tn*R_in*C_in=16*112*112
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
				//cout<<"IFM in Done!"<<endl;
			}
			for(int row=tb_custom_Tr;row<tb_R-tb_custom_Tr;row+=tb_custom_Tr){
				ifm_length=32256;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
									//*((C-1)*tb_custom_stride+tb_custom_k)=16*18*112;
				ifm_addr=ifm+1605632*b+1792*(row-1);//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*128*112*112+row*1*112*16
				for(int ti=0;ti<tb_N;ti+=Tn){
					XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
					ifm_addr+=200704;//Tn*R_in*C_in=16*112*112
					while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
					//cout<<"IFM in Done!"<<endl;
				}
			}
			ifm_length=30464;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
							//*((C-1)*tb_custom_stride+tb_custom_k)=16*17*112;
			ifm_addr=ifm+1605632*b+170240;//+1792*(row-1);//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*128*112*112+row*1*112*16
			for(int ti=0;ti<tb_N;ti+=Tn){
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				ifm_addr+=200704;//Tn*R_in*C_in=16*112*112
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
				//cout<<"IFM in Done!"<<endl;
			}
		}
	}

	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"output in Done!"<<endl;

	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"print loss"<<outnum<<endl;
	//cout<<timeInterval<<" ";

	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for loss"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}

		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void convwu_128_128_112( int outnum,
	DMA_DATA * ifm,
	DMA_DATA * weights,
	DMA_DATA * ofm,
	DMA_DATA weights_print[128][128][3][3]

){
	//update conv4 weights N,M,R,C=128,128,112,112, stride=1,padding=1,kernel=3
	//timer.startTimer();
	tb_statec=2;
	tb_statep=0;
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=0;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=128;
	tb_custom_Tib=4;//[N/2Tn]=6
	//tb_custom_k2=25;
	tb_N=128;
	tb_M=128;
	tb_R=112;
	tb_C=112;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=4;
	tb_custom_Tc=112;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=112;
	tb_C_in=112;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	ofm_length=200704;// Tm*R*C=16*112*112
	weights_addr=weights;
	weights_length=147456;//M*N*tb_custom_k*tb_custom_k
				//=128*128*3*3
	output_addr=weights;
	output_length=147456;//M*N*tb_custom_k*tb_custom_k
		//=128*128*3*3

	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);

	for(int b=0;b<bs;b++){
		for (int to=0;to<8;to++){//repeate[M/Tm]
			ofm_addr=ofm+b*1605632+to*200704;//M_in*R*C*b+toM*M*R*C+to*Tm*R*C=128*112*112*b+to*16*112*112
			for(int ti=0;ti<8;ti++){
				XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
				ifm_length=8960;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
					//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=16*112*5;
				ifm_addr=ifm+b*1605632+ti*200704;//b*R*N*C+ti*Tn*R*C
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				ifm_length=10752;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
								//16*112*6
				for(int row=tb_custom_Tr;row<tb_R-tb_custom_Tr;row+=tb_custom_Tr){
					ifm_addr=ifm+b*1605632+ti*200704+(row-1)*1792;//+1792*row;//b*R*N*C+ti*Tn*R*C+C_in*row*stride*min(Tn,N)+(row-1)*1*16*112
					while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
					//cout<<"IFM in Done!"<<endl;
					XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				}
				ifm_length=8960;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
								//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=16*112*5;
				ifm_addr=ifm+b*1605632+ti*200704+191744;//(row-1)*1792;////b*R*N*C+ti*Tn*R*C+C_in*row*stride*min(Tn,N)+(row-1)*1*16*56
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
				//cout<<"IFM in Done!"<<endl;
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
				//cout<<"IFM in Done!"<<endl;
				while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
				//cout<<"OFM in Done!"<<endl;
			}
		}
	}

	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"Output in Done!"<<endl;

	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"conv"<<outnum<<endl;
	//cout<<timeInterval<<" ";
	//printf("Elapsed Time (seconds) %f\n",timeInterval);

	bf_index=0;
	for(int to=0;to<tb_M_in;to+=Tm){
		for(int ti=0;ti<tb_N;ti+=Tn){
			for (int too=0;too< Tm;too+=4){
				for(int tii=0;tii<Tn;tii++){
					if((too+to)<tb_M_in && (tii+ti)<tb_N){
						for (int r=0;r<tb_custom_k;r++){
							for (int c=0;c<tb_custom_k;c++){
								weights_print[too+to][tii+ti][r][c] = weights[bf_index] ;//output.data;
								weights_print[too+to+1][tii+ti][r][c] = weights[bf_index+1] ;//output.data;
								weights_print[too+to+2][tii+ti][r][c] = weights[bf_index+2] ;//output.data;
								weights_print[too+to+3][tii+ti][r][c] = weights[bf_index+3] ;//output.data;
								bf_index+=4;
							}
						}
					}
				}
			}
		}
	}
	cout<<"1st 3OCs 1st 3ICs" <<endl;
	for (int m=0; m<3;m++){
		for (int n =0; n<3;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"1st 3OCs last 3ICs" <<endl;
	for (int m=0; m<3;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"" <<endl;
	cout<<"last 3OCs 1st 3ICs" <<endl;
	for (int m=tb_M_in-3; m<tb_M_in;m++){
		for (int n =0; n<3;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"last 3OCs last 3ICs" <<endl;
	for (int m=tb_M_in-3; m<tb_M_in;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"" <<endl;
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void convbp_64_128_112( int outnum,
	DMA_DATA * ifm,
	DMA_DATA * weights,
	DMA_DATA * outhat,
	DMA_DATA * gama,
	DMA_DATA * beta,
	DMA_DATA * namuda,
	DMA_DATA * out

){
//backward bn3 N,M,R,C=128,128,112,112, stride=1,padding=1,kernel=3
	tb_statec=0;
	tb_statep=0;
	tb_stateb=2;
//	tb_Relulayerin=0;
//	tb_Relulayerout=1;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=128;
	tb_custom_Tib=1;//[N/2Tn]=1
	//tb_custom_k2=25;
	tb_N=128;
	tb_M=128;
	tb_R=112;
	tb_C=112;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=9;//24;
	tb_custom_Tc=224;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=112;
	tb_C_in=112;


	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ifm_addr=namuda;
	ofm_addr=beta;
	weights_addr=gama;
	ifm_length=tb_M;//M
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip

	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
	//cout<<"IFM in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;
	//ifm_addr=0;
	//ofm_addr=0;
	ofm_length=200704;//Tm*R*C
	ifm_length=200704;//Tm*R*C
	for (int to=0;to<8;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=ifm+1605632*b+200704*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			ofm_addr=outhat+1605632*b+200704*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
			XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
			while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
			//cout<<"IFM in Done!"<<endl;

			while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
			//cout<<"OFM in Done!"<<endl;
		}
	}//=6*3*5*5
	ifm_addr=ifm;
	ofm_addr=outhat;
	output_addr=ifm;
	ofm_length=1605632;//M*R*C
	ifm_length=1605632;//M*R*C
	output_length=1605632;//M*R*C
	for(int b=0;b<bs;b++){
		//ifm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		//ofm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
		XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
		ifm_addr+=ifm_length;
		ofm_addr+=ofm_length;
		output_addr+=output_length;
		while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
		//cout<<"IFM in Done!"<<endl;

		while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
		//cout<<"OFM in Done!"<<endl;

		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"Output in Done!"<<endl;
	}

	wout_addr=gama;
	wout_length=tb_M;//M
	output_addr=beta;
	output_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)wout_addr,wout_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"Weights out Done!"<<endl;
	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"output in Done!"<<endl;
//	output_addr=gama;
//	output_length=tb_M;//M
//	//XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)wout_addr,wout_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
//	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
//	//while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DEVICE_TO_DMA));
//	//cout<<"Weights out Done!"<<endl;
//	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
//	//cout<<"output in Done!"<<endl;
	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;


	cout<<"gama first:"<<endl;
	for(int i=0;i<10;i++ ){
		cout<<gama[i].data<< ";";
	}
	cout<<"gama last:"<<endl;
	for(int i=tb_M-10;i<tb_M;i++ ){
		cout<<gama[i].data<< ";";
	}

	cout<<"beta first:"<<endl;
	cout<<beta[0].data<< ";";
	cout<<beta[1].data<< ";";
	cout<<beta[2].data<< ";";
	cout<<"beta last:"<<endl;
	cout<<beta[tb_M-3].data<< ";";
	cout<<beta[tb_M-2].data<< ";";
	cout<<beta[tb_M-1].data<<endl;


//backward conv3 N,M,R,C=128,64,112,112, stride=1,padding=1,kernel=3
	//timer.startTimer();
	tb_statec=4;
	tb_statep=0;
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=0;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=64;
	tb_custom_Tib=4;//[N/2Tn]=6
	//tb_custom_k2=25;
	tb_N=128;
	tb_M=64;
	tb_R=112;
	tb_C=112;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=16;
	tb_custom_Tc=112;
	tb_R_in=112;
	tb_C_in=112;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	ifm_addr=ifm;
	weights_addr=weights;
	//ifm_length=30464;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
				//*((C-1)*tb_custom_stride+tb_custom_k)=16*17*112;
	weights_length=73728;//M*N*tb_custom_k*tb_custom_k=64*128*3*3


	output_addr=out;
	output_length=bs*802816;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=64*112*112;

	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);


	for(int b=0;b<bs;b++){
		for (int to=0;to<tb_M;to+=Tm){//repeate [M/Tm]
			ifm_length=30464;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
							//*((C-1)*tb_custom_stride+tb_custom_k)=16*17*112;
			ifm_addr=ifm+1605632*b;//+1792*(row-1);//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*128*112*112+row*1*112*16
			for(int ti=0;ti<tb_N;ti+=Tn){
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				ifm_addr+=200704;//Tn*R_in*C_in=16*112*112
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
				//cout<<"IFM in Done!"<<endl;
			}
			for(int row=tb_custom_Tr;row<tb_R-tb_custom_Tr;row+=tb_custom_Tr){
				ifm_length=32256;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
									//*((C-1)*tb_custom_stride+tb_custom_k)=16*18*112;
				ifm_addr=ifm+1605632*b+1792*(row-1);//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*128*112*112+row*1*112*16
				for(int ti=0;ti<tb_N;ti+=Tn){
					XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
					ifm_addr+=200704;//Tn*R_in*C_in=16*112*112
					while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
					//cout<<"IFM in Done!"<<endl;
				}
			}
			ifm_length=30464;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
							//*((C-1)*tb_custom_stride+tb_custom_k)=16*17*112;
			ifm_addr=ifm+1605632*b+170240;//+1792*(row-1);//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*128*112*112+row*1*112*16
			for(int ti=0;ti<tb_N;ti+=Tn){
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				ifm_addr+=200704;//Tn*R_in*C_in=16*112*112
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
				//cout<<"IFM in Done!"<<endl;
			}
		}
	}
	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	//while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"output in Done!"<<endl;

	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"print loss"<<outnum<<endl;
	//cout<<timeInterval<<" ";

	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for loss"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}

		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void convwu_64_128_112( int outnum,
	DMA_DATA * ifm,
	DMA_DATA * weights,
	DMA_DATA * ofm,
	DMA_DATA weights_print[128][64][3][3]

){
	//update conv3 weights N,M,R,C=64,128,112,112, stride=1,padding=1,kernel=3
	//timer.startTimer();
	tb_statec=2;
	tb_statep=0;
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=0;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=128;
	tb_custom_Tib=2;//[N/2Tn]=6
	//tb_custom_k2=25;
	tb_N=64;
	tb_M=128;
	tb_R=112;
	tb_C=112;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=4;
	tb_custom_Tc=112;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=112;
	tb_C_in=112;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	ofm_length=200704;// Tm*R*C=16*112*112
	weights_addr=weights;
	weights_length=73728;//M*N*tb_custom_k*tb_custom_k
			//=64*128*3*3
	output_addr=weights;
	output_length=73728;//M*N*tb_custom_k*tb_custom_k
		//=64*128*3*3

	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);

	for(int b=0;b<bs;b++){
		for (int to=0;to<8;to++){//repeate[M/Tm]
			ofm_addr=ofm+b*1605632+to*200704;//M_in*R*C*b+toM*M*R*C+to*Tm*R*C=128*112*112*b+to*16*112*112
			for(int ti=0;ti<4;ti++){
				XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
				ifm_length=8960;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
					//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=16*112*5;
				ifm_addr=ifm+b*802816+ti*200704;//b*R*N*C+ti*Tn*R*C
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				ifm_length=10752;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
								//16*112*6
				for(int row=tb_custom_Tr;row<tb_R-tb_custom_Tr;row+=tb_custom_Tr){
					ifm_addr=ifm+b*802816+ti*200704+(row-1)*1792;//+1792*row;//b*R*N*C+ti*Tn*R*C+C_in*row*stride*min(Tn,N)+(row-1)*1*16*112
					while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
					//cout<<"IFM in Done!"<<endl;
					XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				}
				ifm_length=8960;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
								//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=16*112*5;
				ifm_addr=ifm+b*802816+ti*200704+191744;//(row-1)*1792;////b*R*N*C+ti*Tn*R*C+C_in*row*stride*min(Tn,N)+(row-1)*1*16*56
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
				//cout<<"IFM in Done!"<<endl;
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
				//cout<<"IFM in Done!"<<endl;
				while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
				//cout<<"OFM in Done!"<<endl;
			}
		}
	}

	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"Output in Done!"<<endl;

	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"conv"<<outnum<<endl;
	//cout<<timeInterval<<" ";
	//printf("Elapsed Time (seconds) %f\n",timeInterval);

	bf_index=0;
	for(int to=0;to<tb_M_in;to+=Tm){
		for(int ti=0;ti<tb_N;ti+=Tn){
			for (int too=0;too< Tm;too+=4){
				for(int tii=0;tii<Tn;tii++){
					if((too+to)<tb_M_in && (tii+ti)<tb_N){
						for (int r=0;r<tb_custom_k;r++){
							for (int c=0;c<tb_custom_k;c++){
								weights_print[too+to][tii+ti][r][c] = weights[bf_index] ;//output.data;
								weights_print[too+to+1][tii+ti][r][c] = weights[bf_index+1] ;//output.data;
								weights_print[too+to+2][tii+ti][r][c] = weights[bf_index+2] ;//output.data;
								weights_print[too+to+3][tii+ti][r][c] = weights[bf_index+3] ;//output.data;
								bf_index+=4;
							}
						}
					}
				}
			}
		}
	}
	cout<<"1st 3OCs 1st 3ICs" <<endl;
	for (int m=0; m<3;m++){
		for (int n =0; n<3;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"1st 3OCs last 3ICs" <<endl;
	for (int m=0; m<3;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"" <<endl;
	cout<<"last 3OCs 1st 3ICs" <<endl;
	for (int m=tb_M_in-3; m<tb_M_in;m++){
		for (int n =0; n<3;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"last 3OCs last 3ICs" <<endl;
	for (int m=tb_M_in-3; m<tb_M_in;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"" <<endl;
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void maxpoolbp_64_112( int outnum,
		DMA_DATA * ifm,
		DMA_DATA * weights,
		DMA_DATA * ofm,
		DMA_DATA * out

){
	//backward pool+relu N,M,R,C=64,64,112,112, stride=2,padding=0,kernel=2
	//timer.startTimer();
	tb_statec=0;
	tb_statep=5;//view=8+4
	tb_stateb=0;
//	tb_Relulayerin=3;
//	tb_Relulayerout=0;
	//tb_Poollayerin=4;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=64;
	tb_custom_Tib=1;//[N/2Tn]=6
	//tb_custom_k2=25;
	tb_N=64;
	tb_M=64;
	tb_R=112;
	tb_C=112;
	tb_custom_stride=2;
	tb_padding=0;
	tb_custom_kb=1;
	tb_custom_k=2;
	tb_custom_Tr=1;
	tb_custom_Tc=112;
//	tb_viewTrc=1;
//	tb_viewTm=1;
	tb_R_in=224;
	tb_C_in=224;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	ifm_addr=ifm;
	ifm_length=bs*802816;//M*tb_custom_Tr*tb_custom_Tc=64*112*112;
	weights_addr=weights;
	weights_length=bs*100352;//r*c*[m/8]=112*112*8
	ofm_addr=ofm;
	ofm_length=bs*3211264;//M*R_in*C_in=64*224*224;
	output_addr=out;
	output_length=bs*3211264;//M*R_in*C_in=64*224*224;

	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip

	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
	//cout<<"IFM in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"Output in Done!"<<endl;


	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"print loss"<<outnum<<endl;
	//cout<<timeInterval<<" ";

	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for loss"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void convbp_64_64_224( int outnum,
		DMA_DATA * ifm,
		DMA_DATA * weights,
		DMA_DATA * outhat,
		DMA_DATA * gama,
		DMA_DATA * beta,
		DMA_DATA * namuda,
		DMA_DATA * ofm,
		DMA_DATA * out

){
//backward bn2 N,M,R,C=64,64,224,224, stride=1,padding=1,kernel=3
	tb_statec=0;
	tb_statep=0;
	tb_stateb=2;
//	tb_Relulayerin=0;
//	tb_Relulayerout=1;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=64;
	tb_custom_Tib=1;//[N/2Tn]=1
	//tb_custom_k2=25;
	tb_N=64;
	tb_M=64;
	tb_R=224;
	tb_C=224;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=9;//24;
	tb_custom_Tc=224;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=224;
	tb_C_in=224;


	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ifm_addr=namuda;
	ofm_addr=beta;
	weights_addr=gama;
	ifm_length=tb_M;//M
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip

	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
	//cout<<"IFM in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//ifm_addr=0;
	//ofm_addr=0;
	ofm_length=802816;//Tm*R*C
	ifm_length=802816;//Tm*R*C
	for (int to=0;to<4;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=ifm+3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			ofm_addr=outhat+3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
			XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
			while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
			//cout<<"IFM in Done!"<<endl;

			while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
			//cout<<"OFM in Done!"<<endl;
		}
	}//=6*3*5*5
	ifm_addr=ifm;
	ofm_addr=outhat;
	output_addr=ifm;
	ofm_length=3211264;//M*R*C
	ifm_length=3211264;//M*R*C
	output_length=3211264;//100352;//M*R*C
	for(int b=0;b<bs;b++){
		//ifm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		//ofm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
		XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
		ifm_addr+=ifm_length;
		ofm_addr+=ofm_length;
		output_addr+=output_length;
		while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
		//cout<<"IFM in Done!"<<endl;

		while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
		//cout<<"OFM in Done!"<<endl;

		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"Output in Done!"<<endl;
	}

	wout_addr=gama;
	wout_length=tb_M;//M
	output_addr=beta;
	output_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)wout_addr,wout_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"Weights out Done!"<<endl;
	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"output in Done!"<<endl;
//	output_addr=gama;
//	output_length=tb_M;//M
//	//XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)wout_addr,wout_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
//	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
//	//while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DEVICE_TO_DMA));
//	//cout<<"Weights out Done!"<<endl;
//	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
//	//cout<<"output in Done!"<<endl;
	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;


	cout<<"gama first:"<<endl;
	for(int i=0;i<10;i++ ){
		cout<<gama[i].data<< ";";
	}
	cout<<"gama last:"<<endl;
	for(int i=tb_M-10;i<tb_M;i++ ){
		cout<<gama[i].data<< ";";
	}

	cout<<"beta first:"<<endl;
	cout<<beta[0].data<< ";";
	cout<<beta[1].data<< ";";
	cout<<beta[2].data<< ";";
	cout<<"beta last:"<<endl;
	cout<<beta[tb_M-3].data<< ";";
	cout<<beta[tb_M-2].data<< ";";
	cout<<beta[tb_M-1].data<<endl;


	//backward conv2+relu N,M,R,C=64,64,224,224, stride=1,padding=1,kernel=3
	//timer.startTimer();
	tb_statec=5;
	tb_statep=0;
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=0;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=64;
	tb_custom_Tib=2;//[N/2Tn]=6
	//tb_custom_k2=25;
	tb_N=64;
	tb_M=64;
	tb_R=224;
	tb_C=224;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=8;
	tb_custom_Tc=224;
	tb_R_in=224;
	tb_C_in=224;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	ifm_addr=ifm;
	weights_addr=weights;
	//ifm_length=30464;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
				//*((C-1)*tb_custom_stride+tb_custom_k)=16*17*112;
	weights_length=36864;//M*N*tb_custom_k*tb_custom_k=64*64*3*3
	ofm_addr=ofm;
	ofm_length=bs*3211264;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=64*224*224;
	output_addr=out;
	output_length=bs*3211264;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=64*224*224;
	XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);

	for(int b=0;b<bs;b++){
		for (int to=0;to<tb_M;to+=Tm){//repeate [M/Tm]
			ifm_length=32256;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
							//*((C-1)*tb_custom_stride+tb_custom_k)=16*9*224;
			ifm_addr=ifm+3211264*b;//+3584*(row-1);//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*64*224*224+row*1*224*16
			for(int ti=0;ti<tb_N;ti+=Tn){
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				ifm_addr+=802816;//Tn*R_in*C_in=16*224*224
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
				//cout<<"IFM in Done!"<<endl;
			}
			for(int row=tb_custom_Tr;row<tb_R-tb_custom_Tr;row+=tb_custom_Tr){
				ifm_length=35840;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
									//*((C-1)*tb_custom_stride+tb_custom_k)=16*10*224;
				ifm_addr=ifm+3211264*b+3584*(row-1);//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*64*224*224+row*1*224*16
				for(int ti=0;ti<tb_N;ti+=Tn){
					XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
					ifm_addr+=802816;//Tn*R_in*C_in=16*224*224
					while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
					//cout<<"IFM in Done!"<<endl;
				}
			}
			ifm_length=32256;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
							//*((C-1)*tb_custom_stride+tb_custom_k)=16*9*224;
			ifm_addr=ifm+3211264*b+770560;//+3584*(row-1);//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*64*224*224+row*1*224*16
			for(int ti=0;ti<tb_N;ti+=Tn){
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				ifm_addr+=802816;//Tn*R_in*C_in=16*224*224
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
				//cout<<"IFM in Done!"<<endl;
			}
		}
	}

	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"output in Done!"<<endl;

	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"print loss"<<outnum<<endl;
	//cout<<timeInterval<<" ";
	//printf("Elapsed Time (seconds) %f\n",timeInterval);

	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for loss"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}

		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index].data<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void convwu_64_64_224( int outnum,
	DMA_DATA * ifm,
	DMA_DATA * weights,
	DMA_DATA * ofm,
	DMA_DATA weights_print[64][64][3][3]

){
	//update conv2 weights N,M,R,C=64,64,224,224, stride=1,padding=1,kernel=3
	//timer.startTimer();
	tb_statec=2;
	tb_statep=0;
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=0;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=64;
	tb_custom_Tib=2;//[N/2Tn]=6
	//tb_custom_k2=25;
	tb_N=64;
	tb_M=64;
	tb_R=224;
	tb_C=224;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=2;
	tb_custom_Tc=224;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=224;
	tb_C_in=224;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	ofm_length=802816;// Tm*R*C=16*224*224
	weights_addr=weights;
	weights_length=36864;//M*N*tb_custom_k*tb_custom_k
				//=64*64*3*3
	output_addr=weights;
	output_length=36864;//M*N*tb_custom_k*tb_custom_k
	//=64*64*3*3

	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);

	for(int b=0;b<bs;b++){
		for (int to=0;to<4;to++){//repeate[M/Tm]
			ofm_addr=ofm+b*3211264+to*802816;//M_in*R*C*b+toM*M*R*C+to*Tm*R*C=64*224*224*b+to*16*224*224
			for(int ti=0;ti<4;ti++){
				XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
				ifm_length=10752;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
					//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=16*224*3;
				ifm_addr=ifm+b*3211264+ti*802816;//b*R*N*C+ti*Tn*R*C
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				ifm_length=14336;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
								//16*224*4
				for(int row=tb_custom_Tr;row<tb_R-tb_custom_Tr;row+=tb_custom_Tr){
					ifm_addr=ifm+b*3211264+ti*802816+(row-1)*3584;//+3584*row;//b*R*N*C+ti*Tn*R*C+C_in*row*stride*min(Tn,N)+(row-1)*1*16*224
					while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
					//cout<<"IFM in Done!"<<endl;
					XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				}
				ifm_length=10752;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
								//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=16*224*3;
				ifm_addr=ifm+b*3211264+ti*802816+792064;//(row-1)*3584;//+3584*row;//b*R*N*C+ti*Tn*R*C+C_in*row*stride*min(Tn,N)+(row-1)*1*16*224
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
				//cout<<"IFM in Done!"<<endl;
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
				//cout<<"IFM in Done!"<<endl;
				while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
				//cout<<"OFM in Done!"<<endl;
			}
		}
	}

	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;


	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"output in Done!"<<endl;


	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"conv"<<outnum<<endl;
	//cout<<timeInterval<<" ";
	//printf("Elapsed Time (seconds) %f\n",timeInterval);

	bf_index=0;
	for(int to=0;to<tb_M_in;to+=Tm){
		for(int ti=0;ti<tb_N;ti+=Tn){
			for (int too=0;too< Tm;too+=4){
				for(int tii=0;tii<Tn;tii++){
					if((too+to)<tb_M_in && (tii+ti)<tb_N){
						for (int r=0;r<tb_custom_k;r++){
							for (int c=0;c<tb_custom_k;c++){
								weights_print[too+to][tii+ti][r][c] = weights[bf_index] ;//output.data;
								weights_print[too+to+1][tii+ti][r][c] = weights[bf_index+1] ;//output.data;
								weights_print[too+to+2][tii+ti][r][c] = weights[bf_index+2] ;//output.data;
								weights_print[too+to+3][tii+ti][r][c] = weights[bf_index+3] ;//output.data;
								bf_index+=4;
							}
						}
					}
				}
			}
		}
	}
	cout<<"1st 3OCs 1st 3ICs" <<endl;
	for (int m=0; m<3;m++){
		for (int n =0; n<3;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"1st 3OCs last 3ICs" <<endl;
	for (int m=0; m<3;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"" <<endl;
	cout<<"last 3OCs 1st 3ICs" <<endl;
	for (int m=tb_M_in-3; m<tb_M_in;m++){
		for (int n =0; n<3;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"last 3OCs last 3ICs" <<endl;
	for (int m=tb_M_in-3; m<tb_M_in;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"" <<endl;
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}

void convbp_3_64_224( int outnum,
		DMA_DATA * ifm,
		//DMA_DATA * weights,
		DMA_DATA * outhat,
		DMA_DATA * gama,
		DMA_DATA * beta,
		DMA_DATA * namuda
	//FPGA_DATA ofm[bs*64*224*224],
	//FPGA_DATA out[bs*64*224*224]

){
//backward bn1 N,M,R,C=64,64,224,224, stride=1,padding=1,kernel=3
	tb_statec=0;
	tb_statep=0;
	tb_stateb=2;
//	tb_Relulayerin=0;
//	tb_Relulayerout=1;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=64;
	tb_custom_Tib=1;//[N/2Tn]=1
	//tb_custom_k2=25;
	tb_N=64;
	tb_M=64;
	tb_R=224;
	tb_C=224;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=9;//24;
	tb_custom_Tc=224;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=224;
	tb_C_in=224;


	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ifm_addr=namuda;
	ofm_addr=beta;
	weights_addr=gama;
	ifm_length=tb_M;//M
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
	XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip

	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
	//cout<<"IFM in Done!"<<endl;

	while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"OFM in Done!"<<endl;
	//ifm_addr=0;
	//ofm_addr=0;
	ofm_length=802816;//Tm*R*C
	ifm_length=802816;//Tm*R*C
	for (int to=0;to<4;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=ifm+3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			ofm_addr=outhat+3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
			XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
			while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
			//cout<<"IFM in Done!"<<endl;

			while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
			//cout<<"OFM in Done!"<<endl;
		}
	}//=6*3*5*5
	ifm_addr=ifm;
	ofm_addr=outhat;
	output_addr=ifm;
	ofm_length=3211264;//M*R*C
	ifm_length=3211264;//M*R*C
	output_length=3211264;//M*R*C
	for(int b=0;b<bs;b++){
		//ifm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		//ofm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
		XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
		XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
		ifm_addr+=ifm_length;
		ofm_addr+=ofm_length;
		output_addr+=output_length;
		while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
		//cout<<"IFM in Done!"<<endl;

		while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
		//cout<<"OFM in Done!"<<endl;

		while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
		//cout<<"Output in Done!"<<endl;
	}

	wout_addr=gama;
	wout_length=tb_M;//M
	output_addr=beta;
	output_length=tb_M;//M
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)wout_addr,wout_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"Weights out Done!"<<endl;
	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"output in Done!"<<endl;
//	output_addr=gama;
//	output_length=tb_M;//M
//	//XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)wout_addr,wout_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
//	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
//	//while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DEVICE_TO_DMA));
//	//cout<<"Weights out Done!"<<endl;
//	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
//	//cout<<"output in Done!"<<endl;
	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;


	cout<<"gama first:"<<endl;
	for(int i=0;i<10;i++ ){
		cout<<gama[i].data<< ";";
	}
	cout<<"gama last:"<<endl;
	for(int i=tb_M-10;i<tb_M;i++ ){
		cout<<gama[i].data<< ";";
	}

	cout<<"beta first:"<<endl;
	cout<<beta[0].data<< ";";
	cout<<beta[1].data<< ";";
	cout<<beta[2].data<< ";";
	cout<<"beta last:"<<endl;
	cout<<beta[tb_M-3].data<< ";";
	cout<<beta[tb_M-2].data<< ";";
	cout<<beta[tb_M-1].data<<endl;

	cout<<"print loss"<<outnum<<endl;


	cout<<"after loopM:"<<endl;


	//FPGA_DATA output1_print[bs][24][24][20]; //[r][c][m]//only print half and last 10 channels

	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for loss"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<ifm[bf_index].data<< ";";

				}
			}
		}

		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<ifm[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<ifm[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<ifm[bf_index].data<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<ifm[bf_index].data<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<ifm[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<ifm[bf_index].data<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<ifm[bf_index].data<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
}

void convwu_3_64_224( int outnum,
	DMA_DATA * ifm,
	DMA_DATA * weights,
	DMA_DATA * ofm,
	DMA_DATA weights_print[64][3][3][3]

){
	//update conv1 weights N,M,R,C=3,64,224,224, stride=1,padding=1,kernel=3
	//timer.startTimer();
	tb_statec=2;
	tb_statep=0;
	tb_stateb=0;
//	tb_Relulayerin=0;
//	tb_Relulayerout=0;
	//tb_Poollayerin=0;
	//tb_Poollayerout=0;
	tb_custom_batch=bs;
	tb_batch_size=bs;
	tb_M_in=64;
	tb_custom_Tib=1;//[N/2Tn]=6
	//tb_custom_k2=25;
	tb_N=3;
	tb_M=64;
	tb_R=224;
	tb_C=224;
	tb_custom_stride=1;
	tb_padding=1;
	tb_custom_kb=1;
	tb_custom_k=3;
	tb_custom_Tr=1;
	tb_custom_Tc=224;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=224;
	tb_C_in=224;
	//tb_flag=0;

	XTop_Set_statec_V(&do_top_simple,tb_statec);
	XTop_Set_statep_V(&do_top_simple,tb_statep);
	XTop_Set_stateb_V(&do_top_simple,tb_stateb);
//	XTop_Set_Relulayerin_V(&do_top_simple,tb_Relulayerin);
//	XTop_Set_Relulayerout_V(&do_top_simple,tb_Relulayerout);
//	XTop_Set_Poollayerin_V(&do_top_simple,tb_Poollayerin);
//	XTop_Set_Poollayerout_V(&do_top_simple,tb_Poollayerout);
	XTop_Set_custom_batch_V(&do_top_simple,tb_custom_batch);
	XTop_Set_batch_size_V(&do_top_simple,tb_batch_size);
	XTop_Set_M_in_V(&do_top_simple,tb_M_in);
	XTop_Set_custom_Tib_V(&do_top_simple,tb_custom_Tib);
	XTop_Set_N_V(&do_top_simple,tb_N);
	XTop_Set_M_V(&do_top_simple,tb_M);
	XTop_Set_R_V(&do_top_simple,tb_R);
	XTop_Set_C_V(&do_top_simple,tb_C);
	XTop_Set_custom_stride_V(&do_top_simple,tb_custom_stride);
	XTop_Set_padding_V(&do_top_simple,tb_padding);
	XTop_Set_custom_kb_V(&do_top_simple,tb_custom_kb);
	XTop_Set_custom_k_V(&do_top_simple,tb_custom_k);//configure hls'custom_k parameter
	XTop_Set_custom_Tr_V(&do_top_simple,tb_custom_Tr);
	XTop_Set_custom_Tc_V(&do_top_simple,tb_custom_Tc);
	XTop_Set_R_in_V(&do_top_simple,tb_R_in);
	XTop_Set_C_in_V(&do_top_simple,tb_C_in);

	XTop_Start(&do_top_simple);//begin run top hls

	ofm_length=802816;// Tm*R*C=16*224*224
	weights_addr=weights;
	weights_length=1728;//M*N*tb_custom_k*tb_custom_k
				//=3*64*3*3
	output_addr=weights;
	output_length=1728;//M*N*tb_custom_k*tb_custom_k
	//=3*64*3*3

	XAxiDma_SimpleTransfer(&do_axi_dma[2],(INTPTR)output_addr,output_length*sizeof(DMA_DATA),XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_SimpleTransfer(&do_axi_dma[3],(INTPTR)weights_addr,weights_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);

	for(int b=0;b<bs;b++){
		for (int to=0;to<4;to++){//repeate[M/Tm]
			ofm_addr=ofm+b*3211264+to*802816;//M_in*R*C*b+toM*M*R*C+to*Tm*R*C=64*224*224*b+to*16*224*224
			//for(int ti=0;ti<4;ti++){
			XAxiDma_SimpleTransfer(&do_axi_dma[1],(INTPTR)ofm_addr,ofm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);
				ifm_length=1792;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
					//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=4*224*2;
				ifm_addr=ifm+b*200704;//+ti*802816;//b*R*N*C+ti*Tn*R*C
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				ifm_length=2688;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
								//4*224*3
				for(int row=tb_custom_Tr;row<tb_R-tb_custom_Tr;row+=tb_custom_Tr){
					ifm_addr=ifm+b*200704+(row-1)*896;//+ti*802816+(row-1)*896;//+3584*row;//b*R*N*C+ti*Tn*R*C+C_in*row*stride*min(Tn,N)+(row-1)*1*4*224
					while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
					//cout<<"IFM in Done!"<<endl;
					XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				}
				ifm_length=1792;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
								//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=4*224*2;
				ifm_addr=ifm+b*200704+198912;//(row-1)*896;//ti*802816+792064;//(row-1)*3584;//+3584*row;//b*R*N*C+ti*Tn*R*C+C_in*row*stride*min(Tn,N)+(row-1)*1*4*224
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
				//cout<<"IFM in Done!"<<endl;
				XAxiDma_SimpleTransfer(&do_axi_dma[0],(INTPTR)ifm_addr,ifm_length*sizeof(DMA_DATA),XAXIDMA_DMA_TO_DEVICE);//write IFM from off-chip
				while(XAxiDma_Busy(&do_axi_dma[0],XAXIDMA_DMA_TO_DEVICE));//
				//cout<<"IFM in Done!"<<endl;
				while(XAxiDma_Busy(&do_axi_dma[1],XAXIDMA_DMA_TO_DEVICE));
				//cout<<"OFM in Done!"<<endl;
			//}
		}
	}


	while(XAxiDma_Busy(&do_axi_dma[3],XAXIDMA_DMA_TO_DEVICE));
	//cout<<"Weights in Done!"<<endl;


	while(XAxiDma_Busy(&do_axi_dma[2],XAXIDMA_DEVICE_TO_DMA));
	//cout<<"output in Done!"<<endl;


	while(!XTop_IsDone(&do_top_simple));//top hls ip implement over
	//cout<<"IP done"<<endl;

	//timer.stopTimer();
	//timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"conv"<<outnum<<endl;
	//cout<<timeInterval<<" ";
	//printf("Elapsed Time (seconds) %f\n",timeInterval);

	bf_index=0;
	for(int to=0;to<tb_M_in;to+=Tm){
		for(int ti=0;ti<tb_N;ti+=Tn){
			for (int too=0;too< Tm;too+=4){
				for(int tii=0;tii<Tn;tii++){
					if((too+to)<tb_M_in && (tii+ti)<tb_N){
						for (int r=0;r<tb_custom_k;r++){
							for (int c=0;c<tb_custom_k;c++){
								weights_print[too+to][tii+ti][r][c] = weights[bf_index] ;//output.data;
								weights_print[too+to+1][tii+ti][r][c] = weights[bf_index+1] ;//output.data;
								weights_print[too+to+2][tii+ti][r][c] = weights[bf_index+2] ;//output.data;
								weights_print[too+to+3][tii+ti][r][c] = weights[bf_index+3] ;//output.data;
								bf_index+=4;
							}
						}
					}
				}
			}
		}
	}
	cout<<"1st 3OCs 1st 3ICs" <<endl;
	for (int m=0; m<3;m++){
		for (int n =0; n<3;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"1st 3OCs last 3ICs" <<endl;
	for (int m=0; m<3;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"" <<endl;
	cout<<"last 3OCs 1st 3ICs" <<endl;
	for (int m=tb_M_in-3; m<tb_M_in;m++){
		for (int n =0; n<3;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"last 3OCs last 3ICs" <<endl;
	for (int m=tb_M_in-3; m<tb_M_in;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c].data << ";";
				}
			}

		}
	}
	cout<<"" <<endl;
	cout<<"finish"<<endl;
	//printf("Elapsed Time (seconds) %f\n",timeInterval);
}











int main()
{
	cout<<"================================================================="<<endl;
	cout<<"===========================start Test=========================="<<endl;
	cout<<"================================================================="<<endl;

    init_platform();
    init();
//
    AxiTimer timer;
    double timeInterval;

	target0[0]=2;
	target0[1]=33;

	for(int b=0; b<bs;b++){
		for (int mo=0; mo<1000;mo++){
			//cout << output3[mo] << endl;
			if(mo==target0[b])
				target[b][mo].data=1.0;
			else
				target[b][mo].data=0.0;
		}
	}


	//FPGA_DATA input_x_buffer[bs*1*28*28];
	bf_index=0;
	for(int b=0;b<bs;b++){
		for(int to=0;to<4;to+=Tm){
			for (int r=0;r<224;r++){
				for (int c=0;c<224;c++){
					for (int too=0;too<Tm;too++){
						if((too+to)<4){
							if(too+to<3){
								input_x_buffer[bf_index].data =(((c+too+to)%29)*(0.5-b))/3.0+(((r+c)%7)*(b-0.5))/6.0-(((c+too+to)%11))/2.0;//output.data;
							}
							else{
								input_x_buffer[bf_index].data=0;
							}
							bf_index++;
						}
					}
				}
			}
		}
	}

	float mul;
	//FPGA_DATA weights_conv1_buffer[64*3*3*3];
	mul = (64%13)*(3%13)*3.0*3.0*400.0;
	bf_index=0;
	for(int to=0;to<64;to+=Tm){
		for(int ti=0;ti<3;ti+=Tn){
			for (int too=0;too<Tm;too+=4){
				for(int tii=0;tii<Tn;tii++){
					if((too+to)<64 && (tii+ti)<3){
						for (int r=0;r<3;r++){
							for (int c=0;c<3;c++){
								weights_conv1_buffer[bf_index].data = (2.0*r+3.0*c+((too+to+c)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv1_buffer[bf_index+1].data =(2.0*r+3.0*c+((too+to+c+1)%11) - ((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv1_buffer[bf_index+2].data = (2.0*r+3.0*c+((too+to+c+2)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv1_buffer[bf_index+3].data = (2.0*r+3.0*c+((too+to+c+3)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								gama1[too+to].data=1.0;
								gama1[too+to+1].data=1.0;
								gama1[too+to+2].data=1.0;
								gama1[too+to+3].data=1.0;
								beta1[too+to].data=0;
								beta1[too+to+1].data=0;
								beta1[too+to+2].data=0;
								beta1[too+to+3].data=0;
								bf_index+=4;
							}
						}
					}
				}
			}
		}
	}

//		cout<<"conv1:"<<endl;
//		bf_index=0;
//		for(int to=0;to<64;to+=Tm){
//			for(int ti=0;ti<3;ti+=Tn){
//				for (int too=0;too< Tm;too+=4){
//					for(int tii=0;tii<Tn;tii++){
//						if((too+to)<64 && (tii+ti)<3){
//							for (int r=0;r<3;r++){
//								for (int c=0;c<3;c++){
//									weights_conv1[too+to][tii+ti][r][c] = weights_conv1_buffer[bf_index] ;//output.data;
//									weights_conv1[too+to+1][tii+ti][r][c] = weights_conv1_buffer[bf_index+1] ;//output.data;
//									weights_conv1[too+to+2][tii+ti][r][c] = weights_conv1_buffer[bf_index+2] ;//output.data;
//									weights_conv1[too+to+3][tii+ti][r][c] = weights_conv1_buffer[bf_index+3] ;//output.data;
//									bf_index+=4;
//								}
//							}
//						}
//					}
//				}
//			}
//		}
//		for (int m=0; m<64;m++){
//			for (int n =0; n<3;n++){
//				for (int r=0;r<3;r++){
//					for (int c=0;c<3;c++){
//						cout<<weights_conv1[m][n][r][c] << ";";
//					}
//				}
//
//			}
//		}

	//FPGA_DATA weights_conv2_buffer[64*64*3*3];
	mul = (64%13)*(64%13)*3.0*3.0*400.0;
	bf_index=0;
	for(int to=0;to<64;to+=Tm){
		for(int ti=0;ti<64;ti+=Tn){
			for (int too=0;too<Tm;too+=4){
				for(int tii=0;tii<Tn;tii++){
					if((too+to)<64 && (tii+ti)<64){
						for (int r=0;r<3;r++){
							for (int c=0;c<3;c++){
								weights_conv2_buffer[bf_index].data = (2.0*r+3.0*c+((too+to+c)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv2_buffer[bf_index+1].data  =(2.0*r+3.0*c+((too+to+c+1)%11) - ((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv2_buffer[bf_index+2].data  = (2.0*r+3.0*c+((too+to+c+2)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv2_buffer[bf_index+3].data  = (2.0*r+3.0*c+((too+to+c+3)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								gama2[too+to].data =1.0;
								gama2[too+to+1].data =1.0;
								gama2[too+to+2].data =1.0;
								gama2[too+to+3].data =1.0;
								beta2[too+to].data =0;
								beta2[too+to+1].data =0;
								beta2[too+to+2].data =0;
								beta2[too+to+3].data =0;
								bf_index+=4;
							}
						}
					}
				}
			}
		}
	}


	//FPGA_DATA weights_conv3_buffer[128*64*3*3];
	mul = (128%13)*(64%13)*3.0*3.0*400.0;
	bf_index=0;
	for(int to=0;to<128;to+=Tm){
		for(int ti=0;ti<64;ti+=Tn){
			for (int too=0;too<Tm;too+=4){
				for(int tii=0;tii<Tn;tii++){
					if((too+to)<128 && (tii+ti)<64){
						for (int r=0;r<3;r++){
							for (int c=0;c<3;c++){
								weights_conv3_buffer[bf_index].data = (2.0*r+3.0*c+((too+to+c)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv3_buffer[bf_index+1].data  =(2.0*r+3.0*c+((too+to+c+1)%11) - ((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv3_buffer[bf_index+2].data  = (2.0*r+3.0*c+((too+to+c+2)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv3_buffer[bf_index+3].data  = (2.0*r+3.0*c+((too+to+c+3)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								gama3[too+to].data =1.0;
								gama3[too+to+1].data =1.0;
								gama3[too+to+2].data =1.0;
								gama3[too+to+3].data =1.0;
								beta3[too+to].data =0;
								beta3[too+to+1].data =0;
								beta3[too+to+2].data =0;
								beta3[too+to+3].data =0;
								bf_index+=4;
							}
						}
					}
				}
			}
		}
	}

	//FPGA_DATA weights_conv4_buffer[128*128*3*3];
	mul = (128%13)*(128%13)*3.0*3.0*400.0;
	bf_index=0;
	for(int to=0;to<128;to+=Tm){
		for(int ti=0;ti<128;ti+=Tn){
			for (int too=0;too<Tm;too+=4){
				for(int tii=0;tii<Tn;tii++){
					if((too+to)<128 && (tii+ti)<128){
						for (int r=0;r<3;r++){
							for (int c=0;c<3;c++){
								weights_conv4_buffer[bf_index].data = (2.0*r+3.0*c+((too+to+c)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv4_buffer[bf_index+1].data  =(2.0*r+3.0*c+((too+to+c+1)%11) - ((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv4_buffer[bf_index+2].data  = (2.0*r+3.0*c+((too+to+c+2)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv4_buffer[bf_index+3].data  = (2.0*r+3.0*c+((too+to+c+3)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								gama4[too+to].data =1.0;
								gama4[too+to+1].data =1.0;
								gama4[too+to+2].data =1.0;
								gama4[too+to+3].data =1.0;
								beta4[too+to].data =0;
								beta4[too+to+1].data =0;
								beta4[too+to+2].data =0;
								beta4[too+to+3].data =0;
								bf_index+=4;
							}
						}
					}
				}
			}
		}
	}

	//FPGA_DATA weights_conv5_buffer[256*128*3*3];
	mul = (256%13)*(128%13)*3.0*3.0*400.0;
	bf_index=0;
	for(int to=0;to<256;to+=Tm){
		for(int ti=0;ti<128;ti+=Tn){
			for (int too=0;too<Tm;too+=4){
				for(int tii=0;tii<Tn;tii++){
					if((too+to)<256 && (tii+ti)<128){
						for (int r=0;r<3;r++){
							for (int c=0;c<3;c++){
								weights_conv5_buffer[bf_index].data = (2.0*r+3.0*c+((too+to+c)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv5_buffer[bf_index+1].data  =(2.0*r+3.0*c+((too+to+c+1)%11) - ((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv5_buffer[bf_index+2].data  = (2.0*r+3.0*c+((too+to+c+2)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv5_buffer[bf_index+3].data  = (2.0*r+3.0*c+((too+to+c+3)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								gama5[too+to].data =1.0;
								gama5[too+to+1].data =1.0;
								gama5[too+to+2].data =1.0;
								gama5[too+to+3].data =1.0;
								beta5[too+to].data =0;
								beta5[too+to+1].data =0;
								beta5[too+to+2].data =0;
								beta5[too+to+3].data =0;
								bf_index+=4;
							}
						}
					}
				}
			}
		}
	}


	//FPGA_DATA weights_conv6_buffer[256*256*3*3];
	mul = (256%13)*(256%13)*3.0*3.0*400.0;
	bf_index=0;
	for(int to=0;to<256;to+=Tm){
		for(int ti=0;ti<256;ti+=Tn){
			for (int too=0;too<Tm;too+=4){
				for(int tii=0;tii<Tn;tii++){
					if((too+to)<256 && (tii+ti)<256){
						for (int r=0;r<3;r++){
							for (int c=0;c<3;c++){
								weights_conv6_buffer[bf_index].data = (2.0*r+3.0*c+((too+to+c)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv6_buffer[bf_index+1].data  =(2.0*r+3.0*c+((too+to+c+1)%11) - ((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv6_buffer[bf_index+2].data  = (2.0*r+3.0*c+((too+to+c+2)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv6_buffer[bf_index+3].data  = (2.0*r+3.0*c+((too+to+c+3)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								gama6[too+to].data =1.0;
								gama6[too+to+1].data =1.0;
								gama6[too+to+2].data =1.0;
								gama6[too+to+3].data =1.0;
								beta6[too+to].data =0;
								beta6[too+to+1].data =0;
								beta6[too+to+2].data =0;
								beta6[too+to+3].data =0;
								bf_index+=4;
							}
						}
					}
				}
			}
		}
	}

	//FPGA_DATA weights_conv7_buffer[256*256*3*3];
	mul = (256%13)*(256%13)*3.0*3.0*400.0;
	bf_index=0;
	for(int to=0;to<256;to+=Tm){
		for(int ti=0;ti<256;ti+=Tn){
			for (int too=0;too<Tm;too+=4){
				for(int tii=0;tii<Tn;tii++){
					if((too+to)<256 && (tii+ti)<256){
						for (int r=0;r<3;r++){
							for (int c=0;c<3;c++){
								weights_conv7_buffer[bf_index].data = (2.0*r+3.0*c+((too+to+c)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv7_buffer[bf_index+1].data  =(2.0*r+3.0*c+((too+to+c+1)%11) - ((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv7_buffer[bf_index+2].data  = (2.0*r+3.0*c+((too+to+c+2)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv7_buffer[bf_index+3].data  = (2.0*r+3.0*c+((too+to+c+3)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								gama7[too+to].data =1.0;
								gama7[too+to+1].data =1.0;
								gama7[too+to+2].data =1.0;
								gama7[too+to+3].data =1.0;
								beta7[too+to].data =0;
								beta7[too+to+1].data =0;
								beta7[too+to+2].data =0;
								beta7[too+to+3].data =0;
								bf_index+=4;
							}
						}
					}
				}
			}
		}
	}

	//FPGA_DATA weights_conv8_buffer[512*256*3*3];
	mul = (512%13)*(256%13)*3.0*3.0*400.0;
	bf_index=0;
	for(int to=0;to<512;to+=Tm){
		for(int ti=0;ti<256;ti+=Tn){
			for (int too=0;too<Tm;too+=4){
				for(int tii=0;tii<Tn;tii++){
					if((too+to)<512 && (tii+ti)<256){
						for (int r=0;r<3;r++){
							for (int c=0;c<3;c++){
								weights_conv8_buffer[bf_index].data = (2.0*r+3.0*c+((too+to+c)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv8_buffer[bf_index+1].data  =(2.0*r+3.0*c+((too+to+c+1)%11) - ((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv8_buffer[bf_index+2].data  = (2.0*r+3.0*c+((too+to+c+2)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv8_buffer[bf_index+3].data  = (2.0*r+3.0*c+((too+to+c+3)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								gama8[too+to].data =1.0;
								gama8[too+to+1].data =1.0;
								gama8[too+to+2].data =1.0;
								gama8[too+to+3].data =1.0;
								beta8[too+to].data =0;
								beta8[too+to+1].data =0;
								beta8[too+to+2].data =0;
								beta8[too+to+3].data =0;
								bf_index+=4;
							}
						}
					}
				}
			}
		}
	}

	//FPGA_DATA weights_conv9_buffer[512*512*3*3];
	mul = (512%13)*(512%13)*3.0*3.0*400.0;
	bf_index=0;
	for(int to=0;to<512;to+=Tm){
		for(int ti=0;ti<512;ti+=Tn){
			for (int too=0;too<Tm;too+=4){
				for(int tii=0;tii<Tn;tii++){
					if((too+to)<512 && (tii+ti)<512){
						for (int r=0;r<3;r++){
							for (int c=0;c<3;c++){
								weights_conv9_buffer[bf_index].data = (2.0*r+3.0*c+((too+to+c)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv9_buffer[bf_index+1].data  =(2.0*r+3.0*c+((too+to+c+1)%11) - ((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv9_buffer[bf_index+2].data  = (2.0*r+3.0*c+((too+to+c+2)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv9_buffer[bf_index+3].data  = (2.0*r+3.0*c+((too+to+c+3)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								gama9[too+to].data =1.0;
								gama9[too+to+1].data =1.0;
								gama9[too+to+2].data =1.0;
								gama9[too+to+3].data =1.0;
								beta9[too+to].data =0;
								beta9[too+to+1].data =0;
								beta9[too+to+2].data =0;
								beta9[too+to+3].data =0;
								bf_index+=4;
							}
						}
					}
				}
			}
		}
	}

	//FPGA_DATA weights_conv10_buffer[512*512*3*3];
	mul = (512%13)*(512%13)*3.0*3.0*400.0;
	bf_index=0;
	for(int to=0;to<512;to+=Tm){
		for(int ti=0;ti<512;ti+=Tn){
			for (int too=0;too<Tm;too+=4){
				for(int tii=0;tii<Tn;tii++){
					if((too+to)<512 && (tii+ti)<512){
						for (int r=0;r<3;r++){
							for (int c=0;c<3;c++){
								weights_conv10_buffer[bf_index].data = (2.0*r+3.0*c+((too+to+c)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv10_buffer[bf_index+1].data  =(2.0*r+3.0*c+((too+to+c+1)%11) - ((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv10_buffer[bf_index+2].data  = (2.0*r+3.0*c+((too+to+c+2)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv10_buffer[bf_index+3].data  = (2.0*r+3.0*c+((too+to+c+3)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								gama10[too+to].data =1.0;
								gama10[too+to+1].data =1.0;
								gama10[too+to+2].data =1.0;
								gama10[too+to+3].data =1.0;
								beta10[too+to].data =0;
								beta10[too+to+1].data =0;
								beta10[too+to+2].data =0;
								beta10[too+to+3].data =0;
								bf_index+=4;
							}
						}
					}
				}
			}
		}
	}

	//FPGA_DATA weights_conv11_buffer[512*512*3*3];
	mul = (512%13)*(512%13)*3.0*3.0*400.0;
	bf_index=0;
	for(int to=0;to<512;to+=Tm){
		for(int ti=0;ti<512;ti+=Tn){
			for (int too=0;too<Tm;too+=4){
				for(int tii=0;tii<Tn;tii++){
					if((too+to)<512 && (tii+ti)<512){
						for (int r=0;r<3;r++){
							for (int c=0;c<3;c++){
								weights_conv11_buffer[bf_index].data = (2.0*r+3.0*c+((too+to+c)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv11_buffer[bf_index+1].data  =(2.0*r+3.0*c+((too+to+c+1)%11) - ((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv11_buffer[bf_index+2].data  = (2.0*r+3.0*c+((too+to+c+2)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv11_buffer[bf_index+3].data  = (2.0*r+3.0*c+((too+to+c+3)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								gama11[too+to].data =1.0;
								gama11[too+to+1].data =1.0;
								gama11[too+to+2].data =1.0;
								gama11[too+to+3].data =1.0;
								beta11[too+to].data =0;
								beta11[too+to+1].data =0;
								beta11[too+to+2].data =0;
								beta11[too+to+3].data =0;
								bf_index+=4;
							}
						}
					}
				}
			}
		}
	}

	//FPGA_DATA weights_conv12_buffer[512*512*3*3];
	mul = (512%13)*(512%13)*3.0*3.0*400.0;
	bf_index=0;
	for(int to=0;to<512;to+=Tm){
		for(int ti=0;ti<512;ti+=Tn){
			for (int too=0;too<Tm;too+=4){
				for(int tii=0;tii<Tn;tii++){
					if((too+to)<512 && (tii+ti)<512){
						for (int r=0;r<3;r++){
							for (int c=0;c<3;c++){
								weights_conv12_buffer[bf_index].data = (2.0*r+3.0*c+((too+to+c)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv12_buffer[bf_index+1].data  =(2.0*r+3.0*c+((too+to+c+1)%11) - ((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv12_buffer[bf_index+2].data  = (2.0*r+3.0*c+((too+to+c+2)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv12_buffer[bf_index+3].data  = (2.0*r+3.0*c+((too+to+c+3)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								gama12[too+to].data =1.0;
								gama12[too+to+1].data =1.0;
								gama12[too+to+2].data =1.0;
								gama12[too+to+3].data =1.0;
								beta12[too+to].data =0;
								beta12[too+to+1].data =0;
								beta12[too+to+2].data =0;
								beta12[too+to+3].data =0;
								bf_index+=4;
							}
						}
					}
				}
			}
		}
	}

	//FPGA_DATA weights_conv13_buffer[512*512*3*3];
	mul = (512%13)*(512%13)*3.0*3.0*400.0;
	bf_index=0;
	for(int to=0;to<512;to+=Tm){
		for(int ti=0;ti<512;ti+=Tn){
			for (int too=0;too<Tm;too+=4){
				for(int tii=0;tii<Tn;tii++){
					if((too+to)<512 && (tii+ti)<512){
						for (int r=0;r<3;r++){
							for (int c=0;c<3;c++){
								weights_conv13_buffer[bf_index].data = (2.0*r+3.0*c+((too+to+c)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv13_buffer[bf_index+1].data  =(2.0*r+3.0*c+((too+to+c+1)%11) - ((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv13_buffer[bf_index+2].data  = (2.0*r+3.0*c+((too+to+c+2)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								weights_conv13_buffer[bf_index+3].data  = (2.0*r+3.0*c+((too+to+c+3)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
								gama13[too+to].data =1.0;
								gama13[too+to+1].data =1.0;
								gama13[too+to+2].data =1.0;
								gama13[too+to+3].data =1.0;
								beta13[too+to].data =0;
								beta13[too+to+1].data =0;
								beta13[too+to+2].data =0;
								beta13[too+to+3].data =0;
								bf_index+=4;
							}
						}
					}
				}
			}
		}
	}


	//FPGA_DATA weights_fc1_buffer[1000*512*1*1];
	mul = (1000%13)*(512%13)*1.0*1.0*400.0;
	bf_index=0;
	for(int to=0;to<1000;to+=Tm){
		for(int ti=0;ti<512;ti+=Tn){
			for (int too=0;too<Tm;too+=4){
				for(int tii=0;tii<Tn;tii++){
					if((too+to)<1000 && (tii+ti)<512){
						for (int r=0;r<1;r++){
							for (int c=0;c<1;c++){
								weights_fc1_buffer[bf_index].data = (2.0*r+3.0*c+((too+to+c)% 11) -((tii+ti+r)%23)+1)/mul;//output.data;
								weights_fc1_buffer[bf_index+1].data = (2.0*r+3.0*c+((too+to+c+1)% 11) - ((tii+ti+r)%23)+1)/mul;//output.data;
								weights_fc1_buffer[bf_index+2].data =(2.0*r+3.0*c+((too+to+c+2)% 11) -((tii+ti+r)%23)+1)/mul;//output.data;
								weights_fc1_buffer[bf_index+3].data = (2.0*r+3.0*c+((too+to+c+3)% 11) -((tii+ti+r)%23)+1)/mul;//output.data;
								bf_index+=4;
							}
						}
					}
				}
			}
		}
	}



	cout<<"Ready to send."<<endl;
	Xil_DCacheDisable();

//	timer.startTimer();
	convfp_3_64_224(1,input_x_buffer,weights_conv1_buffer,output1hat,gama1,beta1,namuda1,output1);
//	timer.stopTimer();
	timeInterval = timer.getElapsedTimerInSeconds();
	cout<<timeInterval<<" ";
	convfp_64_64_224(2,output1,weights_conv2_buffer,output2hat,gama2,beta2,namuda2,output2);
	maxpoolfp_64_112(3,output2,pool1index,output3);
	convfp_64_128_112(4,output3,weights_conv3_buffer,output4hat,gama3,beta3,namuda3,output4);
	convfp_128_128_112(5,output4,weights_conv4_buffer,output5hat,gama4,beta4,namuda4,output5);
	maxpoolfp_128_56(6,output5,pool2index,output6);
	convfp_128_256_56(7,output6,weights_conv5_buffer,output7hat,gama5,beta5,namuda5,output7);
	convfp_256_256_56(8,output7,weights_conv6_buffer,output8hat,gama6,beta6,namuda6,output8);
	convfp_256_256_56(9,output8,weights_conv7_buffer,output9hat,gama7,beta7,namuda7,output9);
	maxpoolfp_256_28(10,output9,pool3index,output10);
	convfp_256_512_28(11,output10,weights_conv8_buffer,output11hat,gama8,beta8,namuda8,output11);
	convfp_512_512_28(12,output11,weights_conv9_buffer,output12hat,gama9,beta9,namuda9,output12);
	convfp_512_512_28(13,output12,weights_conv10_buffer,output13hat,gama10,beta10,namuda10,output13);
	maxpoolfp_512_14(14,output13,pool4index,output14);
	convfp_512_512_14(15,output14,weights_conv11_buffer,output15hat,gama11,beta11,namuda11,output15);
	convfp_512_512_14(16,output15,weights_conv12_buffer,output16hat,gama12,beta12,namuda12,output16);
	convfp_512_512_14(17,output16,weights_conv13_buffer,output17hat,gama13,beta13,namuda13,output17);
	maxpoolfp_512_7(18,output17,pool5index,output18);
	avgpoolfp_512_1(19,output18,output19);
	convfp_512_1000_1(20,output19,weights_fc1_buffer,output20);


// calculate softmax loss
	cout<<"after loopM:"<<";";
	FPGA_DATA sum=0.0;


	    //cout<<tb_flag<<endl;
	   // cout<<"loss1:"<<endl;
	cout<<"XTOP start"<<endl;
	timer.startTimer();
	int loss_addr=0;
	for(int b=0; b<bs;b++){
		//cout<<"batch:"<<b<<"for loss1"<<endl;
		sum=0.0;
		for (int mo=0; mo<1000;mo++){
				loss1[loss_addr+mo].data	= exp(output20[loss_addr+mo].data);
				sum+=loss1[loss_addr+mo].data;
		}
		for (int mo=0; mo<1000;mo++){
			loss1[loss_addr+mo].data = (loss1[loss_addr+mo].data/sum-target[b][mo].data)/bs;
		}
		loss_addr+=1000;
	}


	timer.stopTimer();

	timeInterval = timer.getElapsedTimerInSeconds();
	cout<<"loss1 tim"<<endl;
	cout<<timeInterval<<" ";
	cout<<"loss1:"<<endl;
	loss_addr=0;
	for(int b=0; b<bs;b++){
		cout<<"batch:"<<b<<"for loss1"<<endl;
		for (int mo=0; mo<1000;mo++){
			cout<<loss1[loss_addr+mo].data<<";";
		}
		loss_addr+=10;

	}
	cout<<"finish"<<endl;
	printf("Elapsed Time (seconds) %f\n",timeInterval);


	 convbp_512_1000_1(2,loss1,weights_fc1_buffer,loss2);
	 convwu_512_1000_1(1,output19,weights_fc1_buffer,loss1);
	 avgpoolbp_512_1(3,loss2,loss3);
	 maxpoolbp_512_7(4,loss3,pool5index,output17,loss4);
	 convbp_512_512_14relu(5,loss4,weights_conv13_buffer,output17hat,gama13,beta13,namuda13,output16,loss5);
	 convwu_512_512_14(13,output16,weights_conv13_buffer,loss4,weights_conv13);
	convbp_512_512_14relu(6,loss5,weights_conv12_buffer,output16hat,gama12,beta12,namuda12,output15,loss6);
	convwu_512_512_14(12,output15,weights_conv12_buffer,loss5,weights_conv12);
	convbp_512_512_14(7,loss6,weights_conv11_buffer,output15hat,gama11,beta11,namuda11,loss7);
	convwu_512_512_14(11,output14,weights_conv11_buffer,loss6,weights_conv11);
	maxpoolbp_512_14(8,loss7,pool4index,output13,loss8);
	convbp_512_512_28(9,loss8,weights_conv10_buffer,output13hat,gama10,beta10,namuda10,output12,loss9);
	convwu_512_512_28(10,output12,weights_conv10_buffer,loss8,weights_conv10);
	convbp_512_512_28(10,loss9,weights_conv9_buffer,output12hat,gama9,beta9,namuda9,output11,loss10);
	convwu_512_512_28(9,output11,weights_conv9_buffer,loss9,weights_conv9);
	convbp_256_512_28(11,loss10,weights_conv8_buffer,output11hat,gama8,beta8,namuda8,loss11);
	convwu_256_512_28(8,output10,weights_conv8_buffer,loss10,weights_conv8);
	maxpoolbp_256_28(12,loss11,pool3index,output9,loss12);
	convbp_256_256_56(13,loss12,weights_conv7_buffer,output9hat,gama7,beta7,namuda7,output8,loss13);
	convwu_256_256_56(7,output8,weights_conv7_buffer,loss12,weights_conv7);
	convbp_256_256_56(14,loss13,weights_conv6_buffer,output8hat,gama6,beta6,namuda6,output7,loss14);
	convwu_256_256_56(6,output7,weights_conv6_buffer,loss13,weights_conv6);
	convbp_128_256_56(15,loss14,weights_conv5_buffer,output7hat,gama5,beta5,namuda5,loss15);
	convwu_128_256_56(5,output6,weights_conv5_buffer,loss14,weights_conv5);
	maxpoolbp_128_56(16,loss15,pool2index,output5,loss16);
	convbp_128_128_112(17,loss16,weights_conv4_buffer,output5hat,gama4,beta4,namuda4,output4,loss17);
	convwu_128_128_112(4,output4,weights_conv4_buffer,loss16,weights_conv4);
	convbp_64_128_112(18,loss17,weights_conv3_buffer,output4hat,gama3,beta3,namuda3,loss18);
	convwu_64_128_112(3,output3,weights_conv3_buffer,loss17,weights_conv3);
	maxpoolbp_64_112(19,loss18,pool1index,output2,loss19);
	convbp_64_64_224(20,loss19,weights_conv2_buffer,output2hat,gama2,beta2,namuda2,output1,loss20);
	convwu_64_64_224(2,output1,weights_conv2_buffer,loss19,weights_conv2);
	convbp_3_64_224(21,loss20,output1hat,gama1,beta1,namuda1);
	convwu_3_64_224(1,input_x_buffer,weights_conv1_buffer,loss20,weights_conv1);



	cleanup_platform();
	return 0;
}







