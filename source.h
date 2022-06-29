#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include "hls_math.h"

//#define FLOAT 1
typedef ap_uint<4> int4;
typedef ap_uint<6> int6;
typedef ap_uint<8> int8;
typedef ap_uint<14> int14;

//typedef half fixint8;
//typedef ap_ufixed<5,5,AP_TRN_ZERO, AP_SAT> unit8;
//typedef ap_ufixed<8,8,AP_TRN_ZERO, AP_SAT> int8;

const int Tm=16;//2;//16;
const int Tn=16;//2;//16;
//const int Tr=32;//4;//8;
//const int Tc=32;//4;//8;

const int Trtc=2016;

//const int stride=1;
const int K=3;
const int K2=9;


//const int Weights_bound=K2*20;
const int TiTo=64;
const int BNbank=8;//512/4
const int BNindex=128;//512/4
const int DMAwidth=4;//512/4
//const int Tr_bound=34;//(Tr-1)*stride+K;
//const int Tc_bound=34;//(Tc-1)*stride+K;
const int Trtcin=2486;//(Tc-1)*stride+K;
//const int M1=6;
//const int R1=28;
//const int C1=28;
//const int M2=6;
//const int R2=14;
//const int C2=14;
//const int M3=16;
//const int R3=10;
//const int C3=10;
//const int M4=16;
//const int R4=5;
//const int C4=5;
//const int M5=120;
//const int R5=1;
//const int C5=1;
//const int M6=84;
//const int R6=1;
//const int C6=1;
//const int Mp=5;//[50/11]
//const int Rp=12;
//const int Cp=12;
const int Mp=100352;//112*112*64/8
//const int Rp=112;
//const int Cp=112;
//const int pool_k=2;
const float learn_rate =0.1;
const float minimal=-3.40282e+038 ;
const float eps=1.0e-05 ;

//const int M=128;
//const int N=192;

//const int M=32;
//xuxiao516898
//kikiki426
//Love-xiBaby



//#if FLOAT==1
typedef float FPGA_DATA;
//typedef ap_ufixed<1,1,AP_TRN_ZERO, AP_SAT> Relu_index;//unsigned
//typedef ap_ufixed<1,1,AP_TRN_ZERO, AP_SAT> Pool_index;//ap_uint<1> Pool_index;
//typedef ap_ufixed<32,32,AP_TRN_ZERO, AP_SAT> ufix32;
typedef ap_uint<1> Pool_index;
typedef ap_uint<16> ufix16;
typedef ap_uint<32> ufix32;
//#else
//	typedef ap_fixed<32,10,AP_TRN_ZERO, AP_SAT> FPGA_DATA;//signed
//	typedef ap_fixed<32,1,AP_TRN_ZERO, AP_SAT> FPGA_WEIGHTS;
//#endif

struct Four{
	FPGA_DATA data1;
	FPGA_DATA data2;
	FPGA_DATA data3;
	FPGA_DATA data4;
};

struct DOUBLE{
	FPGA_DATA data1;
	FPGA_DATA data2;
};


//struct DMA_index{
//		ufix16 data;
//		//Four data;
//		bool last;
//};


struct DMA_DATA{
		FPGA_DATA data;
		//Four data;
		bool last;
};

struct DMA_DATA_64{
	DOUBLE data;
	bool last;
};

struct DMA_DATA_128{
	Four data;
	bool last;
};


struct Pool_index_dim2{
	Pool_index data1;
	Pool_index data2;
};


//struct Pool_22{
//	Pool_index_dim2 data1;
//	Pool_index_dim2 data2;
//	Pool_index_dim2 data3;
//	Pool_index_dim2 data4;
//	Pool_index_dim2 data5;
//	Pool_index_dim2 data6;
//	Pool_index_dim2 data7;
//	Pool_index_dim2 data8;
//	Pool_index_dim2 data9;
//	Pool_index_dim2 data10;
//	Pool_index_dim2 data11;
//
//};

