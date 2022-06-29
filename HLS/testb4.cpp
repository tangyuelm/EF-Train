
#include "source.h"
//
using namespace std;

//int min (int a,int b){
//	int y;
//	if (a>b)
//			y = b;
//		else
//			y=a;
//		return y;
//
//}


void top(   hls::stream<DMA_DATA_128> &dma_IFM,
		hls::stream<DMA_DATA_128> &dma_Weights,
		hls::stream<DMA_DATA_128> &dma_Weightsout,
		hls::stream<DMA_DATA_128> &dma_OFM,
		hls::stream<DMA_DATA_128> &dma_Output,
		//hls::stream<DMA_index> &dma_Indexin,
		//hls::stream<DMA_DATA_128> &dma_Indexout,
		//int statec,
		ap_uint<3> statec,
		ap_uint<4> statep,
		ap_uint<4> stateb,
		//int4 Relulayerin,
		//int4 Relulayerout,
		//int4 Poollayerin,
		//int4 Poollayerout,
		int8 custom_batch,
		int8 batch_size,
		int14 M_in,
		int8 custom_Tib,
		//int14 custom_k2,
		//int to,
		int14 N,
		int14 M,
		int8 R,
		int8 C,
		int4 custom_stride,
		int4 padding,
		int4 custom_kb,
		int4 custom_k,
		int8 custom_Tr,
		int8 custom_Tc,
//		int8 viewTrc,
//		int14 viewTm,
		int8 R_in,
		int8 C_in
		//int8& flag
			);


hls::stream<DMA_DATA_128> tb_input_dma_ifm("my_stream_name i");    //ifm
		hls::stream<DMA_DATA_128> tb_input_dma_weights("my_stream_name w");    //weights
		hls::stream<DMA_DATA_128> tb_output_dma_Weightsout("my_stream_name wout");    //bn 2nd out
		hls::stream<DMA_DATA_128> tb_input_dma_ofm("my_stream_name o");    //ofm
		hls::stream<DMA_DATA_128> tb_output_dma_output("my_stream_name out"); //output
		//hls::stream<DMA_index> tb_dma_Indexin("my_stream_name indexin");    //ofm
		//hls::stream<DMA_DATA_128> tb_dma_Indexout("my_stream_name indexout"); //output
		//int tb_statec;
		ap_uint<3> tb_statec;
		ap_uint<4> tb_statep;
		ap_uint<4> tb_stateb;
		//int4 tb_Relulayerin;
		//int4 tb_Relulayerout;
		//int4 tb_Poollayerin;
		//int4 tb_Poollayerout;
		//int tb_to;
		int8 tb_custom_batch;
		int8 tb_batch_size;
		int14 tb_M_in;
		int8 tb_custom_Tib;
		//int14 tb_custom_k2;
		int14 tb_N;
		int14 tb_M;
		int8 tb_R;
		int8 tb_C;
		int4 tb_custom_stride;
		int4 tb_padding;
		int4 tb_custom_kb;
		int4 tb_custom_k;
		int8 tb_custom_Tr;
		int8 tb_custom_Tc;
//		int8 tb_viewTrc;
//		int14 tb_viewTm;
		int8 tb_R_in;
		int8 tb_C_in;
		int8 tb_flag;
		//cout << tb_flag << endl;
		DMA_DATA_128 input_ifm;
		DMA_DATA_128 input_weights;
		DMA_DATA_128 out_weights;
		DMA_DATA_128 input_ofm;
		DMA_DATA_128 output;
		DMA_DATA_128 pool_index;
		int bf_index;

	const  int bs=2;

	static FPGA_DATA input_x_buffer[bs*4*224*224];
	static FPGA_DATA weights_conv1_buffer[64*3*3*3];
	static FPGA_DATA gama1[64];
	static FPGA_DATA beta1[64];
	static FPGA_DATA namuda1[64];
	static FPGA_DATA weights_conv2_buffer[64*64*3*3];
	static FPGA_DATA gama2[64];
	static FPGA_DATA beta2[64];
	static FPGA_DATA namuda2[64];
	static FPGA_DATA weights_conv3_buffer[64*128*3*3];
	static FPGA_DATA gama3[128];
	static FPGA_DATA beta3[128];
	static FPGA_DATA namuda3[128];
	static FPGA_DATA weights_conv4_buffer[128*128*3*3];
	static FPGA_DATA gama4[128];
	static FPGA_DATA beta4[128];
	static FPGA_DATA namuda4[128];
	static FPGA_DATA weights_conv5_buffer[128*256*3*3];
	static FPGA_DATA gama5[256];
	static FPGA_DATA beta5[256];
	static FPGA_DATA namuda5[256];
	static FPGA_DATA weights_conv6_buffer[256*256*3*3];
	static FPGA_DATA gama6[256];
	static FPGA_DATA beta6[256];
	static FPGA_DATA namuda6[256];
	static FPGA_DATA weights_conv7_buffer[256*256*3*3];
	static FPGA_DATA gama7[256];
	static FPGA_DATA beta7[256];
	static FPGA_DATA namuda7[256];
	static FPGA_DATA weights_conv8_buffer[256*512*3*3];
	static FPGA_DATA gama8[512];
	static FPGA_DATA beta8[512];
	static FPGA_DATA namuda8[512];
	static FPGA_DATA weights_conv9_buffer[512*512*3*3];
	static FPGA_DATA gama9[512];
	static FPGA_DATA beta9[512];
	static FPGA_DATA namuda9[512];
	static FPGA_DATA weights_conv10_buffer[512*512*3*3];
	static FPGA_DATA gama10[512];
	static FPGA_DATA beta10[512];
	static FPGA_DATA namuda10[512];
	static FPGA_DATA weights_conv11_buffer[512*512*3*3];
	static FPGA_DATA gama11[512];
	static FPGA_DATA beta11[512];
	static FPGA_DATA namuda11[512];
	static FPGA_DATA weights_conv12_buffer[512*512*3*3];
	static FPGA_DATA gama12[512];
	static FPGA_DATA beta12[512];
	static FPGA_DATA namuda12[512];
	static FPGA_DATA weights_conv13_buffer[512*512*3*3];
	static FPGA_DATA gama13[512];
	static FPGA_DATA beta13[512];
	static FPGA_DATA namuda13[512];
	static FPGA_DATA weights_fc1_buffer[1000*512*1*1];

	static FPGA_DATA output1[bs*224*224*64]; //[r][c][m]
	static FPGA_DATA output1hat[bs*224*224*64]; //[r][c][m]
	static FPGA_DATA output2[bs*224*224*64];//[r][c][m]
	static FPGA_DATA output2hat[bs*224*224*64];//[r][c][m]
	static FPGA_DATA output3[bs*112*112*64];//[r][c][m]
	static FPGA_DATA pool1index[bs*112*112*8];//r*c*[m/8]

	static FPGA_DATA output4[bs*112*112*128];//[r][c][m]
	static FPGA_DATA output4hat[bs*112*112*128];//[r][c][m]
	static FPGA_DATA output5[bs*112*112*128];//[r][c][m]
	static FPGA_DATA output5hat[bs*112*112*128];//[r][c][m]
	static FPGA_DATA output6[bs*56*56*128];//[r][c][m]
	static FPGA_DATA pool2index[bs*56*56*16];//r*c*[m/8]

	static FPGA_DATA output7[bs*56*56*256];//[r][c][m]
	static FPGA_DATA output7hat[bs*56*56*256];//[r][c][m]
	static FPGA_DATA output8[bs*56*56*256];//[r][c][m]
	static FPGA_DATA output8hat[bs*56*56*256];//[r][c][m]
	static FPGA_DATA output9[bs*56*56*256];//[r][c][m]
	static FPGA_DATA output9hat[bs*56*56*256];//[r][c][m]
	static FPGA_DATA output10[bs*28*28*256];//[r][c][m]
	static FPGA_DATA pool3index[bs*28*28*32];//r*c*[m/8]

	static FPGA_DATA output11[bs*28*28*512];//[r][c][m]
	static FPGA_DATA output11hat[bs*28*28*512];//[r][c][m]
	static FPGA_DATA output12[bs*28*28*512];//[r][c][m]
	static FPGA_DATA output12hat[bs*28*28*512];//[r][c][m]
	static FPGA_DATA output13[bs*28*28*512];//[r][c][m]
	static FPGA_DATA output13hat[bs*28*28*512];//[r][c][m]
	static FPGA_DATA output14[bs*14*14*512];//[r][c][m]
	static FPGA_DATA pool4index[bs*14*14*64];//r*c*[m/8]

	static FPGA_DATA output15[bs*14*14*512];//[r][c][m]
	static FPGA_DATA output15hat[bs*14*14*512];//[r][c][m]
	static FPGA_DATA output16[bs*14*14*512];//[r][c][m]
	static FPGA_DATA output16hat[bs*14*14*512];//[r][c][m]
	static FPGA_DATA output17[bs*14*14*512];//[r][c][m]
	static FPGA_DATA output17hat[bs*14*14*512];//[r][c][m]
	static FPGA_DATA output18[bs*7*7*512];//[r][c][m]
	static FPGA_DATA pool5index[bs*7*7*64];//r*c*[m/8]

	static FPGA_DATA output19[bs*1*1*512];//[r][c][m]

	static FPGA_DATA output20[bs*1000];//[r][c][m]


	static FPGA_DATA loss1[bs*1000];
	static FPGA_DATA loss2[bs*1*1*512];
	static FPGA_DATA loss3[bs*7*7*512];
	static FPGA_DATA loss4[bs*14*14*512];
	static FPGA_DATA loss5[bs*14*14*512];//[r][c][m]
	static FPGA_DATA loss6[bs*14*14*512];//[r][c][m]
	static FPGA_DATA loss7[bs*14*14*512];//[r][c][m]
	static FPGA_DATA loss8[bs*28*28*512];//[r][c][m]
	static FPGA_DATA loss9[bs*28*28*512];//[r][c][m]
	static FPGA_DATA loss10[bs*28*28*512];//[r][c][m]
	static FPGA_DATA loss11[bs*28*28*256];//[r][c][m]
	static FPGA_DATA loss12[bs*56*56*256];//[r][c][m]
	static FPGA_DATA loss13[bs*56*56*256];//[r][c][m]
	static FPGA_DATA loss14[bs*56*56*256];//[r][c][m]
	static FPGA_DATA loss15[bs*56*56*128];//[r][c][m]
	static FPGA_DATA loss16[bs*112*112*128];//[r][c][m]
	static FPGA_DATA loss17[bs*112*112*128];//[r][c][m]
	static FPGA_DATA loss18[bs*112*112*64];//[r][c][m]
	static FPGA_DATA loss19[bs*224*224*64];//[r][c][m]
	static FPGA_DATA loss20[bs*224*224*64];//[r][c][m]


//	static FPGA_DATA loss4_print[bs][8][8][50]; //[r][c][m]
//	static FPGA_DATA loss5_print[bs][12][12][20]; //[r][c][m]
//	static FPGA_DATA loss6_print[bs][24][24][20];//[r][c][m]
	static FPGA_DATA weights_conv1[64][3][3][3];//[m][n][r][c]
	static FPGA_DATA weights_conv2[64][64][3][3];
	static FPGA_DATA weights_conv3[128][64][3][3];
	static FPGA_DATA weights_conv4[128][128][3][3];
	static FPGA_DATA weights_conv5[256][128][3][3];
	static FPGA_DATA weights_conv6[256][256][3][3];
	static FPGA_DATA weights_conv7[256][256][3][3];
	static FPGA_DATA weights_conv8[512][256][3][3];
	static FPGA_DATA weights_conv9[512][512][3][3];
	static FPGA_DATA weights_conv10[512][512][3][3];
	static FPGA_DATA weights_conv11[512][512][3][3];
	static FPGA_DATA weights_conv12[512][512][3][3];
	static FPGA_DATA weights_conv13[512][512][3][3];


	int ifm_length;
	int weights_length;
	int wout_length;
	int ofm_length;
	int output_length;
	int ifm_addr;
	int weights_addr;
	int wout_addr;
	int ofm_addr;
	int output_addr;



void convfp_3_64_224( int outnum,
	FPGA_DATA ifm[bs*4*224*224],
	FPGA_DATA weights[64*3*3*3],
	FPGA_DATA outhat[bs*64*224*224],
	FPGA_DATA gama[64],
	FPGA_DATA beta[64],
	FPGA_DATA namuda[64],
	FPGA_DATA out[bs*64*224*224]

){
//forward conv1 N,M,R,C=3,64,224,224, stride=1,padding=1,kernel=3
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


	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ifm_addr=0;
	weights_addr=0;

	weights_length=1728;//M*N*tb_custom_k*tb_custom_k=3*64*3*3
	for(int b=0;b<bs;b++){
		for (int to=0;to<tb_M;to+=Tm){//repeate [M/Tm]
			ifm_length=8960;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
									//*((C-1)*tb_custom_stride+tb_custom_k)=4*224*10//64bits N is multiple of 2
			ifm_addr=200704*b;//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*4*224*224+row*4*224*1
			for(int i=0;i<ifm_length;i+=4){
				input_ifm.data.data1 = ifm[i+ifm_addr];
				input_ifm.data.data2 = ifm[i+ifm_addr+1];
				input_ifm.data.data3 = ifm[i+ifm_addr+2];
				input_ifm.data.data4 = ifm[i+ifm_addr+3];
				tb_input_dma_ifm.write(input_ifm);
			}
			for(int row=tb_custom_Tr;row<tb_R-8;row+=tb_custom_Tr){
				ifm_length=9856;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
										//*((C-1)*tb_custom_stride+tb_custom_k)=4*224*11//64bits N is multiple of 2
				ifm_addr=200704*b+896*(row-1);//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*4*224*224+row*4*224*1
				for(int i=0;i<ifm_length;i+=4){
					input_ifm.data.data1 = ifm[i+ifm_addr];
					input_ifm.data.data2 = ifm[i+ifm_addr+1];
					input_ifm.data.data3 = ifm[i+ifm_addr+2];
					input_ifm.data.data4 = ifm[i+ifm_addr+3];
					tb_input_dma_ifm.write(input_ifm);
				}
			}
			//for(int row=tb_R-1;row<tb_R;row+=tb_custom_Tr){
			ifm_length=8064;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
									//*((C-1)*tb_custom_stride+tb_custom_k)=4*224*9;//64bits N is multiple of 2
			ifm_addr=200704*b+192640;//896*(row-1)=896*(216-1);//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*4*224*224+row*4*224*1
			//for(int ti=0;ti<1;ti++){
				//ifm_length=448;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
										//*((C-1)*tb_custom_stride+tb_custom_k)=1*16*28;
			for(int i=0;i<ifm_length;i+=4){
				input_ifm.data.data1 = ifm[i+ifm_addr];
				input_ifm.data.data2 = ifm[i+ifm_addr+1];
				input_ifm.data.data3 = ifm[i+ifm_addr+2];
				input_ifm.data.data4 = ifm[i+ifm_addr+3];
				tb_input_dma_ifm.write(input_ifm);
			}
		}
	}

	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = weights[i+weights_addr];
		input_weights.data.data2 = weights[i+weights_addr+1];
		input_weights.data.data3 = weights[i+weights_addr+2];
		input_weights.data.data4 = weights[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}								//=6*3*5*5

	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	//cout<<tb_flag<<endl;

	//print output
	output_addr=0;
	output_length=bs*3211264;//min(Tm,M-Tm*to)*min(custom_Tr,R-row*custom_Tr)*C=64*224*224;

	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		out[i+output_addr] = output.data.data1;
		out[i+output_addr+1] = output.data.data2;
		out[i+output_addr+2] = output.data.data3;
		out[i+output_addr+3] = output.data.data4;
	}

	cout<<"out after conv"<<endl;
	cout<<out[0]<<endl;

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


	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ofm_addr=0;
	weights_addr=0;
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = gama[i+weights_addr];
		input_weights.data.data2 = gama[i+weights_addr+1];
		input_weights.data.data3 = gama[i+weights_addr+2];
		input_weights.data.data4 = gama[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}
	for(int i=0;i<ofm_length;i+=4){
		input_ofm.data.data1 = beta[i+ofm_addr];
		input_ofm.data.data2 = beta[i+ofm_addr+1];
		input_ofm.data.data3 = beta[i+ofm_addr+2];
		input_ofm.data.data4 = beta[i+ofm_addr+3];
		tb_input_dma_ofm.write(input_ofm);
	}
	//ifm_addr=0;
	//ofm_addr=0;
	//ofm_length=802816;//Tm*R*C
	ifm_length=802816;//Tm*R*C
	for (int to=0;to<4;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			//ofm_addr=3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			for(int i=0;i<ifm_length;i+=4){
				input_ifm.data.data1 = out[i+ifm_addr];
				input_ifm.data.data2 = out[i+ifm_addr+1];
				input_ifm.data.data3 = out[i+ifm_addr+2];
				input_ifm.data.data4 = out[i+ifm_addr+3];
				tb_input_dma_ifm.write(input_ifm);
			}
		}
	}
	ofm_addr=0;
	ofm_length=3211264;//M*R*C
	for(int b=0;b<bs;b++){
		//ifm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		//ofm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		for(int i=0;i<ofm_length;i+=4){
			input_ofm.data.data1 = out[i+ofm_addr];
			input_ofm.data.data2 = out[i+ofm_addr+1];
			input_ofm.data.data3 = out[i+ofm_addr+2];
			input_ofm.data.data4 = out[i+ofm_addr+3];
			tb_input_dma_ofm.write(input_ofm);
		}
		//ifm_addr+=ifm_length;
		ofm_addr+=ofm_length;
	}
						//=6*3*5*5

	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	//cout<<tb_flag<<endl;

	//print output
	//output_addr=0;
	output_addr=0;
	output_length=3211264;//M*R*C
	for(int b=0;b<bs;b++){
		for(int i=0;i<output_length;i+=4){
			out_weights =tb_output_dma_Weightsout.read();
			outhat[i+output_addr] = out_weights.data.data1;
			outhat[i+output_addr+1] = out_weights.data.data2;
			outhat[i+output_addr+2] = out_weights.data.data3;
			outhat[i+output_addr+3] = out_weights.data.data4;
		}
		for(int i=0;i<output_length;i+=4){
			output =tb_output_dma_output.read();
			out[i+output_addr] = output.data.data1;
			out[i+output_addr+1] = output.data.data2;
			out[i+output_addr+2] = output.data.data3;
			out[i+output_addr+3] = output.data.data4;
		}
		output_addr+=output_length;
	}
	output_addr=0;
	output_length=tb_M;//M
	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		namuda[i+output_addr] = output.data.data1;
		namuda[i+output_addr+1] = output.data.data2;
		namuda[i+output_addr+2] = output.data.data3;
		namuda[i+output_addr+3] = output.data.data4;
	}

	cout<<"print out"<<outnum<<endl;


	cout<<"after loopM:"<<endl;


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
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
	cout<<"namuda"<<endl;
	cout<<namuda[0]<<endl;
	cout<<"outhat"<<endl;
	cout<<outhat[0]<<endl;
}

void convfp_64_64_224( int outnum,
	FPGA_DATA ifm[bs*64*224*224],
	FPGA_DATA weights[64*64*3*3],
	FPGA_DATA outhat[bs*64*224*224],
	FPGA_DATA gama[64],
	FPGA_DATA beta[64],
	FPGA_DATA namuda[64],
	FPGA_DATA out[bs*64*224*224]

){
	//forward conv2 N,M,R,C=64,64,224,224, stride=1,padding=1,kernel=3
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

	//FPGA_DATA output2[bs*24*24*20]; //[r][c][m]
	ifm_addr=0;
	weights_addr=0;

	weights_length=36864;//M*N*tb_custom_k*tb_custom_k=64*64*3*3
	//for (int toM=0;toM<2;toM++){
		for(int b=0;b<bs;b++){
			for (int to=0;to<tb_M;to+=Tm){//repeate [M/Tm]
				ifm_length=32256;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
										//*((C-1)*tb_custom_stride+tb_custom_k)=16*9*224;
				ifm_addr=3211264*b;//+3584*row;//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*64*224*224+row*16*224*1
				for(int ti=0;ti<tb_N;ti+=Tn){
					for(int i=0;i<ifm_length;i+=4){
						input_ifm.data.data1 = ifm[i+ifm_addr];
						input_ifm.data.data2 = ifm[i+ifm_addr+1];
						input_ifm.data.data3 = ifm[i+ifm_addr+2];
						input_ifm.data.data4 = ifm[i+ifm_addr+3];
						tb_input_dma_ifm.write(input_ifm);
					}
					ifm_addr+=802816;//Tn*R_in*C_in=16*224*224
				}
				for(int row=tb_custom_Tr;row<tb_R-tb_custom_Tr;row+=tb_custom_Tr){
					ifm_length=35840;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
															//*((C-1)*tb_custom_stride+tb_custom_k)=16*10*224;
					ifm_addr=3211264*b+(row-1)*3584;//+3584*row;//b*N*C_in*R_in+C_in*(row-padding)*stride*min(Tn,N)=b*96*27*27+(row-2)*1*27*16
					for(int ti=0;ti<tb_N;ti+=Tn){
						for(int i=0;i<ifm_length;i+=4){
							input_ifm.data.data1 = ifm[i+ifm_addr];
							input_ifm.data.data2 = ifm[i+ifm_addr+1];
							input_ifm.data.data3 = ifm[i+ifm_addr+2];
							input_ifm.data.data4 = ifm[i+ifm_addr+3];
							tb_input_dma_ifm.write(input_ifm);
						}
						ifm_addr+=802816;//Tn*R_in*C_in=16*224*224
					}
				}
				ifm_length=32256;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
										//*((C-1)*tb_custom_stride+tb_custom_k)=16*9*224;
				ifm_addr=3211264*b+770560;//+3584*(row-1);//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*64*224*224+row*16*224*1
				for(int ti=0;ti<tb_N;ti+=Tn){
					for(int i=0;i<ifm_length;i+=4){
						input_ifm.data.data1 = ifm[i+ifm_addr];
						input_ifm.data.data2 = ifm[i+ifm_addr+1];
						input_ifm.data.data3 = ifm[i+ifm_addr+2];
						input_ifm.data.data4 = ifm[i+ifm_addr+3];
						tb_input_dma_ifm.write(input_ifm);
					}
					ifm_addr+=802816;//Tn*R_in*C_in=16*224*224
				}
			}
		}
	//}



	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = weights[i+weights_addr];
		input_weights.data.data2 = weights[i+weights_addr+1];
		input_weights.data.data3 = weights[i+weights_addr+2];
		input_weights.data.data4 = weights[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}								//=6*3*5*5

	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	//cout<<tb_flag<<endl;

	//print output
	output_addr=0;
	//for (int toM=0;toM<2;toM++){
		output_length=bs*3211264;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=224*224*64;
		//output_addr=toM*81648;
		//for(int b=0;b<bs;b++){
			for(int i=0;i<output_length;i+=4){
				output =tb_output_dma_output.read();
				out[i+output_addr] = output.data.data1;
				out[i+output_addr+1] = output.data.data2;
				out[i+output_addr+2] = output.data.data3;
				out[i+output_addr+3] = output.data.data4;
			}
			//output_addr+=186624;
		//}
	//}
	cout<<"out after conv"<<endl;
	cout<<out[0]<<endl;

//forward bn2+relu N,M,R,C=64,64,224,224, stride=1,padding=1,kernel=3
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


	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ofm_addr=0;
	weights_addr=0;
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = gama[i+weights_addr];
		input_weights.data.data2 = gama[i+weights_addr+1];
		input_weights.data.data3 = gama[i+weights_addr+2];
		input_weights.data.data4 = gama[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}
	for(int i=0;i<ofm_length;i+=4){
		input_ofm.data.data1 = beta[i+ofm_addr];
		input_ofm.data.data2 = beta[i+ofm_addr+1];
		input_ofm.data.data3 = beta[i+ofm_addr+2];
		input_ofm.data.data4 = beta[i+ofm_addr+3];
		tb_input_dma_ofm.write(input_ofm);
	}
	//ifm_addr=0;
	//ofm_addr=0;
	ifm_length=802816;//Tm*R*C
	for (int to=0;to<4;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			//ofm_addr=3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			for(int i=0;i<ifm_length;i+=4){
				input_ifm.data.data1 = out[i+ifm_addr];
				input_ifm.data.data2 = out[i+ifm_addr+1];
				input_ifm.data.data3 = out[i+ifm_addr+2];
				input_ifm.data.data4 = out[i+ifm_addr+3];
				tb_input_dma_ifm.write(input_ifm);
			}
		}
	}
	ofm_addr=0;
	ofm_length=3211264;//M*R*C
	for(int b=0;b<bs;b++){
		//ifm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		//ofm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		for(int i=0;i<ofm_length;i+=4){
			input_ofm.data.data1 = out[i+ofm_addr];
			input_ofm.data.data2 = out[i+ofm_addr+1];
			input_ofm.data.data3 = out[i+ofm_addr+2];
			input_ofm.data.data4 = out[i+ofm_addr+3];
			tb_input_dma_ofm.write(input_ofm);
		}
		//ifm_addr+=ifm_length;
		ofm_addr+=ofm_length;
	}						//=6*3*5*5

	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	//cout<<tb_flag<<endl;

	output_addr=0;
	output_length=3211264;//M*R*C
	for(int b=0;b<bs;b++){
		for(int i=0;i<output_length;i+=4){
			out_weights =tb_output_dma_Weightsout.read();
			outhat[i+output_addr] = out_weights.data.data1;
			outhat[i+output_addr+1] = out_weights.data.data2;
			outhat[i+output_addr+2] = out_weights.data.data3;
			outhat[i+output_addr+3] = out_weights.data.data4;
		}
		for(int i=0;i<output_length;i+=4){
			output =tb_output_dma_output.read();
			out[i+output_addr] = output.data.data1;
			out[i+output_addr+1] = output.data.data2;
			out[i+output_addr+2] = output.data.data3;
			out[i+output_addr+3] = output.data.data4;
		}
		output_addr+=output_length;
	}
	output_addr=0;
	output_length=tb_M;//M
	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		namuda[i+output_addr] = output.data.data1;
		namuda[i+output_addr+1] = output.data.data2;
		namuda[i+output_addr+2] = output.data.data3;
		namuda[i+output_addr+3] = output.data.data4;
	}

	cout<<"namuda"<<endl;
	cout<<namuda[0]<<endl;
	cout<<"outhat"<<endl;
	cout<<outhat[0]<<endl;


	cout<<"print out"<<outnum<<endl;


	cout<<"after loopM:"<<endl;


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
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<""<<endl;
	}




	cout<<"print out"<<outnum<<endl;


	cout<<"after loopM:"<<endl;


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
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
}

void maxpoolfp_64_112( int outnum,
	FPGA_DATA ifm[bs*64*224*224],
	FPGA_DATA weights[bs*112*112*8],
	FPGA_DATA out[bs*64*112*112]

){
	//forward pool N,M,R,C=64,64,112,112, stride=2,padding=0,kernel=2
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



	ifm_addr=0;
	ifm_length=bs*3211264;//M*R_in*C_in=224*224*64;
	//for(int b=0;b<bs;b++){
		for(int i=0;i<ifm_length;i+=4){
			input_ifm.data.data1 = ifm[i+ifm_addr];
			input_ifm.data.data2 = ifm[i+ifm_addr+1];
			input_ifm.data.data3 = ifm[i+ifm_addr+2];
			input_ifm.data.data4 = ifm[i+ifm_addr+3];
			tb_input_dma_ifm.write(input_ifm);
		}
		//ifm_addr+=ifm_length;
	//}




	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	cout<<tb_flag<<endl;

//print output
	//FPGA_DATA output2[bs*12*12*20];//[r][c][m]
   //FPGA_DATA pool2index[bs*12*12*2];//r*c*[m/11]

	output_addr=0;
	weights_addr=0;
	output_length=802816;//M*min(custom_Tr,R-row*custom_Tr)*C=64*112*112;
	weights_length=100352;//r*c*[m/8]=112*112*8
	for(int b=0;b<bs;b++){
		for(int i=0;i<output_length;i+=4){
			output =tb_output_dma_output.read();
			out[i+output_addr] = output.data.data1;
			out[i+output_addr+1] = output.data.data2;
			out[i+output_addr+2] = output.data.data3;
			out[i+output_addr+3] = output.data.data4;
		}
		output_addr+=output_length;
		for(int i=0;i<weights_length;i+=4){
			output =tb_output_dma_output.read();
			weights[i+weights_addr] = output.data.data1;
			weights[i+weights_addr+1] = output.data.data2;
			weights[i+weights_addr+2] = output.data.data3;
			weights[i+weights_addr+3] = output.data.data4;
		}
		weights_addr+=weights_length;
	}

//	output_addr=0;
//	output_length=bs*1176;//M*min(custom_Tr,R-row*custom_Tr)*C=6*14*14;
//	for(int i=0;i<output_length;i++){
//		output =tb_output_dma_output.read();
//		output2[i+output_addr] = output.data;
//	}
//	weights_addr=0;
//	weights_length=bs*196;//r*c*[m/11]=14*14*1
//	for(int i=0;i<weights_length;i++){
//		input_weights =tb_input_dma_weights.read();
//		pool2index[i+weights_addr] = input_weights.data;
//	}

	cout<<"print out"<<outnum<<endl;
	cout<<"after loopM:"<<endl;
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
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<""<<endl;
	}

}

void convfp_64_128_112( int outnum,
	FPGA_DATA ifm[bs*64*112*112],
	FPGA_DATA weights[64*128*3*3],
	FPGA_DATA outhat[bs*128*112*112],
	FPGA_DATA gama[128],
	FPGA_DATA beta[128],
	FPGA_DATA namuda[128],
	FPGA_DATA out[bs*128*112*112]

){
	//forward conv3 N,M,R,C=64,128,112,112, stride=1,padding=1,kernel=3
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

	//FPGA_DATA output2[bs*24*24*20]; //[r][c][m]
	ifm_addr=0;
	weights_addr=0;

	weights_length=73728;//M*N*tb_custom_k*tb_custom_k=64*128*3*3
	//for (int toM=0;toM<2;toM++){
		for(int b=0;b<bs;b++){
			for (int to=0;to<tb_M;to+=Tm){//repeate [M/Tm]
				ifm_length=30464;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
										//*((C-1)*tb_custom_stride+tb_custom_k)=16*17*112;
				ifm_addr=802816*b;//+3584*row;//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*64*112*112+row*16*224*1
				for(int ti=0;ti<tb_N;ti+=Tn){
					for(int i=0;i<ifm_length;i+=4){
						input_ifm.data.data1 = ifm[i+ifm_addr];
						input_ifm.data.data2 = ifm[i+ifm_addr+1];
						input_ifm.data.data3 = ifm[i+ifm_addr+2];
						input_ifm.data.data4 = ifm[i+ifm_addr+3];
						tb_input_dma_ifm.write(input_ifm);
					}
					ifm_addr+=200704;//Tn*R_in*C_in=16*112*112
				}
				for(int row=tb_custom_Tr;row<tb_R-tb_custom_Tr;row+=tb_custom_Tr){
					ifm_length=32256;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
															//*((C-1)*tb_custom_stride+tb_custom_k)=16*18*112;
					ifm_addr=802816*b+(row-1)*1792;//+1792*row;//b*N*C_in*R_in+C_in*(row-padding)*stride*min(Tn,N)=b*64*112*112+(row-1)*1*112*16
					for(int ti=0;ti<tb_N;ti+=Tn){
						for(int i=0;i<ifm_length;i+=4){
							input_ifm.data.data1 = ifm[i+ifm_addr];
							input_ifm.data.data2 = ifm[i+ifm_addr+1];
							input_ifm.data.data3 = ifm[i+ifm_addr+2];
							input_ifm.data.data4 = ifm[i+ifm_addr+3];
							tb_input_dma_ifm.write(input_ifm);
						}
						ifm_addr+=200704;//Tn*R_in*C_in=16*112*112
					}
				}
				ifm_length=30464;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
										//*((C-1)*tb_custom_stride+tb_custom_k)=16*17*112;
				ifm_addr=802816*b+170240;//+1792*row;//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*64*112*112+row*16*224*1
				for(int ti=0;ti<tb_N;ti+=Tn){
					for(int i=0;i<ifm_length;i+=4){
						input_ifm.data.data1 = ifm[i+ifm_addr];
						input_ifm.data.data2 = ifm[i+ifm_addr+1];
						input_ifm.data.data3 = ifm[i+ifm_addr+2];
						input_ifm.data.data4 = ifm[i+ifm_addr+3];
						tb_input_dma_ifm.write(input_ifm);
					}
					ifm_addr+=200704;//Tn*R_in*C_in=16*112*112
				}
			}
		}
	//}



	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = weights[i+weights_addr];
		input_weights.data.data2 = weights[i+weights_addr+1];
		input_weights.data.data3 = weights[i+weights_addr+2];
		input_weights.data.data4 = weights[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}								//=6*3*5*5

	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	//cout<<tb_flag<<endl;

	//print output
	output_addr=0;
	//for (int toM=0;toM<2;toM++){
		output_length=bs*1605632;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=112*112*128;
		//output_addr=toM*81648;
		//for(int b=0;b<bs;b++){
			for(int i=0;i<output_length;i+=4){
				output =tb_output_dma_output.read();
				out[i+output_addr] = output.data.data1;
				out[i+output_addr+1] = output.data.data2;
				out[i+output_addr+2] = output.data.data3;
				out[i+output_addr+3] = output.data.data4;
			}
			//output_addr+=186624;
		//}
	//}
	cout<<"out after conv"<<endl;
	cout<<out[0]<<endl;

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


	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ofm_addr=0;
	weights_addr=0;
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = gama[i+weights_addr];
		input_weights.data.data2 = gama[i+weights_addr+1];
		input_weights.data.data3 = gama[i+weights_addr+2];
		input_weights.data.data4 = gama[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}
	for(int i=0;i<ofm_length;i+=4){
		input_ofm.data.data1 = beta[i+ofm_addr];
		input_ofm.data.data2 = beta[i+ofm_addr+1];
		input_ofm.data.data3 = beta[i+ofm_addr+2];
		input_ofm.data.data4 = beta[i+ofm_addr+3];
		tb_input_dma_ofm.write(input_ofm);
	}
	//ifm_addr=0;
	//ofm_addr=0;
	ifm_length=200704;//Tm*R*C
	for (int to=0;to<8;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=1605632*b+200704*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			//ofm_addr=3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			for(int i=0;i<ifm_length;i+=4){
				input_ifm.data.data1 = out[i+ifm_addr];
				input_ifm.data.data2 = out[i+ifm_addr+1];
				input_ifm.data.data3 = out[i+ifm_addr+2];
				input_ifm.data.data4 = out[i+ifm_addr+3];
				tb_input_dma_ifm.write(input_ifm);
			}
		}
	}
	ofm_addr=0;
	ofm_length=1605632;//M*R*C
	for(int b=0;b<bs;b++){
		//ifm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		//ofm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		for(int i=0;i<ofm_length;i+=4){
			input_ofm.data.data1 = out[i+ofm_addr];
			input_ofm.data.data2 = out[i+ofm_addr+1];
			input_ofm.data.data3 = out[i+ofm_addr+2];
			input_ofm.data.data4 = out[i+ofm_addr+3];
			tb_input_dma_ofm.write(input_ofm);
		}
		//ifm_addr+=ifm_length;
		ofm_addr+=ofm_length;
	}
							//=6*3*5*5

	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	//cout<<tb_flag<<endl;
	output_addr=0;
	output_length=1605632;//M*R*C
	for(int b=0;b<bs;b++){
		for(int i=0;i<output_length;i+=4){
			out_weights =tb_output_dma_Weightsout.read();
			outhat[i+output_addr] = out_weights.data.data1;
			outhat[i+output_addr+1] = out_weights.data.data2;
			outhat[i+output_addr+2] = out_weights.data.data3;
			outhat[i+output_addr+3] = out_weights.data.data4;
		}
		for(int i=0;i<output_length;i+=4){
			output =tb_output_dma_output.read();
			out[i+output_addr] = output.data.data1;
			out[i+output_addr+1] = output.data.data2;
			out[i+output_addr+2] = output.data.data3;
			out[i+output_addr+3] = output.data.data4;
		}
		output_addr+=output_length;
	}
	//print output

	output_addr=0;
	output_length=tb_M;//M
	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		namuda[i+output_addr] = output.data.data1;
		namuda[i+output_addr+1] = output.data.data2;
		namuda[i+output_addr+2] = output.data.data3;
		namuda[i+output_addr+3] = output.data.data4;
	}

	cout<<"namuda"<<endl;
	cout<<namuda[0]<<endl;
	cout<<"outhat"<<endl;
	cout<<outhat[0]<<endl;


	cout<<"print out"<<outnum<<endl;


	cout<<"after loopM:"<<endl;


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
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
}

void convfp_128_128_112( int outnum,
	FPGA_DATA ifm[bs*128*112*112],
	FPGA_DATA weights[128*128*3*3],
	FPGA_DATA outhat[bs*128*112*112],
	FPGA_DATA gama[128],
	FPGA_DATA beta[128],
	FPGA_DATA namuda[128],
	FPGA_DATA out[bs*128*112*112]

){
	//forward conv4 N,M,R,C=128,128,112,112, stride=1,padding=1,kernel=3
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

	//FPGA_DATA output2[bs*24*24*20]; //[r][c][m]
	ifm_addr=0;
	weights_addr=0;

	weights_length=147456;//M*N*tb_custom_k*tb_custom_k=128*128*3*3
	//for (int toM=0;toM<2;toM++){
		for(int b=0;b<bs;b++){
			for (int to=0;to<tb_M;to+=Tm){//repeate [M/Tm]
				ifm_length=30464;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
										//*((C-1)*tb_custom_stride+tb_custom_k)=16*17*112;
				ifm_addr=1605632*b;//+3584*row;//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*128*112*112+row*16*224*1
				for(int ti=0;ti<tb_N;ti+=Tn){
					for(int i=0;i<ifm_length;i+=4){
						input_ifm.data.data1 = ifm[i+ifm_addr];
						input_ifm.data.data2 = ifm[i+ifm_addr+1];
						input_ifm.data.data3 = ifm[i+ifm_addr+2];
						input_ifm.data.data4 = ifm[i+ifm_addr+3];
						tb_input_dma_ifm.write(input_ifm);
					}
					ifm_addr+=200704;//Tn*R_in*C_in=16*112*112
				}
				for(int row=tb_custom_Tr;row<tb_R-tb_custom_Tr;row+=tb_custom_Tr){
					ifm_length=32256;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
															//*((C-1)*tb_custom_stride+tb_custom_k)=16*18*112;
					ifm_addr=1605632*b+(row-1)*1792;//+1792*row;//b*N*C_in*R_in+C_in*(row-padding)*stride*min(Tn,N)=b*128*112*112+(row-1)*1*112*16
					for(int ti=0;ti<tb_N;ti+=Tn){
						for(int i=0;i<ifm_length;i+=4){
							input_ifm.data.data1 = ifm[i+ifm_addr];
							input_ifm.data.data2 = ifm[i+ifm_addr+1];
							input_ifm.data.data3 = ifm[i+ifm_addr+2];
							input_ifm.data.data4 = ifm[i+ifm_addr+3];
							tb_input_dma_ifm.write(input_ifm);
						}
						ifm_addr+=200704;//Tn*R_in*C_in=16*112*112
					}
				}
				ifm_length=30464;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
										//*((C-1)*tb_custom_stride+tb_custom_k)=16*17*112;
				ifm_addr=1605632*b+170240;//+1792*row;//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*128*112*112+row*16*224*1
				for(int ti=0;ti<tb_N;ti+=Tn){
					for(int i=0;i<ifm_length;i+=4){
						input_ifm.data.data1 = ifm[i+ifm_addr];
						input_ifm.data.data2 = ifm[i+ifm_addr+1];
						input_ifm.data.data3 = ifm[i+ifm_addr+2];
						input_ifm.data.data4 = ifm[i+ifm_addr+3];
						tb_input_dma_ifm.write(input_ifm);
					}
					ifm_addr+=200704;//Tn*R_in*C_in=16*112*112
				}
			}
		}
	//}



	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = weights[i+weights_addr];
		input_weights.data.data2 = weights[i+weights_addr+1];
		input_weights.data.data3 = weights[i+weights_addr+2];
		input_weights.data.data4 = weights[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}								//=6*3*5*5

	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	//cout<<tb_flag<<endl;

	//print output
	output_addr=0;
	//for (int toM=0;toM<2;toM++){
		output_length=bs*1605632;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=112*112*128;
		//output_addr=toM*81648;
		//for(int b=0;b<bs;b++){
			for(int i=0;i<output_length;i+=4){
				output =tb_output_dma_output.read();
				out[i+output_addr] = output.data.data1;
				out[i+output_addr+1] = output.data.data2;
				out[i+output_addr+2] = output.data.data3;
				out[i+output_addr+3] = output.data.data4;
			}
			//output_addr+=186624;
		//}
	//}
	cout<<"out after conv"<<endl;
	cout<<out[0]<<endl;

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


	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ofm_addr=0;
	weights_addr=0;
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = gama[i+weights_addr];
		input_weights.data.data2 = gama[i+weights_addr+1];
		input_weights.data.data3 = gama[i+weights_addr+2];
		input_weights.data.data4 = gama[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}
	for(int i=0;i<ofm_length;i+=4){
		input_ofm.data.data1 = beta[i+ofm_addr];
		input_ofm.data.data2 = beta[i+ofm_addr+1];
		input_ofm.data.data3 = beta[i+ofm_addr+2];
		input_ofm.data.data4 = beta[i+ofm_addr+3];
		tb_input_dma_ofm.write(input_ofm);
	}
	ifm_length=200704;//Tm*R*C
	for (int to=0;to<8;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=1605632*b+200704*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			//ofm_addr=3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			for(int i=0;i<ifm_length;i+=4){
				input_ifm.data.data1 = out[i+ifm_addr];
				input_ifm.data.data2 = out[i+ifm_addr+1];
				input_ifm.data.data3 = out[i+ifm_addr+2];
				input_ifm.data.data4 = out[i+ifm_addr+3];
				tb_input_dma_ifm.write(input_ifm);
			}
		}
	}
	ofm_addr=0;
	ofm_length=1605632;//M*R*C
	for(int b=0;b<bs;b++){
		//ifm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		//ofm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		for(int i=0;i<ofm_length;i+=4){
			input_ofm.data.data1 = out[i+ofm_addr];
			input_ofm.data.data2 = out[i+ofm_addr+1];
			input_ofm.data.data3 = out[i+ofm_addr+2];
			input_ofm.data.data4 = out[i+ofm_addr+3];
			tb_input_dma_ofm.write(input_ofm);
		}
		//ifm_addr+=ifm_length;
		ofm_addr+=ofm_length;
	}
							//=6*3*5*5

	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	//cout<<tb_flag<<endl;
	output_addr=0;
	output_length=1605632;//M*R*C
	for(int b=0;b<bs;b++){
		for(int i=0;i<output_length;i+=4){
			out_weights =tb_output_dma_Weightsout.read();
			outhat[i+output_addr] = out_weights.data.data1;
			outhat[i+output_addr+1] = out_weights.data.data2;
			outhat[i+output_addr+2] = out_weights.data.data3;
			outhat[i+output_addr+3] = out_weights.data.data4;
		}
		for(int i=0;i<output_length;i+=4){
			output =tb_output_dma_output.read();
			out[i+output_addr] = output.data.data1;
			out[i+output_addr+1] = output.data.data2;
			out[i+output_addr+2] = output.data.data3;
			out[i+output_addr+3] = output.data.data4;
		}
		output_addr+=output_length;
	}
		//print output
	output_addr=0;
	output_length=tb_M;//M
	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		namuda[i+output_addr] = output.data.data1;
		namuda[i+output_addr+1] = output.data.data2;
		namuda[i+output_addr+2] = output.data.data3;
		namuda[i+output_addr+3] = output.data.data4;
	}

	cout<<"namuda"<<endl;
	cout<<namuda[0]<<endl;
	cout<<"outhat"<<endl;
	cout<<outhat[0]<<endl;



	cout<<"print out"<<outnum<<endl;


	cout<<"after loopM:"<<endl;


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
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
}


void maxpoolfp_128_56( int outnum,
	FPGA_DATA ifm[bs*128*112*112],
	FPGA_DATA weights[bs*56*56*16],
	FPGA_DATA out[bs*128*56*56]

){
	//forward pool N,M,R,C=128,128,56,56, stride=2,padding=0,kernel=2
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



	ifm_addr=0;
	ifm_length=bs*1605632;//M*R_in*C_in=112*112*128;
	//for(int b=0;b<bs;b++){
		for(int i=0;i<ifm_length;i+=4){
			input_ifm.data.data1 = ifm[i+ifm_addr];
			input_ifm.data.data2 = ifm[i+ifm_addr+1];
			input_ifm.data.data3 = ifm[i+ifm_addr+2];
			input_ifm.data.data4 = ifm[i+ifm_addr+3];
			tb_input_dma_ifm.write(input_ifm);
		}
		//ifm_addr+=ifm_length;
	//}




	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	cout<<tb_flag<<endl;

//print output
	//FPGA_DATA output2[bs*12*12*20];//[r][c][m]
   //FPGA_DATA pool2index[bs*12*12*2];//r*c*[m/11]

	output_addr=0;
	weights_addr=0;
	output_length=401408;//M*min(custom_Tr,R-row*custom_Tr)*C=128*56*56;
	weights_length=50176;//r*c*[m/8]=56*56*16
	for(int b=0;b<bs;b++){
		for(int i=0;i<output_length;i+=4){
			output =tb_output_dma_output.read();
			out[i+output_addr] = output.data.data1;
			out[i+output_addr+1] = output.data.data2;
			out[i+output_addr+2] = output.data.data3;
			out[i+output_addr+3] = output.data.data4;
		}
		output_addr+=output_length;
		for(int i=0;i<weights_length;i+=4){
			output =tb_output_dma_output.read();
			weights[i+weights_addr] = output.data.data1;
			weights[i+weights_addr+1] = output.data.data2;
			weights[i+weights_addr+2] = output.data.data3;
			weights[i+weights_addr+3] = output.data.data4;
		}
		weights_addr+=weights_length;
	}

//	output_addr=0;
//	output_length=bs*1176;//M*min(custom_Tr,R-row*custom_Tr)*C=6*14*14;
//	for(int i=0;i<output_length;i++){
//		output =tb_output_dma_output.read();
//		output2[i+output_addr] = output.data;
//	}
//	weights_addr=0;
//	weights_length=bs*196;//r*c*[m/11]=14*14*1
//	for(int i=0;i<weights_length;i++){
//		input_weights =tb_input_dma_weights.read();
//		pool2index[i+weights_addr] = input_weights.data;
//	}

	cout<<"print out"<<outnum<<endl;
	cout<<"after loopM:"<<endl;
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
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<""<<endl;
	}

}


void convfp_128_256_56( int outnum,
	FPGA_DATA ifm[bs*128*56*56],
	FPGA_DATA weights[128*256*3*3],
	FPGA_DATA outhat[bs*256*56*56],
	FPGA_DATA gama[256],
	FPGA_DATA beta[256],
	FPGA_DATA namuda[256],
	FPGA_DATA out[bs*256*56*56]

){
	//forward conv5 N,M,R,C=128,256,56,56, stride=1,padding=1,kernel=3
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

	//FPGA_DATA output2[bs*24*24*20]; //[r][c][m]
	ifm_addr=0;
	weights_addr=0;
	ifm_length=25984;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
											//*((C-1)*tb_custom_stride+tb_custom_k)=16*29*56;
	weights_length=294912;//M*N*tb_custom_k*tb_custom_k=128*256*3*3
	//for (int toM=0;toM<2;toM++){
		for(int b=0;b<bs;b++){
			for (int to=0;to<tb_M;to+=Tm){//repeate [M/Tm]
				//ifm_length=25984;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
										//*((C-1)*tb_custom_stride+tb_custom_k)=16*29*56;
				ifm_addr=401408*b;//+3584*row;//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*128*56*56+row*16*224*1
				for(int ti=0;ti<tb_N;ti+=Tn){
					for(int i=0;i<ifm_length;i+=4){
						input_ifm.data.data1 = ifm[i+ifm_addr];
						input_ifm.data.data2 = ifm[i+ifm_addr+1];
						input_ifm.data.data3 = ifm[i+ifm_addr+2];
						input_ifm.data.data4 = ifm[i+ifm_addr+3];
						tb_input_dma_ifm.write(input_ifm);
					}
					ifm_addr+=50176;//Tn*R_in*C_in=16*56*56
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
				ifm_addr=401408*b+24192;//+1792*row;//b*N*C_in*R_in+C_in*(row-padding)*stride*min(Tn,N)=b*128*56*56+(row-1)*1*56*16
				for(int ti=0;ti<tb_N;ti+=Tn){
					for(int i=0;i<ifm_length;i+=4){
						input_ifm.data.data1 = ifm[i+ifm_addr];
						input_ifm.data.data2 = ifm[i+ifm_addr+1];
						input_ifm.data.data3 = ifm[i+ifm_addr+2];
						input_ifm.data.data4 = ifm[i+ifm_addr+3];
						tb_input_dma_ifm.write(input_ifm);
					}
					ifm_addr+=50176;//Tn*R_in*C_in=16*56*56
				}
			}
		}
	//}



	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = weights[i+weights_addr];
		input_weights.data.data2 = weights[i+weights_addr+1];
		input_weights.data.data3 = weights[i+weights_addr+2];
		input_weights.data.data4 = weights[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}								//=6*3*5*5

	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	//cout<<tb_flag<<endl;

	//print output
	output_addr=0;
	//for (int toM=0;toM<2;toM++){
		output_length=bs*802816;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=56*56*256;
		//output_addr=toM*81648;
		//for(int b=0;b<bs;b++){
			for(int i=0;i<output_length;i+=4){
				output =tb_output_dma_output.read();
				out[i+output_addr] = output.data.data1;
				out[i+output_addr+1] = output.data.data2;
				out[i+output_addr+2] = output.data.data3;
				out[i+output_addr+3] = output.data.data4;
			}
			//output_addr+=186624;
		//}
	//}
	cout<<"out after conv"<<endl;
	cout<<out[0]<<endl;


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


	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ofm_addr=0;
	weights_addr=0;
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = gama[i+weights_addr];
		input_weights.data.data2 = gama[i+weights_addr+1];
		input_weights.data.data3 = gama[i+weights_addr+2];
		input_weights.data.data4 = gama[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}
	for(int i=0;i<ofm_length;i+=4){
		input_ofm.data.data1 = beta[i+ofm_addr];
		input_ofm.data.data2 = beta[i+ofm_addr+1];
		input_ofm.data.data3 = beta[i+ofm_addr+2];
		input_ofm.data.data4 = beta[i+ofm_addr+3];
		tb_input_dma_ofm.write(input_ofm);
	}
	ifm_length=50176;//Tm*R*C
	for (int to=0;to<16;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=802816*b+50176*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			//ofm_addr=3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			for(int i=0;i<ifm_length;i+=4){
				input_ifm.data.data1 = out[i+ifm_addr];
				input_ifm.data.data2 = out[i+ifm_addr+1];
				input_ifm.data.data3 = out[i+ifm_addr+2];
				input_ifm.data.data4 = out[i+ifm_addr+3];
				tb_input_dma_ifm.write(input_ifm);
			}
		}
	}
	ofm_addr=0;
	ofm_length=802816;//M*R*C
	for(int b=0;b<bs;b++){
		//ifm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		//ofm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		for(int i=0;i<ofm_length;i+=4){
			input_ofm.data.data1 = out[i+ofm_addr];
			input_ofm.data.data2 = out[i+ofm_addr+1];
			input_ofm.data.data3 = out[i+ofm_addr+2];
			input_ofm.data.data4 = out[i+ofm_addr+3];
			tb_input_dma_ofm.write(input_ofm);
		}
		//ifm_addr+=ifm_length;
		ofm_addr+=ofm_length;
	}
							//=6*3*5*5

	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	//cout<<tb_flag<<endl;
	output_addr=0;
	output_length=802816;//M*R*C
	for(int b=0;b<bs;b++){
		for(int i=0;i<output_length;i+=4){
			out_weights =tb_output_dma_Weightsout.read();
			outhat[i+output_addr] = out_weights.data.data1;
			outhat[i+output_addr+1] = out_weights.data.data2;
			outhat[i+output_addr+2] = out_weights.data.data3;
			outhat[i+output_addr+3] = out_weights.data.data4;
		}
		for(int i=0;i<output_length;i+=4){
			output =tb_output_dma_output.read();
			out[i+output_addr] = output.data.data1;
			out[i+output_addr+1] = output.data.data2;
			out[i+output_addr+2] = output.data.data3;
			out[i+output_addr+3] = output.data.data4;
		}
		output_addr+=output_length;
	}
		//print output

	output_addr=0;
	output_length=tb_M;//M
	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		namuda[i+output_addr] = output.data.data1;
		namuda[i+output_addr+1] = output.data.data2;
		namuda[i+output_addr+2] = output.data.data3;
		namuda[i+output_addr+3] = output.data.data4;
	}

	cout<<"namuda"<<endl;
	cout<<namuda[0]<<endl;
	cout<<"outhat"<<endl;
	cout<<outhat[0]<<endl;



	cout<<"print out"<<outnum<<endl;


	cout<<"after loopM:"<<endl;


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
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
}


void convfp_256_256_56( int outnum,
	FPGA_DATA ifm[bs*256*56*56],
	FPGA_DATA weights[256*256*3*3],
	FPGA_DATA outhat[bs*256*56*56],
	FPGA_DATA gama[256],
	FPGA_DATA beta[256],
	FPGA_DATA namuda[256],
	FPGA_DATA out[bs*256*56*56]

){
	//forward conv6,7 N,M,R,C=128,256,56,56, stride=1,padding=1,kernel=3
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

	//FPGA_DATA output2[bs*24*24*20]; //[r][c][m]
	ifm_addr=0;
	weights_addr=0;
	ifm_length=25984;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
											//*((C-1)*tb_custom_stride+tb_custom_k)=16*29*56;
	weights_length=589824;//M*N*tb_custom_k*tb_custom_k=256*256*3*3
	for (int toM=0;toM<2;toM++){
		for(int b=0;b<bs;b++){
			for (int to=0;to<tb_M;to+=Tm){//repeate [M/Tm]
				//ifm_length=25984;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
										//*((C-1)*tb_custom_stride+tb_custom_k)=16*29*56;
				ifm_addr=802816*b;//+3584*row;//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*256*56*56+row*16*224*1
				for(int ti=0;ti<tb_N;ti+=Tn){
					for(int i=0;i<ifm_length;i+=4){
						input_ifm.data.data1 = ifm[i+ifm_addr];
						input_ifm.data.data2 = ifm[i+ifm_addr+1];
						input_ifm.data.data3 = ifm[i+ifm_addr+2];
						input_ifm.data.data4 = ifm[i+ifm_addr+3];
						tb_input_dma_ifm.write(input_ifm);
					}
					ifm_addr+=50176;//Tn*R_in*C_in=16*56*56
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
				ifm_addr=802816*b+24192;//+896*row;//b*N*C_in*R_in+C_in*(row-padding)*stride*min(Tn,N)=b*256*56*56+(row-1)*1*56*16
				for(int ti=0;ti<tb_N;ti+=Tn){
					for(int i=0;i<ifm_length;i+=4){
						input_ifm.data.data1 = ifm[i+ifm_addr];
						input_ifm.data.data2 = ifm[i+ifm_addr+1];
						input_ifm.data.data3 = ifm[i+ifm_addr+2];
						input_ifm.data.data4 = ifm[i+ifm_addr+3];
						tb_input_dma_ifm.write(input_ifm);
					}
					ifm_addr+=50176;//Tn*R_in*C_in=16*56*56
				}
			}
		}
	}



	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = weights[i+weights_addr];
		input_weights.data.data2 = weights[i+weights_addr+1];
		input_weights.data.data3 = weights[i+weights_addr+2];
		input_weights.data.data4 = weights[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}								//=6*3*5*5

	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	//cout<<tb_flag<<endl;

	//print output
	output_addr=0;
	for (int toM=0;toM<2;toM++){
		output_length=401408;//tb_M*R*C=56*56*128;
		output_addr=toM*401408;
		for(int b=0;b<bs;b++){
			for(int i=0;i<output_length;i+=4){
				output =tb_output_dma_output.read();
				out[i+output_addr] = output.data.data1;
				out[i+output_addr+1] = output.data.data2;
				out[i+output_addr+2] = output.data.data3;
				out[i+output_addr+3] = output.data.data4;
			}
			output_addr+=802816;
		}
	}

	cout<<"out after conv"<<endl;
	cout<<out[0]<<endl;


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


	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ofm_addr=0;
	weights_addr=0;
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = gama[i+weights_addr];
		input_weights.data.data2 = gama[i+weights_addr+1];
		input_weights.data.data3 = gama[i+weights_addr+2];
		input_weights.data.data4 = gama[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}
	for(int i=0;i<ofm_length;i+=4){
		input_ofm.data.data1 = beta[i+ofm_addr];
		input_ofm.data.data2 = beta[i+ofm_addr+1];
		input_ofm.data.data3 = beta[i+ofm_addr+2];
		input_ofm.data.data4 = beta[i+ofm_addr+3];
		tb_input_dma_ofm.write(input_ofm);
	}
	ifm_length=50176;//Tm*R*C
	for (int to=0;to<16;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=802816*b+50176*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			//ofm_addr=3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			for(int i=0;i<ifm_length;i+=4){
				input_ifm.data.data1 = out[i+ifm_addr];
				input_ifm.data.data2 = out[i+ifm_addr+1];
				input_ifm.data.data3 = out[i+ifm_addr+2];
				input_ifm.data.data4 = out[i+ifm_addr+3];
				tb_input_dma_ifm.write(input_ifm);
			}
		}
	}
	ofm_addr=0;
	ofm_length=802816;//M*R*C
	for(int b=0;b<bs;b++){
		//ifm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		//ofm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		for(int i=0;i<ofm_length;i+=4){
			input_ofm.data.data1 = out[i+ofm_addr];
			input_ofm.data.data2 = out[i+ofm_addr+1];
			input_ofm.data.data3 = out[i+ofm_addr+2];
			input_ofm.data.data4 = out[i+ofm_addr+3];
			tb_input_dma_ofm.write(input_ofm);
		}
		//ifm_addr+=ifm_length;
		ofm_addr+=ofm_length;
	}
							//=6*3*5*5

	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	//cout<<tb_flag<<endl;
	output_addr=0;
	output_length=802816;//M*R*C
	for(int b=0;b<bs;b++){
		for(int i=0;i<output_length;i+=4){
			out_weights =tb_output_dma_Weightsout.read();
			outhat[i+output_addr] = out_weights.data.data1;
			outhat[i+output_addr+1] = out_weights.data.data2;
			outhat[i+output_addr+2] = out_weights.data.data3;
			outhat[i+output_addr+3] = out_weights.data.data4;
		}
		for(int i=0;i<output_length;i+=4){
			output =tb_output_dma_output.read();
			out[i+output_addr] = output.data.data1;
			out[i+output_addr+1] = output.data.data2;
			out[i+output_addr+2] = output.data.data3;
			out[i+output_addr+3] = output.data.data4;
		}
		output_addr+=output_length;
	}
	output_addr=0;
	output_length=tb_M;//M
	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		namuda[i+output_addr] = output.data.data1;
		namuda[i+output_addr+1] = output.data.data2;
		namuda[i+output_addr+2] = output.data.data3;
		namuda[i+output_addr+3] = output.data.data4;
	}

	cout<<"namuda"<<endl;
	cout<<namuda[0]<<endl;
	cout<<"outhat"<<endl;
	cout<<outhat[0]<<endl;




	cout<<"print out"<<outnum<<endl;


	cout<<"after loopM:"<<endl;


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
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
}


void maxpoolfp_256_28( int outnum,
	FPGA_DATA ifm[bs*256*56*56],
	FPGA_DATA weights[bs*28*28*32],
	FPGA_DATA out[bs*256*28*28]

){
	//forward pool N,M,R,C=256,256,28,28, stride=2,padding=0,kernel=2
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



	ifm_addr=0;
	ifm_length=bs*802816;//M*R_in*C_in=56*56*256;
	//for(int b=0;b<bs;b++){
		for(int i=0;i<ifm_length;i+=4){
			input_ifm.data.data1 = ifm[i+ifm_addr];
			input_ifm.data.data2 = ifm[i+ifm_addr+1];
			input_ifm.data.data3 = ifm[i+ifm_addr+2];
			input_ifm.data.data4 = ifm[i+ifm_addr+3];
			tb_input_dma_ifm.write(input_ifm);
		}
		//ifm_addr+=ifm_length;
	//}




	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	cout<<tb_flag<<endl;

//print output
	//FPGA_DATA output2[bs*12*12*20];//[r][c][m]
   //FPGA_DATA pool2index[bs*12*12*2];//r*c*[m/11]

	output_addr=0;
	weights_addr=0;
	output_length=200704;//M*min(custom_Tr,R-row*custom_Tr)*C=256*28*28;
	weights_length=25088;//r*c*[m/8]=28*28*32
	for(int b=0;b<bs;b++){
		for(int i=0;i<output_length;i+=4){
			output =tb_output_dma_output.read();
			out[i+output_addr] = output.data.data1;
			out[i+output_addr+1] = output.data.data2;
			out[i+output_addr+2] = output.data.data3;
			out[i+output_addr+3] = output.data.data4;
		}
		output_addr+=output_length;
		for(int i=0;i<weights_length;i+=4){
			output =tb_output_dma_output.read();
			weights[i+weights_addr] = output.data.data1;
			weights[i+weights_addr+1] = output.data.data2;
			weights[i+weights_addr+2] = output.data.data3;
			weights[i+weights_addr+3] = output.data.data4;
		}
		weights_addr+=weights_length;
	}

//	output_addr=0;
//	output_length=bs*1176;//M*min(custom_Tr,R-row*custom_Tr)*C=6*14*14;
//	for(int i=0;i<output_length;i++){
//		output =tb_output_dma_output.read();
//		output2[i+output_addr] = output.data;
//	}
//	weights_addr=0;
//	weights_length=bs*196;//r*c*[m/11]=14*14*1
//	for(int i=0;i<weights_length;i++){
//		input_weights =tb_input_dma_weights.read();
//		pool2index[i+weights_addr] = input_weights.data;
//	}

	cout<<"print out"<<outnum<<endl;
	cout<<"after loopM:"<<endl;
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
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<""<<endl;
	}

}


void convfp_256_512_28( int outnum,
	FPGA_DATA ifm[bs*256*28*28],
	FPGA_DATA weights[256*512*3*3],
	FPGA_DATA outhat[bs*512*28*28],
	FPGA_DATA gama[512],
	FPGA_DATA beta[512],
	FPGA_DATA namuda[512],
	FPGA_DATA out[bs*512*28*28]

){
	//forward conv8 N,M,R,C=256,512,28,28, stride=1,padding=1,kernel=3
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

	//FPGA_DATA output2[bs*24*24*20]; //[r][c][m]

	weights_addr=0;
	ifm_length=200704;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
											//*((C-1)*tb_custom_stride+tb_custom_k)=256*28*28;
	weights_length=1179648;//M*N*tb_custom_k*tb_custom_k=512*256*3*3
	for (int toM=0;toM<4;toM++){
		ifm_addr=0;
		for(int b=0;b<bs;b++){
			for (int to=0;to<tb_M;to+=Tm){//repeate [M/Tm]
				//ifm_length=200704;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
										//*((C-1)*tb_custom_stride+tb_custom_k)=256*28*28;
				//ifm_addr=200704*b;//+3584*row;//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*256*28*28+row*16*28*1
				//for(int ti=0;ti<tb_N;ti+=Tn){
					for(int i=0;i<ifm_length;i+=4){
						input_ifm.data.data1 = ifm[i+ifm_addr];
						input_ifm.data.data2 = ifm[i+ifm_addr+1];
						input_ifm.data.data3 = ifm[i+ifm_addr+2];
						input_ifm.data.data4 = ifm[i+ifm_addr+3];
						tb_input_dma_ifm.write(input_ifm);
					}
					//ifm_addr+=12544;//Tn*R_in*C_in=16*28*28
				//}
//				ifm_length=3584;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
//										//*((C-1)*tb_custom_stride+tb_custom_k)=16*8*28;
//				ifm_addr=200704*b;//+3584*row;//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*256*28*28+row*16*28*1
//				for(int ti=0;ti<tb_N;ti+=Tn){
//					for(int i=0;i<ifm_length;i+=4){
//						input_ifm.data.data1 = ifm[i+ifm_addr];
//						input_ifm.data.data2 = ifm[i+ifm_addr+1];
//						input_ifm.data.data3 = ifm[i+ifm_addr+2];
//						input_ifm.data.data4 = ifm[i+ifm_addr+3];
//						tb_input_dma_ifm.write(input_ifm);
//					}
//					ifm_addr+=12544;//Tn*R_in*C_in=16*28*28
//				}
//				for(int row=tb_custom_Tr;row<tb_R-tb_custom_Tr;row+=tb_custom_Tr){
//					ifm_length=4032;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
//															//*((C-1)*tb_custom_stride+tb_custom_k)=16*9*28;
//					ifm_addr=200704*b+(row-1)*448;//+1792*row;//b*N*C_in*R_in+C_in*(row-padding)*stride*min(Tn,N)=b*256*28*28+(row-1)*1*28*16
//					for(int ti=0;ti<tb_N;ti+=Tn){
//						for(int i=0;i<ifm_length;i+=4){
//							input_ifm.data.data1 = ifm[i+ifm_addr];
//							input_ifm.data.data2 = ifm[i+ifm_addr+1];
//							input_ifm.data.data3 = ifm[i+ifm_addr+2];
//							input_ifm.data.data4 = ifm[i+ifm_addr+3];
//							tb_input_dma_ifm.write(input_ifm);
//						}
//						ifm_addr+=12544;//Tn*R_in*C_in=16*28*28
//					}
//				}
//				ifm_length=3584;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
//								//*((C-1)*tb_custom_stride+tb_custom_k)=16*8*28;
//				ifm_addr=200704*b+8960;//+3584*row;//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*256*28*28+row*16*28*1
//				for(int ti=0;ti<tb_N;ti+=Tn){
//					for(int i=0;i<ifm_length;i+=4){
//						input_ifm.data.data1 = ifm[i+ifm_addr];
//						input_ifm.data.data2 = ifm[i+ifm_addr+1];
//						input_ifm.data.data3 = ifm[i+ifm_addr+2];
//						input_ifm.data.data4 = ifm[i+ifm_addr+3];
//						tb_input_dma_ifm.write(input_ifm);
//					}
//					ifm_addr+=12544;//Tn*R_in*C_in=16*28*28
//				}
			}
			ifm_addr+=200704;//N*R_in*C_in=256*28*28
		}
	}



	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = weights[i+weights_addr];
		input_weights.data.data2 = weights[i+weights_addr+1];
		input_weights.data.data3 = weights[i+weights_addr+2];
		input_weights.data.data4 = weights[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}								//=6*3*5*5

	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	//cout<<tb_flag<<endl;

	//print output
	output_addr=0;
	for (int toM=0;toM<4;toM++){
		output_length=100352;//tb_M*R*C=28*28*128;
		output_addr=toM*100352;
		for(int b=0;b<bs;b++){
			for(int i=0;i<output_length;i+=4){
				output =tb_output_dma_output.read();
				out[i+output_addr] = output.data.data1;
				out[i+output_addr+1] = output.data.data2;
				out[i+output_addr+2] = output.data.data3;
				out[i+output_addr+3] = output.data.data4;
			}
			output_addr+=401408;
		}
	}

	cout<<"out after conv"<<endl;
	cout<<out[0]<<endl;

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


	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ofm_addr=0;
	weights_addr=0;
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = gama[i+weights_addr];
		input_weights.data.data2 = gama[i+weights_addr+1];
		input_weights.data.data3 = gama[i+weights_addr+2];
		input_weights.data.data4 = gama[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}
	for(int i=0;i<ofm_length;i+=4){
		input_ofm.data.data1 = beta[i+ofm_addr];
		input_ofm.data.data2 = beta[i+ofm_addr+1];
		input_ofm.data.data3 = beta[i+ofm_addr+2];
		input_ofm.data.data4 = beta[i+ofm_addr+3];
		tb_input_dma_ofm.write(input_ofm);
	}

	ifm_length=12544;//Tm*R*C
	for (int to=0;to<32;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=401408*b+12544*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			//ofm_addr=3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			for(int i=0;i<ifm_length;i+=4){
				input_ifm.data.data1 = out[i+ifm_addr];
				input_ifm.data.data2 = out[i+ifm_addr+1];
				input_ifm.data.data3 = out[i+ifm_addr+2];
				input_ifm.data.data4 = out[i+ifm_addr+3];
				tb_input_dma_ifm.write(input_ifm);
			}
		}
	}
	ofm_addr=0;
	ofm_length=401408;//M*R*C
	for(int b=0;b<bs;b++){
		//ifm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		//ofm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		for(int i=0;i<ofm_length;i+=4){
			input_ofm.data.data1 = out[i+ofm_addr];
			input_ofm.data.data2 = out[i+ofm_addr+1];
			input_ofm.data.data3 = out[i+ofm_addr+2];
			input_ofm.data.data4 = out[i+ofm_addr+3];
			tb_input_dma_ofm.write(input_ofm);
		}
		//ifm_addr+=ifm_length;
		ofm_addr+=ofm_length;
	}
							//=6*3*5*5

	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	//cout<<tb_flag<<endl;
	output_addr=0;
	output_length=401408;//M*R*C
	for(int b=0;b<bs;b++){
		for(int i=0;i<output_length;i+=4){
			out_weights =tb_output_dma_Weightsout.read();
			outhat[i+output_addr] = out_weights.data.data1;
			outhat[i+output_addr+1] = out_weights.data.data2;
			outhat[i+output_addr+2] = out_weights.data.data3;
			outhat[i+output_addr+3] = out_weights.data.data4;
		}
		for(int i=0;i<output_length;i+=4){
			output =tb_output_dma_output.read();
			out[i+output_addr] = output.data.data1;
			out[i+output_addr+1] = output.data.data2;
			out[i+output_addr+2] = output.data.data3;
			out[i+output_addr+3] = output.data.data4;
		}
		output_addr+=output_length;
	}

	output_addr=0;
	output_length=tb_M;//M
	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		namuda[i+output_addr] = output.data.data1;
		namuda[i+output_addr+1] = output.data.data2;
		namuda[i+output_addr+2] = output.data.data3;
		namuda[i+output_addr+3] = output.data.data4;
	}

	cout<<"namuda"<<endl;
	cout<<namuda[0]<<endl;
	cout<<"outhat"<<endl;
	cout<<outhat[0]<<endl;


	cout<<"print out"<<outnum<<endl;


	cout<<"after loopM:"<<endl;


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
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
}


void convfp_512_512_28( int outnum,
	FPGA_DATA ifm[bs*512*28*28],
	FPGA_DATA weights[512*512*3*3],
	FPGA_DATA outhat[bs*512*28*28],
	FPGA_DATA gama[512],
	FPGA_DATA beta[512],
	FPGA_DATA namuda[512],
	FPGA_DATA out[bs*512*28*28]

){
	//forward conv9,10 N,M,R,C=512,512,28,28, stride=1,padding=1,kernel=3
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

	//FPGA_DATA output2[bs*24*24*20]; //[r][c][m]

	weights_addr=0;
	ifm_length=401408;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
											//*((C-1)*tb_custom_stride+tb_custom_k)=512*28*28;
	weights_length=2359296;//M*N*tb_custom_k*tb_custom_k=512*512*3*3
	for (int toM=0;toM<8;toM++){
		ifm_addr=0;
		for(int b=0;b<bs;b++){
			for (int to=0;to<tb_M;to+=Tm){//repeate [M/Tm]
				//ifm_length=401408;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
										//*((C-1)*tb_custom_stride+tb_custom_k)=512*28*28;
				//ifm_addr=401408*b;//+3584*row;//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*512*28*28+row*16*28*1
				//for(int ti=0;ti<tb_N;ti+=Tn){
					for(int i=0;i<ifm_length;i+=4){
						input_ifm.data.data1 = ifm[i+ifm_addr];
						input_ifm.data.data2 = ifm[i+ifm_addr+1];
						input_ifm.data.data3 = ifm[i+ifm_addr+2];
						input_ifm.data.data4 = ifm[i+ifm_addr+3];
						tb_input_dma_ifm.write(input_ifm);
					}
					//ifm_addr+=12544;//Tn*R_in*C_in=16*28*28
				//}
//				ifm_length=3584;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
//										//*((C-1)*tb_custom_stride+tb_custom_k)=16*8*28;
//				ifm_addr=401408*b;//+3584*row;//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*512*28*28+row*16*28*1
//				for(int ti=0;ti<tb_N;ti+=Tn){
//					for(int i=0;i<ifm_length;i+=4){
//						input_ifm.data.data1 = ifm[i+ifm_addr];
//						input_ifm.data.data2 = ifm[i+ifm_addr+1];
//						input_ifm.data.data3 = ifm[i+ifm_addr+2];
//						input_ifm.data.data4 = ifm[i+ifm_addr+3];
//						tb_input_dma_ifm.write(input_ifm);
//					}
//					ifm_addr+=12544;//Tn*R_in*C_in=16*28*28
//				}
//				for(int row=tb_custom_Tr;row<tb_R-tb_custom_Tr;row+=tb_custom_Tr){
//					ifm_length=4032;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
//															//*((C-1)*tb_custom_stride+tb_custom_k)=16*9*28;
//					ifm_addr=401408*b+(row-1)*448;//+1792*row;//b*N*C_in*R_in+C_in*(row-padding)*stride*min(Tn,N)=b*512*28*28+(row-1)*1*28*16
//					for(int ti=0;ti<tb_N;ti+=Tn){
//						for(int i=0;i<ifm_length;i+=4){
//							input_ifm.data.data1 = ifm[i+ifm_addr];
//							input_ifm.data.data2 = ifm[i+ifm_addr+1];
//							input_ifm.data.data3 = ifm[i+ifm_addr+2];
//							input_ifm.data.data4 = ifm[i+ifm_addr+3];
//							tb_input_dma_ifm.write(input_ifm);
//						}
//						ifm_addr+=12544;//Tn*R_in*C_in=16*28*28
//					}
//				}
//				ifm_length=3584;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
//								//*((C-1)*tb_custom_stride+tb_custom_k)=16*8*28;
//				ifm_addr=401408*b+8960;//+3584*row;//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*512*28*28+row*16*28*1
//				for(int ti=0;ti<tb_N;ti+=Tn){
//					for(int i=0;i<ifm_length;i+=4){
//						input_ifm.data.data1 = ifm[i+ifm_addr];
//						input_ifm.data.data2 = ifm[i+ifm_addr+1];
//						input_ifm.data.data3 = ifm[i+ifm_addr+2];
//						input_ifm.data.data4 = ifm[i+ifm_addr+3];
//						tb_input_dma_ifm.write(input_ifm);
//					}
//					ifm_addr+=12544;//Tn*R_in*C_in=16*28*28
//				}
			}
			ifm_addr+=401408;//N*R_in*C_in=512*28*28
		}
	}



	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = weights[i+weights_addr];
		input_weights.data.data2 = weights[i+weights_addr+1];
		input_weights.data.data3 = weights[i+weights_addr+2];
		input_weights.data.data4 = weights[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}								//=6*3*5*5

	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	//cout<<tb_flag<<endl;

	//print output
	output_addr=0;
	for (int toM=0;toM<8;toM++){
		output_length=50176;//tb_M*R*C=28*28*64;
		output_addr=toM*50176;
		for(int b=0;b<bs;b++){
			for(int i=0;i<output_length;i+=4){
				output =tb_output_dma_output.read();
				out[i+output_addr] = output.data.data1;
				out[i+output_addr+1] = output.data.data2;
				out[i+output_addr+2] = output.data.data3;
				out[i+output_addr+3] = output.data.data4;
			}
			output_addr+=401408;
		}
	}

	cout<<"out after conv"<<endl;
	cout<<out[0]<<endl;

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


	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ofm_addr=0;
	weights_addr=0;
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = gama[i+weights_addr];
		input_weights.data.data2 = gama[i+weights_addr+1];
		input_weights.data.data3 = gama[i+weights_addr+2];
		input_weights.data.data4 = gama[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}
	for(int i=0;i<ofm_length;i+=4){
		input_ofm.data.data1 = beta[i+ofm_addr];
		input_ofm.data.data2 = beta[i+ofm_addr+1];
		input_ofm.data.data3 = beta[i+ofm_addr+2];
		input_ofm.data.data4 = beta[i+ofm_addr+3];
		tb_input_dma_ofm.write(input_ofm);
	}
	//ifm_addr=0;
	//ofm_addr=0;
	ifm_length=12544;//Tm*R*C
	for (int to=0;to<32;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=401408*b+12544*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			//ofm_addr=3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			for(int i=0;i<ifm_length;i+=4){
				input_ifm.data.data1 = out[i+ifm_addr];
				input_ifm.data.data2 = out[i+ifm_addr+1];
				input_ifm.data.data3 = out[i+ifm_addr+2];
				input_ifm.data.data4 = out[i+ifm_addr+3];
				tb_input_dma_ifm.write(input_ifm);
			}
		}
	}
	ofm_addr=0;
	ofm_length=401408;//M*R*C
	for(int b=0;b<bs;b++){
		//ifm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		//ofm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		for(int i=0;i<ofm_length;i+=4){
			input_ofm.data.data1 = out[i+ofm_addr];
			input_ofm.data.data2 = out[i+ofm_addr+1];
			input_ofm.data.data3 = out[i+ofm_addr+2];
			input_ofm.data.data4 = out[i+ofm_addr+3];
			tb_input_dma_ofm.write(input_ofm);
		}
		//ifm_addr+=ifm_length;
		ofm_addr+=ofm_length;
	}
							//=6*3*5*5

	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	//cout<<tb_flag<<endl;
	output_addr=0;
	output_length=401408;//M*R*C
	for(int b=0;b<bs;b++){
		for(int i=0;i<output_length;i+=4){
			out_weights =tb_output_dma_Weightsout.read();
			outhat[i+output_addr] = out_weights.data.data1;
			outhat[i+output_addr+1] = out_weights.data.data2;
			outhat[i+output_addr+2] = out_weights.data.data3;
			outhat[i+output_addr+3] = out_weights.data.data4;
		}
		for(int i=0;i<output_length;i+=4){
			output =tb_output_dma_output.read();
			out[i+output_addr] = output.data.data1;
			out[i+output_addr+1] = output.data.data2;
			out[i+output_addr+2] = output.data.data3;
			out[i+output_addr+3] = output.data.data4;
		}
		output_addr+=output_length;
	}
	output_addr=0;
	output_length=tb_M;//M
	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		namuda[i+output_addr] = output.data.data1;
		namuda[i+output_addr+1] = output.data.data2;
		namuda[i+output_addr+2] = output.data.data3;
		namuda[i+output_addr+3] = output.data.data4;
	}

	cout<<"namuda"<<endl;
	cout<<namuda[0]<<endl;
	cout<<"outhat"<<endl;
	cout<<outhat[0]<<endl;

	cout<<"print out"<<outnum<<endl;


	cout<<"after loopM:"<<endl;


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
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
}


void maxpoolfp_512_14( int outnum,
	FPGA_DATA ifm[bs*512*28*28],
	FPGA_DATA weights[bs*14*14*64],
	FPGA_DATA out[bs*512*14*14]

){
	//forward pool N,M,R,C=512,512,14,14, stride=2,padding=0,kernel=2
	tb_statec=0;
	tb_statep=6;
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



	ifm_addr=0;
	ifm_length=bs*401408;//M*R_in*C_in=28*28*512;
	//for(int b=0;b<bs;b++){
		for(int i=0;i<ifm_length;i+=4){
			input_ifm.data.data1 = ifm[i+ifm_addr];
			input_ifm.data.data2 = ifm[i+ifm_addr+1];
			input_ifm.data.data3 = ifm[i+ifm_addr+2];
			input_ifm.data.data4 = ifm[i+ifm_addr+3];
			tb_input_dma_ifm.write(input_ifm);
		}
		//ifm_addr+=ifm_length;
	//}




	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	cout<<tb_flag<<endl;

//print output
	//FPGA_DATA output2[bs*12*12*20];//[r][c][m]
   //FPGA_DATA pool2index[bs*12*12*2];//r*c*[m/11]

	output_addr=0;
	weights_addr=0;
	output_length=100352;//M*min(custom_Tr,R-row*custom_Tr)*C=512*14*14;
	weights_length=12544;//r*c*[m/8]=14*14*64
	for(int b=0;b<bs;b++){
		for(int i=0;i<output_length;i+=4){
			output =tb_output_dma_output.read();
			out[i+output_addr] = output.data.data1;
			out[i+output_addr+1] = output.data.data2;
			out[i+output_addr+2] = output.data.data3;
			out[i+output_addr+3] = output.data.data4;
		}
		output_addr+=output_length;
		for(int i=0;i<weights_length;i+=4){
			output =tb_output_dma_output.read();
			weights[i+weights_addr] = output.data.data1;
			weights[i+weights_addr+1] = output.data.data2;
			weights[i+weights_addr+2] = output.data.data3;
			weights[i+weights_addr+3] = output.data.data4;
		}
		weights_addr+=weights_length;
	}

//	output_addr=0;
//	output_length=bs*1176;//M*min(custom_Tr,R-row*custom_Tr)*C=6*14*14;
//	for(int i=0;i<output_length;i++){
//		output =tb_output_dma_output.read();
//		output2[i+output_addr] = output.data;
//	}
//	weights_addr=0;
//	weights_length=bs*196;//r*c*[m/11]=14*14*1
//	for(int i=0;i<weights_length;i++){
//		input_weights =tb_input_dma_weights.read();
//		pool2index[i+weights_addr] = input_weights.data;
//	}

	cout<<"print out"<<outnum<<endl;
	cout<<"after loopM:"<<endl;
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
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<""<<endl;
	}

}


void convfp_512_512_14( int outnum,
	FPGA_DATA ifm[bs*512*14*14],
	FPGA_DATA weights[512*512*3*3],
	FPGA_DATA outhat[bs*512*14*14],
	FPGA_DATA gama[512],
	FPGA_DATA beta[512],
	FPGA_DATA namuda[512],
	FPGA_DATA out[bs*512*14*14]

){
	//forward conv11,12,13 N,M,R,C=512,512,14,14, stride=1,padding=1,kernel=3
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

	//FPGA_DATA output2[bs*24*24*20]; //[r][c][m]

	weights_addr=0;

	weights_length=2359296;//M*N*tb_custom_k*tb_custom_k=512*512*3*3
	for (int toM=0;toM<8;toM++){
		ifm_addr=0;
		for(int b=0;b<bs;b++){
			for (int to=0;to<tb_M;to+=Tm){//repeate [M/Tm]
				ifm_length=100352;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
										//*((C-1)*tb_custom_stride+tb_custom_k)=512*14*14;
				//ifm_addr=401408*b;//+3584*row;//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*512*28*28+row*16*28*1
				//for(int ti=0;ti<tb_N;ti+=Tn){
					for(int i=0;i<ifm_length;i+=4){
						input_ifm.data.data1 = ifm[i+ifm_addr];
						input_ifm.data.data2 = ifm[i+ifm_addr+1];
						input_ifm.data.data3 = ifm[i+ifm_addr+2];
						input_ifm.data.data4 = ifm[i+ifm_addr+3];
						tb_input_dma_ifm.write(input_ifm);
					}
					//ifm_addr+=12544;//Tn*R_in*C_in=16*28*28
				//}
//				ifm_length=3584;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
//										//*((C-1)*tb_custom_stride+tb_custom_k)=16*8*28;
//				ifm_addr=401408*b;//+3584*row;//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*512*28*28+row*16*28*1
//				for(int ti=0;ti<tb_N;ti+=Tn){
//					for(int i=0;i<ifm_length;i+=4){
//						input_ifm.data.data1 = ifm[i+ifm_addr];
//						input_ifm.data.data2 = ifm[i+ifm_addr+1];
//						input_ifm.data.data3 = ifm[i+ifm_addr+2];
//						input_ifm.data.data4 = ifm[i+ifm_addr+3];
//						tb_input_dma_ifm.write(input_ifm);
//					}
//					ifm_addr+=12544;//Tn*R_in*C_in=16*28*28
//				}
//				for(int row=tb_custom_Tr;row<tb_R-tb_custom_Tr;row+=tb_custom_Tr){
//					ifm_length=4032;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
//															//*((C-1)*tb_custom_stride+tb_custom_k)=16*9*28;
//					ifm_addr=401408*b+(row-1)*448;//+1792*row;//b*N*C_in*R_in+C_in*(row-padding)*stride*min(Tn,N)=b*512*28*28+(row-1)*1*28*16
//					for(int ti=0;ti<tb_N;ti+=Tn){
//						for(int i=0;i<ifm_length;i+=4){
//							input_ifm.data.data1 = ifm[i+ifm_addr];
//							input_ifm.data.data2 = ifm[i+ifm_addr+1];
//							input_ifm.data.data3 = ifm[i+ifm_addr+2];
//							input_ifm.data.data4 = ifm[i+ifm_addr+3];
//							tb_input_dma_ifm.write(input_ifm);
//						}
//						ifm_addr+=12544;//Tn*R_in*C_in=16*28*28
//					}
//				}
//				ifm_length=3584;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
//								//*((C-1)*tb_custom_stride+tb_custom_k)=16*8*28;
//				ifm_addr=401408*b+8960;//+3584*row;//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*512*28*28+row*16*28*1
//				for(int ti=0;ti<tb_N;ti+=Tn){
//					for(int i=0;i<ifm_length;i+=4){
//						input_ifm.data.data1 = ifm[i+ifm_addr];
//						input_ifm.data.data2 = ifm[i+ifm_addr+1];
//						input_ifm.data.data3 = ifm[i+ifm_addr+2];
//						input_ifm.data.data4 = ifm[i+ifm_addr+3];
//						tb_input_dma_ifm.write(input_ifm);
//					}
//					ifm_addr+=12544;//Tn*R_in*C_in=16*28*28
//				}
			}
			ifm_addr+=100352;//N*R_in*C_in=512*14*14
		}
	}



	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = weights[i+weights_addr];
		input_weights.data.data2 = weights[i+weights_addr+1];
		input_weights.data.data3 = weights[i+weights_addr+2];
		input_weights.data.data4 = weights[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}								//=6*3*5*5

	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	//cout<<tb_flag<<endl;

	//print output
	output_addr=0;
	for (int toM=0;toM<8;toM++){
		output_length=12544;//tb_M*R*C=14*14*64;
		output_addr=toM*12544;
		for(int b=0;b<bs;b++){
			for(int i=0;i<output_length;i+=4){
				output =tb_output_dma_output.read();
				out[i+output_addr] = output.data.data1;
				out[i+output_addr+1] = output.data.data2;
				out[i+output_addr+2] = output.data.data3;
				out[i+output_addr+3] = output.data.data4;
			}
			output_addr+=100352;
		}
	}

	cout<<"out after conv"<<endl;
	cout<<out[0]<<endl;

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


	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ofm_addr=0;
	weights_addr=0;
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = gama[i+weights_addr];
		input_weights.data.data2 = gama[i+weights_addr+1];
		input_weights.data.data3 = gama[i+weights_addr+2];
		input_weights.data.data4 = gama[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}
	for(int i=0;i<ofm_length;i+=4){
		input_ofm.data.data1 = beta[i+ofm_addr];
		input_ofm.data.data2 = beta[i+ofm_addr+1];
		input_ofm.data.data3 = beta[i+ofm_addr+2];
		input_ofm.data.data4 = beta[i+ofm_addr+3];
		tb_input_dma_ofm.write(input_ofm);
	}
	//ifm_addr=0;
	//ofm_addr=0;
	ifm_length=3136;//Tm*R*C
	for (int to=0;to<32;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=100352*b+3136*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			//ofm_addr=3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			for(int i=0;i<ifm_length;i+=4){
				input_ifm.data.data1 = out[i+ifm_addr];
				input_ifm.data.data2 = out[i+ifm_addr+1];
				input_ifm.data.data3 = out[i+ifm_addr+2];
				input_ifm.data.data4 = out[i+ifm_addr+3];
				tb_input_dma_ifm.write(input_ifm);
			}
		}
	}
	ofm_addr=0;
	ofm_length=100352;//M*R*C
	for(int b=0;b<bs;b++){
		//ifm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		//ofm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		for(int i=0;i<ofm_length;i+=4){
			input_ofm.data.data1 = out[i+ofm_addr];
			input_ofm.data.data2 = out[i+ofm_addr+1];
			input_ofm.data.data3 = out[i+ofm_addr+2];
			input_ofm.data.data4 = out[i+ofm_addr+3];
			tb_input_dma_ofm.write(input_ofm);
		}
		//ifm_addr+=ifm_length;
		ofm_addr+=ofm_length;
	}
							//=6*3*5*5

	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	//cout<<tb_flag<<endl;
	output_addr=0;
	output_length=100352;//M*R*C
	for(int b=0;b<bs;b++){
		for(int i=0;i<output_length;i+=4){
			out_weights =tb_output_dma_Weightsout.read();
			outhat[i+output_addr] = out_weights.data.data1;
			outhat[i+output_addr+1] = out_weights.data.data2;
			outhat[i+output_addr+2] = out_weights.data.data3;
			outhat[i+output_addr+3] = out_weights.data.data4;
		}
		for(int i=0;i<output_length;i+=4){
			output =tb_output_dma_output.read();
			out[i+output_addr] = output.data.data1;
			out[i+output_addr+1] = output.data.data2;
			out[i+output_addr+2] = output.data.data3;
			out[i+output_addr+3] = output.data.data4;
		}
		output_addr+=output_length;
	}

	output_addr=0;
	output_length=tb_M;//M
	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		namuda[i+output_addr] = output.data.data1;
		namuda[i+output_addr+1] = output.data.data2;
		namuda[i+output_addr+2] = output.data.data3;
		namuda[i+output_addr+3] = output.data.data4;
	}

	cout<<"namuda"<<endl;
	cout<<namuda[0]<<endl;
	cout<<"outhat"<<endl;
	cout<<outhat[0]<<endl;


	cout<<"print out"<<outnum<<endl;


	cout<<"after loopM:"<<endl;


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
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
}


void maxpoolfp_512_7( int outnum,
	FPGA_DATA ifm[bs*512*14*14],
	FPGA_DATA weights[bs*7*7*64],
	FPGA_DATA out[bs*512*7*7]

){
	//forward pool N,M,R,C=512,512,7,7, stride=2,padding=0,kernel=2
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



	ifm_addr=0;
	ifm_length=bs*100352;//M*R_in*C_in=14*14*512;
	//for(int b=0;b<bs;b++){
		for(int i=0;i<ifm_length;i+=4){
			input_ifm.data.data1 = ifm[i+ifm_addr];
			input_ifm.data.data2 = ifm[i+ifm_addr+1];
			input_ifm.data.data3 = ifm[i+ifm_addr+2];
			input_ifm.data.data4 = ifm[i+ifm_addr+3];
			tb_input_dma_ifm.write(input_ifm);
		}
		//ifm_addr+=ifm_length;
	//}




	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	cout<<tb_flag<<endl;

//print output
	//FPGA_DATA output2[bs*12*12*20];//[r][c][m]
   //FPGA_DATA pool2index[bs*12*12*2];//r*c*[m/11]

	output_addr=0;
	weights_addr=0;
	output_length=25088;//M*min(custom_Tr,R-row*custom_Tr)*C=512*7*7;
	weights_length=3136;//r*c*[m/8]=7*7*64
	for(int b=0;b<bs;b++){
		for(int i=0;i<output_length;i+=4){
			output =tb_output_dma_output.read();
			out[i+output_addr] = output.data.data1;
			out[i+output_addr+1] = output.data.data2;
			out[i+output_addr+2] = output.data.data3;
			out[i+output_addr+3] = output.data.data4;
		}
		output_addr+=output_length;
		for(int i=0;i<weights_length;i+=4){
			output =tb_output_dma_output.read();
			weights[i+weights_addr] = output.data.data1;
			weights[i+weights_addr+1] = output.data.data2;
			weights[i+weights_addr+2] = output.data.data3;
			weights[i+weights_addr+3] = output.data.data4;
		}
		weights_addr+=weights_length;
	}

//	output_addr=0;
//	output_length=bs*1176;//M*min(custom_Tr,R-row*custom_Tr)*C=6*14*14;
//	for(int i=0;i<output_length;i++){
//		output =tb_output_dma_output.read();
//		output2[i+output_addr] = output.data;
//	}
//	weights_addr=0;
//	weights_length=bs*196;//r*c*[m/11]=14*14*1
//	for(int i=0;i<weights_length;i++){
//		input_weights =tb_input_dma_weights.read();
//		pool2index[i+weights_addr] = input_weights.data;
//	}

	cout<<"print out"<<outnum<<endl;
	cout<<"after loopM:"<<endl;
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
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<""<<endl;
	}

}

void avgpoolfp_512_1( int outnum,
	FPGA_DATA ifm[bs*512*7*7],
	FPGA_DATA out[bs*512*1*1]

){
	//forward avgpool N,M,R,C=512,512,1,1, stride=1,padding=0,kernel=6
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
	tb_custom_kb=4;
	tb_custom_k=7;
	tb_custom_Tr=1;//12;
	tb_custom_Tc=6;
//	tb_viewTrc=0;
//	tb_viewTm=0;
	tb_R_in=7;
	tb_C_in=7;
	//tb_flag=0;



	ifm_addr=0;
	ifm_length=bs*25088;//M*R_in*C_in=512*7*7;
	//for(int b=0;b<bs;b++){
		for(int i=0;i<ifm_length;i+=4){
			input_ifm.data.data1 = ifm[i+ifm_addr];
			input_ifm.data.data2 = ifm[i+ifm_addr+1];
			input_ifm.data.data3 = ifm[i+ifm_addr+2];
			input_ifm.data.data4 = ifm[i+ifm_addr+3];
			tb_input_dma_ifm.write(input_ifm);
		}
	//	ifm_addr+=ifm_length;
	//}




	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	cout<<tb_flag<<endl;

//print output
	//FPGA_DATA output2[bs*12*12*20];//[r][c][m]
   //FPGA_DATA pool2index[bs*12*12*2];//r*c*[m/11]

	output_addr=0;
	output_length=bs*512;//M*min(custom_Tr,R-row*custom_Tr)*C=512*1*1;

	//for(int b=0;b<bs;b++){
		for(int i=0;i<output_length;i+=4){
			output =tb_output_dma_output.read();
			out[i+output_addr] = output.data.data1;
			out[i+output_addr+1] = output.data.data2;
			out[i+output_addr+2] = output.data.data3;
			out[i+output_addr+3] = output.data.data4;
		}
		//output_addr+=output_length;
//   		for(int i=0;i<weights_length;i+=4){
//   			output =tb_output_dma_output.read();
//   			pool3index[i+weights_addr] = output.data.data1;
//   			pool3index[i+weights_addr+1] = output.data.data2;
//   			pool3index[i+weights_addr+2] = output.data.data3;
//   			pool3index[i+weights_addr+3] = output.data.data4;
//   		}
//   		weights_addr+=weights_length;
//   	}

//	output_addr=0;
//	output_length=bs*1176;//M*min(custom_Tr,R-row*custom_Tr)*C=6*14*14;
//	for(int i=0;i<output_length;i++){
//		output =tb_output_dma_output.read();
//		output2[i+output_addr] = output.data;
//	}
//	weights_addr=0;
//	weights_length=bs*196;//r*c*[m/11]=14*14*1
//	for(int i=0;i<weights_length;i++){
//		input_weights =tb_input_dma_weights.read();
//		pool2index[i+weights_addr] = input_weights.data;
//	}



	cout<<"print out"<<outnum<<endl;
	cout<<"after loopM:"<<endl;
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
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<""<<endl;
	}

}


void convfp_512_1000_1( int outnum,
	FPGA_DATA ifm[bs*512*1*1],
	FPGA_DATA weights[512*1000*1*1],
	FPGA_DATA out[bs*1000*1*1]

){
	//forward fc1 N,M,R,C=512,1000,1,1, stride=1,padding=0,kernel=1
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



	//FPGA_DATA output2[bs*24*24*20]; //[r][c][m]
	//ifm_addr=0;
	weights_addr=0;
	ifm_length=512;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
						//*((C-1)*tb_custom_stride+tb_custom_k)=9216;
	weights_length=512000;//M*N*tb_custom_k*tb_custom_k=512*1000*1*1
	for (int toM=0;toM<1;toM++){
		ifm_addr=0;
		for(int b=0;b<bs;b++){
			for (int to=0;to<tb_M;to+=Tm){//repeate[M/Tm]
				for(int i=0;i<ifm_length;i+=4){
					input_ifm.data.data1 = ifm[i+ifm_addr];
					input_ifm.data.data2 = ifm[i+ifm_addr+1];
					input_ifm.data.data3 = ifm[i+ifm_addr+2];
					input_ifm.data.data4 = ifm[i+ifm_addr+3];
					tb_input_dma_ifm.write(input_ifm);
				}
			}
			ifm_addr+=ifm_length;
		}
	}
	for (int toM=1;toM<2;toM++){
		ifm_addr=0;
		for(int b=0;b<bs;b++){
			for (int to=0;to<488;to+=Tm){//repeate[(tb_M_in-toM*M)/Tm]
				for(int i=0;i<ifm_length;i+=4){
					input_ifm.data.data1 = ifm[i+ifm_addr];
					input_ifm.data.data2 = ifm[i+ifm_addr+1];
					input_ifm.data.data3 = ifm[i+ifm_addr+2];
					input_ifm.data.data4 = ifm[i+ifm_addr+3];
					tb_input_dma_ifm.write(input_ifm);
				}
			}
			ifm_addr+=ifm_length;
		}
	}


//			for(int row=0;row<tb_R;row+=tb_custom_Tr){
//				ifm_addr=784*b+28*row;//b*N*C_in*R_in+C_in*row*stride*Tn=b*1*28*28+row*1*28*1=784b+28*row=784b+28row
//				for(int ti=0;ti<1;ti++){
//					//ifm_length=448;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
//											//*((C-1)*tb_custom_stride+tb_custom_k)=1*16*28;
//					for(int i=0;i<ifm_length;i++){
//						input_ifm.data = input_x_buffer[i+ifm_addr];
//						 tb_input_dma_ifm.write(input_ifm);
//					}
//					ifm_addr+=4704;//Tn*R_in*C_in=6*28*28
//				}
//
//			}


	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = weights[i+weights_addr];
		input_weights.data.data2 = weights[i+weights_addr+1];
		input_weights.data.data3 = weights[i+weights_addr+2];
		input_weights.data.data4 = weights[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}								//=6*3*5*5

	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	//cout<<tb_flag<<endl;

	//print output
	output_addr=0;
	for (int toM=0;toM<1;toM++){
		output_length=512;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=48;
		output_addr=toM*512;
		for(int b=0;b<bs;b++){
			for(int i=0;i<output_length;i+=4){
				output =tb_output_dma_output.read();
				out[i+output_addr] = output.data.data1;
				out[i+output_addr+1] = output.data.data2;
				out[i+output_addr+2] = output.data.data3;
				out[i+output_addr+3] = output.data.data4;
			}
			output_addr+=1000;
		}
	}
	for (int toM=1;toM<2;toM++){
		output_length=488;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=40;
		output_addr=toM*512;
		for(int b=0;b<bs;b++){
			for(int i=0;i<output_length;i+=4){
				output =tb_output_dma_output.read();
				out[i+output_addr] = output.data.data1;
				out[i+output_addr+1] = output.data.data2;
				out[i+output_addr+2] = output.data.data3;
				out[i+output_addr+3] = output.data.data4;
			}
			output_addr+=1000;
		}
	}

	cout<<"print out"<<outnum<<endl;


	cout<<"after loopM:"<<endl;


	//FPGA_DATA output1_print[bs][24][24][20]; //[r][c][m]//only print half and last 10 channels

	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for out"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		//cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<1;r++){
			for (int c=0;c<1;c++){
				for (int too=0;too<Tm;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}

		cout <<" last channel tile  " <<endl;
		//cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<1;r++){
			for (int c=0;c<1;c++){
				for (int too=0;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}

		cout<<""<<endl;
	}
}

void convbp_512_1000_1( int outnum,
	FPGA_DATA ifm[bs*1000*1*1],
	FPGA_DATA weights[512*1000*1*1],
	FPGA_DATA out[bs*512*1*1]

){
	//backward fc1 N,M,R,C=1000,512,1,1, stride=1,padding=0,kernel=1
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


	//ifm_addr=0;
	ifm_length=1000;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
						//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=1000*1*1;
	weights_addr=0;
	for (int toM=0;toM<2;toM++){
		ifm_addr=0;
		for(int b=0;b<bs;b++){
			for(int to=0;to<tb_M;to+=Tm){
				for(int i=0;i<ifm_length;i+=4){
					input_ifm.data.data1 = ifm[i+ifm_addr];
					input_ifm.data.data2 = ifm[i+ifm_addr+1];
					input_ifm.data.data3 = ifm[i+ifm_addr+2];
					input_ifm.data.data4 = ifm[i+ifm_addr+3];
					tb_input_dma_ifm.write(input_ifm);
				}
			}
			ifm_addr+=ifm_length;
		}
		weights_addr=4096*toM;//
		for (int ti=0; ti<62; ti++){

			weights_length=4096;//16*Tm*min(Tn,N-Tn*ti)*tb_custom_k*tb_custom_k
								//=256*16*1*1
			for(int i=0;i<weights_length;i+=4){
				input_weights.data.data1 = weights[i+weights_addr];
				input_weights.data.data2 = weights[i+weights_addr+1];
				input_weights.data.data3 = weights[i+weights_addr+2];
				input_weights.data.data4 = weights[i+weights_addr+3];
				tb_input_dma_weights.write(input_weights);
			}
			weights_addr+=8192;//Tm*N'(M)*tb_custom_k*tb_custom_k=16*512*1*1;
		}
		for (int ti=62; ti<63; ti++){
			weights_length=2048;//16*Tm*min(Tn,N-Tn*ti)*tb_custom_k*tb_custom_k
								//=256*8*1*1
			weights_addr=507904+2048*toM;//ti*Tn*M_in+toM*16*Tm*8=62*16*512+toM*16*16*8
			for(int i=0;i<weights_length;i+=4){
				input_weights.data.data1 = weights[i+weights_addr];
				input_weights.data.data2 = weights[i+weights_addr+1];
				input_weights.data.data3 = weights[i+weights_addr+2];
				input_weights.data.data4 = weights[i+weights_addr+3];
				tb_input_dma_weights.write(input_weights);
			}
			//weights_addr+=4800;//Tm*N'(M)*tb_custom_k*tb_custom_k=6*800*1*1;
		}

	}


		cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	cout<<tb_flag<<endl;

	for (int toM=0;toM<2;toM++){
		output_length=256;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=256*1*1;
		output_addr=toM*256;
		for(int b=0;b<bs;b++){
			for(int i=0;i<output_length;i+=4){
				output =tb_output_dma_output.read();
				out[i+output_addr] = output.data.data1;
				out[i+output_addr+1] = output.data.data2;
				out[i+output_addr+2] = output.data.data3;
				out[i+output_addr+3] = output.data.data4;
			}
			output_addr+=512;
		}
	}

	cout<<"print loss"<<outnum<<endl;


	cout<<"after loopM:"<<endl;


	//FPGA_DATA output1_print[bs][24][24][20]; //[r][c][m]//only print half and last 10 channels

	for(int b=0;b<bs;b++){
		cout << "batch"<<b<<"for loss"<<outnum<<endl;
		//for(int to=0;to<tb_M_in;to+=Tm){
		cout <<" 1st channel tile  " <<endl;
		//cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<1;r++){
			for (int c=0;c<1;c++){
				for (int too=0;too<Tm;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}

		cout <<" last channel tile  " <<endl;
		//cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<1;r++){
			for (int c=0;c<1;c++){
				for (int too=0;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}

		cout<<""<<endl;
	}
}

void convwu_512_1000_1( int outnum,
	FPGA_DATA ifm[bs*512*1*1],
	FPGA_DATA weights[512*1000*1*1],
	FPGA_DATA ofm[bs*1000*1*1]

){

	//update fc1 weights N,M,R,C=512,1000,1,1, stride=1,padding=0,kernel=1
	tb_statec=3;
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





	ifm_length=512;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
				//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=512*1*1;
	ofm_addr=0;
	for (int toM=0;toM<1;toM++){
		ifm_addr=0;
		ofm_length=tb_M;
		ofm_addr=toM*tb_M;
		for(int b=0;b<bs;b++){
			for(int i=0;i<ofm_length;i+=4){
				input_ofm.data.data1 = ofm[i+ofm_addr];
				input_ofm.data.data2 = ofm[i+ofm_addr+1];
				input_ofm.data.data3 = ofm[i+ofm_addr+2];
				input_ofm.data.data4 = ofm[i+ofm_addr+3];
				tb_input_dma_ofm.write(input_ofm);
			}
			for (int to=0;to<tb_M;to+=Tm){//repeate[M/Tm]
				for(int i=0;i<ifm_length;i+=4){
					input_ifm.data.data1 = ifm[i+ifm_addr];
					input_ifm.data.data2 = ifm[i+ifm_addr+1];
					input_ifm.data.data3 = ifm[i+ifm_addr+2];
					input_ifm.data.data4 = ifm[i+ifm_addr+3];
					tb_input_dma_ifm.write(input_ifm);
				}
			}
			ifm_addr+=ifm_length;
			ofm_addr+=tb_M_in;
		}
	}
	for (int toM=1;toM<2;toM++){
		ifm_addr=0;
		ofm_length=488;
		ofm_addr=toM*tb_M;
		for(int b=0;b<bs;b++){
			for(int i=0;i<ofm_length;i+=4){
				input_ofm.data.data1 = ofm[i+ofm_addr];
				input_ofm.data.data2 = ofm[i+ofm_addr+1];
				input_ofm.data.data3 = ofm[i+ofm_addr+2];
				input_ofm.data.data4 = ofm[i+ofm_addr+3];
				tb_input_dma_ofm.write(input_ofm);
			}
			for (int to=0;to<488;to+=Tm){//repeate[M/Tm]
				for(int i=0;i<ifm_length;i+=4){
					input_ifm.data.data1 = ifm[i+ifm_addr];
					input_ifm.data.data2 = ifm[i+ifm_addr+1];
					input_ifm.data.data3 = ifm[i+ifm_addr+2];
					input_ifm.data.data4 = ifm[i+ifm_addr+3];
					tb_input_dma_ifm.write(input_ifm);
				}
			}
			ifm_addr+=ifm_length;
			ofm_addr+=tb_M_in;
		}
	}


	weights_addr=0;

	weights_length=512000;//M*N*tb_custom_k*tb_custom_k
			//=1000*512*1*1

	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = weights[i+weights_addr];
		input_weights.data.data2 = weights[i+weights_addr+1];
		input_weights.data.data3 = weights[i+weights_addr+2];
		input_weights.data.data4 = weights[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}




		cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	cout<<tb_flag<<endl;

	//print output
	output_addr=0;
	output_length=512000;//M*N*tb_custom_k*tb_custom_k
	//=1000*9216*1*1
	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		weights[i+output_addr] = output.data.data1;
		weights[i+output_addr+1] = output.data.data2;
		weights[i+output_addr+2] = output.data.data3;
		weights[i+output_addr+3] = output.data.data4;
	}


	cout<<"after loopM:"<<endl;

	cout<<tb_flag<<endl;
	cout<<"fc1:"<<endl;
	bf_index=0;
	cout<<tb_flag<<endl;

	for (int n=0; n<16;n++){
		cout << weights[n] << ";";
	}
	cout << "last 16 channels" <<endl;
	for (int n=tb_M_in*tb_N*tb_custom_k*tb_custom_k-16; n<tb_M_in*tb_N*tb_custom_k*tb_custom_k;n++){
		cout << weights[n] << ";";
	}
		cout << "" << endl;

}

void avgpoolbp_512_1( int outnum,
	FPGA_DATA ifm[bs*512*1*1],
	FPGA_DATA out[bs*512*7*7]

){
	//backward avgpool N,M,R,C=512,512,1,1, stride=1,padding=0,kernel=7
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


	//FPGA_DATA loss5[bs*10*10*16];//[r][c][m]
	ifm_addr=0;
	ifm_length=bs*512;//M*tb_custom_Tr*tb_custom_Tc=512*1*1;
	//for(int b=0;b<bs;b++){
		for(int i=0;i<ifm_length;i+=4){
			input_ifm.data.data1 = ifm[i+ifm_addr];
			input_ifm.data.data2 = ifm[i+ifm_addr+1];
			input_ifm.data.data3 = ifm[i+ifm_addr+2];
			input_ifm.data.data4 = ifm[i+ifm_addr+3];
			tb_input_dma_ifm.write(input_ifm);
			//cout<<input_ifm.data<<endl;
		}




	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	cout<<tb_flag<<endl;

//print output

	output_addr=0;
	output_length=bs*25088;//M*R_in*C_in=512*7*7;
	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		out[i+output_addr] = output.data.data1;
		out[i+output_addr+1] = output.data.data2;
		out[i+output_addr+2] = output.data.data3;
		out[i+output_addr+3] = output.data.data4;
		//cout << loss3[i+output_addr] << ";";
	}



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
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<""<<endl;
	}

}

void maxpoolbp_512_7( int outnum,
	FPGA_DATA ifm[bs*512*7*7],
	FPGA_DATA weights[bs*7*7*64],
	FPGA_DATA ofm[bs*512*14*14],
	FPGA_DATA out[bs*512*14*14]

){
	//backward pool+relu N,M,R,C=512,512,7,7, stride=2,padding=0,kernel=2
	tb_statec=0;
	tb_statep=5;//view=8+4
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


	//FPGA_DATA loss5[bs*10*10*16];//[r][c][m]
	ifm_addr=0;
	ifm_length=bs*25088;//M*tb_custom_Tr*tb_custom_Tc=512*7*7;
	//for(int b=0;b<bs;b++){
		for(int i=0;i<ifm_length;i+=4){
			input_ifm.data.data1 = ifm[i+ifm_addr];
			input_ifm.data.data2 = ifm[i+ifm_addr+1];
			input_ifm.data.data3 = ifm[i+ifm_addr+2];
			input_ifm.data.data4 = ifm[i+ifm_addr+3];
			tb_input_dma_ifm.write(input_ifm);
			//cout<<input_ifm.data<<endl;
		}
		//ifm_addr+=ifm_length;
	//}

	weights_addr=0;
	weights_length=bs*3136;//r*c*[m/4]=7*7*64
	for(int i=0;i<weights_length;i+=4){
		pool_index.data.data1 =weights[i+weights_addr];
		pool_index.data.data2 =weights[i+weights_addr+1];
		pool_index.data.data3 =weights[i+weights_addr+2];
		pool_index.data.data4 =weights[i+weights_addr+3];
		tb_input_dma_weights.write(pool_index);

	}

	ofm_addr=0;
	ofm_length=bs*100352;//M*R_in*C_in=512*14*14;
	for(int i=0;i<ofm_length;i+=4){
		input_ofm.data.data1 = ofm[i+ofm_addr];
		input_ofm.data.data2 = ofm[i+ofm_addr+1];
		input_ofm.data.data3 = ofm[i+ofm_addr+2];
		input_ofm.data.data4 = ofm[i+ofm_addr+3];
		tb_input_dma_ofm.write(input_ofm);
	}

	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	cout<<tb_flag<<endl;

//print output

	output_addr=0;
	output_length=bs*100352;//M*R_in*C_in=512*14*14;
	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		out[i+output_addr] = output.data.data1;
		out[i+output_addr+1] = output.data.data2;
		out[i+output_addr+2] = output.data.data3;
		out[i+output_addr+3] = output.data.data4;
		//cout << loss3[i+output_addr] << ";";
	}




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
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<""<<endl;
	}

}

void convbp_512_512_14( int outnum,
	ap_uint<4> reluornot,//conv11=4;conv12,13=5
	FPGA_DATA ifm[bs*512*14*14],
	FPGA_DATA weights[512*512*3*3],
	FPGA_DATA outhat[bs*512*14*14],
	FPGA_DATA gama[512],
	FPGA_DATA beta[512],
	FPGA_DATA namuda[512],
	FPGA_DATA ofm[bs*512*14*14],
	FPGA_DATA out[bs*512*14*14]

){

//backward bn11,12,13 N,M,R,C=512,512,14,14, stride=1,padding=1,kernel=3
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


	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ifm_addr=0;
	ofm_addr=0;
	weights_addr=0;
	ifm_length=tb_M;//M
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = gama[i+weights_addr];
		input_weights.data.data2 = gama[i+weights_addr+1];
		input_weights.data.data3 = gama[i+weights_addr+2];
		input_weights.data.data4 = gama[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}
	for(int i=0;i<ofm_length;i+=4){
		input_ofm.data.data1 = beta[i+ofm_addr];
		input_ofm.data.data2 = beta[i+ofm_addr+1];
		input_ofm.data.data3 = beta[i+ofm_addr+2];
		input_ofm.data.data4 = beta[i+ofm_addr+3];
		tb_input_dma_ofm.write(input_ofm);
	}
	for(int i=0;i<ifm_length;i+=4){
		input_ifm.data.data1 = namuda[i+ifm_addr];
		input_ifm.data.data2 = namuda[i+ifm_addr+1];
		input_ifm.data.data3 = namuda[i+ifm_addr+2];
		input_ifm.data.data4 = namuda[i+ifm_addr+3];
		tb_input_dma_ifm.write(input_ifm);
	}
	//ifm_addr=0;
	//ofm_addr=0;
	ofm_length=3136;//Tm*R*C
	ifm_length=3136;//Tm*R*C
	for (int to=0;to<32;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=100352*b+3136*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			ofm_addr=100352*b+3136*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			for(int i=0;i<ifm_length;i+=4){
				input_ifm.data.data1 = ifm[i+ifm_addr];
				input_ifm.data.data2 = ifm[i+ifm_addr+1];
				input_ifm.data.data3 = ifm[i+ifm_addr+2];
				input_ifm.data.data4 = ifm[i+ifm_addr+3];
				tb_input_dma_ifm.write(input_ifm);
			}
			for(int i=0;i<ofm_length;i+=4){
				input_ofm.data.data1 = outhat[i+ofm_addr];
				input_ofm.data.data2 = outhat[i+ofm_addr+1];
				input_ofm.data.data3 = outhat[i+ofm_addr+2];
				input_ofm.data.data4 = outhat[i+ofm_addr+3];
				tb_input_dma_ofm.write(input_ofm);
			}
		}
	}//=6*3*5*5
	ifm_addr=0;
	ofm_addr=0;
	ofm_length=100352;//M*R*C
	ifm_length=100352;//M*R*C
	for(int b=0;b<bs;b++){
		//ifm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		//ofm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		for(int i=0;i<ifm_length;i+=4){
			input_ifm.data.data1 = ifm[i+ifm_addr];
			input_ifm.data.data2 = ifm[i+ifm_addr+1];
			input_ifm.data.data3 = ifm[i+ifm_addr+2];
			input_ifm.data.data4 = ifm[i+ifm_addr+3];
			tb_input_dma_ifm.write(input_ifm);
		}
		for(int i=0;i<ofm_length;i+=4){
			input_ofm.data.data1 = outhat[i+ofm_addr];
			input_ofm.data.data2 = outhat[i+ofm_addr+1];
			input_ofm.data.data3 = outhat[i+ofm_addr+2];
			input_ofm.data.data4 = outhat[i+ofm_addr+3];
			tb_input_dma_ofm.write(input_ofm);
		}
		ifm_addr+=ifm_length;
		ofm_addr+=ofm_length;
	}

	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	//cout<<tb_flag<<endl;

	//print output
	output_addr=0;
	output_length=100352;//M*R*C
	for(int b=0;b<bs;b++){
		for(int i=0;i<output_length;i+=4){
			output =tb_output_dma_output.read();
			ifm[i+output_addr] = output.data.data1;
			ifm[i+output_addr+1] = output.data.data2;
			ifm[i+output_addr+2] = output.data.data3;
			ifm[i+output_addr+3] = output.data.data4;
		}
		output_addr+=output_length;
	}
	output_addr=0;
	output_length=tb_M;//M
	for(int i=0;i<output_length;i+=4){
		out_weights =tb_output_dma_Weightsout.read();
		gama[i+output_addr] = out_weights.data.data1;
		gama[i+output_addr+1] = out_weights.data.data2;
		gama[i+output_addr+2] = out_weights.data.data3;
		gama[i+output_addr+3] = out_weights.data.data4;
	}
	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		beta[i+output_addr] = output.data.data1;
		beta[i+output_addr+1] = output.data.data2;
		beta[i+output_addr+2] = output.data.data3;
		beta[i+output_addr+3] = output.data.data4;
	}

	cout<<"gama first:"<<endl;
	for(int i=0;i<10;i++ ){
		cout<<gama[i]<< ";";
	}
	cout<<"gama last:"<<endl;
	for(int i=tb_M-10;i<tb_M;i++ ){
		cout<<gama[i]<< ";";
	}

	cout<<"beta first:"<<endl;
	cout<<beta[0]<< ";";
	cout<<beta[1]<< ";";
	cout<<beta[2]<< ";";
	cout<<"beta last:"<<endl;
	cout<<beta[tb_M-3]<< ";";
	cout<<beta[tb_M-2]<< ";";
	cout<<beta[tb_M-1]<<endl;

//	for(int b=0;b<bs;b++){
//		cout << "batch"<<b<<"for loss after bn"<<outnum<<endl;
//		//for(int to=0;to<tb_M_in;to+=Tm){
//		cout <<" 1st channel tile  " <<endl;
//		cout<<"1st 3rows 1st 3cols"<<":";
//		for (int r=0;r<3;r++){
//			for (int c=0;c<3;c++){
//				for (int too=0;too<3;too++){
//					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
//					cout <<ifm[bf_index]<< ";";
//
//				}
//			}
//		}
//
//		cout<<"1st 3rows last 3cols"<<":";
//		for (int r=0;r<3;r++){
//			for (int c=tb_C-3;c<tb_C;c++){
//				for (int too=0;too<3;too++){
//					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
//					cout <<ifm[bf_index]<< ";";
//
//				}
//			}
//		}
//		cout<<"last 3rows 1st 3cols"<<":";
//		for (int r=tb_R-3;r<tb_R;r++){
//			for (int c=0;c<3;c++){
//				for (int too=0;too<3;too++){
//					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
//					cout <<ifm[bf_index]<< ";";
//
//				}
//			}
//		}
//		cout<<"last 3rows last 3cols"<<":";
//		for (int r=tb_R-3;r<tb_R;r++){
//			for (int c=tb_C-3;c<tb_C;c++){
//				for (int too=0;too<3;too++){
//					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
//					cout <<ifm[bf_index]<< ";";
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
//					cout <<ifm[bf_index]<< ";";
//
//				}
//			}
//		}
//		cout<<"1st 3rows last 3cols"<<":";
//		for (int r=0;r<3;r++){
//			for (int c=tb_C-3;c<tb_C;c++){
//				for (int too=Tm-3;too<Tm;too++){
//					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
//					cout <<ifm[bf_index]<< ";";
//
//				}
//			}
//		}
//		cout<<"last 3rows 1st 3cols"<<":";
//		for (int r=tb_R-3;r<tb_R;r++){
//			for (int c=0;c<3;c++){
//				for (int too=Tm-3;too<Tm;too++){
//					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
//					cout <<ifm[bf_index]<< ";";
//
//				}
//			}
//		}
//		cout<<"last 3rows last 3cols"<<":";
//		for (int r=tb_R-3;r<tb_R;r++){
//			for (int c=tb_C-3;c<tb_C;c++){
//				for (int too=Tm-3;too<Tm;too++){
//					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
//					cout <<ifm[bf_index]<< ";";
//
//				}
//			}
//		}
//		cout<<""<<endl;
//	}

//backward conv11,12+relu,13+relu N,M,R,C=512,512,14,14, stride=1,padding=1,kernel=3
	tb_statec=reluornot;//11->4;12,13->5
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


	//FPGA_DATA loss6[bs*14*14*6];
	//ifm_addr=0;
	ifm_length=100352;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
				//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=512*14*14;
	ofm_length=12544;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=64*14*14;
	weights_addr=0;
	for (int toM=0;toM<8;toM++){
		ifm_addr=0;
		ofm_addr=toM*12544;
		for(int b=0;b<bs;b++){
			for (int to=0;to<tb_M;to+=Tm){//repeate[M/Tm]
				for(int i=0;i<ifm_length;i+=4){
					input_ifm.data.data1 = ifm[i+ifm_addr];
					input_ifm.data.data2 = ifm[i+ifm_addr+1];
					input_ifm.data.data3 = ifm[i+ifm_addr+2];
					input_ifm.data.data4 = ifm[i+ifm_addr+3];
					tb_input_dma_ifm.write(input_ifm);
				}

			}
			ifm_addr+=ifm_length;
			if(reluornot>=5){
				for(int i=0;i<ofm_length;i+=4){
					input_ofm.data.data1 = ofm[i+ofm_addr];
					input_ofm.data.data2 = ofm[i+ofm_addr+1];
					input_ofm.data.data3 = ofm[i+ofm_addr+2];
					input_ofm.data.data4 = ofm[i+ofm_addr+3];
					tb_input_dma_ofm.write(input_ofm);
				}
				ofm_addr+=100352;
			}
		}
		weights_addr=9216*toM;//
		for (int ti=0; ti<tb_N; ti+=Tn){

			weights_length=9216;//4*Tm*min(Tn,N-Tn*ti)*tb_custom_k*tb_custom_k
								//=64*16*3*3
			for(int i=0;i<weights_length;i+=4){
				input_weights.data.data1 = weights[i+weights_addr];
				input_weights.data.data2 = weights[i+weights_addr+1];
				input_weights.data.data3 = weights[i+weights_addr+2];
				input_weights.data.data4 = weights[i+weights_addr+3];
				tb_input_dma_weights.write(input_weights);
			}
			weights_addr+=73728;//Tm*N'(M)*tb_custom_k*tb_custom_k=16*512*3*3;
		}
	}



		cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	cout<<tb_flag<<endl;

	for (int toM=0;toM<8;toM++){
		output_length=12544;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=64*14*14;
		output_addr=toM*12544;
		for(int b=0;b<bs;b++){
			for(int i=0;i<output_length;i+=4){
				output =tb_output_dma_output.read();
				out[i+output_addr] = output.data.data1;
				out[i+output_addr+1] = output.data.data2;
				out[i+output_addr+2] = output.data.data3;
				out[i+output_addr+3] = output.data.data4;
			}
			output_addr+=100352;
		}
	}

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
					cout <<out[bf_index]<< ";";

				}
			}
		}

		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
}

void convwu_512_512_14( int outnum,
	FPGA_DATA ifm[bs*512*14*14],
	FPGA_DATA weights[512*512*3*3],
	FPGA_DATA ofm[bs*512*14*14],
	FPGA_DATA weights_print[512][512][3][3]

){
	//update conv11,12,13 weights N,M,R,C=512,512,14,14, stride=1,padding=1,kernel=3
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

	ifm_length=100352;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
					//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=512*14*14;
	for (int toM=0;toM<8;toM++){
		ifm_addr=0;
		ofm_length=12544;//tb_M*tb_R*tb_C=64*14*14;
		ofm_addr=toM*12544;
		for(int b=0;b<bs;b++){
			for(int i=0;i<ofm_length;i+=4){
				input_ofm.data.data1 = ofm[i+ofm_addr];
				input_ofm.data.data2 = ofm[i+ofm_addr+1];
				input_ofm.data.data3 = ofm[i+ofm_addr+2];
				input_ofm.data.data4 = ofm[i+ofm_addr+3];
				tb_input_dma_ofm.write(input_ofm);
			}
			for (int to=0;to<tb_M;to+=Tm){//repeate[M/Tm]
				for(int i=0;i<ifm_length;i+=4){
					input_ifm.data.data1 = ifm[i+ifm_addr];
					input_ifm.data.data2 = ifm[i+ifm_addr+1];
					input_ifm.data.data3 = ifm[i+ifm_addr+2];
					input_ifm.data.data4 = ifm[i+ifm_addr+3];
					tb_input_dma_ifm.write(input_ifm);
				}
			}
			ifm_addr+=ifm_length;
			ofm_addr+=100352;//tb_M_in*tb_R*tb_C=512*14*14;
		}
	}


	weights_addr=0;

	weights_length=2359296;//M*N*tb_custom_k*tb_custom_k
			//=512*512*3*3

	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = weights[i+weights_addr];
		input_weights.data.data2 = weights[i+weights_addr+1];
		input_weights.data.data3 = weights[i+weights_addr+2];
		input_weights.data.data4 = weights[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}




		cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	cout<<tb_flag<<endl;

	//print output
	output_addr=0;
	output_length=2359296;//M*N*tb_custom_k*tb_custom_k
	//=512*512*3*3
	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		weights[i+output_addr] = output.data.data1;
		weights[i+output_addr+1] = output.data.data2;
		weights[i+output_addr+2] = output.data.data3;
		weights[i+output_addr+3] = output.data.data4;
	}



	cout<<"after loopM:"<<endl;

	cout<<tb_flag<<endl;
	cout<<"conv"<<outnum<<endl;
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
					cout<<weights_print[m][n][r][c] << ";";
				}
			}

		}
	}
	cout<<"1st 3OCs last 3ICs" <<endl;
	for (int m=0; m<3;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c] << ";";
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
					cout<<weights_print[m][n][r][c] << ";";
				}
			}

		}
	}
	cout<<"last 3OCs last 3ICs" <<endl;
	for (int m=tb_M_in-3; m<tb_M_in;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c] << ";";
				}
			}

		}
	}
}

void maxpoolbp_512_14( int outnum,
	FPGA_DATA ifm[bs*512*14*14],
	FPGA_DATA weights[bs*14*14*64],
	FPGA_DATA ofm[bs*512*28*28],
	FPGA_DATA out[bs*512*28*28]

){
	//backward pool+relu N,M,R,C=512,512,14,14, stride=2,padding=0,kernel=2
	tb_statec=0;
	tb_statep=5;//view=8+4
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


	//FPGA_DATA loss5[bs*10*10*16];//[r][c][m]
	ifm_addr=0;
	ifm_length=bs*100352;//M*tb_custom_Tr*tb_custom_Tc=512*14*14;
	//for(int b=0;b<bs;b++){
		for(int i=0;i<ifm_length;i+=4){
			input_ifm.data.data1 = ifm[i+ifm_addr];
			input_ifm.data.data2 = ifm[i+ifm_addr+1];
			input_ifm.data.data3 = ifm[i+ifm_addr+2];
			input_ifm.data.data4 = ifm[i+ifm_addr+3];
			tb_input_dma_ifm.write(input_ifm);
			//cout<<input_ifm.data<<endl;
		}
		//ifm_addr+=ifm_length;
	//}

	weights_addr=0;
	weights_length=bs*12544;//r*c*[m/8]=14*14*64
	for(int i=0;i<weights_length;i+=4){
		pool_index.data.data1 =weights[i+weights_addr];
		pool_index.data.data2 =weights[i+weights_addr+1];
		pool_index.data.data3 =weights[i+weights_addr+2];
		pool_index.data.data4 =weights[i+weights_addr+3];
		tb_input_dma_weights.write(pool_index);

	}


	ofm_addr=0;
	ofm_length=bs*401408;//M*R_in*C_in=512*28*28;
	for(int i=0;i<ofm_length;i+=4){
		input_ofm.data.data1 = ofm[i+ofm_addr];
		input_ofm.data.data2 = ofm[i+ofm_addr+1];
		input_ofm.data.data3 = ofm[i+ofm_addr+2];
		input_ofm.data.data4 = ofm[i+ofm_addr+3];
		tb_input_dma_ofm.write(input_ofm);
	}


	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	cout<<tb_flag<<endl;

//print output

	output_addr=0;
	output_length=bs*401408;//M*R_in*C_in=512*28*28;
	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		out[i+output_addr] = output.data.data1;
		out[i+output_addr+1] = output.data.data2;
		out[i+output_addr+2] = output.data.data3;
		out[i+output_addr+3] = output.data.data4;
		//cout << loss3[i+output_addr] << ";";
	}




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
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
}

void convbp_512_512_28( int outnum,
	FPGA_DATA ifm[bs*512*28*28],
	FPGA_DATA weights[512*512*3*3],
	FPGA_DATA outhat[bs*512*28*28],
	FPGA_DATA gama[512],
	FPGA_DATA beta[512],
	FPGA_DATA namuda[512],
	FPGA_DATA ofm[bs*512*28*28],
	FPGA_DATA out[bs*512*28*28]

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


	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ifm_addr=0;
	ofm_addr=0;
	weights_addr=0;
	ifm_length=tb_M;//M
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = gama[i+weights_addr];
		input_weights.data.data2 = gama[i+weights_addr+1];
		input_weights.data.data3 = gama[i+weights_addr+2];
		input_weights.data.data4 = gama[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}
	for(int i=0;i<ofm_length;i+=4){
		input_ofm.data.data1 = beta[i+ofm_addr];
		input_ofm.data.data2 = beta[i+ofm_addr+1];
		input_ofm.data.data3 = beta[i+ofm_addr+2];
		input_ofm.data.data4 = beta[i+ofm_addr+3];
		tb_input_dma_ofm.write(input_ofm);
	}
	for(int i=0;i<ifm_length;i+=4){
		input_ifm.data.data1 = namuda[i+ifm_addr];
		input_ifm.data.data2 = namuda[i+ifm_addr+1];
		input_ifm.data.data3 = namuda[i+ifm_addr+2];
		input_ifm.data.data4 = namuda[i+ifm_addr+3];
		tb_input_dma_ifm.write(input_ifm);
	}
	//ifm_addr=0;
	//ofm_addr=0;
	ofm_length=12544;//Tm*R*C
	ifm_length=12544;//Tm*R*C
	for (int to=0;to<32;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=401408*b+12544*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			ofm_addr=401408*b+12544*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			for(int i=0;i<ifm_length;i+=4){
				input_ifm.data.data1 = ifm[i+ifm_addr];
				input_ifm.data.data2 = ifm[i+ifm_addr+1];
				input_ifm.data.data3 = ifm[i+ifm_addr+2];
				input_ifm.data.data4 = ifm[i+ifm_addr+3];
				tb_input_dma_ifm.write(input_ifm);
			}
			for(int i=0;i<ofm_length;i+=4){
				input_ofm.data.data1 = outhat[i+ofm_addr];
				input_ofm.data.data2 = outhat[i+ofm_addr+1];
				input_ofm.data.data3 = outhat[i+ofm_addr+2];
				input_ofm.data.data4 = outhat[i+ofm_addr+3];
				tb_input_dma_ofm.write(input_ofm);
			}
		}
	}//=6*3*5*5
	ifm_addr=0;
	ofm_addr=0;
	ofm_length=401408;//M*R*C
	ifm_length=401408;//M*R*C
	for(int b=0;b<bs;b++){
		//ifm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		//ofm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		for(int i=0;i<ifm_length;i+=4){
			input_ifm.data.data1 = ifm[i+ifm_addr];
			input_ifm.data.data2 = ifm[i+ifm_addr+1];
			input_ifm.data.data3 = ifm[i+ifm_addr+2];
			input_ifm.data.data4 = ifm[i+ifm_addr+3];
			tb_input_dma_ifm.write(input_ifm);
		}
		for(int i=0;i<ofm_length;i+=4){
			input_ofm.data.data1 = outhat[i+ofm_addr];
			input_ofm.data.data2 = outhat[i+ofm_addr+1];
			input_ofm.data.data3 = outhat[i+ofm_addr+2];
			input_ofm.data.data4 = outhat[i+ofm_addr+3];
			tb_input_dma_ofm.write(input_ofm);
		}
		ifm_addr+=ifm_length;
		ofm_addr+=ofm_length;
	}

	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	//cout<<tb_flag<<endl;

	//print output
	output_addr=0;
	output_length=401408;//M*R*C
	for(int b=0;b<bs;b++){
		for(int i=0;i<output_length;i+=4){
			output =tb_output_dma_output.read();
			ifm[i+output_addr] = output.data.data1;
			ifm[i+output_addr+1] = output.data.data2;
			ifm[i+output_addr+2] = output.data.data3;
			ifm[i+output_addr+3] = output.data.data4;
		}
		output_addr+=output_length;
	}
	output_addr=0;
	output_length=tb_M;//M
	for(int i=0;i<output_length;i+=4){
		out_weights =tb_output_dma_Weightsout.read();
		gama[i+output_addr] = out_weights.data.data1;
		gama[i+output_addr+1] = out_weights.data.data2;
		gama[i+output_addr+2] = out_weights.data.data3;
		gama[i+output_addr+3] = out_weights.data.data4;
	}
	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		beta[i+output_addr] = output.data.data1;
		beta[i+output_addr+1] = output.data.data2;
		beta[i+output_addr+2] = output.data.data3;
		beta[i+output_addr+3] = output.data.data4;
	}

	cout<<"gama first:"<<endl;
	for(int i=0;i<10;i++ ){
		cout<<gama[i]<< ";";
	}
	cout<<"gama last:"<<endl;
	for(int i=tb_M-10;i<tb_M;i++ ){
		cout<<gama[i]<< ";";
	}

	cout<<"beta first:"<<endl;
	cout<<beta[0]<< ";";
	cout<<beta[1]<< ";";
	cout<<beta[2]<< ";";
	cout<<"beta last:"<<endl;
	cout<<beta[tb_M-3]<< ";";
	cout<<beta[tb_M-2]<< ";";
	cout<<beta[tb_M-1]<<endl;


	//backward conv9,10+relu N,M,R,C=512,512,28,28, stride=1,padding=1,kernel=3
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


	//FPGA_DATA loss6[bs*14*14*6];
	//ifm_addr=0;
	ifm_length=401408;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
				//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=512*28*28;
	ofm_length=50176;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=64*28*28;
	weights_addr=0;
	for (int toM=0;toM<8;toM++){
		ifm_addr=0;
		ofm_addr=toM*50176;
		for(int b=0;b<bs;b++){
			for (int to=0;to<tb_M;to+=Tm){//repeate[M/Tm]
				for(int i=0;i<ifm_length;i+=4){
					input_ifm.data.data1 = ifm[i+ifm_addr];
					input_ifm.data.data2 = ifm[i+ifm_addr+1];
					input_ifm.data.data3 = ifm[i+ifm_addr+2];
					input_ifm.data.data4 = ifm[i+ifm_addr+3];
					tb_input_dma_ifm.write(input_ifm);
				}
			}
			ifm_addr+=ifm_length;
			for(int i=0;i<ofm_length;i+=4){
				input_ofm.data.data1 = ofm[i+ofm_addr];
				input_ofm.data.data2 = ofm[i+ofm_addr+1];
				input_ofm.data.data3 = ofm[i+ofm_addr+2];
				input_ofm.data.data4 = ofm[i+ofm_addr+3];
				tb_input_dma_ofm.write(input_ofm);
			}
			ofm_addr+=401408;
		}
		weights_addr=9216*toM;//
		for (int ti=0; ti<tb_N; ti+=Tn){

			weights_length=9216;//4*Tm*min(Tn,N-Tn*ti)*tb_custom_k*tb_custom_k
								//=64*16*3*3
			for(int i=0;i<weights_length;i+=4){
				input_weights.data.data1 = weights[i+weights_addr];
				input_weights.data.data2 = weights[i+weights_addr+1];
				input_weights.data.data3 = weights[i+weights_addr+2];
				input_weights.data.data4 = weights[i+weights_addr+3];
				tb_input_dma_weights.write(input_weights);
			}
			weights_addr+=73728;//Tm*N'(M)*tb_custom_k*tb_custom_k=16*512*3*3;
		}
	}



		cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	cout<<tb_flag<<endl;

	for (int toM=0;toM<8;toM++){
		output_length=50176;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=64*28*28;
		output_addr=toM*50176;
		for(int b=0;b<bs;b++){
			for(int i=0;i<output_length;i+=4){
				output =tb_output_dma_output.read();
				out[i+output_addr] = output.data.data1;
				out[i+output_addr+1] = output.data.data2;
				out[i+output_addr+2] = output.data.data3;
				out[i+output_addr+3] = output.data.data4;
			}
			output_addr+=401408;
		}
	}

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
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
}

void convwu_512_512_28( int outnum,
	FPGA_DATA ifm[bs*512*28*28],
	FPGA_DATA weights[512*512*3*3],
	FPGA_DATA ofm[bs*512*28*28],
	FPGA_DATA weights_print[512][512][3][3]

){
	//update conv9,10 weights N,M,R,C=512,512,28,28, stride=1,padding=1,kernel=3
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

	ifm_length=401408;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
					//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=512*28*28;
	for (int toM=0;toM<8;toM++){
		ifm_addr=0;
		ofm_length=50176;//tb_M*tb_R*tb_C=64*28*28;
		ofm_addr=toM*50176;
		for(int b=0;b<bs;b++){
			for(int i=0;i<ofm_length;i+=4){
				input_ofm.data.data1 = ofm[i+ofm_addr];
				input_ofm.data.data2 = ofm[i+ofm_addr+1];
				input_ofm.data.data3 = ofm[i+ofm_addr+2];
				input_ofm.data.data4 = ofm[i+ofm_addr+3];
				tb_input_dma_ofm.write(input_ofm);
			}
			for (int to=0;to<tb_M;to+=Tm){//repeate[M/Tm]
				for(int i=0;i<ifm_length;i+=4){
					input_ifm.data.data1 = ifm[i+ifm_addr];
					input_ifm.data.data2 = ifm[i+ifm_addr+1];
					input_ifm.data.data3 = ifm[i+ifm_addr+2];
					input_ifm.data.data4 = ifm[i+ifm_addr+3];
					tb_input_dma_ifm.write(input_ifm);
				}
			}
			ifm_addr+=ifm_length;
			ofm_addr+=401408;//tb_M_in*tb_R*tb_C=512*28*28;
		}
	}


	weights_addr=0;

	weights_length=2359296;//M*N*tb_custom_k*tb_custom_k
			//=512*512*3*3

	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = weights[i+weights_addr];
		input_weights.data.data2 = weights[i+weights_addr+1];
		input_weights.data.data3 = weights[i+weights_addr+2];
		input_weights.data.data4 = weights[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}




		cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	cout<<tb_flag<<endl;

	//print output
	output_addr=0;
	output_length=2359296;//M*N*tb_custom_k*tb_custom_k
	//=512*512*3*3
	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		weights[i+output_addr] = output.data.data1;
		weights[i+output_addr+1] = output.data.data2;
		weights[i+output_addr+2] = output.data.data3;
		weights[i+output_addr+3] = output.data.data4;
	}



	cout<<"after loopM:"<<endl;

	cout<<tb_flag<<endl;
	cout<<"conv"<<outnum<<endl;
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
					cout<<weights_print[m][n][r][c] << ";";
				}
			}

		}
	}
	cout<<"1st 3OCs last 3ICs" <<endl;
	for (int m=0; m<3;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c] << ";";
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
					cout<<weights_print[m][n][r][c] << ";";
				}
			}

		}
	}
	cout<<"last 3OCs last 3ICs" <<endl;
	for (int m=tb_M_in-3; m<tb_M_in;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c] << ";";
				}
			}

		}
	}
}

void convbp_256_512_28( int outnum,
	FPGA_DATA ifm[bs*512*28*28],
	FPGA_DATA weights[256*512*3*3],
	FPGA_DATA outhat[bs*512*28*28],
	FPGA_DATA gama[512],
	FPGA_DATA beta[512],
	FPGA_DATA namuda[512],
	FPGA_DATA out[bs*256*28*28]

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


	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ifm_addr=0;
	ofm_addr=0;
	weights_addr=0;
	ifm_length=tb_M;//M
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = gama[i+weights_addr];
		input_weights.data.data2 = gama[i+weights_addr+1];
		input_weights.data.data3 = gama[i+weights_addr+2];
		input_weights.data.data4 = gama[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}
	for(int i=0;i<ofm_length;i+=4){
		input_ofm.data.data1 = beta[i+ofm_addr];
		input_ofm.data.data2 = beta[i+ofm_addr+1];
		input_ofm.data.data3 = beta[i+ofm_addr+2];
		input_ofm.data.data4 = beta[i+ofm_addr+3];
		tb_input_dma_ofm.write(input_ofm);
	}
	for(int i=0;i<ifm_length;i+=4){
		input_ifm.data.data1 = namuda[i+ifm_addr];
		input_ifm.data.data2 = namuda[i+ifm_addr+1];
		input_ifm.data.data3 = namuda[i+ifm_addr+2];
		input_ifm.data.data4 = namuda[i+ifm_addr+3];
		tb_input_dma_ifm.write(input_ifm);
	}
	//ifm_addr=0;
	//ofm_addr=0;
	ofm_length=12544;//Tm*R*C
	ifm_length=12544;//Tm*R*C
	for (int to=0;to<32;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=401408*b+12544*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			ofm_addr=401408*b+12544*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			for(int i=0;i<ifm_length;i+=4){
				input_ifm.data.data1 = ifm[i+ifm_addr];
				input_ifm.data.data2 = ifm[i+ifm_addr+1];
				input_ifm.data.data3 = ifm[i+ifm_addr+2];
				input_ifm.data.data4 = ifm[i+ifm_addr+3];
				tb_input_dma_ifm.write(input_ifm);
			}
			for(int i=0;i<ofm_length;i+=4){
				input_ofm.data.data1 = outhat[i+ofm_addr];
				input_ofm.data.data2 = outhat[i+ofm_addr+1];
				input_ofm.data.data3 = outhat[i+ofm_addr+2];
				input_ofm.data.data4 = outhat[i+ofm_addr+3];
				tb_input_dma_ofm.write(input_ofm);
			}
		}
	}//=6*3*5*5
	ifm_addr=0;
	ofm_addr=0;
	ofm_length=401408;//M*R*C
	ifm_length=401408;//M*R*C
	for(int b=0;b<bs;b++){
		//ifm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		//ofm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		for(int i=0;i<ifm_length;i+=4){
			input_ifm.data.data1 = ifm[i+ifm_addr];
			input_ifm.data.data2 = ifm[i+ifm_addr+1];
			input_ifm.data.data3 = ifm[i+ifm_addr+2];
			input_ifm.data.data4 = ifm[i+ifm_addr+3];
			tb_input_dma_ifm.write(input_ifm);
		}
		for(int i=0;i<ofm_length;i+=4){
			input_ofm.data.data1 = outhat[i+ofm_addr];
			input_ofm.data.data2 = outhat[i+ofm_addr+1];
			input_ofm.data.data3 = outhat[i+ofm_addr+2];
			input_ofm.data.data4 = outhat[i+ofm_addr+3];
			tb_input_dma_ofm.write(input_ofm);
		}
		ifm_addr+=ifm_length;
		ofm_addr+=ofm_length;
	}

	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	//cout<<tb_flag<<endl;

	//print output
	output_addr=0;
	output_length=401408;//M*R*C
	for(int b=0;b<bs;b++){
		for(int i=0;i<output_length;i+=4){
			output =tb_output_dma_output.read();
			ifm[i+output_addr] = output.data.data1;
			ifm[i+output_addr+1] = output.data.data2;
			ifm[i+output_addr+2] = output.data.data3;
			ifm[i+output_addr+3] = output.data.data4;
		}
		output_addr+=output_length;
	}
	output_addr=0;
	output_length=tb_M;//M
	for(int i=0;i<output_length;i+=4){
		out_weights =tb_output_dma_Weightsout.read();
		gama[i+output_addr] = out_weights.data.data1;
		gama[i+output_addr+1] = out_weights.data.data2;
		gama[i+output_addr+2] = out_weights.data.data3;
		gama[i+output_addr+3] = out_weights.data.data4;
	}
	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		beta[i+output_addr] = output.data.data1;
		beta[i+output_addr+1] = output.data.data2;
		beta[i+output_addr+2] = output.data.data3;
		beta[i+output_addr+3] = output.data.data4;
	}

	cout<<"gama first:"<<endl;
	for(int i=0;i<10;i++ ){
		cout<<gama[i]<< ";";
	}
	cout<<"gama last:"<<endl;
	for(int i=tb_M-10;i<tb_M;i++ ){
		cout<<gama[i]<< ";";
	}

	cout<<"beta first:"<<endl;
	cout<<beta[0]<< ";";
	cout<<beta[1]<< ";";
	cout<<beta[2]<< ";";
	cout<<"beta last:"<<endl;
	cout<<beta[tb_M-3]<< ";";
	cout<<beta[tb_M-2]<< ";";
	cout<<beta[tb_M-1]<<endl;


	//backward conv8 N,M,R,C=512,256,28,28, stride=1,padding=1,kernel=3
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


	//FPGA_DATA loss6[bs*14*14*6];
	//ifm_addr=0;
	ifm_length=401408;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
				//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=512*28*28;

	weights_addr=0;
	for (int toM=0;toM<4;toM++){
		ifm_addr=0;
		for(int b=0;b<bs;b++){
			for (int to=0;to<tb_M;to+=Tm){//repeate[M/Tm]
				for(int i=0;i<ifm_length;i+=4){
					input_ifm.data.data1 = ifm[i+ifm_addr];
					input_ifm.data.data2 = ifm[i+ifm_addr+1];
					input_ifm.data.data3 = ifm[i+ifm_addr+2];
					input_ifm.data.data4 = ifm[i+ifm_addr+3];
					tb_input_dma_ifm.write(input_ifm);
				}
			}
			ifm_addr+=ifm_length;
		}
		weights_addr=9216*toM;//
		for (int ti=0; ti<tb_N; ti+=Tn){

			weights_length=9216;//4*Tm*min(Tn,N-Tn*ti)*tb_custom_k*tb_custom_k
								//=64*16*3*3
			for(int i=0;i<weights_length;i+=4){
				input_weights.data.data1 = weights[i+weights_addr];
				input_weights.data.data2 = weights[i+weights_addr+1];
				input_weights.data.data3 = weights[i+weights_addr+2];
				input_weights.data.data4 = weights[i+weights_addr+3];
				tb_input_dma_weights.write(input_weights);
			}
			weights_addr+=36864;//Tm*N'(M)*tb_custom_k*tb_custom_k=16*256*3*3;
		}
	}



		cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	cout<<tb_flag<<endl;

	for (int toM=0;toM<4;toM++){
		output_length=50176;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=64*28*28;
		output_addr=toM*50176;
		for(int b=0;b<bs;b++){
			for(int i=0;i<output_length;i+=4){
				output =tb_output_dma_output.read();
				out[i+output_addr] = output.data.data1;
				out[i+output_addr+1] = output.data.data2;
				out[i+output_addr+2] = output.data.data3;
				out[i+output_addr+3] = output.data.data4;
			}
			output_addr+=200704;//M*R*C=256*28*28
		}
	}

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
					cout <<out[bf_index]<< ";";

				}
			}
		}

		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
}

void convwu_256_512_28( int outnum,
	FPGA_DATA ifm[bs*256*28*28],
	FPGA_DATA weights[256*512*3*3],
	FPGA_DATA ofm[bs*512*28*28],
	FPGA_DATA weights_print[512][256][3][3]

){
	//update conv8 weights N,M,R,C=256,512,28,28, stride=1,padding=1,kernel=3
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

	ifm_length=200704;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
					//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=256*28*28;
	for (int toM=0;toM<4;toM++){
		ifm_addr=0;
		ofm_length=100352;//tb_M*tb_R*tb_C=128*28*28;
		ofm_addr=toM*100352;
		for(int b=0;b<bs;b++){
			for(int i=0;i<ofm_length;i+=4){
				input_ofm.data.data1 = ofm[i+ofm_addr];
				input_ofm.data.data2 = ofm[i+ofm_addr+1];
				input_ofm.data.data3 = ofm[i+ofm_addr+2];
				input_ofm.data.data4 = ofm[i+ofm_addr+3];
				tb_input_dma_ofm.write(input_ofm);
			}
			for (int to=0;to<tb_M;to+=Tm){//repeate[M/Tm]
				for(int i=0;i<ifm_length;i+=4){
					input_ifm.data.data1 = ifm[i+ifm_addr];
					input_ifm.data.data2 = ifm[i+ifm_addr+1];
					input_ifm.data.data3 = ifm[i+ifm_addr+2];
					input_ifm.data.data4 = ifm[i+ifm_addr+3];
					tb_input_dma_ifm.write(input_ifm);
				}
			}
			ifm_addr+=ifm_length;
			ofm_addr+=401408;//tb_M_in*tb_R*tb_C=512*28*28;
		}
	}


	weights_addr=0;

	weights_length=1179648;//M*N*tb_custom_k*tb_custom_k
			//=256*512*3*3

	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = weights[i+weights_addr];
		input_weights.data.data2 = weights[i+weights_addr+1];
		input_weights.data.data3 = weights[i+weights_addr+2];
		input_weights.data.data4 = weights[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}




		cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	cout<<tb_flag<<endl;

	//print output
	output_addr=0;
	output_length=1179648;//M*N*tb_custom_k*tb_custom_k
	//=256*512*3*3
	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		weights[i+output_addr] = output.data.data1;
		weights[i+output_addr+1] = output.data.data2;
		weights[i+output_addr+2] = output.data.data3;
		weights[i+output_addr+3] = output.data.data4;
	}



	cout<<"after loopM:"<<endl;

	cout<<tb_flag<<endl;
	cout<<"conv"<<outnum<<endl;
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
					cout<<weights_print[m][n][r][c] << ";";
				}
			}

		}
	}
	cout<<"1st 3OCs last 3ICs" <<endl;
	for (int m=0; m<3;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c] << ";";
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
					cout<<weights_print[m][n][r][c] << ";";
				}
			}

		}
	}
	cout<<"last 3OCs last 3ICs" <<endl;
	for (int m=tb_M_in-3; m<tb_M_in;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c] << ";";
				}
			}

		}
	}
}

void maxpoolbp_256_28( int outnum,
	FPGA_DATA ifm[bs*256*28*28],
	FPGA_DATA weights[bs*28*28*32],
	FPGA_DATA ofm[bs*256*56*56],
	FPGA_DATA out[bs*256*56*56]

){
	//backward pool+relu N,M,R,C=256,256,28,28, stride=2,padding=0,kernel=2
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


	//FPGA_DATA loss5[bs*10*10*16];//[r][c][m]
	ifm_addr=0;
	ifm_length=bs*200704;//M*tb_custom_Tr*tb_custom_Tc=256*28*28;
	//for(int b=0;b<bs;b++){
		for(int i=0;i<ifm_length;i+=4){
			input_ifm.data.data1 = ifm[i+ifm_addr];
			input_ifm.data.data2 = ifm[i+ifm_addr+1];
			input_ifm.data.data3 = ifm[i+ifm_addr+2];
			input_ifm.data.data4 = ifm[i+ifm_addr+3];
			tb_input_dma_ifm.write(input_ifm);
			//cout<<input_ifm.data<<endl;
		}
		//ifm_addr+=ifm_length;
	//}

	weights_addr=0;
	weights_length=bs*25088;//r*c*[m/8]=28*28*32
	for(int i=0;i<weights_length;i+=4){
		pool_index.data.data1 =weights[i+weights_addr];
		pool_index.data.data2 =weights[i+weights_addr+1];
		pool_index.data.data3 =weights[i+weights_addr+2];
		pool_index.data.data4 =weights[i+weights_addr+3];
		tb_input_dma_weights.write(pool_index);

	}

	ofm_addr=0;
	ofm_length=bs*802816;//M*R_in*C_in=256*56*56;
	for(int i=0;i<ofm_length;i+=4){
		input_ofm.data.data1 = ofm[i+ofm_addr];
		input_ofm.data.data2 = ofm[i+ofm_addr+1];
		input_ofm.data.data3 = ofm[i+ofm_addr+2];
		input_ofm.data.data4 = ofm[i+ofm_addr+3];
		tb_input_dma_ofm.write(input_ofm);
	}


	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	cout<<tb_flag<<endl;

//print output

	output_addr=0;
	output_length=bs*802816;//M*R_in*C_in=256*56*56;
	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		out[i+output_addr] = output.data.data1;
		out[i+output_addr+1] = output.data.data2;
		out[i+output_addr+2] = output.data.data3;
		out[i+output_addr+3] = output.data.data4;
		//cout << loss3[i+output_addr] << ";";
	}




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
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
}

void convbp_256_256_56( int outnum,
	FPGA_DATA ifm[bs*256*56*56],
	FPGA_DATA weights[256*256*3*3],
	FPGA_DATA outhat[bs*256*56*56],
	FPGA_DATA gama[256],
	FPGA_DATA beta[256],
	FPGA_DATA namuda[256],
	FPGA_DATA ofm[bs*256*56*56],
	FPGA_DATA out[bs*256*56*56]

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
	ifm_addr=0;
	ofm_addr=0;
	weights_addr=0;
	ifm_length=tb_M;//M
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = gama[i+weights_addr];
		input_weights.data.data2 = gama[i+weights_addr+1];
		input_weights.data.data3 = gama[i+weights_addr+2];
		input_weights.data.data4 = gama[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}
	for(int i=0;i<ofm_length;i+=4){
		input_ofm.data.data1 = beta[i+ofm_addr];
		input_ofm.data.data2 = beta[i+ofm_addr+1];
		input_ofm.data.data3 = beta[i+ofm_addr+2];
		input_ofm.data.data4 = beta[i+ofm_addr+3];
		tb_input_dma_ofm.write(input_ofm);
	}
	for(int i=0;i<ifm_length;i+=4){
		input_ifm.data.data1 = namuda[i+ifm_addr];
		input_ifm.data.data2 = namuda[i+ifm_addr+1];
		input_ifm.data.data3 = namuda[i+ifm_addr+2];
		input_ifm.data.data4 = namuda[i+ifm_addr+3];
		tb_input_dma_ifm.write(input_ifm);
	}
	//ifm_addr=0;
	//ofm_addr=0;
	ofm_length=50176;//Tm*R*C
	ifm_length=50176;//Tm*R*C
	for (int to=0;to<16;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=802816*b+50176*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			ofm_addr=802816*b+50176*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			for(int i=0;i<ifm_length;i+=4){
				input_ifm.data.data1 = ifm[i+ifm_addr];
				input_ifm.data.data2 = ifm[i+ifm_addr+1];
				input_ifm.data.data3 = ifm[i+ifm_addr+2];
				input_ifm.data.data4 = ifm[i+ifm_addr+3];
				tb_input_dma_ifm.write(input_ifm);
			}
			for(int i=0;i<ofm_length;i+=4){
				input_ofm.data.data1 = outhat[i+ofm_addr];
				input_ofm.data.data2 = outhat[i+ofm_addr+1];
				input_ofm.data.data3 = outhat[i+ofm_addr+2];
				input_ofm.data.data4 = outhat[i+ofm_addr+3];
				tb_input_dma_ofm.write(input_ofm);
			}
		}
	}//=6*3*5*5
	ifm_addr=0;
	ofm_addr=0;
	ofm_length=802816;//M*R*C
	ifm_length=802816;//M*R*C
	for(int b=0;b<bs;b++){
		//ifm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		//ofm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		for(int i=0;i<ifm_length;i+=4){
			input_ifm.data.data1 = ifm[i+ifm_addr];
			input_ifm.data.data2 = ifm[i+ifm_addr+1];
			input_ifm.data.data3 = ifm[i+ifm_addr+2];
			input_ifm.data.data4 = ifm[i+ifm_addr+3];
			tb_input_dma_ifm.write(input_ifm);
		}
		for(int i=0;i<ofm_length;i+=4){
			input_ofm.data.data1 = outhat[i+ofm_addr];
			input_ofm.data.data2 = outhat[i+ofm_addr+1];
			input_ofm.data.data3 = outhat[i+ofm_addr+2];
			input_ofm.data.data4 = outhat[i+ofm_addr+3];
			tb_input_dma_ofm.write(input_ofm);
		}
		ifm_addr+=ifm_length;
		ofm_addr+=ofm_length;
	}

	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	//cout<<tb_flag<<endl;

	//print output
	output_addr=0;
	output_length=802816;//M*R*C
	for(int b=0;b<bs;b++){
		for(int i=0;i<output_length;i+=4){
			output =tb_output_dma_output.read();
			ifm[i+output_addr] = output.data.data1;
			ifm[i+output_addr+1] = output.data.data2;
			ifm[i+output_addr+2] = output.data.data3;
			ifm[i+output_addr+3] = output.data.data4;
		}
		output_addr+=output_length;
	}
	output_addr=0;
	output_length=tb_M;//M
	for(int i=0;i<output_length;i+=4){
		out_weights =tb_output_dma_Weightsout.read();
		gama[i+output_addr] = out_weights.data.data1;
		gama[i+output_addr+1] = out_weights.data.data2;
		gama[i+output_addr+2] = out_weights.data.data3;
		gama[i+output_addr+3] = out_weights.data.data4;
	}
	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		beta[i+output_addr] = output.data.data1;
		beta[i+output_addr+1] = output.data.data2;
		beta[i+output_addr+2] = output.data.data3;
		beta[i+output_addr+3] = output.data.data4;
	}

	cout<<"gama first:"<<endl;
	for(int i=0;i<10;i++ ){
		cout<<gama[i]<< ";";
	}
	cout<<"gama last:"<<endl;
	for(int i=tb_M-10;i<tb_M;i++ ){
		cout<<gama[i]<< ";";
	}

	cout<<"beta first:"<<endl;
	cout<<beta[0]<< ";";
	cout<<beta[1]<< ";";
	cout<<beta[2]<< ";";
	cout<<"beta last:"<<endl;
	cout<<beta[tb_M-3]<< ";";
	cout<<beta[tb_M-2]<< ";";
	cout<<beta[tb_M-1]<<endl;


	//backward conv6,7+relu N,M,R,C=256,256,56,56, stride=1,padding=1,kernel=3
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

	ifm_length=25984;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
					//*((C-1)*tb_custom_stride+tb_custom_k)=16*29*56;
	ofm_length=401408;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=128*56*56;
	for (int toM=0;toM<2;toM++){
		ofm_addr=toM*401408;
		for(int i=0;i<ofm_length;i+=4){
			input_ofm.data.data1 = ofm[i+ofm_addr];
			input_ofm.data.data2 = ofm[i+ofm_addr+1];
			input_ofm.data.data3 = ofm[i+ofm_addr+2];
			input_ofm.data.data4 = ofm[i+ofm_addr+3];
			tb_input_dma_ofm.write(input_ofm);
		}
		ofm_addr+=802816;
		for (int to=0;to<tb_M;to+=Tm){//repeate[M/Tm]
			//ifm_length=25984;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
									//*((C-1)*tb_custom_stride+tb_custom_k)=16*29*56;
			ifm_addr=0;
			weights_addr=18432*toM;
			for(int ti=0;ti<tb_N;ti+=Tn){
				for(int i=0;i<ifm_length;i+=4){
					input_ifm.data.data1 = ifm[i+ifm_addr];
					input_ifm.data.data2 = ifm[i+ifm_addr+1];
					input_ifm.data.data3 = ifm[i+ifm_addr+2];
					input_ifm.data.data4 = ifm[i+ifm_addr+3];
					tb_input_dma_ifm.write(input_ifm);
				}
				ifm_addr+=50176;//Tn*R_in*C_in=16*56*56
				if(to==0){
					weights_length=18432;//8*Tm*min(Tn,N-Tn*ti)*tb_custom_k*tb_custom_k
										//=128*16*3*3
					for(int i=0;i<weights_length;i+=4){
						input_weights.data.data1 = weights[i+weights_addr];
						input_weights.data.data2 = weights[i+weights_addr+1];
						input_weights.data.data3 = weights[i+weights_addr+2];
						input_weights.data.data4 = weights[i+weights_addr+3];
						tb_input_dma_weights.write(input_weights);
					}
					weights_addr+=36864;//Tm*N'(M)*tb_custom_k*tb_custom_k=16*256*3*3;
				}
			}
//			for(int row=tb_custom_Tr;row<tb_R-tb_custom_Tr;row+=tb_custom_Tr){
//				ifm_length=3456;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
//														//*((C-1)*tb_custom_stride+tb_custom_k)=16*6*56;
//				ifm_addr=(row-2)*432;//+3632*row;//b*N*C_in*R_in+C_in*(row-padding)*stride*min(Tn,N)=b*96*27*27+(row-2)*1*27*16
//				for(int ti=0;ti<tb_N;ti+=Tn){
//					for(int i=0;i<ifm_length;i+=4){
			//			input_ifm.data.data1 = ifm[i+ifm_addr];
			//			input_ifm.data.data2 = ifm[i+ifm_addr+1];
			//			input_ifm.data.data3 = ifm[i+ifm_addr+2];
			//			input_ifm.data.data4 = ifm[i+ifm_addr+3];
			//			tb_input_dma_ifm.write(input_ifm);
			//		}
//					ifm_addr+=11664;//Tn*R_in*C_in=16*27*27
//				}
//			}
			//ifm_length=25984;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
												//*((C-1)*tb_custom_stride+tb_custom_k)=16*29*56;
			ifm_addr=24192;//+896*(row-1);//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*256*56*56+row*1*56*16
			for(int ti=0;ti<tb_N;ti+=Tn){
				for(int i=0;i<ifm_length;i+=4){
					input_ifm.data.data1 = ifm[i+ifm_addr];
					input_ifm.data.data2 = ifm[i+ifm_addr+1];
					input_ifm.data.data3 = ifm[i+ifm_addr+2];
					input_ifm.data.data4 = ifm[i+ifm_addr+3];
					tb_input_dma_ifm.write(input_ifm);
				}
				ifm_addr+=50176;//Tn*R_in*C_in=16*56*56
			}
		}
		for(int b=1;b<bs;b++){
			for(int i=0;i<ofm_length;i+=4){
				input_ofm.data.data1 = ofm[i+ofm_addr];
				input_ofm.data.data2 = ofm[i+ofm_addr+1];
				input_ofm.data.data3 = ofm[i+ofm_addr+2];
				input_ofm.data.data4 = ofm[i+ofm_addr+3];
				tb_input_dma_ofm.write(input_ofm);
			}
			ofm_addr+=802816;
			for (int to=0;to<tb_M;to+=Tm){//repeate[M/Tm]
				//ifm_length=25984;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
													//*((C-1)*tb_custom_stride+tb_custom_k)=16*29*56;
				ifm_addr=802816*b;//+3632*row;//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*256*56*56+row*1*56*16
				for(int ti=0;ti<tb_N;ti+=Tn){
					for(int i=0;i<ifm_length;i+=4){
						input_ifm.data.data1 = ifm[i+ifm_addr];
						input_ifm.data.data2 = ifm[i+ifm_addr+1];
						input_ifm.data.data3 = ifm[i+ifm_addr+2];
						input_ifm.data.data4 = ifm[i+ifm_addr+3];
						tb_input_dma_ifm.write(input_ifm);
					}
					ifm_addr+=50176;//Tn*R_in*C_in=16*56*56
				}
//				for(int row=tb_custom_Tr;row<tb_R-tb_custom_Tr;row+=tb_custom_Tr){
//					ifm_length=3456;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
//															//*((C-1)*tb_custom_stride+tb_custom_k)=16*8*27;
//					ifm_addr=186624*b+(row-2)*432;//+3632*row;//b*N*C_in*R_in+C_in*(row-padding)*stride*min(Tn,N)=b*256*27*27+(row-2)*1*27*16
//					for(int ti=0;ti<tb_N;ti+=Tn){
//						for(int i=0;i<ifm_length;i+=4){
//							input_ifm.data.data1 = ifm[i+ifm_addr];
//							input_ifm.data.data2 = ifm[i+ifm_addr+1];
//							input_ifm.data.data3 = ifm[i+ifm_addr+2];
//							input_ifm.data.data4 = ifm[i+ifm_addr+3];
//							tb_input_dma_ifm.write(input_ifm);
//						}
//						ifm_addr+=11664;//Tn*R_in*C_in=16*27*27
//					}
//				}
				//ifm_length=25984;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
													//*((C-1)*tb_custom_stride+tb_custom_k)=16*29*56;
				ifm_addr=802816*b+24192;//+896*(row-1);//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*256*56*56+row*1*56*16
				for(int ti=0;ti<tb_N;ti+=Tn){
					for(int i=0;i<ifm_length;i+=4){
						input_ifm.data.data1 = ifm[i+ifm_addr];
						input_ifm.data.data2 = ifm[i+ifm_addr+1];
						input_ifm.data.data3 = ifm[i+ifm_addr+2];
						input_ifm.data.data4 = ifm[i+ifm_addr+3];
						tb_input_dma_ifm.write(input_ifm);
					}
					ifm_addr+=50176;//Tn*R_in*C_in=16*56*56
				}
			}
		}
	}

		cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	cout<<tb_flag<<endl;

	for (int toM=0;toM<2;toM++){
		output_length=401408;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=128*56*56;
		output_addr=toM*401408;
		for(int b=0;b<bs;b++){
			for(int i=0;i<output_length;i+=4){
				output =tb_output_dma_output.read();
				out[i+output_addr] = output.data.data1;
				out[i+output_addr+1] = output.data.data2;
				out[i+output_addr+2] = output.data.data3;
				out[i+output_addr+3] = output.data.data4;
			}
			output_addr+=802816;
		}
	}

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
					cout <<out[bf_index]<< ";";

				}
			}
		}

		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
}

void convwu_256_256_56( int outnum,
	FPGA_DATA ifm[bs*256*56*56],
	FPGA_DATA weights[256*256*3*3],
	FPGA_DATA ofm[bs*256*56*56],
	FPGA_DATA weights_print[256][256][3][3]

){
	//update conv6,7 weights N,M,R,C=256,256,56,56, stride=1,padding=1,kernel=3
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


	for (int toM=0;toM<2;toM++){
		ofm_length=50176;// Tm*R*C=16*56*56
		for(int b=0;b<bs;b++){
			for (int to=0;to<8;to++){//repeate[M/Tm]
				ofm_addr=b*802816+toM*401408+to*50176;//M_in*R*C*b+toM*M*R*C+ti*Tm*R*C=256*56*56*b+toM*128*56*56+to*16*56*56
				for(int ti=0;ti<16;ti++){
					for(int i=0;i<ofm_length;i+=4){
						input_ofm.data.data1 = ofm[i+ofm_addr];
						input_ofm.data.data2 = ofm[i+ofm_addr+1];
						input_ofm.data.data3 = ofm[i+ofm_addr+2];
						input_ofm.data.data4 = ofm[i+ofm_addr+3];
						tb_input_dma_ofm.write(input_ofm);
					}
					ifm_length=4480;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
						//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=16*56*5;
					ifm_addr=b*802816+ti*50176;//b*R*N*C+ti*Tn*R*C
					for(int i=0;i<ifm_length;i+=4){
						input_ifm.data.data1 = ifm[i+ifm_addr];
						input_ifm.data.data2 = ifm[i+ifm_addr+1];
						input_ifm.data.data3 = ifm[i+ifm_addr+2];
						input_ifm.data.data4 = ifm[i+ifm_addr+3];
						tb_input_dma_ifm.write(input_ifm);
					}
					ifm_length=5376;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
									//16*56*6
					for(int row=tb_custom_Tr;row<tb_R-tb_custom_Tr;row+=tb_custom_Tr){
						ifm_addr=b*802816+ti*50176+(row-1)*896;//+3632*row;//b*R*N*C+ti*Tn*R*C+C_in*row*stride*min(Tn,N)+(row-1)*1*16*56
						for(int i=0;i<ifm_length;i+=4){
							input_ifm.data.data1 = ifm[i+ifm_addr];
							input_ifm.data.data2 = ifm[i+ifm_addr+1];
							input_ifm.data.data3 = ifm[i+ifm_addr+2];
							input_ifm.data.data4 = ifm[i+ifm_addr+3];
							tb_input_dma_ifm.write(input_ifm);
						}
					}
					ifm_length=4480;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
									//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=16*56*5;
					ifm_addr=b*802816+ti*50176+45696;//(row-1)*896;//+3632*row;//b*R*N*C+ti*Tn*R*C+C_in*row*stride*min(Tn,N)+(row-1)*1*16*56
					for(int i=0;i<ifm_length;i+=4){
						input_ifm.data.data1 = ifm[i+ifm_addr];
						input_ifm.data.data2 = ifm[i+ifm_addr+1];
						input_ifm.data.data3 = ifm[i+ifm_addr+2];
						input_ifm.data.data4 = ifm[i+ifm_addr+3];
						tb_input_dma_ifm.write(input_ifm);
					}
				}
			}
		}
	}


	weights_addr=0;

	weights_length=589824;//M*N*tb_custom_k*tb_custom_k
			//=256*256*3*3

	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = weights[i+weights_addr];
		input_weights.data.data2 = weights[i+weights_addr+1];
		input_weights.data.data3 = weights[i+weights_addr+2];
		input_weights.data.data4 = weights[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}




		cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	cout<<tb_flag<<endl;

	//print output
	output_addr=0;
	output_length=589824;//M*N*tb_custom_k*tb_custom_k
	//=256*256*3*3
	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		weights[i+output_addr] = output.data.data1;
		weights[i+output_addr+1] = output.data.data2;
		weights[i+output_addr+2] = output.data.data3;
		weights[i+output_addr+3] = output.data.data4;
	}



	cout<<"after loopM:"<<endl;

	cout<<tb_flag<<endl;
	cout<<"conv"<<outnum<<endl;
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
					cout<<weights_print[m][n][r][c] << ";";
				}
			}

		}
	}
	cout<<"1st 3OCs last 3ICs" <<endl;
	for (int m=0; m<3;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c] << ";";
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
					cout<<weights_print[m][n][r][c] << ";";
				}
			}

		}
	}
	cout<<"last 3OCs last 3ICs" <<endl;
	for (int m=tb_M_in-3; m<tb_M_in;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c] << ";";
				}
			}

		}
	}
}

void convbp_128_256_56( int outnum,
	FPGA_DATA ifm[bs*256*56*56],
	FPGA_DATA weights[128*256*3*3],
	FPGA_DATA outhat[bs*256*56*56],
	FPGA_DATA gama[256],
	FPGA_DATA beta[256],
	FPGA_DATA namuda[256],
	FPGA_DATA out[bs*128*56*56]

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


	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ifm_addr=0;
	ofm_addr=0;
	weights_addr=0;
	ifm_length=tb_M;//M
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = gama[i+weights_addr];
		input_weights.data.data2 = gama[i+weights_addr+1];
		input_weights.data.data3 = gama[i+weights_addr+2];
		input_weights.data.data4 = gama[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}
	for(int i=0;i<ofm_length;i+=4){
		input_ofm.data.data1 = beta[i+ofm_addr];
		input_ofm.data.data2 = beta[i+ofm_addr+1];
		input_ofm.data.data3 = beta[i+ofm_addr+2];
		input_ofm.data.data4 = beta[i+ofm_addr+3];
		tb_input_dma_ofm.write(input_ofm);
	}
	for(int i=0;i<ifm_length;i+=4){
		input_ifm.data.data1 = namuda[i+ifm_addr];
		input_ifm.data.data2 = namuda[i+ifm_addr+1];
		input_ifm.data.data3 = namuda[i+ifm_addr+2];
		input_ifm.data.data4 = namuda[i+ifm_addr+3];
		tb_input_dma_ifm.write(input_ifm);
	}
	//ifm_addr=0;
	//ofm_addr=0;
	ofm_length=50176;//Tm*R*C
	ifm_length=50176;//Tm*R*C
	for (int to=0;to<16;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=802816*b+50176*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			ofm_addr=802816*b+50176*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			for(int i=0;i<ifm_length;i+=4){
				input_ifm.data.data1 = ifm[i+ifm_addr];
				input_ifm.data.data2 = ifm[i+ifm_addr+1];
				input_ifm.data.data3 = ifm[i+ifm_addr+2];
				input_ifm.data.data4 = ifm[i+ifm_addr+3];
				tb_input_dma_ifm.write(input_ifm);
			}
			for(int i=0;i<ofm_length;i+=4){
				input_ofm.data.data1 = outhat[i+ofm_addr];
				input_ofm.data.data2 = outhat[i+ofm_addr+1];
				input_ofm.data.data3 = outhat[i+ofm_addr+2];
				input_ofm.data.data4 = outhat[i+ofm_addr+3];
				tb_input_dma_ofm.write(input_ofm);
			}
		}
	}//=6*3*5*5
	ifm_addr=0;
	ofm_addr=0;
	ofm_length=802816;//M*R*C
	ifm_length=802816;//M*R*C
	for(int b=0;b<bs;b++){
		//ifm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		//ofm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		for(int i=0;i<ifm_length;i+=4){
			input_ifm.data.data1 = ifm[i+ifm_addr];
			input_ifm.data.data2 = ifm[i+ifm_addr+1];
			input_ifm.data.data3 = ifm[i+ifm_addr+2];
			input_ifm.data.data4 = ifm[i+ifm_addr+3];
			tb_input_dma_ifm.write(input_ifm);
		}
		for(int i=0;i<ofm_length;i+=4){
			input_ofm.data.data1 = outhat[i+ofm_addr];
			input_ofm.data.data2 = outhat[i+ofm_addr+1];
			input_ofm.data.data3 = outhat[i+ofm_addr+2];
			input_ofm.data.data4 = outhat[i+ofm_addr+3];
			tb_input_dma_ofm.write(input_ofm);
		}
		ifm_addr+=ifm_length;
		ofm_addr+=ofm_length;
	}

	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	//cout<<tb_flag<<endl;

	//print output
	output_addr=0;
	output_length=802816;//M*R*C
	for(int b=0;b<bs;b++){
		for(int i=0;i<output_length;i+=4){
			output =tb_output_dma_output.read();
			ifm[i+output_addr] = output.data.data1;
			ifm[i+output_addr+1] = output.data.data2;
			ifm[i+output_addr+2] = output.data.data3;
			ifm[i+output_addr+3] = output.data.data4;
		}
		output_addr+=output_length;
	}
	output_addr=0;
	output_length=tb_M;//M
	for(int i=0;i<output_length;i+=4){
		out_weights =tb_output_dma_Weightsout.read();
		gama[i+output_addr] = out_weights.data.data1;
		gama[i+output_addr+1] = out_weights.data.data2;
		gama[i+output_addr+2] = out_weights.data.data3;
		gama[i+output_addr+3] = out_weights.data.data4;
	}
	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		beta[i+output_addr] = output.data.data1;
		beta[i+output_addr+1] = output.data.data2;
		beta[i+output_addr+2] = output.data.data3;
		beta[i+output_addr+3] = output.data.data4;
	}

	cout<<"gama first:"<<endl;
	for(int i=0;i<10;i++ ){
		cout<<gama[i]<< ";";
	}
	cout<<"gama last:"<<endl;
	for(int i=tb_M-10;i<tb_M;i++ ){
		cout<<gama[i]<< ";";
	}

	cout<<"beta first:"<<endl;
	cout<<beta[0]<< ";";
	cout<<beta[1]<< ";";
	cout<<beta[2]<< ";";
	cout<<"beta last:"<<endl;
	cout<<beta[tb_M-3]<< ";";
	cout<<beta[tb_M-2]<< ";";
	cout<<beta[tb_M-1]<<endl;


	//backward conv5 N,M,R,C=256,128,56,56, stride=1,padding=1,kernel=3
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


	ifm_addr=0;
	weights_addr=0;
	ifm_length=25984;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
				//*((C-1)*tb_custom_stride+tb_custom_k)=16*29*56;
	weights_length=294912;//M*N*tb_custom_k*tb_custom_k=`128*256*3*3
	for(int b=0;b<bs;b++){
		for (int to=0;to<tb_M;to+=Tm){//repeate [M/Tm]
			//for(int row=0;row<tb_R;row+=tb_custom_Tr){
			ifm_addr=802816*b;//+256*row;//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*256*56*56+row*1*56*16
			for(int ti=0;ti<tb_N;ti+=Tn){
				//ifm_length=448;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
										//*((C-1)*tb_custom_stride+tb_custom_k)=1*16*28;
				for(int i=0;i<ifm_length;i+=4){
					input_ifm.data.data1 = ifm[i+ifm_addr];
					input_ifm.data.data2 = ifm[i+ifm_addr+1];
					input_ifm.data.data3 = ifm[i+ifm_addr+2];
					input_ifm.data.data4 = ifm[i+ifm_addr+3];
					tb_input_dma_ifm.write(input_ifm);
				}
				ifm_addr+=50176;//Tn*R_in*C_in=16*56*56
			}
			ifm_addr=802816*b+24192;//+896*(row-1);//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*256*56*56+row*1*56*16
			for(int ti=0;ti<tb_N;ti+=Tn){
				//ifm_length=448;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
										//*((C-1)*tb_custom_stride+tb_custom_k)=1*16*28;
				for(int i=0;i<ifm_length;i+=4){
					input_ifm.data.data1 = ifm[i+ifm_addr];
					input_ifm.data.data2 = ifm[i+ifm_addr+1];
					input_ifm.data.data3 = ifm[i+ifm_addr+2];
					input_ifm.data.data4 = ifm[i+ifm_addr+3];
					tb_input_dma_ifm.write(input_ifm);
				}
				ifm_addr+=50176;//Tn*R_in*C_in=16*56*56
			}
		}
	}

	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = weights[i+weights_addr];
		input_weights.data.data2 = weights[i+weights_addr+1];
		input_weights.data.data3 = weights[i+weights_addr+2];
		input_weights.data.data4 = weights[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}

		cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	cout<<tb_flag<<endl;
	output_addr=0;
	//for (int toM=0;toM<2;toM++){
		output_length=bs*401408;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=128*56*56;
		//output_addr=toM*401408;
		//for(int b=0;b<bs;b++){
			for(int i=0;i<output_length;i+=4){
				output =tb_output_dma_output.read();
				out[i+output_addr] = output.data.data1;
				out[i+output_addr+1] = output.data.data2;
				out[i+output_addr+2] = output.data.data3;
				out[i+output_addr+3] = output.data.data4;
			}
			//output_addr+=802816;
		//}
	//}

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
					cout <<out[bf_index]<< ";";

				}
			}
		}

		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
}

void convwu_128_256_56( int outnum,
	FPGA_DATA ifm[bs*128*56*56],
	FPGA_DATA weights[128*256*3*3],
	FPGA_DATA ofm[bs*256*56*56],
	FPGA_DATA weights_print[256][128][3][3]

){
	//update conv5 weights N,M,R,C=128,256,56,56, stride=1,padding=1,kernel=3
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


	//for (int toM=0;toM<2;toM++){
		ofm_length=50176;// Tm*R*C=16*56*56
		for(int b=0;b<bs;b++){
			for (int to=0;to<16;to++){//repeate[M/Tm]
				ofm_addr=b*802816+to*50176;//M_in*R*C*b+toM*M*R*C+to*Tm*R*C=256*56*56*b+to*16*56*56
				for(int ti=0;ti<8;ti++){
					for(int i=0;i<ofm_length;i+=4){
						input_ofm.data.data1 = ofm[i+ofm_addr];
						input_ofm.data.data2 = ofm[i+ofm_addr+1];
						input_ofm.data.data3 = ofm[i+ofm_addr+2];
						input_ofm.data.data4 = ofm[i+ofm_addr+3];
						tb_input_dma_ofm.write(input_ofm);
					}
					ifm_length=4480;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
						//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=16*56*5;
					ifm_addr=b*401408+ti*50176;//b*R*N*C+ti*Tn*R*C
					for(int i=0;i<ifm_length;i+=4){
						input_ifm.data.data1 = ifm[i+ifm_addr];
						input_ifm.data.data2 = ifm[i+ifm_addr+1];
						input_ifm.data.data3 = ifm[i+ifm_addr+2];
						input_ifm.data.data4 = ifm[i+ifm_addr+3];
						tb_input_dma_ifm.write(input_ifm);
					}
					ifm_length=5376;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
									//16*56*6
					for(int row=tb_custom_Tr;row<tb_R-tb_custom_Tr;row+=tb_custom_Tr){
						ifm_addr=b*401408+ti*50176+(row-1)*896;//+3632*row;//b*R*N*C+ti*Tn*R*C+C_in*row*stride*min(Tn,N)+(row-1)*1*16*56
						for(int i=0;i<ifm_length;i+=4){
							input_ifm.data.data1 = ifm[i+ifm_addr];
							input_ifm.data.data2 = ifm[i+ifm_addr+1];
							input_ifm.data.data3 = ifm[i+ifm_addr+2];
							input_ifm.data.data4 = ifm[i+ifm_addr+3];
							tb_input_dma_ifm.write(input_ifm);
						}
					}
					ifm_length=4480;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
									//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=16*56*5;
					ifm_addr=b*401408+ti*50176+45696;//(row-1)*896;//+3632*row;//b*R*N*C+ti*Tn*R*C+C_in*row*stride*min(Tn,N)+(row-1)*1*16*56
					for(int i=0;i<ifm_length;i+=4){
						input_ifm.data.data1 = ifm[i+ifm_addr];
						input_ifm.data.data2 = ifm[i+ifm_addr+1];
						input_ifm.data.data3 = ifm[i+ifm_addr+2];
						input_ifm.data.data4 = ifm[i+ifm_addr+3];
						tb_input_dma_ifm.write(input_ifm);
					}
				}
			}
		}
	//}


	weights_addr=0;

	weights_length=294912;//M*N*tb_custom_k*tb_custom_k
			//=128*256*3*3

	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = weights[i+weights_addr];
		input_weights.data.data2 = weights[i+weights_addr+1];
		input_weights.data.data3 = weights[i+weights_addr+2];
		input_weights.data.data4 = weights[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}




		cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	cout<<tb_flag<<endl;

	//print output
	output_addr=0;
	output_length=294912;//M*N*tb_custom_k*tb_custom_k
				//=128*256*3*3
	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		weights[i+output_addr] = output.data.data1;
		weights[i+output_addr+1] = output.data.data2;
		weights[i+output_addr+2] = output.data.data3;
		weights[i+output_addr+3] = output.data.data4;
	}



	cout<<"after loopM:"<<endl;

	cout<<tb_flag<<endl;
	cout<<"conv"<<outnum<<endl;
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
					cout<<weights_print[m][n][r][c] << ";";
				}
			}

		}
	}
	cout<<"1st 3OCs last 3ICs" <<endl;
	for (int m=0; m<3;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c] << ";";
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
					cout<<weights_print[m][n][r][c] << ";";
				}
			}

		}
	}
	cout<<"last 3OCs last 3ICs" <<endl;
	for (int m=tb_M_in-3; m<tb_M_in;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c] << ";";
				}
			}

		}
	}
}

void maxpoolbp_128_56( int outnum,
	FPGA_DATA ifm[bs*128*56*56],
	FPGA_DATA weights[bs*56*56*16],
	FPGA_DATA ofm[bs*128*112*112],
	FPGA_DATA out[bs*128*112*112]

){
	//backward pool+relu N,M,R,C=128,128,56,56, stride=2,padding=0,kernel=2
	tb_statec=0;
	tb_statep=5;//view=8+4
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


	//FPGA_DATA loss5[bs*10*10*16];//[r][c][m]
	ifm_addr=0;
	ifm_length=bs*401408;//M*tb_custom_Tr*tb_custom_Tc=128*56*56;
	//for(int b=0;b<bs;b++){
		for(int i=0;i<ifm_length;i+=4){
			input_ifm.data.data1 = ifm[i+ifm_addr];
			input_ifm.data.data2 = ifm[i+ifm_addr+1];
			input_ifm.data.data3 = ifm[i+ifm_addr+2];
			input_ifm.data.data4 = ifm[i+ifm_addr+3];
			tb_input_dma_ifm.write(input_ifm);
			//cout<<input_ifm.data<<endl;
		}
		//ifm_addr+=ifm_length;
	//}

	weights_addr=0;
	weights_length=bs*50176;//r*c*[m/8]=56*56*16
	for(int i=0;i<weights_length;i+=4){
		pool_index.data.data1 =weights[i+weights_addr];
		pool_index.data.data2 =weights[i+weights_addr+1];
		pool_index.data.data3 =weights[i+weights_addr+2];
		pool_index.data.data4 =weights[i+weights_addr+3];
		tb_input_dma_weights.write(pool_index);

	}

	ofm_addr=0;
	ofm_length=bs*1605632;//M*R_in*C_in=128*112*112;
	for(int i=0;i<ofm_length;i+=4){
		input_ofm.data.data1 = ofm[i+ofm_addr];
		input_ofm.data.data2 = ofm[i+ofm_addr+1];
		input_ofm.data.data3 = ofm[i+ofm_addr+2];
		input_ofm.data.data4 = ofm[i+ofm_addr+3];
		tb_input_dma_ofm.write(input_ofm);
	}


	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	cout<<tb_flag<<endl;

//print output

	output_addr=0;
	output_length=bs*1605632;//M*R_in*C_in=128*112*112;
	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		out[i+output_addr] = output.data.data1;
		out[i+output_addr+1] = output.data.data2;
		out[i+output_addr+2] = output.data.data3;
		out[i+output_addr+3] = output.data.data4;
		//cout << loss3[i+output_addr] << ";";
	}




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
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
}

void convbp_128_128_112( int outnum,
	FPGA_DATA ifm[bs*128*112*112],
	FPGA_DATA weights[128*128*3*3],
	FPGA_DATA outhat[bs*128*112*112],
	FPGA_DATA gama[128],
	FPGA_DATA beta[128],
	FPGA_DATA namuda[128],
	FPGA_DATA ofm[bs*128*112*112],
	FPGA_DATA out[bs*128*112*112]

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
	ifm_addr=0;
	ofm_addr=0;
	weights_addr=0;
	ifm_length=tb_M;//M
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = gama[i+weights_addr];
		input_weights.data.data2 = gama[i+weights_addr+1];
		input_weights.data.data3 = gama[i+weights_addr+2];
		input_weights.data.data4 = gama[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}
	for(int i=0;i<ofm_length;i+=4){
		input_ofm.data.data1 = beta[i+ofm_addr];
		input_ofm.data.data2 = beta[i+ofm_addr+1];
		input_ofm.data.data3 = beta[i+ofm_addr+2];
		input_ofm.data.data4 = beta[i+ofm_addr+3];
		tb_input_dma_ofm.write(input_ofm);
	}
	for(int i=0;i<ifm_length;i+=4){
		input_ifm.data.data1 = namuda[i+ifm_addr];
		input_ifm.data.data2 = namuda[i+ifm_addr+1];
		input_ifm.data.data3 = namuda[i+ifm_addr+2];
		input_ifm.data.data4 = namuda[i+ifm_addr+3];
		tb_input_dma_ifm.write(input_ifm);
	}
	//ifm_addr=0;
	//ofm_addr=0;
	ofm_length=200704;//Tm*R*C
	ifm_length=200704;//Tm*R*C
	for (int to=0;to<8;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=1605632*b+200704*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			ofm_addr=1605632*b+200704*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			for(int i=0;i<ifm_length;i+=4){
				input_ifm.data.data1 = ifm[i+ifm_addr];
				input_ifm.data.data2 = ifm[i+ifm_addr+1];
				input_ifm.data.data3 = ifm[i+ifm_addr+2];
				input_ifm.data.data4 = ifm[i+ifm_addr+3];
				tb_input_dma_ifm.write(input_ifm);
			}
			for(int i=0;i<ofm_length;i+=4){
				input_ofm.data.data1 = outhat[i+ofm_addr];
				input_ofm.data.data2 = outhat[i+ofm_addr+1];
				input_ofm.data.data3 = outhat[i+ofm_addr+2];
				input_ofm.data.data4 = outhat[i+ofm_addr+3];
				tb_input_dma_ofm.write(input_ofm);
			}
		}
	}//=6*3*5*5
	ifm_addr=0;
	ofm_addr=0;
	ofm_length=1605632;//M*R*C
	ifm_length=1605632;//M*R*C
	for(int b=0;b<bs;b++){
		//ifm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		//ofm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		for(int i=0;i<ifm_length;i+=4){
			input_ifm.data.data1 = ifm[i+ifm_addr];
			input_ifm.data.data2 = ifm[i+ifm_addr+1];
			input_ifm.data.data3 = ifm[i+ifm_addr+2];
			input_ifm.data.data4 = ifm[i+ifm_addr+3];
			tb_input_dma_ifm.write(input_ifm);
		}
		for(int i=0;i<ofm_length;i+=4){
			input_ofm.data.data1 = outhat[i+ofm_addr];
			input_ofm.data.data2 = outhat[i+ofm_addr+1];
			input_ofm.data.data3 = outhat[i+ofm_addr+2];
			input_ofm.data.data4 = outhat[i+ofm_addr+3];
			tb_input_dma_ofm.write(input_ofm);
		}
		ifm_addr+=ifm_length;
		ofm_addr+=ofm_length;
	}

	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	//cout<<tb_flag<<endl;

	//print output
	output_addr=0;
	output_length=1605632;//M*R*C
	for(int b=0;b<bs;b++){
		for(int i=0;i<output_length;i+=4){
			output =tb_output_dma_output.read();
			ifm[i+output_addr] = output.data.data1;
			ifm[i+output_addr+1] = output.data.data2;
			ifm[i+output_addr+2] = output.data.data3;
			ifm[i+output_addr+3] = output.data.data4;
		}
		output_addr+=output_length;
	}
	output_addr=0;
	output_length=tb_M;//M
	for(int i=0;i<output_length;i+=4){
		out_weights =tb_output_dma_Weightsout.read();
		gama[i+output_addr] = out_weights.data.data1;
		gama[i+output_addr+1] = out_weights.data.data2;
		gama[i+output_addr+2] = out_weights.data.data3;
		gama[i+output_addr+3] = out_weights.data.data4;
	}
	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		beta[i+output_addr] = output.data.data1;
		beta[i+output_addr+1] = output.data.data2;
		beta[i+output_addr+2] = output.data.data3;
		beta[i+output_addr+3] = output.data.data4;
	}

	cout<<"gama first:"<<endl;
	for(int i=0;i<10;i++ ){
		cout<<gama[i]<< ";";
	}
	cout<<"gama last:"<<endl;
	for(int i=tb_M-10;i<tb_M;i++ ){
		cout<<gama[i]<< ";";
	}

	cout<<"beta first:"<<endl;
	cout<<beta[0]<< ";";
	cout<<beta[1]<< ";";
	cout<<beta[2]<< ";";
	cout<<"beta last:"<<endl;
	cout<<beta[tb_M-3]<< ";";
	cout<<beta[tb_M-2]<< ";";
	cout<<beta[tb_M-1]<<endl;


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


	ifm_addr=0;
	weights_addr=0;
	//ifm_length=30464;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
				//*((C-1)*tb_custom_stride+tb_custom_k)=16*17*112;
	weights_length=147456;//M*N*tb_custom_k*tb_custom_k=128*128*3*3
	ofm_length=bs*1605632;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=128*112*112;
	ofm_addr=0;
	for(int i=0;i<ofm_length;i+=4){
		input_ofm.data.data1 = ofm[i+ofm_addr];
		input_ofm.data.data2 = ofm[i+ofm_addr+1];
		input_ofm.data.data3 = ofm[i+ofm_addr+2];
		input_ofm.data.data4 = ofm[i+ofm_addr+3];
		tb_input_dma_ofm.write(input_ofm);
	}

	for(int b=0;b<bs;b++){
		for (int to=0;to<tb_M;to+=Tm){//repeate [M/Tm]
			ifm_length=30464;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
							//*((C-1)*tb_custom_stride+tb_custom_k)=16*17*112;
			ifm_addr=1605632*b;//+1792*(row-1);//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*128*112*112+row*1*112*16
			for(int ti=0;ti<tb_N;ti+=Tn){
				for(int i=0;i<ifm_length;i+=4){
					input_ifm.data.data1 = ifm[i+ifm_addr];
					input_ifm.data.data2 = ifm[i+ifm_addr+1];
					input_ifm.data.data3 = ifm[i+ifm_addr+2];
					input_ifm.data.data4 = ifm[i+ifm_addr+3];
					tb_input_dma_ifm.write(input_ifm);
				}
				ifm_addr+=200704;//Tn*R_in*C_in=16*112*112
			}
			for(int row=tb_custom_Tr;row<tb_R-tb_custom_Tr;row+=tb_custom_Tr){
				ifm_length=32256;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
									//*((C-1)*tb_custom_stride+tb_custom_k)=16*18*112;
				ifm_addr=1605632*b+1792*(row-1);//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*128*112*112+row*1*112*16
				for(int ti=0;ti<tb_N;ti+=Tn){
					for(int i=0;i<ifm_length;i+=4){
						input_ifm.data.data1 = ifm[i+ifm_addr];
						input_ifm.data.data2 = ifm[i+ifm_addr+1];
						input_ifm.data.data3 = ifm[i+ifm_addr+2];
						input_ifm.data.data4 = ifm[i+ifm_addr+3];
						tb_input_dma_ifm.write(input_ifm);
					}
					ifm_addr+=200704;//Tn*R_in*C_in=16*112*112
				}
			}
			ifm_length=30464;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
							//*((C-1)*tb_custom_stride+tb_custom_k)=16*17*112;
			ifm_addr=1605632*b+170240;//+1792*(row-1);//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*128*112*112+row*1*112*16
			for(int ti=0;ti<tb_N;ti+=Tn){
				for(int i=0;i<ifm_length;i+=4){
					input_ifm.data.data1 = ifm[i+ifm_addr];
					input_ifm.data.data2 = ifm[i+ifm_addr+1];
					input_ifm.data.data3 = ifm[i+ifm_addr+2];
					input_ifm.data.data4 = ifm[i+ifm_addr+3];
					tb_input_dma_ifm.write(input_ifm);
				}
				ifm_addr+=200704;//Tn*R_in*C_in=16*112*112
			}
		}
	}

	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = weights[i+weights_addr];
		input_weights.data.data2 = weights[i+weights_addr+1];
		input_weights.data.data3 = weights[i+weights_addr+2];
		input_weights.data.data4 = weights[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}

		cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	cout<<tb_flag<<endl;
	output_addr=0;
	//for (int toM=0;toM<2;toM++){
		output_length=bs*1605632;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=128*112*112;
		//output_addr=toM*401408;
		//for(int b=0;b<bs;b++){
			for(int i=0;i<output_length;i+=4){
				output =tb_output_dma_output.read();
				out[i+output_addr] = output.data.data1;
				out[i+output_addr+1] = output.data.data2;
				out[i+output_addr+2] = output.data.data3;
				out[i+output_addr+3] = output.data.data4;
			}
			//output_addr+=802816;
		//}
	//}

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
					cout <<out[bf_index]<< ";";

				}
			}
		}

		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
}

void convwu_128_128_112( int outnum,
	FPGA_DATA ifm[bs*128*112*112],
	FPGA_DATA weights[128*128*3*3],
	FPGA_DATA ofm[bs*128*112*112],
	FPGA_DATA weights_print[128][128][3][3]

){
	//update conv4 weights N,M,R,C=128,128,112,112, stride=1,padding=1,kernel=3
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


	//for (int toM=0;toM<2;toM++){
		ofm_length=200704;// Tm*R*C=16*112*112
		for(int b=0;b<bs;b++){
			for (int to=0;to<8;to++){//repeate[M/Tm]
				ofm_addr=b*1605632+to*200704;//M_in*R*C*b+toM*M*R*C+to*Tm*R*C=128*112*112*b+to*16*112*112
				for(int ti=0;ti<8;ti++){
					for(int i=0;i<ofm_length;i+=4){
						input_ofm.data.data1 = ofm[i+ofm_addr];
						input_ofm.data.data2 = ofm[i+ofm_addr+1];
						input_ofm.data.data3 = ofm[i+ofm_addr+2];
						input_ofm.data.data4 = ofm[i+ofm_addr+3];
						tb_input_dma_ofm.write(input_ofm);
					}
					ifm_length=8960;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
						//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=16*112*5;
					ifm_addr=b*1605632+ti*200704;//b*R*N*C+ti*Tn*R*C
					for(int i=0;i<ifm_length;i+=4){
						input_ifm.data.data1 = ifm[i+ifm_addr];
						input_ifm.data.data2 = ifm[i+ifm_addr+1];
						input_ifm.data.data3 = ifm[i+ifm_addr+2];
						input_ifm.data.data4 = ifm[i+ifm_addr+3];
						tb_input_dma_ifm.write(input_ifm);
					}
					ifm_length=10752;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
									//16*112*6
					for(int row=tb_custom_Tr;row<tb_R-tb_custom_Tr;row+=tb_custom_Tr){
						ifm_addr=b*1605632+ti*200704+(row-1)*1792;//+1792*row;//b*R*N*C+ti*Tn*R*C+C_in*row*stride*min(Tn,N)+(row-1)*1*16*112
						for(int i=0;i<ifm_length;i+=4){
							input_ifm.data.data1 = ifm[i+ifm_addr];
							input_ifm.data.data2 = ifm[i+ifm_addr+1];
							input_ifm.data.data3 = ifm[i+ifm_addr+2];
							input_ifm.data.data4 = ifm[i+ifm_addr+3];
							tb_input_dma_ifm.write(input_ifm);
						}
					}
					ifm_length=8960;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
									//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=16*112*5;
					ifm_addr=b*1605632+ti*200704+191744;//(row-1)*1792;////b*R*N*C+ti*Tn*R*C+C_in*row*stride*min(Tn,N)+(row-1)*1*16*56
					for(int i=0;i<ifm_length;i+=4){
						input_ifm.data.data1 = ifm[i+ifm_addr];
						input_ifm.data.data2 = ifm[i+ifm_addr+1];
						input_ifm.data.data3 = ifm[i+ifm_addr+2];
						input_ifm.data.data4 = ifm[i+ifm_addr+3];
						tb_input_dma_ifm.write(input_ifm);
					}
				}
			}
		}
	//}


	weights_addr=0;

	weights_length=147456;//M*N*tb_custom_k*tb_custom_k
			//=128*128*3*3

	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = weights[i+weights_addr];
		input_weights.data.data2 = weights[i+weights_addr+1];
		input_weights.data.data3 = weights[i+weights_addr+2];
		input_weights.data.data4 = weights[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}




		cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	cout<<tb_flag<<endl;

	//print output
	output_addr=0;
	output_length=147456;//M*N*tb_custom_k*tb_custom_k
	//=128*128*3*3
	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		weights[i+output_addr] = output.data.data1;
		weights[i+output_addr+1] = output.data.data2;
		weights[i+output_addr+2] = output.data.data3;
		weights[i+output_addr+3] = output.data.data4;
	}



	cout<<"after loopM:"<<endl;

	cout<<tb_flag<<endl;
	cout<<"conv"<<outnum<<endl;
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
					cout<<weights_print[m][n][r][c] << ";";
				}
			}

		}
	}
	cout<<"1st 3OCs last 3ICs" <<endl;
	for (int m=0; m<3;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c] << ";";
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
					cout<<weights_print[m][n][r][c] << ";";
				}
			}

		}
	}
	cout<<"last 3OCs last 3ICs" <<endl;
	for (int m=tb_M_in-3; m<tb_M_in;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c] << ";";
				}
			}

		}
	}
}

void convbp_64_128_112( int outnum,
	FPGA_DATA ifm[bs*64*112*112],
	FPGA_DATA weights[64*128*3*3],
	FPGA_DATA outhat[bs*128*112*112],
	FPGA_DATA gama[128],
	FPGA_DATA beta[128],
	FPGA_DATA namuda[256],
	FPGA_DATA out[bs*128*112*112]

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


	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ifm_addr=0;
	ofm_addr=0;
	weights_addr=0;
	ifm_length=tb_M;//M
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = gama[i+weights_addr];
		input_weights.data.data2 = gama[i+weights_addr+1];
		input_weights.data.data3 = gama[i+weights_addr+2];
		input_weights.data.data4 = gama[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}
	for(int i=0;i<ofm_length;i+=4){
		input_ofm.data.data1 = beta[i+ofm_addr];
		input_ofm.data.data2 = beta[i+ofm_addr+1];
		input_ofm.data.data3 = beta[i+ofm_addr+2];
		input_ofm.data.data4 = beta[i+ofm_addr+3];
		tb_input_dma_ofm.write(input_ofm);
	}
	for(int i=0;i<ifm_length;i+=4){
		input_ifm.data.data1 = namuda[i+ifm_addr];
		input_ifm.data.data2 = namuda[i+ifm_addr+1];
		input_ifm.data.data3 = namuda[i+ifm_addr+2];
		input_ifm.data.data4 = namuda[i+ifm_addr+3];
		tb_input_dma_ifm.write(input_ifm);
	}
	//ifm_addr=0;
	//ofm_addr=0;
	ofm_length=200704;//Tm*R*C
	ifm_length=200704;//Tm*R*C
	for (int to=0;to<8;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=1605632*b+200704*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			ofm_addr=1605632*b+200704*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			for(int i=0;i<ifm_length;i+=4){
				input_ifm.data.data1 = ifm[i+ifm_addr];
				input_ifm.data.data2 = ifm[i+ifm_addr+1];
				input_ifm.data.data3 = ifm[i+ifm_addr+2];
				input_ifm.data.data4 = ifm[i+ifm_addr+3];
				tb_input_dma_ifm.write(input_ifm);
			}
			for(int i=0;i<ofm_length;i+=4){
				input_ofm.data.data1 = outhat[i+ofm_addr];
				input_ofm.data.data2 = outhat[i+ofm_addr+1];
				input_ofm.data.data3 = outhat[i+ofm_addr+2];
				input_ofm.data.data4 = outhat[i+ofm_addr+3];
				tb_input_dma_ofm.write(input_ofm);
			}
		}
	}//=6*3*5*5
	ifm_addr=0;
	ofm_addr=0;
	ofm_length=1605632;//M*R*C
	ifm_length=1605632;//M*R*C
	for(int b=0;b<bs;b++){
		//ifm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		//ofm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		for(int i=0;i<ifm_length;i+=4){
			input_ifm.data.data1 = ifm[i+ifm_addr];
			input_ifm.data.data2 = ifm[i+ifm_addr+1];
			input_ifm.data.data3 = ifm[i+ifm_addr+2];
			input_ifm.data.data4 = ifm[i+ifm_addr+3];
			tb_input_dma_ifm.write(input_ifm);
		}
		for(int i=0;i<ofm_length;i+=4){
			input_ofm.data.data1 = outhat[i+ofm_addr];
			input_ofm.data.data2 = outhat[i+ofm_addr+1];
			input_ofm.data.data3 = outhat[i+ofm_addr+2];
			input_ofm.data.data4 = outhat[i+ofm_addr+3];
			tb_input_dma_ofm.write(input_ofm);
		}
		ifm_addr+=ifm_length;
		ofm_addr+=ofm_length;
	}

	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	//cout<<tb_flag<<endl;

	//print output
	output_addr=0;
	output_length=1605632;//M*R*C
	for(int b=0;b<bs;b++){
		for(int i=0;i<output_length;i+=4){
			output =tb_output_dma_output.read();
			ifm[i+output_addr] = output.data.data1;
			ifm[i+output_addr+1] = output.data.data2;
			ifm[i+output_addr+2] = output.data.data3;
			ifm[i+output_addr+3] = output.data.data4;
		}
		output_addr+=output_length;
	}
	output_addr=0;
	output_length=tb_M;//M
	for(int i=0;i<output_length;i+=4){
		out_weights =tb_output_dma_Weightsout.read();
		gama[i+output_addr] = out_weights.data.data1;
		gama[i+output_addr+1] = out_weights.data.data2;
		gama[i+output_addr+2] = out_weights.data.data3;
		gama[i+output_addr+3] = out_weights.data.data4;
	}
	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		beta[i+output_addr] = output.data.data1;
		beta[i+output_addr+1] = output.data.data2;
		beta[i+output_addr+2] = output.data.data3;
		beta[i+output_addr+3] = output.data.data4;
	}

	cout<<"gama first:"<<endl;
	for(int i=0;i<10;i++ ){
		cout<<gama[i]<< ";";
	}
	cout<<"gama last:"<<endl;
	for(int i=tb_M-10;i<tb_M;i++ ){
		cout<<gama[i]<< ";";
	}

	cout<<"beta first:"<<endl;
	cout<<beta[0]<< ";";
	cout<<beta[1]<< ";";
	cout<<beta[2]<< ";";
	cout<<"beta last:"<<endl;
	cout<<beta[tb_M-3]<< ";";
	cout<<beta[tb_M-2]<< ";";
	cout<<beta[tb_M-1]<<endl;


	//backward conv3 N,M,R,C=128,64,112,112, stride=1,padding=1,kernel=3
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


	ifm_addr=0;
	weights_addr=0;
	//ifm_length=30464;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
				//*((C-1)*tb_custom_stride+tb_custom_k)=16*17*112;
	weights_length=73728;//M*N*tb_custom_k*tb_custom_k=64*128*3*3
	for(int b=0;b<bs;b++){
		for (int to=0;to<tb_M;to+=Tm){//repeate [M/Tm]
			ifm_length=30464;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
							//*((C-1)*tb_custom_stride+tb_custom_k)=16*17*112;
			ifm_addr=1605632*b;//+1792*(row-1);//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*128*112*112+row*1*112*16
			for(int ti=0;ti<tb_N;ti+=Tn){
				for(int i=0;i<ifm_length;i+=4){
					input_ifm.data.data1 = ifm[i+ifm_addr];
					input_ifm.data.data2 = ifm[i+ifm_addr+1];
					input_ifm.data.data3 = ifm[i+ifm_addr+2];
					input_ifm.data.data4 = ifm[i+ifm_addr+3];
					tb_input_dma_ifm.write(input_ifm);
				}
				ifm_addr+=200704;//Tn*R_in*C_in=16*112*112
			}
			for(int row=tb_custom_Tr;row<tb_R-tb_custom_Tr;row+=tb_custom_Tr){
				ifm_length=32256;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
									//*((C-1)*tb_custom_stride+tb_custom_k)=16*18*112;
				ifm_addr=1605632*b+1792*(row-1);//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*128*112*112+row*1*112*16
				for(int ti=0;ti<tb_N;ti+=Tn){
					for(int i=0;i<ifm_length;i+=4){
						input_ifm.data.data1 = ifm[i+ifm_addr];
						input_ifm.data.data2 = ifm[i+ifm_addr+1];
						input_ifm.data.data3 = ifm[i+ifm_addr+2];
						input_ifm.data.data4 = ifm[i+ifm_addr+3];
						tb_input_dma_ifm.write(input_ifm);
					}
					ifm_addr+=200704;//Tn*R_in*C_in=16*112*112
				}
			}
			ifm_length=30464;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
							//*((C-1)*tb_custom_stride+tb_custom_k)=16*17*112;
			ifm_addr=1605632*b+170240;//+1792*(row-1);//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*128*112*112+row*1*112*16
			for(int ti=0;ti<tb_N;ti+=Tn){
				for(int i=0;i<ifm_length;i+=4){
					input_ifm.data.data1 = ifm[i+ifm_addr];
					input_ifm.data.data2 = ifm[i+ifm_addr+1];
					input_ifm.data.data3 = ifm[i+ifm_addr+2];
					input_ifm.data.data4 = ifm[i+ifm_addr+3];
					tb_input_dma_ifm.write(input_ifm);
				}
				ifm_addr+=200704;//Tn*R_in*C_in=16*112*112
			}
		}
	}

	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = weights[i+weights_addr];
		input_weights.data.data2 = weights[i+weights_addr+1];
		input_weights.data.data3 = weights[i+weights_addr+2];
		input_weights.data.data4 = weights[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}

		cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	cout<<tb_flag<<endl;
	output_addr=0;
	//for (int toM=0;toM<2;toM++){
		output_length=bs*802816;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=64*112*112;
		//output_addr=toM*401408;
		//for(int b=0;b<bs;b++){
			for(int i=0;i<output_length;i+=4){
				output =tb_output_dma_output.read();
				out[i+output_addr] = output.data.data1;
				out[i+output_addr+1] = output.data.data2;
				out[i+output_addr+2] = output.data.data3;
				out[i+output_addr+3] = output.data.data4;
			}
			//output_addr+=802816;
		//}
	//}

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
					cout <<out[bf_index]<< ";";

				}
			}
		}

		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
}

void convwu_64_128_112( int outnum,
	FPGA_DATA ifm[bs*64*112*112],
	FPGA_DATA weights[64*128*3*3],
	FPGA_DATA ofm[bs*128*112*112],
	FPGA_DATA weights_print[128][64][3][3]

){
	//update conv3 weights N,M,R,C=64,128,112,112, stride=1,padding=1,kernel=3
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


	//for (int toM=0;toM<2;toM++){
		ofm_length=200704;// Tm*R*C=16*112*112
		for(int b=0;b<bs;b++){
			for (int to=0;to<8;to++){//repeate[M/Tm]
				ofm_addr=b*1605632+to*200704;//M_in*R*C*b+toM*M*R*C+to*Tm*R*C=128*112*112*b+to*16*112*112
				for(int ti=0;ti<4;ti++){
					for(int i=0;i<ofm_length;i+=4){
						input_ofm.data.data1 = ofm[i+ofm_addr];
						input_ofm.data.data2 = ofm[i+ofm_addr+1];
						input_ofm.data.data3 = ofm[i+ofm_addr+2];
						input_ofm.data.data4 = ofm[i+ofm_addr+3];
						tb_input_dma_ofm.write(input_ofm);
					}
					ifm_length=8960;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
						//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=16*112*5;
					ifm_addr=b*802816+ti*200704;//b*R*N*C+ti*Tn*R*C
					for(int i=0;i<ifm_length;i+=4){
						input_ifm.data.data1 = ifm[i+ifm_addr];
						input_ifm.data.data2 = ifm[i+ifm_addr+1];
						input_ifm.data.data3 = ifm[i+ifm_addr+2];
						input_ifm.data.data4 = ifm[i+ifm_addr+3];
						tb_input_dma_ifm.write(input_ifm);
					}
					ifm_length=10752;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
									//16*112*6
					for(int row=tb_custom_Tr;row<tb_R-tb_custom_Tr;row+=tb_custom_Tr){
						ifm_addr=b*802816+ti*200704+(row-1)*1792;//+1792*row;//b*R*N*C+ti*Tn*R*C+C_in*row*stride*min(Tn,N)+(row-1)*1*16*112
						for(int i=0;i<ifm_length;i+=4){
							input_ifm.data.data1 = ifm[i+ifm_addr];
							input_ifm.data.data2 = ifm[i+ifm_addr+1];
							input_ifm.data.data3 = ifm[i+ifm_addr+2];
							input_ifm.data.data4 = ifm[i+ifm_addr+3];
							tb_input_dma_ifm.write(input_ifm);
						}
					}
					ifm_length=8960;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
									//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=16*112*5;
					ifm_addr=b*802816+ti*200704+191744;//(row-1)*1792;////b*R*N*C+ti*Tn*R*C+C_in*row*stride*min(Tn,N)+(row-1)*1*16*56
					for(int i=0;i<ifm_length;i+=4){
						input_ifm.data.data1 = ifm[i+ifm_addr];
						input_ifm.data.data2 = ifm[i+ifm_addr+1];
						input_ifm.data.data3 = ifm[i+ifm_addr+2];
						input_ifm.data.data4 = ifm[i+ifm_addr+3];
						tb_input_dma_ifm.write(input_ifm);
					}
				}
			}
		}
	//}


	weights_addr=0;

	weights_length=73728;//M*N*tb_custom_k*tb_custom_k
			//=64*128*3*3

	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = weights[i+weights_addr];
		input_weights.data.data2 = weights[i+weights_addr+1];
		input_weights.data.data3 = weights[i+weights_addr+2];
		input_weights.data.data4 = weights[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}




		cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	cout<<tb_flag<<endl;

	//print output
	output_addr=0;
	output_length=73728;//M*N*tb_custom_k*tb_custom_k
	//=64*128*3*3
	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		weights[i+output_addr] = output.data.data1;
		weights[i+output_addr+1] = output.data.data2;
		weights[i+output_addr+2] = output.data.data3;
		weights[i+output_addr+3] = output.data.data4;
	}



	cout<<"after loopM:"<<endl;

	cout<<tb_flag<<endl;
	cout<<"conv"<<outnum<<endl;
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
					cout<<weights_print[m][n][r][c] << ";";
				}
			}

		}
	}
	cout<<"1st 3OCs last 3ICs" <<endl;
	for (int m=0; m<3;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c] << ";";
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
					cout<<weights_print[m][n][r][c] << ";";
				}
			}

		}
	}
	cout<<"last 3OCs last 3ICs" <<endl;
	for (int m=tb_M_in-3; m<tb_M_in;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c] << ";";
				}
			}

		}
	}
}

void maxpoolbp_64_112( int outnum,
	FPGA_DATA ifm[bs*64*112*112],
	FPGA_DATA weights[bs*112*112*8],
	FPGA_DATA ofm[bs*64*224*224],
	FPGA_DATA out[bs*64*224*224]

){
	//backward pool+relu N,M,R,C=64,64,112,112, stride=2,padding=0,kernel=2
	tb_statec=0;
	tb_statep=5;//view=8+4
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


	//FPGA_DATA loss5[bs*10*10*16];//[r][c][m]
	ifm_addr=0;
	ifm_length=bs*802816;//M*tb_custom_Tr*tb_custom_Tc=64*112*112;
	//for(int b=0;b<bs;b++){
		for(int i=0;i<ifm_length;i+=4){
			input_ifm.data.data1 = ifm[i+ifm_addr];
			input_ifm.data.data2 = ifm[i+ifm_addr+1];
			input_ifm.data.data3 = ifm[i+ifm_addr+2];
			input_ifm.data.data4 = ifm[i+ifm_addr+3];
			tb_input_dma_ifm.write(input_ifm);
			//cout<<input_ifm.data<<endl;
		}
		//ifm_addr+=ifm_length;
	//}

	weights_addr=0;
	weights_length=bs*100352;//r*c*[m/8]=112*112*8
	for(int i=0;i<weights_length;i+=4){
		pool_index.data.data1 =weights[i+weights_addr];
		pool_index.data.data2 =weights[i+weights_addr+1];
		pool_index.data.data3 =weights[i+weights_addr+2];
		pool_index.data.data4 =weights[i+weights_addr+3];
		tb_input_dma_weights.write(pool_index);

	}

	ofm_addr=0;
	ofm_length=bs*3211264;//M*R_in*C_in=64*224*224;
	for(int i=0;i<ofm_length;i+=4){
		input_ofm.data.data1 = ofm[i+ofm_addr];
		input_ofm.data.data2 = ofm[i+ofm_addr+1];
		input_ofm.data.data3 = ofm[i+ofm_addr+2];
		input_ofm.data.data4 = ofm[i+ofm_addr+3];
		tb_input_dma_ofm.write(input_ofm);
	}


	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	cout<<tb_flag<<endl;

//print output

	output_addr=0;
	output_length=bs*3211264;//M*R_in*C_in=64*224*224;
	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		out[i+output_addr] = output.data.data1;
		out[i+output_addr+1] = output.data.data2;
		out[i+output_addr+2] = output.data.data3;
		out[i+output_addr+3] = output.data.data4;
		//cout << loss3[i+output_addr] << ";";
	}




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
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R_in*tb_C_in*tb_M_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R_in-3;r<tb_R_in;r++){
			for (int c=tb_C_in-3;c<tb_C_in;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R_in*tb_C_in+r*tb_C_in*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
}

void convbp_64_64_224( int outnum,
	FPGA_DATA ifm[bs*64*224*224],
	FPGA_DATA weights[64*64*3*3],
	FPGA_DATA outhat[bs*64*224*224],
	FPGA_DATA gama[64],
	FPGA_DATA beta[64],
	FPGA_DATA namuda[64],
	FPGA_DATA ofm[bs*64*224*224],
	FPGA_DATA out[bs*64*224*224]

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


	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ifm_addr=0;
	ofm_addr=0;
	weights_addr=0;
	ifm_length=tb_M;//M
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = gama[i+weights_addr];
		input_weights.data.data2 = gama[i+weights_addr+1];
		input_weights.data.data3 = gama[i+weights_addr+2];
		input_weights.data.data4 = gama[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}
	for(int i=0;i<ofm_length;i+=4){
		input_ofm.data.data1 = beta[i+ofm_addr];
		input_ofm.data.data2 = beta[i+ofm_addr+1];
		input_ofm.data.data3 = beta[i+ofm_addr+2];
		input_ofm.data.data4 = beta[i+ofm_addr+3];
		tb_input_dma_ofm.write(input_ofm);
	}
	for(int i=0;i<ifm_length;i+=4){
		input_ifm.data.data1 = namuda[i+ifm_addr];
		input_ifm.data.data2 = namuda[i+ifm_addr+1];
		input_ifm.data.data3 = namuda[i+ifm_addr+2];
		input_ifm.data.data4 = namuda[i+ifm_addr+3];
		tb_input_dma_ifm.write(input_ifm);
	}
	//ifm_addr=0;
	//ofm_addr=0;
	ofm_length=802816;//Tm*R*C
	ifm_length=802816;//Tm*R*C
	for (int to=0;to<4;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			ofm_addr=3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			for(int i=0;i<ifm_length;i+=4){
				input_ifm.data.data1 = ifm[i+ifm_addr];
				input_ifm.data.data2 = ifm[i+ifm_addr+1];
				input_ifm.data.data3 = ifm[i+ifm_addr+2];
				input_ifm.data.data4 = ifm[i+ifm_addr+3];
				tb_input_dma_ifm.write(input_ifm);
			}
			for(int i=0;i<ofm_length;i+=4){
				input_ofm.data.data1 = outhat[i+ofm_addr];
				input_ofm.data.data2 = outhat[i+ofm_addr+1];
				input_ofm.data.data3 = outhat[i+ofm_addr+2];
				input_ofm.data.data4 = outhat[i+ofm_addr+3];
				tb_input_dma_ofm.write(input_ofm);
			}
		}
	}//=6*3*5*5
	ifm_addr=0;
	ofm_addr=0;
	ofm_length=3211264;//M*R*C
	ifm_length=3211264;//M*R*C
	for(int b=0;b<bs;b++){
		//ifm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		//ofm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		for(int i=0;i<ifm_length;i+=4){
			input_ifm.data.data1 = ifm[i+ifm_addr];
			input_ifm.data.data2 = ifm[i+ifm_addr+1];
			input_ifm.data.data3 = ifm[i+ifm_addr+2];
			input_ifm.data.data4 = ifm[i+ifm_addr+3];
			tb_input_dma_ifm.write(input_ifm);
		}
		for(int i=0;i<ofm_length;i+=4){
			input_ofm.data.data1 = outhat[i+ofm_addr];
			input_ofm.data.data2 = outhat[i+ofm_addr+1];
			input_ofm.data.data3 = outhat[i+ofm_addr+2];
			input_ofm.data.data4 = outhat[i+ofm_addr+3];
			tb_input_dma_ofm.write(input_ofm);
		}
		ifm_addr+=ifm_length;
		ofm_addr+=ofm_length;
	}

	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	//cout<<tb_flag<<endl;

	//print output
	output_addr=0;
	output_length=3211264;//M*R*C
	for(int b=0;b<bs;b++){
		for(int i=0;i<output_length;i+=4){
			output =tb_output_dma_output.read();
			ifm[i+output_addr] = output.data.data1;
			ifm[i+output_addr+1] = output.data.data2;
			ifm[i+output_addr+2] = output.data.data3;
			ifm[i+output_addr+3] = output.data.data4;
		}
		output_addr+=output_length;
	}
	output_addr=0;
	output_length=tb_M;//M
	for(int i=0;i<output_length;i+=4){
		out_weights =tb_output_dma_Weightsout.read();
		gama[i+output_addr] = out_weights.data.data1;
		gama[i+output_addr+1] = out_weights.data.data2;
		gama[i+output_addr+2] = out_weights.data.data3;
		gama[i+output_addr+3] = out_weights.data.data4;
	}
	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		beta[i+output_addr] = output.data.data1;
		beta[i+output_addr+1] = output.data.data2;
		beta[i+output_addr+2] = output.data.data3;
		beta[i+output_addr+3] = output.data.data4;
	}

	cout<<"gama first:"<<endl;
	for(int i=0;i<10;i++ ){
		cout<<gama[i]<< ";";
	}
	cout<<"gama last:"<<endl;
	for(int i=tb_M-10;i<tb_M;i++ ){
		cout<<gama[i]<< ";";
	}

	cout<<"beta first:"<<endl;
	cout<<beta[0]<< ";";
	cout<<beta[1]<< ";";
	cout<<beta[2]<< ";";
	cout<<"beta last:"<<endl;
	cout<<beta[tb_M-3]<< ";";
	cout<<beta[tb_M-2]<< ";";
	cout<<beta[tb_M-1]<<endl;



	//backward conv2+relu N,M,R,C=64,64,224,224, stride=1,padding=1,kernel=3
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


	ifm_addr=0;
	weights_addr=0;
	//ifm_length=30464;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
				//*((C-1)*tb_custom_stride+tb_custom_k)=16*17*112;
	weights_length=36864;//M*N*tb_custom_k*tb_custom_k=64*64*3*3
	ofm_length=bs*3211264;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=64*224*224;
	ofm_addr=0;
	for(int i=0;i<ofm_length;i+=4){
		input_ofm.data.data1 = ofm[i+ofm_addr];
		input_ofm.data.data2 = ofm[i+ofm_addr+1];
		input_ofm.data.data3 = ofm[i+ofm_addr+2];
		input_ofm.data.data4 = ofm[i+ofm_addr+3];
		tb_input_dma_ofm.write(input_ofm);
	}
	for(int b=0;b<bs;b++){
		for (int to=0;to<tb_M;to+=Tm){//repeate [M/Tm]
			ifm_length=32256;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
							//*((C-1)*tb_custom_stride+tb_custom_k)=16*9*224;
			ifm_addr=3211264*b;//+3584*(row-1);//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*64*224*224+row*1*224*16
			for(int ti=0;ti<tb_N;ti+=Tn){
				for(int i=0;i<ifm_length;i+=4){
					input_ifm.data.data1 = ifm[i+ifm_addr];
					input_ifm.data.data2 = ifm[i+ifm_addr+1];
					input_ifm.data.data3 = ifm[i+ifm_addr+2];
					input_ifm.data.data4 = ifm[i+ifm_addr+3];
					tb_input_dma_ifm.write(input_ifm);
				}
				ifm_addr+=802816;//Tn*R_in*C_in=16*224*224
			}
			for(int row=tb_custom_Tr;row<tb_R-tb_custom_Tr;row+=tb_custom_Tr){
				ifm_length=35840;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
									//*((C-1)*tb_custom_stride+tb_custom_k)=16*10*224;
				ifm_addr=3211264*b+3584*(row-1);//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*64*224*224+row*1*224*16
				for(int ti=0;ti<tb_N;ti+=Tn){
					for(int i=0;i<ifm_length;i+=4){
						input_ifm.data.data1 = ifm[i+ifm_addr];
						input_ifm.data.data2 = ifm[i+ifm_addr+1];
						input_ifm.data.data3 = ifm[i+ifm_addr+2];
						input_ifm.data.data4 = ifm[i+ifm_addr+3];
						tb_input_dma_ifm.write(input_ifm);
					}
					ifm_addr+=802816;//Tn*R_in*C_in=16*224*224
				}
			}
			ifm_length=32256;//N*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k)
							//*((C-1)*tb_custom_stride+tb_custom_k)=16*9*224;
			ifm_addr=3211264*b+770560;//+3584*(row-1);//b*N*C_in*R_in+C_in*row*stride*min(Tn,N)=b*64*224*224+row*1*224*16
			for(int ti=0;ti<tb_N;ti+=Tn){
				for(int i=0;i<ifm_length;i+=4){
					input_ifm.data.data1 = ifm[i+ifm_addr];
					input_ifm.data.data2 = ifm[i+ifm_addr+1];
					input_ifm.data.data3 = ifm[i+ifm_addr+2];
					input_ifm.data.data4 = ifm[i+ifm_addr+3];
					tb_input_dma_ifm.write(input_ifm);
				}
				ifm_addr+=802816;//Tn*R_in*C_in=16*224*224
			}
		}
	}

	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = weights[i+weights_addr];
		input_weights.data.data2 = weights[i+weights_addr+1];
		input_weights.data.data3 = weights[i+weights_addr+2];
		input_weights.data.data4 = weights[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}

		cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	cout<<tb_flag<<endl;
	output_addr=0;
	//for (int toM=0;toM<2;toM++){
		output_length=bs*3211264;//tb_M*min(custom_Tr,R-row*custom_Tr)*C=64*224*224;
		//output_addr=toM*401408;
		//for(int b=0;b<bs;b++){
			for(int i=0;i<output_length;i+=4){
				output =tb_output_dma_output.read();
				out[i+output_addr] = output.data.data1;
				out[i+output_addr+1] = output.data.data2;
				out[i+output_addr+2] = output.data.data3;
				out[i+output_addr+3] = output.data.data4;
			}
			//output_addr+=802816;
		//}
	//}

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
					cout <<out[bf_index]<< ";";

				}
			}
		}

		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<out[bf_index]<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
}

void convwu_64_64_224( int outnum,
	FPGA_DATA ifm[bs*64*224*224],
	FPGA_DATA weights[64*64*3*3],
	FPGA_DATA ofm[bs*64*224*224],
	FPGA_DATA weights_print[64][64][3][3]

){
	//update conv2 weights N,M,R,C=64,64,224,224, stride=1,padding=1,kernel=3
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


	//for (int toM=0;toM<2;toM++){
		ofm_length=802816;// Tm*R*C=16*224*224
		for(int b=0;b<bs;b++){
			for (int to=0;to<4;to++){//repeate[M/Tm]
				ofm_addr=b*3211264+to*802816;//M_in*R*C*b+toM*M*R*C+to*Tm*R*C=64*224*224*b+to*16*224*224
				for(int ti=0;ti<4;ti++){
					for(int i=0;i<ofm_length;i+=4){
						input_ofm.data.data1 = ofm[i+ofm_addr];
						input_ofm.data.data2 = ofm[i+ofm_addr+1];
						input_ofm.data.data3 = ofm[i+ofm_addr+2];
						input_ofm.data.data4 = ofm[i+ofm_addr+3];
						tb_input_dma_ofm.write(input_ofm);
					}
					ifm_length=10752;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
						//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=16*224*3;
					ifm_addr=b*3211264+ti*802816;//b*R*N*C+ti*Tn*R*C
					for(int i=0;i<ifm_length;i+=4){
						input_ifm.data.data1 = ifm[i+ifm_addr];
						input_ifm.data.data2 = ifm[i+ifm_addr+1];
						input_ifm.data.data3 = ifm[i+ifm_addr+2];
						input_ifm.data.data4 = ifm[i+ifm_addr+3];
						tb_input_dma_ifm.write(input_ifm);
					}
					ifm_length=14336;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
									//16*224*4
					for(int row=tb_custom_Tr;row<tb_R-tb_custom_Tr;row+=tb_custom_Tr){
						ifm_addr=b*3211264+ti*802816+(row-1)*3584;//+3584*row;//b*R*N*C+ti*Tn*R*C+C_in*row*stride*min(Tn,N)+(row-1)*1*16*224
						for(int i=0;i<ifm_length;i+=4){
							input_ifm.data.data1 = ifm[i+ifm_addr];
							input_ifm.data.data2 = ifm[i+ifm_addr+1];
							input_ifm.data.data3 = ifm[i+ifm_addr+2];
							input_ifm.data.data4 = ifm[i+ifm_addr+3];
							tb_input_dma_ifm.write(input_ifm);
						}
					}
					ifm_length=10752;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
									//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=16*224*3;
					ifm_addr=b*3211264+ti*802816+792064;//(row-1)*3584;//+3584*row;//b*R*N*C+ti*Tn*R*C+C_in*row*stride*min(Tn,N)+(row-1)*1*16*224
					for(int i=0;i<ifm_length;i+=4){
						input_ifm.data.data1 = ifm[i+ifm_addr];
						input_ifm.data.data2 = ifm[i+ifm_addr+1];
						input_ifm.data.data3 = ifm[i+ifm_addr+2];
						input_ifm.data.data4 = ifm[i+ifm_addr+3];
						tb_input_dma_ifm.write(input_ifm);
					}
				}
			}
		}
	//}


	weights_addr=0;

	weights_length=36864;//M*N*tb_custom_k*tb_custom_k
			//=64*64*3*3

	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = weights[i+weights_addr];
		input_weights.data.data2 = weights[i+weights_addr+1];
		input_weights.data.data3 = weights[i+weights_addr+2];
		input_weights.data.data4 = weights[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}




		cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	cout<<tb_flag<<endl;

	//print output
	output_addr=0;
	output_length=36864;//M*N*tb_custom_k*tb_custom_k
	//=64*64*3*3
	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		weights[i+output_addr] = output.data.data1;
		weights[i+output_addr+1] = output.data.data2;
		weights[i+output_addr+2] = output.data.data3;
		weights[i+output_addr+3] = output.data.data4;
	}



	cout<<"after loopM:"<<endl;

	cout<<tb_flag<<endl;
	cout<<"conv"<<outnum<<endl;
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
					cout<<weights_print[m][n][r][c] << ";";
				}
			}

		}
	}
	cout<<"1st 3OCs last 3ICs" <<endl;
	for (int m=0; m<3;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c] << ";";
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
					cout<<weights_print[m][n][r][c] << ";";
				}
			}

		}
	}
	cout<<"last 3OCs last 3ICs" <<endl;
	for (int m=tb_M_in-3; m<tb_M_in;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c] << ";";
				}
			}

		}
	}
}

void convbp_3_64_224( int outnum,
	FPGA_DATA ifm[bs*64*224*224],
	//FPGA_DATA weights[64*64*3*3],
	FPGA_DATA outhat[bs*64*224*224],
	FPGA_DATA gama[64],
	FPGA_DATA beta[64],
	FPGA_DATA namuda[64]
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


	//FPGA_DATA output1[bs*24*24*20]; //[r][c][m]
	ifm_addr=0;
	ofm_addr=0;
	weights_addr=0;
	ifm_length=tb_M;//M
	ofm_length=tb_M;//M
	weights_length=tb_M;//M
	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = gama[i+weights_addr];
		input_weights.data.data2 = gama[i+weights_addr+1];
		input_weights.data.data3 = gama[i+weights_addr+2];
		input_weights.data.data4 = gama[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}
	for(int i=0;i<ofm_length;i+=4){
		input_ofm.data.data1 = beta[i+ofm_addr];
		input_ofm.data.data2 = beta[i+ofm_addr+1];
		input_ofm.data.data3 = beta[i+ofm_addr+2];
		input_ofm.data.data4 = beta[i+ofm_addr+3];
		tb_input_dma_ofm.write(input_ofm);
	}
	for(int i=0;i<ifm_length;i+=4){
		input_ifm.data.data1 = namuda[i+ifm_addr];
		input_ifm.data.data2 = namuda[i+ifm_addr+1];
		input_ifm.data.data3 = namuda[i+ifm_addr+2];
		input_ifm.data.data4 = namuda[i+ifm_addr+3];
		tb_input_dma_ifm.write(input_ifm);
	}
	//ifm_addr=0;
	//ofm_addr=0;
	ofm_length=802816;//Tm*R*C
	ifm_length=802816;//Tm*R*C
	for (int to=0;to<4;to++){//repeate [M/Tm]
		for(int b=0;b<bs;b++){
			ifm_addr=3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			ofm_addr=3211264*b+802816*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
			for(int i=0;i<ifm_length;i+=4){
				input_ifm.data.data1 = ifm[i+ifm_addr];
				input_ifm.data.data2 = ifm[i+ifm_addr+1];
				input_ifm.data.data3 = ifm[i+ifm_addr+2];
				input_ifm.data.data4 = ifm[i+ifm_addr+3];
				tb_input_dma_ifm.write(input_ifm);
			}
			for(int i=0;i<ofm_length;i+=4){
				input_ofm.data.data1 = outhat[i+ofm_addr];
				input_ofm.data.data2 = outhat[i+ofm_addr+1];
				input_ofm.data.data3 = outhat[i+ofm_addr+2];
				input_ofm.data.data4 = outhat[i+ofm_addr+3];
				tb_input_dma_ofm.write(input_ofm);
			}
		}
	}//=6*3*5*5
	ifm_addr=0;
	ofm_addr=0;
	ofm_length=3211264;//M*R*C
	ifm_length=3211264;//M*R*C
	for(int b=0;b<bs;b++){
		//ifm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		//ofm_addr=4096*b+1024*to;//+128*row;//b*M*C*R+to*Tm*R*C=b*16*32*32+to*16*32*32
		for(int i=0;i<ifm_length;i+=4){
			input_ifm.data.data1 = ifm[i+ifm_addr];
			input_ifm.data.data2 = ifm[i+ifm_addr+1];
			input_ifm.data.data3 = ifm[i+ifm_addr+2];
			input_ifm.data.data4 = ifm[i+ifm_addr+3];
			tb_input_dma_ifm.write(input_ifm);
		}
		for(int i=0;i<ofm_length;i+=4){
			input_ofm.data.data1 = outhat[i+ofm_addr];
			input_ofm.data.data2 = outhat[i+ofm_addr+1];
			input_ofm.data.data3 = outhat[i+ofm_addr+2];
			input_ofm.data.data4 = outhat[i+ofm_addr+3];
			tb_input_dma_ofm.write(input_ofm);
		}
		ifm_addr+=ifm_length;
		ofm_addr+=ofm_length;
	}

	cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	//cout<<tb_flag<<endl;

	//print output
	output_addr=0;
	output_length=3211264;//M*R*C
	for(int b=0;b<bs;b++){
		for(int i=0;i<output_length;i+=4){
			output =tb_output_dma_output.read();
			ifm[i+output_addr] = output.data.data1;
			ifm[i+output_addr+1] = output.data.data2;
			ifm[i+output_addr+2] = output.data.data3;
			ifm[i+output_addr+3] = output.data.data4;
		}
		output_addr+=output_length;
	}
	output_addr=0;
	output_length=tb_M;//M
	for(int i=0;i<output_length;i+=4){
		out_weights =tb_output_dma_Weightsout.read();
		gama[i+output_addr] = out_weights.data.data1;
		gama[i+output_addr+1] = out_weights.data.data2;
		gama[i+output_addr+2] = out_weights.data.data3;
		gama[i+output_addr+3] = out_weights.data.data4;
	}
	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		beta[i+output_addr] = output.data.data1;
		beta[i+output_addr+1] = output.data.data2;
		beta[i+output_addr+2] = output.data.data3;
		beta[i+output_addr+3] = output.data.data4;
	}

	cout<<"gama first:"<<endl;
	for(int i=0;i<10;i++ ){
		cout<<gama[i]<< ";";
	}
	cout<<"gama last:"<<endl;
	for(int i=tb_M-10;i<tb_M;i++ ){
		cout<<gama[i]<< ";";
	}

	cout<<"beta first:"<<endl;
	cout<<beta[0]<< ";";
	cout<<beta[1]<< ";";
	cout<<beta[2]<< ";";
	cout<<"beta last:"<<endl;
	cout<<beta[tb_M-3]<< ";";
	cout<<beta[tb_M-2]<< ";";
	cout<<beta[tb_M-1]<<endl;


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
					cout <<ifm[bf_index]<< ";";

				}
			}
		}

		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<ifm[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<ifm[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=0;too<3;too++){
					bf_index=b*tb_R*tb_C*tb_M_in+r*tb_C*Tm+c*Tm+too;
					cout <<ifm[bf_index]<< ";";

				}
			}
		}
		cout <<" last channel tile  " <<endl;
		cout<<"1st 3rows 1st 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<ifm[bf_index]<< ";";

				}
			}
		}
		cout<<"1st 3rows last 3cols"<<":";
		for (int r=0;r<3;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<ifm[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows 1st 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=0;c<3;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<ifm[bf_index]<< ";";

				}
			}
		}
		cout<<"last 3rows last 3cols"<<":";
		for (int r=tb_R-3;r<tb_R;r++){
			for (int c=tb_C-3;c<tb_C;c++){
				for (int too=Tm-3;too<Tm;too++){
					bf_index=((b+1)*tb_M_in-Tm)*tb_R*tb_C+r*tb_C*Tm+c*Tm+too;
					cout <<ifm[bf_index]<< ";";

				}
			}
		}
		cout<<""<<endl;
	}
}

void convwu_3_64_224( int outnum,
	FPGA_DATA ifm[bs*4*224*224],
	FPGA_DATA weights[3*64*3*3],
	FPGA_DATA ofm[bs*64*224*224],
	FPGA_DATA weights_print[64][3][3][3]

){
	//update conv1 weights N,M,R,C=3,64,224,224, stride=1,padding=1,kernel=3
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


	//for (int toM=0;toM<2;toM++){
		ofm_length=802816;// Tm*R*C=16*224*224
		for(int b=0;b<bs;b++){
			for (int to=0;to<4;to++){//repeate[M/Tm]
				ofm_addr=b*3211264+to*802816;//M_in*R*C*b+toM*M*R*C+to*Tm*R*C=64*224*224*b+to*16*224*224
				//for(int ti=0;ti<4;ti++){
					for(int i=0;i<ofm_length;i+=4){
						input_ofm.data.data1 = ofm[i+ofm_addr];
						input_ofm.data.data2 = ofm[i+ofm_addr+1];
						input_ofm.data.data3 = ofm[i+ofm_addr+2];
						input_ofm.data.data4 = ofm[i+ofm_addr+3];
						tb_input_dma_ofm.write(input_ofm);
					}
					ifm_length=1792;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
						//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=4*224*2;
					ifm_addr=b*200704;//+ti*802816;//b*R*N*C+ti*Tn*R*C
					for(int i=0;i<ifm_length;i+=4){
						input_ifm.data.data1 = ifm[i+ifm_addr];
						input_ifm.data.data2 = ifm[i+ifm_addr+1];
						input_ifm.data.data3 = ifm[i+ifm_addr+2];
						input_ifm.data.data4 = ifm[i+ifm_addr+3];
						tb_input_dma_ifm.write(input_ifm);
					}
					ifm_length=2688;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
									//4*224*3
					for(int row=tb_custom_Tr;row<tb_R-tb_custom_Tr;row+=tb_custom_Tr){
						ifm_addr=b*200704+(row-1)*896;//+ti*802816+(row-1)*896;//+3584*row;//b*R*N*C+ti*Tn*R*C+C_in*row*stride*min(Tn,N)+(row-1)*1*4*224
						for(int i=0;i<ifm_length;i+=4){
							input_ifm.data.data1 = ifm[i+ifm_addr];
							input_ifm.data.data2 = ifm[i+ifm_addr+1];
							input_ifm.data.data3 = ifm[i+ifm_addr+2];
							input_ifm.data.data4 = ifm[i+ifm_addr+3];
							tb_input_dma_ifm.write(input_ifm);
						}
					}
					ifm_length=1792;//Tn*((min(custom_Tr,R-row*custom_Tr)-1)*tb_custom_stride+tb_custom_k-2*padding)
									//*((C-1)*tb_custom_stride+tb_custom_k-2*padding)=4*224*2;
					ifm_addr=b*200704+198912;//(row-1)*896;//ti*802816+792064;//(row-1)*3584;//+3584*row;//b*R*N*C+ti*Tn*R*C+C_in*row*stride*min(Tn,N)+(row-1)*1*4*224
					for(int i=0;i<ifm_length;i+=4){
						input_ifm.data.data1 = ifm[i+ifm_addr];
						input_ifm.data.data2 = ifm[i+ifm_addr+1];
						input_ifm.data.data3 = ifm[i+ifm_addr+2];
						input_ifm.data.data4 = ifm[i+ifm_addr+3];
						tb_input_dma_ifm.write(input_ifm);
					}
				//}
			}
		}
	//}


	weights_addr=0;

	weights_length=1728;//M*N*tb_custom_k*tb_custom_k
			//=3*64*3*3

	for(int i=0;i<weights_length;i+=4){
		input_weights.data.data1 = weights[i+weights_addr];
		input_weights.data.data2 = weights[i+weights_addr+1];
		input_weights.data.data3 = weights[i+weights_addr+2];
		input_weights.data.data4 = weights[i+weights_addr+3];
		tb_input_dma_weights.write(input_weights);
	}




		cout<<"before top:"<<endl;

	top( tb_input_dma_ifm, tb_input_dma_weights,tb_output_dma_Weightsout,tb_input_dma_ofm, tb_output_dma_output,
	tb_statec,tb_statep,tb_stateb,
	tb_custom_batch,tb_batch_size,tb_M_in,tb_custom_Tib,tb_N,tb_M,tb_R,tb_C,
	tb_custom_stride,tb_padding,tb_custom_kb,tb_custom_k,tb_custom_Tr,tb_custom_Tc,
	tb_R_in,tb_C_in);
	cout<<"after top:"<<endl;

	cout<<tb_flag<<endl;

	//print output
	output_addr=0;
	output_length=1728;//M*N*tb_custom_k*tb_custom_k
	//=3*64*3*3
	for(int i=0;i<output_length;i+=4){
		output =tb_output_dma_output.read();
		weights[i+output_addr] = output.data.data1;
		weights[i+output_addr+1] = output.data.data2;
		weights[i+output_addr+2] = output.data.data3;
		weights[i+output_addr+3] = output.data.data4;
	}



	cout<<"after loopM:"<<endl;

	cout<<tb_flag<<endl;
	cout<<"conv"<<outnum<<endl;
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
					cout<<weights_print[m][n][r][c] << ";";
				}
			}

		}
	}
	cout<<"1st 3OCs last 3ICs" <<endl;
	for (int m=0; m<3;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c] << ";";
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
					cout<<weights_print[m][n][r][c] << ";";
				}
			}

		}
	}
	cout<<"last 3OCs last 3ICs" <<endl;
	for (int m=tb_M_in-3; m<tb_M_in;m++){
		for (int n =tb_N-3; n<tb_N;n++){
			for (int r=0;r<tb_custom_k;r++){
				for (int c=0;c<tb_custom_k;c++){
					cout<<weights_print[m][n][r][c] << ";";
				}
			}

		}
	}
}














int main(){
	//FPGA_DATA input_x_buffer[bs*1*28*28];
		bf_index=0;
		for(int b=0;b<bs;b++){
			for(int to=0;to<4;to+=Tm){
				for (int r=0;r<224;r++){
					for (int c=0;c<224;c++){
						for (int too=0;too<Tm;too++){
							if((too+to)<4){
								if(too+to<3){
									input_x_buffer[bf_index] =(((c+too+to)%29)*(0.5-b))/3.0+(((r+c)%7)*(b-0.5))/6.0-(((c+too+to)%11))/2.0;//output.data;
								}
								else{
									input_x_buffer[bf_index]=0;
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
									weights_conv1_buffer[bf_index] = (2.0*r+3.0*c+((too+to+c)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv1_buffer[bf_index+1] =(2.0*r+3.0*c+((too+to+c+1)%11) - ((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv1_buffer[bf_index+2] = (2.0*r+3.0*c+((too+to+c+2)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv1_buffer[bf_index+3] = (2.0*r+3.0*c+((too+to+c+3)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									gama1[too+to]=1.0;
									gama1[too+to+1]=1.0;
									gama1[too+to+2]=1.0;
									gama1[too+to+3]=1.0;
									beta1[too+to]=0;
									beta1[too+to+1]=0;
									beta1[too+to+2]=0;
									beta1[too+to+3]=0;
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
									weights_conv2_buffer[bf_index] = (2.0*r+3.0*c+((too+to+c)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv2_buffer[bf_index+1] =(2.0*r+3.0*c+((too+to+c+1)%11) - ((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv2_buffer[bf_index+2] = (2.0*r+3.0*c+((too+to+c+2)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv2_buffer[bf_index+3] = (2.0*r+3.0*c+((too+to+c+3)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									gama2[too+to]=1.0;
									gama2[too+to+1]=1.0;
									gama2[too+to+2]=1.0;
									gama2[too+to+3]=1.0;
									beta2[too+to]=0;
									beta2[too+to+1]=0;
									beta2[too+to+2]=0;
									beta2[too+to+3]=0;
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
									weights_conv3_buffer[bf_index] = (2.0*r+3.0*c+((too+to+c)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv3_buffer[bf_index+1] =(2.0*r+3.0*c+((too+to+c+1)%11) - ((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv3_buffer[bf_index+2] = (2.0*r+3.0*c+((too+to+c+2)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv3_buffer[bf_index+3] = (2.0*r+3.0*c+((too+to+c+3)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									gama3[too+to]=1.0;
									gama3[too+to+1]=1.0;
									gama3[too+to+2]=1.0;
									gama3[too+to+3]=1.0;
									beta3[too+to]=0;
									beta3[too+to+1]=0;
									beta3[too+to+2]=0;
									beta3[too+to+3]=0;
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
									weights_conv4_buffer[bf_index] = (2.0*r+3.0*c+((too+to+c)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv4_buffer[bf_index+1] =(2.0*r+3.0*c+((too+to+c+1)%11) - ((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv4_buffer[bf_index+2] = (2.0*r+3.0*c+((too+to+c+2)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv4_buffer[bf_index+3] = (2.0*r+3.0*c+((too+to+c+3)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									gama4[too+to]=1.0;
									gama4[too+to+1]=1.0;
									gama4[too+to+2]=1.0;
									gama4[too+to+3]=1.0;
									beta4[too+to]=0;
									beta4[too+to+1]=0;
									beta4[too+to+2]=0;
									beta4[too+to+3]=0;
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
									weights_conv5_buffer[bf_index] = (2.0*r+3.0*c+((too+to+c)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv5_buffer[bf_index+1] =(2.0*r+3.0*c+((too+to+c+1)%11) - ((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv5_buffer[bf_index+2] = (2.0*r+3.0*c+((too+to+c+2)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv5_buffer[bf_index+3] = (2.0*r+3.0*c+((too+to+c+3)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									gama5[too+to]=1.0;
									gama5[too+to+1]=1.0;
									gama5[too+to+2]=1.0;
									gama5[too+to+3]=1.0;
									beta5[too+to]=0;
									beta5[too+to+1]=0;
									beta5[too+to+2]=0;
									beta5[too+to+3]=0;
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
									weights_conv6_buffer[bf_index] = (2.0*r+3.0*c+((too+to+c)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv6_buffer[bf_index+1] =(2.0*r+3.0*c+((too+to+c+1)%11) - ((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv6_buffer[bf_index+2] = (2.0*r+3.0*c+((too+to+c+2)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv6_buffer[bf_index+3] = (2.0*r+3.0*c+((too+to+c+3)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									gama6[too+to]=1.0;
									gama6[too+to+1]=1.0;
									gama6[too+to+2]=1.0;
									gama6[too+to+3]=1.0;
									beta6[too+to]=0;
									beta6[too+to+1]=0;
									beta6[too+to+2]=0;
									beta6[too+to+3]=0;
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
									weights_conv7_buffer[bf_index] = (2.0*r+3.0*c+((too+to+c)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv7_buffer[bf_index+1] =(2.0*r+3.0*c+((too+to+c+1)%11) - ((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv7_buffer[bf_index+2] = (2.0*r+3.0*c+((too+to+c+2)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv7_buffer[bf_index+3] = (2.0*r+3.0*c+((too+to+c+3)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									gama7[too+to]=1.0;
									gama7[too+to+1]=1.0;
									gama7[too+to+2]=1.0;
									gama7[too+to+3]=1.0;
									beta7[too+to]=0;
									beta7[too+to+1]=0;
									beta7[too+to+2]=0;
									beta7[too+to+3]=0;
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
									weights_conv8_buffer[bf_index] = (2.0*r+3.0*c+((too+to+c)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv8_buffer[bf_index+1] =(2.0*r+3.0*c+((too+to+c+1)%11) - ((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv8_buffer[bf_index+2] = (2.0*r+3.0*c+((too+to+c+2)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv8_buffer[bf_index+3] = (2.0*r+3.0*c+((too+to+c+3)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									gama8[too+to]=1.0;
									gama8[too+to+1]=1.0;
									gama8[too+to+2]=1.0;
									gama8[too+to+3]=1.0;
									beta8[too+to]=0;
									beta8[too+to+1]=0;
									beta8[too+to+2]=0;
									beta8[too+to+3]=0;
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
									weights_conv9_buffer[bf_index] = (2.0*r+3.0*c+((too+to+c)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv9_buffer[bf_index+1] =(2.0*r+3.0*c+((too+to+c+1)%11) - ((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv9_buffer[bf_index+2] = (2.0*r+3.0*c+((too+to+c+2)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv9_buffer[bf_index+3] = (2.0*r+3.0*c+((too+to+c+3)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									gama9[too+to]=1.0;
									gama9[too+to+1]=1.0;
									gama9[too+to+2]=1.0;
									gama9[too+to+3]=1.0;
									beta9[too+to]=0;
									beta9[too+to+1]=0;
									beta9[too+to+2]=0;
									beta9[too+to+3]=0;
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
									weights_conv10_buffer[bf_index] = (2.0*r+3.0*c+((too+to+c)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv10_buffer[bf_index+1] =(2.0*r+3.0*c+((too+to+c+1)%11) - ((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv10_buffer[bf_index+2] = (2.0*r+3.0*c+((too+to+c+2)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv10_buffer[bf_index+3] = (2.0*r+3.0*c+((too+to+c+3)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									gama10[too+to]=1.0;
									gama10[too+to+1]=1.0;
									gama10[too+to+2]=1.0;
									gama10[too+to+3]=1.0;
									beta10[too+to]=0;
									beta10[too+to+1]=0;
									beta10[too+to+2]=0;
									beta10[too+to+3]=0;
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
									weights_conv11_buffer[bf_index] = (2.0*r+3.0*c+((too+to+c)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv11_buffer[bf_index+1] =(2.0*r+3.0*c+((too+to+c+1)%11) - ((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv11_buffer[bf_index+2] = (2.0*r+3.0*c+((too+to+c+2)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv11_buffer[bf_index+3] = (2.0*r+3.0*c+((too+to+c+3)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									gama11[too+to]=1.0;
									gama11[too+to+1]=1.0;
									gama11[too+to+2]=1.0;
									gama11[too+to+3]=1.0;
									beta11[too+to]=0;
									beta11[too+to+1]=0;
									beta11[too+to+2]=0;
									beta11[too+to+3]=0;
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
									weights_conv12_buffer[bf_index] = (2.0*r+3.0*c+((too+to+c)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv12_buffer[bf_index+1] =(2.0*r+3.0*c+((too+to+c+1)%11) - ((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv12_buffer[bf_index+2] = (2.0*r+3.0*c+((too+to+c+2)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv12_buffer[bf_index+3] = (2.0*r+3.0*c+((too+to+c+3)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									gama12[too+to]=1.0;
									gama12[too+to+1]=1.0;
									gama12[too+to+2]=1.0;
									gama12[too+to+3]=1.0;
									beta12[too+to]=0;
									beta12[too+to+1]=0;
									beta12[too+to+2]=0;
									beta12[too+to+3]=0;
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
									weights_conv13_buffer[bf_index] = (2.0*r+3.0*c+((too+to+c)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv13_buffer[bf_index+1] =(2.0*r+3.0*c+((too+to+c+1)%11) - ((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv13_buffer[bf_index+2] = (2.0*r+3.0*c+((too+to+c+2)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									weights_conv13_buffer[bf_index+3] = (2.0*r+3.0*c+((too+to+c+3)%11) -((tii+ti+r)%23)+1.0)/mul;//output.data;
									gama13[too+to]=1.0;
									gama13[too+to+1]=1.0;
									gama13[too+to+2]=1.0;
									gama13[too+to+3]=1.0;
									beta13[too+to]=0;
									beta13[too+to+1]=0;
									beta13[too+to+2]=0;
									beta13[too+to+3]=0;
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
									weights_fc1_buffer[bf_index] = (2.0*r+3.0*c+((too+to+c)% 11) -((tii+ti+r)%23)+1)/mul;//output.data;
									weights_fc1_buffer[bf_index+1] = (2.0*r+3.0*c+((too+to+c+1)% 11) - ((tii+ti+r)%23)+1)/mul;//output.data;
									weights_fc1_buffer[bf_index+2] =(2.0*r+3.0*c+((too+to+c+2)% 11) -((tii+ti+r)%23)+1)/mul;//output.data;
									weights_fc1_buffer[bf_index+3] = (2.0*r+3.0*c+((too+to+c+3)% 11) -((tii+ti+r)%23)+1)/mul;//output.data;
									bf_index+=4;
								}
							}
						}
					}
				}
			}
		}


		cout <<"buffer"<< endl;
		cout << input_x_buffer[0] << endl;
		cout << input_x_buffer[1] << endl;
		cout << weights_conv1_buffer[0] << endl;
		cout << weights_conv1_buffer[64*3*3*3-1] << endl;
		cout << weights_conv1_buffer[4] << endl;
		cout << weights_conv2_buffer[0] << endl;
		cout << weights_fc1_buffer[0] << endl;
		//cout << weights_fc2_buffer[0] << endl;
		//cout << weights_fc3_buffer[0] << endl;


	convfp_3_64_224(1,input_x_buffer,weights_conv1_buffer,output1hat,gama1,beta1,namuda1,output1);
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



	cout<<"after loopM:"<<endl;
	//FPGA_DATA loss1[bs*10];
	FPGA_DATA sum=0.0;
	FPGA_DATA target0[bs];
	target0[0]=2;
	target0[1]=33;
//	target0[2]=2;
//	target0[3]=0;
	FPGA_DATA target[bs][1000];
	for(int b=0; b<bs;b++){
		for (int mo=0; mo<1000;mo++){
			//cout << output3[mo] << endl;
			if(mo==target0[b])
				target[b][mo]=1.0;
			else
				target[b][mo]=0.0;
		}
	}
    cout<<tb_flag<<endl;
    //cout<<"loss1:"<<endl;
    int loss_addr=0;
    for(int b=0; b<bs;b++){
    	cout<<"batch:"<<b<<"for loss1"<<endl;
    	sum=0.0;
		for (int mo=0; mo<1000;mo++){
				loss1[loss_addr+mo]	= exp(output20[loss_addr+mo]);
				sum+=loss1[loss_addr+mo];
		}
		for (int mo=0; mo<1000;mo++){
			loss1[loss_addr+mo] = (loss1[loss_addr+mo]/sum-target[b][mo])/bs;
			cout<<loss1[loss_addr+mo]<<endl;
		}
		loss_addr+=1000;
	}


    convbp_512_1000_1(2,loss1,weights_fc1_buffer,loss2);
    convwu_512_1000_1(1,output19,weights_fc1_buffer,loss1);
    avgpoolbp_512_1(3,loss2,loss3);
    maxpoolbp_512_7(4,loss3,pool5index,output17,loss4);
    convbp_512_512_14(5,5,loss4,weights_conv13_buffer,output17hat,gama13,beta13,namuda13,output16,loss5);
    convwu_512_512_14(13,output16,weights_conv13_buffer,loss4,weights_conv13);
    convbp_512_512_14(6,5,loss5,weights_conv12_buffer,output16hat,gama12,beta12,namuda12,output15,loss6);
    convwu_512_512_14(12,output15,weights_conv12_buffer,loss5,weights_conv12);
    convbp_512_512_14(7,4,loss6,weights_conv11_buffer,output15hat,gama11,beta11,namuda11,output14,loss7);
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


	return 0;

}
