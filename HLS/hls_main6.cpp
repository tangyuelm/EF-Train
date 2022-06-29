#include "source.h"

//int min (int a,int b){
//	int y;
//	if (a>b)
//			y = b;
//		else
//			y=a;
//		return y;
//
//}
void Initialize_OFM( FPGA_DATA Output1[Tm][Trtc],//[Tr][Tc],
					FPGA_DATA Output1_DB[Tm][Trtc],//[Tr][Tc],
					int8 custom_Tr,
					int8 custom_Tc
					){
	for(int8 trr=0;trr<custom_Tr;trr++){
	#pragma HLS loop_tripcount min=8 max=8 avg=8
		for(int8 tcc=0;tcc<custom_Tc;tcc++){
	#pragma HLS loop_tripcount min=24 max=24 avg=24
	#pragma HLS PIPELINE II=1
			int14 trtcindex=trr*custom_Tc+tcc;
			for(int14 too=0;too<Tm;too++){
				//Output1[too][trr - row][tcc - col]=0;
				Output1[too][trtcindex]=0;
				Output1_DB[too][trtcindex]=0;
			}
		}
	}
}

void Initialize_IFM( FPGA_DATA IFM[Tn][Trtcin],//[Tr][Tc],
					FPGA_DATA IFM_DB[Tn][Trtcin],//[Tr][Tc],
					int8 custom_Trin,
					int8 custom_Tcin
					){
	for(int8 trr=0;trr<custom_Trin;trr++){
	#pragma HLS loop_tripcount min=8 max=8 avg=8
		for(int8 tcc=0;tcc<custom_Tcin;tcc++){
	#pragma HLS loop_tripcount min=24 max=24 avg=24
	#pragma HLS PIPELINE II=1
			int14 trtcindex=trr*custom_Tcin+tcc;
			for(int14 too=0;too<Tn;too++){
				//Output1[too][trr - row][tcc - col]=0;
				IFM[too][trtcindex]=0;
				IFM_DB[too][trtcindex]=0;
			}
		}
	}
}



void Initialize_WEIGHTS( FPGA_DATA Output2[Tm][Tn][TiTo][K][K],
						FPGA_DATA Output2_DB[Tm][Tn][TiTo][K][K]
						//int14 custom_Tib,
						//int4 custom_k,
						//int14 to_index
					){
	//int14 uppertito=custom_Tib*to_index;
//	if(custom_k>1){//common conv
//		for(int14 b=0;b<TiTo;b++){
//
//		//for(int14 b=0;b<uppertito;b++){
//		//#pragma HLS loop_tripcount min=9 max=9 avg=9
//				//int8 k_index=i*custom_k+j;
//		for(int4 i=0;i<custom_k;i++){
//			#pragma HLS loop_tripcount min=3 max=3 avg=3
//				for(int4 j=0;j<custom_k;j++){
//					#pragma HLS loop_tripcount min=3 max=3 avg=3
//							//int8 k_index=i*custom_k+j;
//					#pragma HLS PIPELINE II=1
//					for(int14 too=0;too<Tm;too++){
//						for(int14 tii=0;tii<Tn;tii++){
//							 Output2[too][tii][b][i][j]=0;
//							 Output2_DB[too][tii][b][i][j]=0;
//						}
//					}
//				}
//			}
//		}
//	}
//	else{//FC
	for(int14 b=0;b<TiTo;b++){
	//for(int14 b=0;b<uppertito;b++){
		//#pragma HLS loop_tripcount min=9 max=9 avg=9
				//int8 k_index=i*custom_k+j;
		for(int4 i=0;i<K;i++){
		#pragma HLS loop_tripcount min=3 max=3 avg=3
			for(int4 j=0;j<K;j++){
				#pragma HLS loop_tripcount min=3 max=3 avg=3
						//int8 k_index=i*custom_k+j;
				#pragma HLS PIPELINE II=1
				for(int14 too=0;too<Tm;too++){
					for(int14 tii=0;tii<Tn;tii++){
						 Output2[too][tii][b][i][j]=0;
						 Output2_DB[too][tii][b][i][j]=0;
					}
				}
			}
		}
	}
}
	//}




void LOAD_poolfor(hls::stream<DMA_DATA_128> &dma_IFM,
		//Relu_index Reluindex1[M1][R1][C1],
		//int8 Relulayerin,
		FPGA_DATA IFM[Tn][Trtcin],
		ap_uint<4> state,
		//int state,
		int14 ti,
		int8 row,
		//int8 col,
		int14 M,
		int8 R,
		int8 C,
		//int8 custom_Tr,
		//int8 custom_Tc,
		int4 pool_s,
		int4 pool_k,
		int8 custom_Tr,
		int8 custom_Tc,
		int8 R_in,
		int8 C_in
		//int8& flag
		){
	DMA_DATA_128 ifm_input_data;
	//if(state[1]==1){//11Xforward
	int8 i_upper=(custom_Tr-1)*pool_s+pool_k;
	int8 j_upper=(custom_Tc-1)*pool_s+pool_k;
	int8 i_current_sub=(R - 1) * pool_s + pool_k;
	int8 j_current_sub=(C - 1) * pool_s + pool_k;
	for(int8 i=0;i<i_upper;i++){
		#pragma HLS loop_tripcount min=24 max=24 avg=24
		for(int8 j=0;j<j_upper;j++){
		#pragma HLS loop_tripcount min=24 max=24 avg=24
			for(int14 tii=0;tii<Tn;tii+=4){
		#pragma HLS loop_tripcount min=3 max=3 avg=3
		#pragma HLS PIPELINE II=1
		#pragma HLS dependence variable=IFM intra false
				int8 i_current=i+row*pool_s;
				int14 tii_current = tii+ti;
				int14 trtcindex=i*j_upper+j;
				//FPGA_DATA ifm_input;
				if(	i_current >= 0 && i_current <i_current_sub && j <j_current_sub && tii_current<M ){
					ifm_input_data=dma_IFM.read();
					IFM[tii][trtcindex] = ifm_input_data.data.data1;
					IFM[tii+1][trtcindex] = ifm_input_data.data.data2;
					IFM[tii+2][trtcindex] = ifm_input_data.data.data3;
					IFM[tii+3][trtcindex] = ifm_input_data.data.data4;
					//flag=122;
				}
				else{
					IFM[tii][trtcindex] = 0;
					IFM[tii+1][trtcindex] = 0;
					IFM[tii+2][trtcindex] = 0;
					IFM[tii+3][trtcindex] = 0;
					//flag=IFM[tii][i][j];
				}
				//IFM[too][i][j] = ifm_input;
			}
		}
	}
}


void LOAD_poolback(hls::stream<DMA_DATA_128> &dma_IFM,
		//Relu_index Reluindex1[M1][R1][C1],
		//int8 Relulayerin,
		FPGA_DATA OFM[Tm][Trtc],
//		ap_uint<4> state,
		//int state,
		int14 ti,
		int8 row,
		//int8 col,
		int14 M,
		int8 R,
		int8 C,
		//int8 custom_Tr,
		//int8 custom_Tc,
		int4 pool_s,
		int4 pool_k,
		int8 custom_Tr,
		int8 custom_Tc,
		int8 R_in,
		int8 C_in
		//int8& flag
		){
//	else{//10Xbackward
	DMA_DATA_128 ifm_input_data;
//		if(state[3]==0){//common pool
			for(int8 trr=0;trr<custom_Tr;trr++){
			#pragma HLS loop_tripcount min=12 max=12 avg=12
				int8 trr_current=trr+row;
				for(int8 tcc=0;tcc<custom_Tc;tcc++){
			#pragma HLS loop_tripcount min=12 max=12 avg=12
					//int8 tcc_current=tcc;
					for(int14 tii=0;tii<Tn;tii+=4){
			#pragma HLS loop_tripcount min=3 max=3 avg=3
						int14 tii_current=tii+ti;
			#pragma HLS PIPELINE II=1
			#pragma HLS dependence variable=IFM intra false
						//FPGA_DATA ifm_input;

						int14 trtcindex=trr*custom_Tc+tcc;
						if(tii_current<M && trr_current<R && tcc<C ){
							ifm_input_data=dma_IFM.read();
							OFM[tii][trtcindex] = ifm_input_data.data.data1;
							OFM[tii+1][trtcindex] = ifm_input_data.data.data2;
							OFM[tii+2][trtcindex] = ifm_input_data.data.data3;
							OFM[tii+3][trtcindex] = ifm_input_data.data.data4;


							//flag=122;
						}
						else{
							OFM[tii][trtcindex] = 0;
							OFM[tii+1][trtcindex] = 0;
							OFM[tii+2][trtcindex] = 0;
							OFM[tii+3][trtcindex] = 0;
							//flag=IFM[tii][i][j];
						}
						//IFM[too][trr][tcc] = ifm_input;
					}
				}
			}
//		}
//		else{//needs to view
//			int8 RmulC=R*C;
//			int8 TrmulTc=custom_Tr*custom_Tc;
//			for(int14 too=0;too<Tm;too++){
//			#pragma HLS loop_tripcount min=6 max=6 avg=6
//				int14 too_current=too+to;
//				//for(int8 trr=0;trr<custom_Tr;trr++){
//				//#pragma HLS loop_tripcount min=4 max=4 avg=4
//				//	int8 trr_current=trr+row;
//				//	for(int8 tcc=0;tcc<custom_Tc;tcc+=4){
//				//#pragma HLS loop_tripcount min=4 max=4 avg=4
//						//int8 tcc_current=tcc;
//				for(int8 trc=0;trc<TrmulTc;trc+=4){
//			#pragma HLS loop_tripcount min=36 max=36 avg=36
//			#pragma HLS PIPELINE //II=1
//			#pragma HLS dependence variable=IFM intra false
//						//FPGA_DATA ifm_input;
//						//int14 trtcindex=trr*custom_Tc+tcc;
//						int14 too1=too;
//						int8 trc1=trc+1;
//						//int8 tcc1=tcc+1;
//						if(trc1>=RmulC){
//							trc1=0;
//							too1++;
//						}
////						if(trr1+row>=R){
////							trr1=0;
////							too1++;
////						}
//						int14 too2=too1;
//						int8 trc2=trc1+1;
//						//int8 tcc1=tcc+1;
//						if(trc2>=RmulC){
//							trc2=0;
//							too2++;
//						}
//						int14 too3=too2;
//						int8 trc3=trc2+1;
//						//int8 tcc1=tcc+1;
//						if(trc3>=RmulC){
//							trc3=0;
//							too3++;
//						}
//						if(too_current<M && trc<RmulC){
//
//							ifm_input_data=dma_IFM.read();
//							OFM[too][trc] = ifm_input_data.data.data1;
//							OFM[too1][trc1] = ifm_input_data.data.data2;
//							OFM[too2][trc2] = ifm_input_data.data.data3;
//							OFM[too3][trc3] = ifm_input_data.data.data4;
//							//flag=122;
//						}
//						else{
//							OFM[too][trc] = 0;
//							OFM[too1][trc1] = 0;
//							OFM[too2][trc2] = 0;
//							OFM[too3][trc3] = 0;
//							//flag=IFM[tii][i][j];
//						}
//						//IFM[too][trr][tcc] = ifm_input;
//					}
//				//}
//			}
//		}
}


void STORE_poolfor( FPGA_DATA OFM[Tm][Trtc],
				//Relu_index Reluindex1[4][M1][R1][C1],
				//Relu_index Reluindex3[4][M3][R3][C3],
				//int4 Relulayerin,
				//int4 Relulayerout,
				//int state,
				ap_uint<4> state,
				//int8 batch_size,
				//int8 ba,
				int14 ti,
				int8 row,
				//int8 col,
				int14 M,
				int8 R,
				int8 C,
				//int8 custom_Tr,
				//int8 custom_Tc,
				int4 pool_s,
				int4 pool_k,
				int8 custom_Tr,
				int8 custom_Tc,
				//hls::stream<DMA_DATA_128> &dma_OFM,
				hls::stream<DMA_DATA_128> &dma_Output,
				int8 R_in,
				int8 C_in){
				//int8& flag){
	DMA_DATA_128 output_dma_O_data;
	FPGA_DATA avgk=pool_k*pool_k;
	//DMA_DATA_128 ofm_dma_O_data;
	//int8 i_clear=custom_Tr;
	//Relu_index Reluindex_data;
//	if(state[1]==1){//11Xforward
//		if(state[3]==0){//nomal store
			for(int8 trr=0;trr<custom_Tr;trr++){
			#pragma HLS loop_tripcount min=12 max=12 avg=12
				int8 trr_current=trr+row;
				for(int8 tcc=0;tcc<custom_Tc;tcc++){
			#pragma HLS loop_tripcount min=12 max=12 avg=12
					//flag=150;
					//int8 tcc_current=tcc+col;
					for(int14 tii=0;tii<Tn;tii+=4){
			#pragma HLS loop_tripcount min=3 max=3 avg=3
						//int8 i_current=i+row*pool_s;
						int14 tii_current = tii+ti;
			#pragma HLS PIPELINE II=1
			//#pragma HLS dependence variable=Output1 intra false
			#pragma HLS dependence variable=OFM intra false
						int14 trtcindex=trr*custom_Tc+tcc;
						if(	trr_current<R && tcc<C && tii_current<M ){
							FPGA_DATA Read_Output1;
							FPGA_DATA Read_Output2;
							FPGA_DATA Read_Output3;
							FPGA_DATA Read_Output4;
							FPGA_DATA Write_dma_O1;
							FPGA_DATA Write_dma_O2;
							FPGA_DATA Write_dma_O3;
							FPGA_DATA Write_dma_O4;
							//Output[too][trr][tcc]=0;
							if(state[3]==0){//maxpool
								Read_Output1 = OFM[tii][trtcindex];
								Read_Output2 = OFM[tii+1][trtcindex];
								Read_Output3 = OFM[tii+2][trtcindex];
								Read_Output4 = OFM[tii+3][trtcindex];
							}
							else{//avgpool
								Read_Output1 = OFM[tii][trtcindex]/avgk;
								Read_Output2 = OFM[tii+1][trtcindex]/avgk;
								Read_Output3 = OFM[tii+2][trtcindex]/avgk;
								Read_Output4 = OFM[tii+3][trtcindex]/avgk;
							}

							if(state[0]==1){//XX1 relu after conv need to be processed
								 //if (Output1[too - to][trr - row][tcc - col] > 0){
								if (Read_Output1 > 0){
									//flag=OFM[too][trr][tcc];
									Write_dma_O1 =Read_Output1;// Output1[too - to][trr - row][tcc - col];
									//Reluindex_data = 1;
								}
								else{
									//flag=151;
									Write_dma_O1 = 0;//Output1[too - to][trr - row][tcc - col];
									//Reluindex_data = 0;
								}
								if (Read_Output2 > 0){
									//flag=OFM[too][trr][tcc];
									Write_dma_O2 =Read_Output2;// Output1[too - to][trr - row][tcc - col];
									//Reluindex_data = 1;
								}
								else{
									//flag=151;
									Write_dma_O2 = 0;//Output1[too - to][trr - row][tcc - col];
									//Reluindex_data = 0;
								}
								if (Read_Output3 > 0){
									//flag=OFM[too][trr][tcc];
									Write_dma_O3 =Read_Output3;// Output1[too - to][trr - row][tcc - col];
									//Reluindex_data = 1;
								}
								else{
									//flag=151;
									Write_dma_O3 = 0;//Output1[too - to][trr - row][tcc - col];
									//Reluindex_data = 0;
								}
								if (Read_Output4 > 0){
									//flag=OFM[too][trr][tcc];
									Write_dma_O4 =Read_Output4;// Output1[too - to][trr - row][tcc - col];
									//Reluindex_data = 1;
								}
								else{
									//flag=151;
									Write_dma_O4 = 0;//Output1[too - to][trr - row][tcc - col];
									//Reluindex_data = 0;
								}
							}
							else{
								//flag=153;
								Write_dma_O1 = Read_Output1;//Output1[too - to][trr - row][tcc - col];
								Write_dma_O2 = Read_Output2;
								Write_dma_O3 = Read_Output3;
								Write_dma_O4 = Read_Output4;
							}
							output_dma_O_data.data.data1=Write_dma_O1;
							output_dma_O_data.data.data2=Write_dma_O2;
							output_dma_O_data.data.data3=Write_dma_O3;
							output_dma_O_data.data.data4=Write_dma_O4;
							//if(custom_Tr>=R){//feature map small, can load and store together
							if( trr_current+1>=R && tcc+1>=C && tii_current+4>=M){
								output_dma_O_data.last=1;
							}
							else{
								output_dma_O_data.last=0;
							}
							dma_Output.write(output_dma_O_data);
						}
					}
				}
			}
//		}
//		else{//needs to view for fc
//			int8 RmulC=R*C;
//			int8 TrmulTc=custom_Tr*custom_Tc;
//			for(int14 too=0;too<Tm;too++){
//			#pragma HLS loop_tripcount min=6 max=6 avg=6
////				for(int8 trr=0;trr<custom_Tr;trr++){
////				#pragma HLS loop_tripcount min=4 max=4 avg=4
////					int8 trr_current=trr+row;
////					for(int8 tcc=0;tcc<custom_Tc;tcc+=4){
////				#pragma HLS loop_tripcount min=4 max=4 avg=4
//				for(int8 trc=0;trc<TrmulTc;trc+=4){
//				#pragma HLS loop_tripcount min=36 max=36 avg=36
//					//flag=150;
//					//int8 tcc_current=tcc+col;
//						//int8 i_current=i+row*pool_s;
//						int14 too_current = too+to;
//			#pragma HLS PIPELINE //II=1
//			//#pragma HLS dependence variable=Output1 intra false
//			#pragma HLS dependence variable=OFM intra false
//
//						if(	trc<RmulC && too_current<M ){
//							//int14 trtcindex=trc;
//							int14 too1=too;
//							int8 trc1=trc+1;
//							//int8 tcc1=tcc+1;
//							if(trc1>=RmulC){
//								trc1=0;
//								too1++;
//							}
//							int14 too2=too1;
//							int8 trc2=trc1+1;
//							//int8 tcc1=tcc+1;
//							if(trc2>=RmulC){
//								trc2=0;
//								too2++;
//							}
//							int14 too3=too2;
//							int8 trc3=trc2+1;
//							//int8 tcc1=tcc+1;
//							if(trc3>=RmulC){
//								trc3=0;
//								too3++;
//							}
//							FPGA_DATA Read_Output1;
//							FPGA_DATA Read_Output2;
//							FPGA_DATA Read_Output3;
//							FPGA_DATA Read_Output4;
//							FPGA_DATA Write_dma_O1;
//							FPGA_DATA Write_dma_O2;
//							FPGA_DATA Write_dma_O3;
//							FPGA_DATA Write_dma_O4;
//							Read_Output1 = OFM[too][trc];
//							Read_Output2 = OFM[too1][trc1];
//							Read_Output3 = OFM[too2][trc2];
//							Read_Output4 = OFM[too3][trc3];
//							if(state[0]==1){//XX1 relu after conv need to be processed
//								 //if (Output1[too - to][trr - row][tcc - col] > 0){
//								if (Read_Output1 > 0){
//									//flag=OFM[too][trr][tcc];
//									Write_dma_O1 =Read_Output1;// Output1[too - to][trr - row][tcc - col];
//									//Reluindex_data = 1;
//								}
//								else{
//									//flag=151;
//									Write_dma_O1 = 0;//Output1[too - to][trr - row][tcc - col];
//									//Reluindex_data = 0;
//								}
//								if (Read_Output2 > 0){
//									//flag=OFM[too][trr][tcc];
//									Write_dma_O2 =Read_Output2;// Output1[too - to][trr - row][tcc - col];
//									//Reluindex_data = 1;
//								}
//								else{
//									//flag=151;
//									Write_dma_O2 = 0;//Output1[too - to][trr - row][tcc - col];
//									//Reluindex_data = 0;
//								}
//								if (Read_Output3 > 0){
//									//flag=OFM[too][trr][tcc];
//									Write_dma_O3 =Read_Output3;// Output1[too - to][trr - row][tcc - col];
//									//Reluindex_data = 1;
//								}
//								else{
//									//flag=151;
//									Write_dma_O3 = 0;//Output1[too - to][trr - row][tcc - col];
//									//Reluindex_data = 0;
//								}
//								if (Read_Output4 > 0){
//									//flag=OFM[too][trr][tcc];
//									Write_dma_O4 =Read_Output4;// Output1[too - to][trr - row][tcc - col];
//									//Reluindex_data = 1;
//								}
//								else{
//									//flag=151;
//									Write_dma_O4 = 0;//Output1[too - to][trr - row][tcc - col];
//									//Reluindex_data = 0;
//								}
//							}
//							else{
//								//flag=153;
//								Write_dma_O1 = Read_Output1;//Output1[too - to][trr - row][tcc - col];
//								Write_dma_O2 = Read_Output2;
//								Write_dma_O3 = Read_Output3;
//								Write_dma_O4 = Read_Output4;
//							}
//							output_dma_O_data.data.data1=Write_dma_O1;
//							output_dma_O_data.data.data2=Write_dma_O2;
//							output_dma_O_data.data.data3=Write_dma_O3;
//							output_dma_O_data.data.data4=Write_dma_O4;
//							//if(custom_Tr>=R){//feature map small, can load and store together
//							//if(ba==custom_batch-1 && trr_current==R-1 && tcc==C-2 && too_current==M-1){
//							//if(trr_current==R-1 && tcc==C-2 && too_current==M-1){
//							if(trc+4>=RmulC && too_current+1>=M){
//								output_dma_O_data.last=1;
//							}
//							else{
//								output_dma_O_data.last=0;
//							}
//							dma_Output.write(output_dma_O_data);
//						}
//					}
//				//}
//			}
//		}
		for(int8 trr=0;trr<custom_Tr;trr++){
		#pragma HLS loop_tripcount min=24 max=24 avg=24
			for(int8 tcc=0;tcc<custom_Tc;tcc++){
		#pragma HLS loop_tripcount min=24 max=24 avg=24
		#pragma HLS PIPELINE II=1
				for(int14 tii=0;tii<Tn;tii++){
					int14 trtcindex=trr*custom_Tc+tcc;
					//Output1[too][trr - row][tcc - col]=0;
					OFM[tii][trtcindex]=0;
				}
			}
		}
}


void STORE_poolback( FPGA_DATA IFM[Tm][Trtcin],
				//Relu_index Reluindex1[4][M1][R1][C1],
				//Relu_index Reluindex3[4][M3][R3][C3],
				//int4 Relulayerin,
				//int4 Relulayerout,
				//int state,
				ap_uint<4> state,
				int8 custom_batch,
				int8 batch_size,
				int8 ba,
				int14 ti,
				int8 row,
				//int8 col,
				int14 M,
				int8 R,
				int8 C,
				//int8 custom_Tr,
				//int8 custom_Tc,
				int4 pool_s,
				int4 pool_k,
				int8 custom_Tr,
				int8 custom_Tc,
				hls::stream<DMA_DATA_128> &dma_OFM,
				hls::stream<DMA_DATA_128> &dma_Output,
				int8 R_in,
				int8 C_in){
				//int8& flag){
	DMA_DATA_128 output_dma_O_data;
	DMA_DATA_128 ofm_dma_O_data;
	//int8 i_clear;

//else{//10Xbackward
		int8 i_upper=(custom_Tr-1)*pool_s+pool_k;
		int8 j_upper=(custom_Tc-1)*pool_s+pool_k;
		int8 i_current_sub=(R - 1) * pool_s + pool_k;
		int8 j_current_sub=(C - 1) * pool_s + pool_k;
		FPGA_DATA avgk=pool_k*pool_k;
		//int8 i_current=0;
		//i_clear=i_upper;
		for(int8 i=0;i<i_upper;i++){
		#pragma HLS loop_tripcount min=24 max=24 avg=24
			for(int8 j=0;j<j_upper;j++){
		#pragma HLS loop_tripcount min=24 max=24 avg=24
				for(int14 tii=0;tii<Tn;tii+=4){
		#pragma HLS loop_tripcount min=6 max=6 avg=6
		#pragma HLS PIPELINE II=1

		#pragma HLS dependence variable=IFM intra false
					int14 tii_current=tii+ti;
					int8 i_current=i+row*pool_s;
					if(tii_current<M && i_current <i_current_sub && j<j_current_sub){
						int14 trtcindex=i*j_upper+j;
						FPGA_DATA Read_Output1;
						FPGA_DATA Read_Output2;
						FPGA_DATA Read_Output3;
						FPGA_DATA Read_Output4;
						FPGA_DATA Write_dma_O1;
						FPGA_DATA Write_dma_O2;
						FPGA_DATA Write_dma_O3;
						FPGA_DATA Write_dma_O4;
						if(state[3]==0){//maxpool
							Read_Output1 =IFM[tii][trtcindex];
							Read_Output2 = IFM[tii+1][trtcindex];
							Read_Output3 = IFM[tii+2][trtcindex];
							Read_Output4 = IFM[tii+3][trtcindex];
						}
						else{//avgpool
							Read_Output1 = IFM[tii][trtcindex]/avgk;
							Read_Output2 = IFM[tii+1][trtcindex]/avgk;
							Read_Output3 = IFM[tii+2][trtcindex]/avgk;
							Read_Output4 = IFM[tii+3][trtcindex]/avgk;
						}
						if(state[0]==1){//XX1 relu after conv need to be processed
							FPGA_DATA Relu_activ1;
							FPGA_DATA Relu_activ2;
							FPGA_DATA Relu_activ3;
							FPGA_DATA Relu_activ4;
							ofm_dma_O_data=dma_OFM.read();
							Relu_activ1=ofm_dma_O_data.data.data1;
							Relu_activ2=ofm_dma_O_data.data.data2;
							Relu_activ3=ofm_dma_O_data.data.data3;
							Relu_activ4=ofm_dma_O_data.data.data4;
							if (Relu_activ1 <= 0){
								Write_dma_O1 = 0;
							 }
							else{
								//flag=156;
								Write_dma_O1 = Read_Output1;//Output1[too - to][trr - row][tcc - col];
							}
							if (Relu_activ2 <= 0){
								Write_dma_O2 = 0;
							 }
							else{
								//flag=156;
								Write_dma_O2 = Read_Output2;//Output1[too - to][trr - row][tcc - col];
							}
							if (Relu_activ3 <= 0){
								Write_dma_O3 = 0;
							 }
							else{
								//flag=156;
								Write_dma_O3 = Read_Output3;//Output1[too - to][trr - row][tcc - col];
							}
							if (Relu_activ4 <= 0){
								Write_dma_O4 = 0;
							 }
							else{
								//flag=156;
								Write_dma_O4 = Read_Output4;//Output1[too - to][trr - row][tcc - col];
							}
						}
						else{
							//flag=153;
							Write_dma_O1 = Read_Output1;//Output1[too - to][trr - row][tcc - col];
							Write_dma_O2 = Read_Output2;
							Write_dma_O3 = Read_Output3;
							Write_dma_O4 = Read_Output4;
						}
						output_dma_O_data.data.data1=Write_dma_O1;
						output_dma_O_data.data.data2=Write_dma_O2;
						output_dma_O_data.data.data3=Write_dma_O3;
						output_dma_O_data.data.data4=Write_dma_O4;
						//if(custom_Tr>=R){//feature map small, can load and store together
						//if(((batch_size<64 && (ba%custom_batch)==(custom_batch-1)|| (custom_batch>=64 && ba==31)) && (i_current==R_in-1 && j==C_in-1 && too_current==M-2)){
						if((ba+1>=batch_size||(ba%custom_batch)==(custom_batch-1)) && (i_current+1>=R_in && j+1>=C_in && tii_current+4>=M)){
							output_dma_O_data.last=1;
						}
						else{
							output_dma_O_data.last=0;
						}
						dma_Output.write(output_dma_O_data);
					}
				}
			}
		}
//		if(i_current_sub<R_in && i_current>=i_current_sub-1){//
//			for(int8 i=i_current;i<R_in;i++){
//				#pragma HLS loop_tripcount min=1 max=1 avg=1
//				for(int8 j=0;j<C_in;j++){
//				#pragma HLS loop_tripcount min=24 max=24 avg=24
//					for(int14 too=0;too<Tm;too++){
//				#pragma HLS loop_tripcount min=6 max=6 avg=6
//				#pragma HLS PIPELINE II=1
//				#pragma HLS dependence variable=OFM intra false
//						FPGA_DATA Relu_activ;
//						int14 too_current=too+to;
//						FPGA_DATA Read_Output;
//						FPGA_DATA Write_dma_O;
//						Read_Output = 0;
//						if(too_current<M ){
//							if(state[0]==1){//XX1 relu after conv need to be transmited
//								ofm_dma_O_data=dma_OFM.read();
//								Relu_activ=ofm_dma_O_data.data;
//							}
//							output_dma_O_data.data=0;
//
//							//if(custom_Tr>=R){//feature map small, can load and store together
//							if( ba==custom_batch-1 && (i_current==R_in-1 && j==C_in-1 && too_current==M-1)){
//								output_dma_O_data.last=1;
//							}
//							else{
//								output_dma_O_data.last=0;
//							}
//							dma_Output.write(output_dma_O_data);
//						}
//					}
//				}
//			}
//		}
//	}

	for(int8 trr=0;trr<i_upper;trr++){
	#pragma HLS loop_tripcount min=24 max=24 avg=24
		for(int8 tcc=0;tcc<j_upper;tcc++){
	#pragma HLS loop_tripcount min=24 max=24 avg=24
	#pragma HLS PIPELINE II=1
			for(int14 tii=0;tii<Tn;tii++){
				int14 trtcindex=trr*j_upper+tcc;
				//Output1[too][trr - row][tcc - col]=0;
				IFM[tii][trtcindex]=0;
			}
		}
	}
}


void Poolindex_Store(
		ufix16 Poolindex[Mp],
		ap_uint<4> state,
		int14 M,
		int8 R,
		int8 C,
		//int8 ba,
		//int8 custom_batch,
		hls::stream<DMA_DATA_128> &dma_Indexout
		){
		//int8& flag){
	DMA_DATA_128 output_dma_O_data;
	//Relu_index Reluindex_data;

	int toindex=0;
	int Mall=R*C*M;
	if(state[3]==0){
		for(int to=0;to<Mall;to+=32){//the valid width of DMA DATA is 23bits
		#pragma HLS loop_tripcount min=432 max=432 avg=432

		#pragma HLS PIPELINE II=1
		#pragma HLS dependence variable=Poolindex intra false
			//FPGA_DATA output;
//				Pool_22 Poolindex22;
			ufix32 Poolindex32=0;
			ufix32 Poolindex322=0;
			ufix32 Poolindex323=0;
			ufix32 Poolindex324=0;
			Poolindex32(15,0)=Poolindex[toindex];
			Poolindex322(15,0)=Poolindex[toindex+1];
			Poolindex323(15,0)=Poolindex[toindex+2];
			Poolindex324(15,0)=Poolindex[toindex+3];
			//output=Poolindex32;
			output_dma_O_data.data.data1=Poolindex32;
			output_dma_O_data.data.data2=Poolindex322;
			output_dma_O_data.data.data3=Poolindex323;
			output_dma_O_data.data.data4=Poolindex324;
			//if(ba==custom_batch-1 && trr==R-1 && tcc==C-1 && to>=M-15){
			if( to+32>=Mall){
				output_dma_O_data.last=1;
			}
			else{
				output_dma_O_data.last=0;
			}
			dma_Indexout.write(output_dma_O_data);
			toindex+=4;
		}
	}
}

void Poolindex_Load(hls::stream<DMA_DATA_128> &dma_Indexin,//only backward

		ufix16 Poolindex[Mp],
		ap_uint<4> state,
		int14 M,
		int8 R,
		int8 C

		){
	DMA_DATA_128 ifm_input_data;
	int Mall=R*C*M;
	int toindex=0;
	if(state[3]==0){
		for(int to=0;to<Mall;to+=32){//the valid width of DMA DATA is 23bits
		#pragma HLS loop_tripcount min=432 max=432 avg=432
		#pragma HLS PIPELINE II=1
		#pragma HLS dependence variable=Poolindex intra false
			//FPGA_DATA input;
//				Pool_22 Poolindex22;
			ufix32 Poolindex32=0;
			ufix32 Poolindex322=0;
			ufix32 Poolindex323=0;
			ufix32 Poolindex324=0;
			ifm_input_data=dma_Indexin.read();
			Poolindex32=ifm_input_data.data.data1;
			Poolindex322=ifm_input_data.data.data2;
			Poolindex323=ifm_input_data.data.data3;
			Poolindex324=ifm_input_data.data.data4;
			//Poolindex32=input;
			Poolindex[toindex]=Poolindex32(15,0);
			Poolindex[toindex+1]=Poolindex322(15,0);
			Poolindex[toindex+2]=Poolindex323(15,0);
			Poolindex[toindex+3]=Poolindex324(15,0);
			toindex+=4;
		}
	}
}

void Comparefor(FPGA_DATA IFM[Tn][Trtcin],
		FPGA_DATA OFM[Tm][Trtc],
		ufix16 Poolindex[Mp],
		//Pool_index_dim2 Poolindex4[4][M4][R4][C4],
		//int4 Poollayerin,
		//int4 Poollayerout,
		ap_uint<4> state,
		//int state,
		//int8 ba,
		int8 ticount,
		int8 row,
		//int8 col,
		int14 M,
		int8 R,
		int8 C,
		//int8 custom_Tr,
		//int8 custom_Tc,
		int4 pool_s,
		int4 pool_k,
		int8 custom_Tr,
		int8 custom_Tc
//		int8 viewTrc,
//		int8 viewTm,
		//int8 R_in,
		//int8 C_in
		//int8& flag
		){

	//FPGA_DATA pmax;
	int8 tiicount=0;
	int8 j_upper=(custom_Tc-1)*pool_s+pool_k;
	if(state[3]==0){//maxpool
		for(int14 tii=0;tii<Tn;tii+=8){
			//int8 tii_current_count=ticount+tiicount;
			for(int4 ip=0;ip<pool_k;ip++){
		#pragma HLS loop_tripcount min=3 max=3 avg=3
				for(int4 jp=0;jp<pool_k;jp++){
		#pragma HLS loop_tripcount min=3 max=3 avg=3
					for(int8 trr=0;trr<custom_Tr;trr++){
			#pragma HLS loop_tripcount min=12 max=12 avg=12
						//int8 trr_current=trr+row;
						for(int8 tcc=0;tcc<custom_Tc;tcc++){
			#pragma HLS loop_tripcount min=12 max=12 avg=12


			#pragma HLS PIPELINE II=1

							//FPGA_DATA OFM_reg[3];
			//#pragma HLS ARRAY_PARTITION variable=OFM_reg complete dim=0
			#pragma HLS dependence variable=Poolindex inter false
			#pragma HLS dependence variable=OFM inter false
			//#pragma HLS dependence variable=OFM_reg inter false
							int trr_current=trr+row;
							int tii_current_count=ticount+tiicount;
							int Mpindex=tii_current_count*C*R+trr_current*R+tcc;
							int8 ifm_r= pool_s*trr+ip;
							int8 ifm_c= pool_s*tcc+jp;
							int14 trtcindex=trr*custom_Tc+tcc;
							int14 trtcindexifm=ifm_r*j_upper+ifm_c;

							Pool_index imax[8];
							Pool_index jmax[8];
							ufix16 Poolindex32;
							FPGA_DATA OFM_reg1;
							FPGA_DATA OFM_reg2;
							FPGA_DATA OFM_reg3;
							FPGA_DATA OFM_reg4;
							FPGA_DATA OFM_reg5;
							FPGA_DATA OFM_reg6;
							FPGA_DATA OFM_reg7;
							FPGA_DATA OFM_reg8;
							Poolindex32=Poolindex[Mpindex];
							jmax[0]=Poolindex32[0];
							imax[0]=Poolindex32[1];
							jmax[1]=Poolindex32[2];
							imax[1]=Poolindex32[3];
							jmax[2]=Poolindex32[4];
							imax[2]=Poolindex32[5];
							jmax[3]=Poolindex32[6];
							imax[3]=Poolindex32[7];
							jmax[4]=Poolindex32[8];
							imax[4]=Poolindex32[9];
							jmax[5]=Poolindex32[10];
							imax[5]=Poolindex32[11];
							jmax[6]=Poolindex32[12];
							imax[6]=Poolindex32[13];
							jmax[7]=Poolindex32[14];
							imax[7]=Poolindex32[15];


							if(jp==0 && ip==0){
								OFM_reg1=minimal;
								OFM_reg2=minimal;
								OFM_reg3=minimal;
								OFM_reg4=minimal;
								OFM_reg5=minimal;
								OFM_reg6=minimal;
								OFM_reg7=minimal;
								OFM_reg8=minimal;
							}
							else{
								OFM_reg1=OFM[tii][trtcindex];
								OFM_reg2=OFM[tii+1][trtcindex];
								OFM_reg3=OFM[tii+2][trtcindex];
								OFM_reg4=OFM[tii+3][trtcindex];
								OFM_reg5=OFM[tii+4][trtcindex];
								OFM_reg6=OFM[tii+5][trtcindex];
								OFM_reg7=OFM[tii+6][trtcindex];
								OFM_reg8=OFM[tii+7][trtcindex];
							}

							FPGA_DATA ifmin1=IFM[tii][trtcindexifm];
							FPGA_DATA ifmin2=IFM[tii+1][trtcindexifm];
							FPGA_DATA ifmin3=IFM[tii+2][trtcindexifm];
							FPGA_DATA ifmin4=IFM[tii+3][trtcindexifm];
							FPGA_DATA ifmin5=IFM[tii+4][trtcindexifm];
							FPGA_DATA ifmin6=IFM[tii+5][trtcindexifm];
							FPGA_DATA ifmin7=IFM[tii+6][trtcindexifm];
							FPGA_DATA ifmin8=IFM[tii+7][trtcindexifm];
							if(ifmin1>OFM_reg1){
								imax[0]=ip;
								jmax[0]=jp;
								OFM_reg1=ifmin1;

							}
							if(ifmin2>OFM_reg2){
								imax[1]=ip;
								jmax[1]=jp;
								OFM_reg2=ifmin2;

							}
							if(ifmin3>OFM_reg3){
								imax[2]=ip;
								jmax[2]=jp;
								OFM_reg3=ifmin3;

							}
							if(ifmin4>OFM_reg4){
								imax[3]=ip;
								jmax[3]=jp;
								OFM_reg4=ifmin4;

							}
							if(ifmin5>OFM_reg5){
								imax[4]=ip;
								jmax[4]=jp;
								OFM_reg5=ifmin5;

							}
							if(ifmin6>OFM_reg6){
								imax[5]=ip;
								jmax[5]=jp;
								OFM_reg6=ifmin6;

							}
							if(ifmin7>OFM_reg7){
								imax[6]=ip;
								jmax[6]=jp;
								OFM_reg7=ifmin7;

							}
							if(ifmin8>OFM_reg8){
								imax[7]=ip;
								jmax[7]=jp;
								OFM_reg8=ifmin8;

							}
							OFM[tii][trtcindex]=OFM_reg1;
							OFM[tii+1][trtcindex]=OFM_reg2;
							OFM[tii+2][trtcindex]=OFM_reg3;
							OFM[tii+3][trtcindex]=OFM_reg4;
							OFM[tii+4][trtcindex]=OFM_reg5;
							OFM[tii+5][trtcindex]=OFM_reg6;
							OFM[tii+6][trtcindex]=OFM_reg7;
							OFM[tii+7][trtcindex]=OFM_reg8;
							Poolindex32[0]=jmax[0];
							Poolindex32[1]=imax[0];
							Poolindex32[2]=jmax[1];
							Poolindex32[3]=imax[1];
							Poolindex32[4]=jmax[2];
							Poolindex32[5]=imax[2];
							Poolindex32[6]=jmax[3];
							Poolindex32[7]=imax[3];
							Poolindex32[8]=jmax[4];
							Poolindex32[9]=imax[4];
							Poolindex32[10]=jmax[5];
							Poolindex32[11]=imax[5];
							Poolindex32[12]=jmax[6];
							Poolindex32[13]=imax[6];
							Poolindex32[14]=jmax[7];
							Poolindex32[15]=imax[7];
							Poolindex[Mpindex]=Poolindex32;
						}
					}
				}

			}
			tiicount++;
		}
	}
	else{//avgpool
		for(int14 tii=0;tii<Tn;tii+=8){
			//int8 too_current_count=tocount+toocount;
			for(int4 ip=0;ip<pool_k;ip++){
		#pragma HLS loop_tripcount min=3 max=3 avg=3
				for(int4 jp=0;jp<pool_k;jp++){
		#pragma HLS loop_tripcount min=3 max=3 avg=3
					for(int8 trr=0;trr<custom_Tr;trr++){
			#pragma HLS loop_tripcount min=12 max=12 avg=12
						//int8 trr_current=trr+row;
						for(int8 tcc=0;tcc<custom_Tc;tcc++){
			#pragma HLS loop_tripcount min=12 max=12 avg=12


			#pragma HLS PIPELINE II=1

							//FPGA_DATA OFM_reg[3];
			//#pragma HLS ARRAY_PARTITION variable=OFM_reg complete dim=0

			#pragma HLS dependence variable=OFM inter false
			//#pragma HLS dependence variable=OFM_reg inter false
							int8 ifm_r= pool_s*trr+ip;
							int8 ifm_c= pool_s*tcc+jp;

							int14 trtcindex=trr*custom_Tc+tcc;
							int14 trtcindexifm=ifm_r*j_upper+ifm_c;

							OFM[tii][trtcindex]+=IFM[tii][trtcindexifm];
							OFM[tii+1][trtcindex]+=IFM[tii+1][trtcindexifm];
							OFM[tii+2][trtcindex]+=IFM[tii+2][trtcindexifm];
							OFM[tii+3][trtcindex]+=IFM[tii+3][trtcindexifm];
							OFM[tii+4][trtcindex]+=IFM[tii+4][trtcindexifm];
							OFM[tii+5][trtcindex]+=IFM[tii+5][trtcindexifm];
							OFM[tii+6][trtcindex]+=IFM[tii+6][trtcindexifm];
							OFM[tii+7][trtcindex]+=IFM[tii+7][trtcindexifm];

						}
					}
				}
			}
		}
	}
}

void Compareback(FPGA_DATA IFM[Tn][Trtcin],
		FPGA_DATA OFM[Tm][Trtc],
		ufix16 Poolindex[Mp],
		//Pool_index_dim2 Poolindex4[4][M4][R4][C4],
		//int4 Poollayerin,
		//int4 Poollayerout,
		ap_uint<4> state,
		//int state,
		//int8 ba,
		int8 ticount,
		int8 row,
		//int8 col,
		int14 M,
		int8 R,
		int8 C,
		//int8 custom_Tr,
		//int8 custom_Tc,
		int4 pool_s,
		int4 pool_k,
		int8 custom_Tr,
		int8 custom_Tc
//		int8 viewTrc,
//		int8 viewTm,
		//int8 R_in,
		//int8 C_in
		//int8& flag
		){
	int8 tiicount=0;
	int8 j_upper=(custom_Tc-1)*pool_s+pool_k;
	if(state[3]==0){//maxpool
		for(int14 tii=0;tii<Tm;tii+=8){
			for(int4 ip=0;ip<pool_k;ip++){
			#pragma HLS loop_tripcount min=3 max=3 avg=3
				for(int4 jp=0;jp<pool_k;jp++){
			#pragma HLS loop_tripcount min=3 max=3 avg=3
					for(int8 trr=0;trr<custom_Tr;trr++){
			#pragma HLS loop_tripcount min=12 max=12 avg=12
						//int8 trr_current=trr+row;
						for(int8 tcc=0;tcc<custom_Tc;tcc++){
			#pragma HLS loop_tripcount min=12 max=12 avg=12
//					//int8 tcc_current=tcc;
//					ufix16 Poolindex32=Poolindex[trr_current][tcc][too_current];
//					Pool_index jmax=Poolindex32[2*too];
//					Pool_index imax=Poolindex32[2*too+1];
////					imax(Tm,0)=Poolindex32(5,0);
////					jmax(Tm,0)=Poolindex32(11,6);
//					FPGA_DATA OFM_reg;

			#pragma HLS PIPELINE II=1
							//FPGA_DATA OFM_reg[3];
			//#pragma HLS ARRAY_PARTITION variable=OFM_reg complete dim=0
			#pragma HLS dependence variable=IFM inter false
							int8 ifm_r= pool_s*trr+ip;
							int8 ifm_c= pool_s*tcc+jp;
							int14 trtcindex=trr*custom_Tc+tcc;
							int14 trtcindexifm=ifm_r*j_upper+ifm_c;
							int trr_current=trr+row;
							int tii_current_count=ticount+tiicount;
							int Mpindex=tii_current_count*C*R+trr_current*R+tcc;
							Pool_index imax[8];
							Pool_index jmax[8];
							ufix16 Poolindex32;
							FPGA_DATA OFM_reg1;
							FPGA_DATA OFM_reg2;
							FPGA_DATA OFM_reg3;
							FPGA_DATA OFM_reg4;
							FPGA_DATA OFM_reg5;
							FPGA_DATA OFM_reg6;
							FPGA_DATA OFM_reg7;
							FPGA_DATA OFM_reg8;

							FPGA_DATA ifmin1=OFM[tii][trtcindex];
							FPGA_DATA ifmin2=OFM[tii+1][trtcindex];
							FPGA_DATA ifmin3=OFM[tii+2][trtcindex];
							FPGA_DATA ifmin4=OFM[tii+3][trtcindex];
							FPGA_DATA ifmin5=OFM[tii+4][trtcindex];
							FPGA_DATA ifmin6=OFM[tii+5][trtcindex];
							FPGA_DATA ifmin7=OFM[tii+6][trtcindex];
							FPGA_DATA ifmin8=OFM[tii+7][trtcindex];
							Poolindex32=Poolindex[Mpindex];
							jmax[0]=Poolindex32[0];
							imax[0]=Poolindex32[1];
							jmax[1]=Poolindex32[2];
							imax[1]=Poolindex32[3];
							jmax[2]=Poolindex32[4];
							imax[2]=Poolindex32[5];
							jmax[3]=Poolindex32[6];
							imax[3]=Poolindex32[7];
							jmax[4]=Poolindex32[8];
							imax[4]=Poolindex32[9];
							jmax[5]=Poolindex32[10];
							imax[5]=Poolindex32[11];
							jmax[6]=Poolindex32[12];
							imax[6]=Poolindex32[13];
							jmax[7]=Poolindex32[14];
							imax[7]=Poolindex32[15];
							if(imax[0]==ip && jmax[0]==jp){
								OFM_reg1=ifmin1;
							}
							else{
								OFM_reg1=0;
							}
							if(imax[1]==ip && jmax[1]==jp){
								OFM_reg2=ifmin2;
							}
							else{
								OFM_reg2=0;
							}
							if(imax[2]==ip && jmax[2]==jp){
								OFM_reg3=ifmin3;
							}
							else{
								OFM_reg3=0;
							}
							if(imax[3]==ip && jmax[3]==jp){
								OFM_reg4=ifmin4;
							}
							else{
								OFM_reg4=0;
							}
							if(imax[4]==ip && jmax[4]==jp){
								OFM_reg5=ifmin5;
							}
							else{
								OFM_reg5=0;
							}
							if(imax[5]==ip && jmax[5]==jp){
								OFM_reg6=ifmin6;
							}
							else{
								OFM_reg6=0;
							}
							if(imax[6]==ip && jmax[6]==jp){
								OFM_reg7=ifmin7;
							}
							else{
								OFM_reg7=0;
							}
							if(imax[7]==ip && jmax[7]==jp){
								OFM_reg8=ifmin8;
							}
							else{
								OFM_reg8=0;
							}
							IFM[tii][trtcindexifm]+=OFM_reg1;
							IFM[tii+1][trtcindexifm]+=OFM_reg2;
							IFM[tii+2][trtcindexifm]+=OFM_reg3;
							IFM[tii+3][trtcindexifm]+=OFM_reg4;
							IFM[tii+4][trtcindexifm]+=OFM_reg5;
							IFM[tii+5][trtcindexifm]+=OFM_reg6;
							IFM[tii+6][trtcindexifm]+=OFM_reg7;
							IFM[tii+7][trtcindexifm]+=OFM_reg8;
						}
					}
				}
			}
			tiicount++;
		}
	}
	else{//avgpool
		for(int14 tii=0;tii<Tn;tii+=8){
			for(int4 ip=0;ip<pool_k;ip++){
			#pragma HLS loop_tripcount min=3 max=3 avg=3
				for(int4 jp=0;jp<pool_k;jp++){
			#pragma HLS loop_tripcount min=3 max=3 avg=3
					for(int8 trr=0;trr<custom_Tr;trr++){
			#pragma HLS loop_tripcount min=12 max=12 avg=12
						//int8 trr_current=trr+row;
						for(int8 tcc=0;tcc<custom_Tc;tcc++){
			#pragma HLS loop_tripcount min=12 max=12 avg=12

			#pragma HLS PIPELINE II=1
							//FPGA_DATA OFM_reg[3];
			//#pragma HLS ARRAY_PARTITION variable=OFM_reg complete dim=0
			#pragma HLS dependence variable=IFM inter false
							int8 ifm_r= pool_s*trr+ip;
							int8 ifm_c= pool_s*tcc+jp;

							int14 trtcindex=trr*custom_Tc+tcc;
							int14 trtcindexifm=ifm_r*j_upper+ifm_c;

							IFM[tii][trtcindexifm]+=OFM[tii][trtcindex];
							IFM[tii+1][trtcindexifm]+=OFM[tii+1][trtcindex];
							IFM[tii+2][trtcindexifm]+=OFM[tii+2][trtcindex];
							IFM[tii+3][trtcindexifm]+=OFM[tii+3][trtcindex];
							IFM[tii+4][trtcindexifm]+=OFM[tii+4][trtcindex];
							IFM[tii+5][trtcindexifm]+=OFM[tii+5][trtcindex];
							IFM[tii+6][trtcindexifm]+=OFM[tii+6][trtcindex];
							IFM[tii+7][trtcindexifm]+=OFM[tii+7][trtcindex];
						}
					}
				}
			}
		}
	}
}

void Compare_STORE_pool(
				FPGA_DATA IFM[Tn][Trtcin],
				FPGA_DATA OFM[Tm][Trtc],
				ufix16 Poolindex[Mp],
				//int state,
				ap_uint<4> state,
				int8 custom_batch,
				int8 ba,
				int14 to,
				int8 tocount,
				int8 row,
				//int8 col,
				int14 M,
				int8 R,
				int8 C,
				//int8 custom_Tr,
				//int8 custom_Tc,
				int4 pool_s,
				int4 pool_k,
				int8 custom_Tr,
				int8 custom_Tc,
				//hls::stream<DMA_DATA_64> &dma_OFM,
				hls::stream<DMA_DATA_128> &dma_Output,
				int8 R_in,
				int8 C_in
				){
				Comparefor(IFM,OFM,Poolindex,state,tocount,row,M,R,C,pool_s,pool_k,custom_Tr,custom_Tc);
				STORE_poolfor(OFM,state,to,row,M,R,C,pool_s,pool_k,custom_Tr,custom_Tc,//dma_OFM,
				dma_Output,R_in,C_in);
}

void LOAD_Compare_pool(
		hls::stream<DMA_DATA_128> &dma_IFM,
		FPGA_DATA IFM[Tn][Trtcin],
		FPGA_DATA OFM[Tm][Trtc],
		ufix16 Poolindex[Mp],
		ap_uint<4> state,
		//int state,
		int14 to,
		int8 tocount,
		int8 row,
		//int8 col,
		int14 M,
		int8 R,
		int8 C,
		//int8 custom_Tr,
		//int8 custom_Tc,
		int4 pool_s,
		int4 pool_k,
		int8 custom_Tr,
		int8 custom_Tc,
		int8 R_in,
		int8 C_in
		//int8& flag
		){
	LOAD_poolback(dma_IFM,OFM,to,row,M,R,C,pool_s,pool_k,custom_Tr,custom_Tc,R_in,C_in);
	Compareback(IFM,OFM,Poolindex,state,tocount,row,M,R,C,pool_s,pool_k,custom_Tr,custom_Tc);
}



void pool(	hls::stream<DMA_DATA_128> &dma_IFM,
			//hls::stream<DMA_index> &dma_Indexin,
			hls::stream<DMA_DATA_128> &dma_Weights,
			//hls::stream<DMA_DATA_128> &dma_Indexout,
			FPGA_DATA IFM[Tn][Trtcin],
			FPGA_DATA IFM_DB[Tn][Trtcin],
			FPGA_DATA OFM[Tm][Trtc],
			FPGA_DATA OFM_DB[Tm][Trtc],
//			Relu_index Reluindex1[4][M1][R1][C1],
//			Relu_index Reluindex3[4][M3][R3][C3],
//			int4 Relulayerin,
//			int4 Relulayerout,
			//Pool_index_dim2 Poolindex[Rp][Cp][Mp],
			ufix16 Poolindex[Mp],
			//Pool_index_dim2 Poolindex4[4][M4][R4][C4],
			//int4 Poollayerin,
			//int4 Poollayerout,
			//int state,
			ap_uint<4> state,
			int8 custom_batch,
			int8 batch_size,
			//int to,
			int14 N,
			int8 R,
			int8 C,
			int4 pool_s,
			int4 pool_k,
			int8 custom_Tr,
			int8 custom_Tc,
			hls::stream<DMA_DATA_128> &dma_OFM,
			hls::stream<DMA_DATA_128> &dma_Output,
//			int8 viewTrc,
//			int14 viewTm,
			int8 R_in,
			int8 C_in
			//int8& flag
			){
	int8 r_clear=(custom_Tr-1)*pool_s+pool_k;
	int8 c_clear=(custom_Tc-1)*pool_s+pool_k;
	if(state[1]==0){//backward
		Initialize_IFM(IFM,IFM_DB,r_clear,c_clear);
	}
	for(int8 ba=0;ba<batch_size;ba++){
	#pragma HLS loop_tripcount min=4 max=4 avg=4
		int num = 0;
		//int num2 =0;
		//int8 colp;
		int8 rowp=0;
		int14 tip=0;
		int8 ticount=0;
		int8 ticountp=0;
		if(state[1]==1){//11Xforward
			for(int14 ti=0;ti<N;ti+=Tn){
			#pragma HLS loop_tripcount min=6 max=6 avg=6
				for(int8 row=0;row<R;row+=custom_Tr){
			#pragma HLS loop_tripcount min=3 max=3 avg=3
					if(num==0){
						LOAD_poolfor(dma_IFM,IFM,state,ti,row,N,R,C,pool_s,pool_k,custom_Tr,custom_Tc,R_in,C_in);
					}
					else{
						if(num%2==0){
							LOAD_poolfor(dma_IFM,IFM,state,ti,row,N,R,C,pool_s,pool_k,custom_Tr,custom_Tc,R_in,C_in);
							Compare_STORE_pool(IFM_DB,OFM,Poolindex,state,custom_batch,ba,tip,ticountp,rowp,
							N,R,C,pool_s,pool_k,custom_Tr,custom_Tc,dma_Output,R_in,C_in);
							//Compare(IFM_DB,OFM,Poolindex,state[1],ticountp,rowp,custom_Tr,custom_Tc);
							//STORE_pool(OFM,state,custom_batch,ba,tip,rowp,N,R,C,custom_Tr,custom_Tc,//dma_OFM,
							//dma_Output);
						}
						else{
							LOAD_poolfor(dma_IFM,IFM_DB,state,ti,row,N,R,C,pool_s,pool_k,custom_Tr,custom_Tc,R_in,C_in);
							Compare_STORE_pool(IFM,OFM,Poolindex,state,custom_batch,ba,tip,ticountp,rowp,
							N,R,C,pool_s,pool_k,custom_Tr,custom_Tc,dma_Output,R_in,C_in);
							//Compare(IFM,OFM,Poolindex,state[1],ticountp,rowp,custom_Tr,custom_Tc);
							//STORE_pool(OFM,state,custom_batch,ba,tip,rowp,N,R,C,custom_Tr,custom_Tc,//dma_OFM,
							//dma_Output);
						}
					}
					num++;
					tip=ti;
					ticountp=ticount;
					rowp=row;
				}
				ticount+=2;
			}
			if(num%2==0){
				//Compare_STORE_pool(IFM_DB,OFM,Poolindex,state,custom_batch,ba,tip,ticountp,rowp,
				//N,R,C,custom_Tr,custom_Tc,dma_Output);
				Comparefor(IFM_DB,OFM,Poolindex,state,ticountp,rowp,N,R,C,pool_s,pool_k,custom_Tr,custom_Tc);
				STORE_poolfor(OFM,state,tip,rowp,N,R,C,pool_s,pool_k,custom_Tr,custom_Tc,//dma_OFM,
				dma_Output,R_in,C_in);
				Poolindex_Store(Poolindex,state,N,R,C,dma_Output);
			}
			else{
				//Compare_STORE_pool(IFM,OFM,Poolindex,state,custom_batch,ba,tip,ticountp,rowp,
				//N,R,C,custom_Tr,custom_Tc,dma_Output);
				Comparefor(IFM,OFM,Poolindex,state,ticountp,rowp,N,R,C,pool_s,pool_k,custom_Tr,custom_Tc);
				STORE_poolfor(OFM,state,tip,rowp,N,R,C,pool_s,pool_k,custom_Tr,custom_Tc,//dma_OFM,
				dma_Output,R_in,C_in);
				Poolindex_Store(Poolindex,state,N,R,C,dma_Output);
			}

		}
		else{//10Xbackward
			for(int14 ti=0;ti<N;ti+=Tn){
			#pragma HLS loop_tripcount min=6 max=6 avg=6
				for(int8 row=0;row<R;row+=custom_Tr){
			#pragma HLS loop_tripcount min=3 max=3 avg=3
					if(num==0){
						Poolindex_Load(dma_Weights,Poolindex,state,N,R,C);
						//LOAD_Compare_pool(dma_IFM,IFM,OFM,Poolindex,state,ti,ticount,row,
						//N,R,C,custom_Tr,custom_Tc);
						LOAD_poolback(dma_IFM,OFM,ti,row,N,R,C,pool_s,pool_k,custom_Tr,custom_Tc,R_in,C_in);
						Compareback(IFM,OFM,Poolindex,state,ticount,row,N,R,C,pool_s,pool_k,custom_Tr,custom_Tc);
					}
					else{
						if(num%2==0){
							LOAD_Compare_pool(dma_IFM,IFM,OFM,Poolindex,state,ti,ticount,row,
							N,R,C,pool_s,pool_k,custom_Tr,custom_Tc,R_in,C_in);
							//LOAD_pool(dma_IFM,IFM,state,ti,row,N,R,C,custom_Tr,custom_Tc);
							//Compare(IFM,OFM,Poolindex,state[1],ticount,row,custom_Tr,custom_Tc);
							STORE_poolback(IFM_DB,state,custom_batch,batch_size,ba,tip,rowp,N,R,C,pool_s,pool_k,
							custom_Tr,custom_Tc,dma_OFM,dma_Output,R_in,C_in);
						}
						else{
							LOAD_Compare_pool(dma_IFM,IFM_DB,OFM,Poolindex,state,ti,ticount,row,
							N,R,C,pool_s,pool_k,custom_Tr,custom_Tc,R_in,C_in);
							//LOAD_pool(dma_IFM,IFM,state,ti,row,N,R,C,custom_Tr,custom_Tc);
							//Compare(IFM,OFM_DB,Poolindex,state[1],ticount,row,custom_Tr,custom_Tc);
							STORE_poolback(IFM,state,custom_batch,batch_size,ba,tip,rowp,N,R,C,pool_s,pool_k,
							custom_Tr,custom_Tc,dma_OFM,dma_Output,R_in,C_in);
						}
					}
					num++;
					tip=ti;
					//ticountp=ticount;
					rowp=row;
				}
				ticount+=2;

			}
			if(num%2==0){
				STORE_poolback(IFM_DB,state,custom_batch,batch_size,ba,tip,rowp,N,R,C,pool_s,pool_k,
				custom_Tr,custom_Tc,dma_OFM,dma_Output,R_in,C_in);
			}
			else{
				STORE_poolback(IFM,state,custom_batch,batch_size,ba,tip,rowp,N,R,C,pool_s,pool_k,
				custom_Tr,custom_Tc,dma_OFM,dma_Output,R_in,C_in);
			}
		}
	}
	Initialize_OFM(OFM,OFM_DB,r_clear,c_clear);
}



void LOAD_Weights_bn(hls::stream<DMA_DATA_128> &dma_Weights,
					hls::stream<DMA_DATA_128> &dma_OFM,
					hls::stream<DMA_DATA_128> &dma_IFM,
					FPGA_DATA BNW[DMAwidth][BNbank][BNindex],
					 ap_uint<4> state,
					 //int state,
					 //int8 custom_Tib,
					 //int8 to_index,
					 //int8 ti_index,
					 //int4 ki_index,
					 //int4 kj_index,
					 //int13 ti,
					 //int14 to,
					 //int13 N,
					 int14 M
					 //FPGA_DATA Fnumdiv
					 ){

	DMA_DATA_128 weight_input_dma;
	DMA_DATA_128 ofm_input_dma;
	DMA_DATA_128 ifm_input_dma;
	//int8 too_index=to_index;
	int14 too_index=0;
	for (int14 too=0;too<M;too+=4){
		#pragma HLS loop_tripcount min=6 max=6 avg=6
		//int14 too_current =too+to;
		#pragma HLS PIPELINE II=1
		#pragma HLS dependence variable=BNW inter false
		//if(too_current<M ){//&& M_current<M_in){
			weight_input_dma=dma_Weights.read();
			ofm_input_dma=dma_OFM.read();
			BNW[0][2][too_index] = weight_input_dma.data.data1;//gama
			BNW[1][2][too_index] = weight_input_dma.data.data2;
			BNW[2][2][too_index] = weight_input_dma.data.data3;
			BNW[3][2][too_index] = weight_input_dma.data.data4;
			BNW[0][3][too_index] = ofm_input_dma.data.data1;//beta
			BNW[1][3][too_index] = ofm_input_dma.data.data2;
			BNW[2][3][too_index] = ofm_input_dma.data.data3;
			BNW[3][3][too_index] = ofm_input_dma.data.data4;
			if(state<4){//BP
				ifm_input_dma=dma_IFM.read();
				BNW[0][6][too_index] = ifm_input_dma.data.data1;//BP:namuda
				BNW[1][6][too_index] = ifm_input_dma.data.data2;
				BNW[2][6][too_index] = ifm_input_dma.data.data3;
				BNW[3][6][too_index] = ifm_input_dma.data.data4;
			}
			 //flag=140;
		//}
		too_index++;
	}
}




void LOAD_bnfor(hls::stream<DMA_DATA_128> &dma_IFM,
		//hls::stream<DMA_DATA_128> &dma_OFM,
		FPGA_DATA BNW[DMAwidth][BNbank][BNindex],
		//ap_uint<3> state,
		int8 custom_batch,
		//int state,
		//int13 ti,
		int14 to,
		int14 to_index,
		int14 M,
		//int13 M_index,
		//int13 M_in,
		int8 R,
		int8 C
		//FPGA_DATA Fnumdiv
		){
	DMA_DATA_128 ifm_input_data;
	//DMA_DATA_128 ofm_input_data;
	int14 too_index=to_index;
	//for(int8 ba=0;ba<custom_batch;ba++){
	//	#pragma HLS loop_tripcount min=16 max=16 avg=16
		for(int8 trr=0;trr<R;trr++){
		#pragma HLS loop_tripcount min=32 max=32 avg=32
			for(int8 tcc=0;tcc<C;tcc++){
				too_index=to_index;
		#pragma HLS loop_tripcount min=32 max=32 avg=32
				for(int14 too=0;too<8*4;too+=4){
		#pragma HLS loop_tripcount min=4 max=4 avg=4
					int14 too_current=too+to;
					//int13 M_current=M_index+too_current;
		#pragma HLS PIPELINE II=1
		#pragma HLS dependence variable=BNW inter false
					FPGA_DATA ifm_input1;
					FPGA_DATA ifm_input2;
					FPGA_DATA ifm_input3;
					FPGA_DATA ifm_input4;
					if(too_current<M && too<Tm){// && M_current<M_in){
						ifm_input_data=dma_IFM.read();
						ifm_input1 = ifm_input_data.data.data1;
						ifm_input2 = ifm_input_data.data.data2;
						ifm_input3 = ifm_input_data.data.data3;
						ifm_input4 = ifm_input_data.data.data4;
					}
					else{
						ifm_input1 = 0;
						ifm_input2 = 0;
						ifm_input3 = 0;
						ifm_input4 = 0;
					}
					BNW[0][0][too_index] += ifm_input1;
					BNW[1][0][too_index] += ifm_input2;
					BNW[2][0][too_index] += ifm_input3;
					BNW[3][0][too_index] += ifm_input4;
					BNW[0][1][too_index] += ifm_input1*ifm_input1;
					BNW[1][1][too_index] += ifm_input2*ifm_input2;
					BNW[2][1][too_index] += ifm_input3*ifm_input3;
					BNW[3][1][too_index] += ifm_input4*ifm_input4;
					too_index++;

				}
			}
		}
	//}
}

void Weights_bnfor(//hls::stream<DMA_DATA_64> &dma_IFM,
		//hls::stream<DMA_DATA_128> &dma_OFM,
		FPGA_DATA BNW[DMAwidth][BNbank][BNindex],
		//ap_uint<3> state,
		//int8 custom_batch,
		//int state,
		//int13 ti,
		//int14 to,
		//int8 to_index,
		int14 M,
		//int13 M_index,
		//int13 M_in,
		//int8 R,
		//int8 C,
		FPGA_DATA Fnumdiv
		){

	int14 too_index=0;
	int14 too_num=0;
	for (int14 too=0;too<M;too++){
		#pragma HLS loop_tripcount min=64 max=64 avg=64
		//int14 too_current =too+to;
		#pragma HLS PIPELINE II=1
		#pragma HLS dependence variable=BNW inter false
		//if(too_current<M ){//&& M_current<M_in){
			FPGA_DATA mean;
			FPGA_DATA variance;
			mean=BNW[too_num][0][too_index]*Fnumdiv;
			variance=BNW[too_num][1][too_index]*Fnumdiv-mean*mean;
			BNW[too_num][5][too_index] = 1/hls::sqrtf(variance+eps);//namuda
			BNW[too_num][4][too_index] = mean;//mean
			//BNW[too_num][6][too_index] = variance;
			 //flag=140;
		//}
		too_num++;
		if(too_num>=4){
			too_index++;
			too_num=0;
		}
	}
}

void LOAD_bnback(hls::stream<DMA_DATA_128> &dma_IFM,
		hls::stream<DMA_DATA_128> &dma_OFM,
		FPGA_DATA BNW[DMAwidth][BNbank][BNindex],
		//ap_uint<3> state,
		int8 custom_batch,
		//int state,
		//int13 ti,
		int14 to,
		int14 to_index,
		int14 M,
		//int13 M_index,
		//int13 M_in,
		int8 R,
		int8 C
		//FPGA_DATA Fnumdiv
		){
	DMA_DATA_128 ifm_input_data;
	DMA_DATA_128 ofm_input_data;
	int14 too_index=to_index;
	//for(int8 ba=0;ba<custom_batch;ba++){
	//	#pragma HLS loop_tripcount min=16 max=16 avg=16
		for(int8 trr=0;trr<R;trr++){
		#pragma HLS loop_tripcount min=32 max=32 avg=32
			for(int8 tcc=0;tcc<C;tcc++){
				too_index=to_index;
		#pragma HLS loop_tripcount min=32 max=32 avg=32
				for(int14 too=0;too<8*4;too+=4){
		#pragma HLS loop_tripcount min=6 max=6 avg=6
					int14 too_current=too+to;
					//int13 M_current=M_index+too_current;
		#pragma HLS PIPELINE II=1
		#pragma HLS dependence variable=BNW inter false
					FPGA_DATA ifm_input1;
					FPGA_DATA ifm_input2;
					FPGA_DATA ifm_input3;
					FPGA_DATA ifm_input4;
					FPGA_DATA ofm_input1;
					FPGA_DATA ofm_input2;
					FPGA_DATA ofm_input3;
					FPGA_DATA ofm_input4;
					if(too_current<M && too<Tm){// && M_current<M_in){
						ifm_input_data=dma_IFM.read();
						ofm_input_data=dma_OFM.read();
						ifm_input1 = ifm_input_data.data.data1;
						ifm_input2 = ifm_input_data.data.data2;
						ifm_input3 = ifm_input_data.data.data3;
						ifm_input4 = ifm_input_data.data.data4;
						ofm_input1 = ofm_input_data.data.data1;
						ofm_input2 = ofm_input_data.data.data2;
						ofm_input3 = ofm_input_data.data.data3;
						ofm_input4 = ofm_input_data.data.data4;

					}
					else{
						ifm_input1 = 0;
						ifm_input2 = 0;
						ifm_input3 = 0;
						ifm_input4 = 0;
						ofm_input1 = 0;
						ofm_input2 = 0;
						ofm_input3 = 0;
						ofm_input4 = 0;
					}
					BNW[0][0][too_index] += ifm_input1;
					BNW[1][0][too_index] += ifm_input2;
					BNW[2][0][too_index] += ifm_input3;
					BNW[3][0][too_index] += ifm_input4;
					BNW[0][1][too_index] += ifm_input1*ofm_input1;
					BNW[1][1][too_index] += ifm_input2*ofm_input2;
					BNW[2][1][too_index] += ifm_input3*ofm_input3;
					BNW[3][1][too_index] += ifm_input4*ofm_input4;
					too_index++;
				}
			}
		}
	//}
}


void Weights_bnback(//hls::stream<DMA_DATA_64> &dma_IFM,
		//hls::stream<DMA_DATA_64> &dma_OFM,
		FPGA_DATA BNW[DMAwidth][BNbank][BNindex],
		//ap_uint<3> state,
		//int8 custom_batch,
		//int state,
		//int13 ti,
		//int14 to,
		//int8 to_index,
		int14 M,
		//int13 M_index,
		//int13 M_in,
		//int8 R,
		//int8 C,
		FPGA_DATA Fnumdiv
		){
	int14 too_index=0;
	int14 too_num=0;
	for (int14 too=0;too<M;too++){
		#pragma HLS loop_tripcount min=64 max=64 avg=64
		//int14 too_current =too+to;
		#pragma HLS PIPELINE II=1
		#pragma HLS dependence variable=BNW inter false
		//if(too_current<M ){//&& M_current<M_in){
			FPGA_DATA mean;
			FPGA_DATA variance;
			mean=BNW[too_num][0][too_index];
			variance=BNW[too_num][1][too_index];
			BNW[too_num][4][too_index]=BNW[too_num][2][too_index]-learn_rate*variance;//gama
			BNW[too_num][5][too_index] =BNW[too_num][3][too_index]-learn_rate*mean;//beta
			BNW[too_num][7][too_index] = BNW[too_num][2][too_index]*BNW[too_num][6][too_index];//namuda
			BNW[too_num][0][too_index] = mean*Fnumdiv;//mean
			BNW[too_num][1][too_index] = variance*Fnumdiv;//variance
		//}
		too_num++;
		if(too_num>=4){
			too_index++;
			too_num=0;
		}
	}
}



void Weights_STORE_bn( FPGA_DATA BNW[DMAwidth][BNbank][BNindex],
					//FPGA_DATA WEIGHT[Tm][Tn][K][K],
					ap_uint<4> state,
					//int8 batch_size,
					//int8 custom_batch,
					//int8 ba,
					//int8 custom_Tib,
					//int8 to_index,
					//int8 ti_index,
					//int4 ki_index,
					//int4 kj_index,
					//int13 ti,
					//int14 to,
					//int13 N,
					int14 M,
					//int13 M_index,
					//int13 M_in,
					//int4 custom_k,
					hls::stream<DMA_DATA_128> &dma_Weightsout,
					hls::stream<DMA_DATA_128> &dma_Output
					){

	DMA_DATA_128 output_dma_O_data;
	DMA_DATA_128 weightsout_dma_O_data;
	int14 too_index=0;
	//if(print==0){
		for(int14 too=0;too<M;too+=4){
		#pragma HLS loop_tripcount min=32 max=32 avg=32
			//int14 too_current=too+to;
		#pragma HLS PIPELINE II=1

		#pragma HLS dependence variable=BNW inter false
		//#pragma HLS dependence variable=Output intra false
			//if(too_current<M ){//&& M_current<M_in){
				output_dma_O_data.data.data1 =BNW[0][5][too_index];//FP namuda;BP beta
				output_dma_O_data.data.data2 =BNW[1][5][too_index];
				output_dma_O_data.data.data3 =BNW[2][5][too_index];
				output_dma_O_data.data.data4 =BNW[3][5][too_index];
				weightsout_dma_O_data.data.data1 =BNW[0][4][too_index];//BP gama
				weightsout_dma_O_data.data.data2 =BNW[1][4][too_index];
				weightsout_dma_O_data.data.data3 =BNW[2][4][too_index];
				weightsout_dma_O_data.data.data4 =BNW[3][4][too_index];
				//if(ba==batch_size-1 &&(i==custom_k-1 && j==custom_k-1 && M_current==M_in-4 &&  tii_current==N-1)){
				if(too+4>=M){
					output_dma_O_data.last=1;
					weightsout_dma_O_data.last=1;
				}
				else{
					output_dma_O_data.last=0;
					weightsout_dma_O_data.last=0;
				}
				dma_Output.write(output_dma_O_data);
				if(state<4){//BP
					dma_Weightsout.write(weightsout_dma_O_data);
				}
			//}
			too_index++;
		}
	//}

//	else{//FP also print mean, variance
//		too_index=to_index;
//		for(int14 too=0;too<Tm;too+=2){
//		#pragma HLS loop_tripcount min=3 max=3 avg=3
//			int14 too_current=too+to;
//		#pragma HLS PIPELINE II=1
//
//		#pragma HLS dependence variable=BNW inter false
//		//#pragma HLS dependence variable=Output intra false
//			if(too_current<M ){//&& M_current<M_in){
//				output_dma_O_data.data.data1 =BNW[0][4][too_index];//mean
//				output_dma_O_data.data.data2 =BNW[1][4][too_index];
//				//output_dma_O_data.data.data3 =BNW[2][4][too_index];
//				//output_dma_O_data.data.data4 =BNW[3][4][too_index];
//				weightsout_dma_O_data.data.data1 =BNW[0][6][too_index];//variance
//				weightsout_dma_O_data.data.data2 =BNW[1][6][too_index];
//				//weightsout_dma_O_data.data.data3 =BNW[2][6][too_index];
//				//weightsout_dma_O_data.data.data4 =BNW[3][6][too_index];
//				//if(ba==batch_size-1 &&(i==custom_k-1 && j==custom_k-1 && M_current==M_in-4 &&  tii_current==N-1)){
//				if(too_current+2>=M){
//					output_dma_O_data.last=1;
//					weightsout_dma_O_data.last=1;
//				}
//				else{
//					output_dma_O_data.last=0;
//					weightsout_dma_O_data.last=0;
//				}
//				dma_Output.write(output_dma_O_data);
//				dma_Weightsout.write(weightsout_dma_O_data);
//			}
//			too_index++;
//		}
//	}

	too_index=0;
	for(int14 too=0;too<M;too+=4){
	#pragma HLS loop_tripcount min=32 max=32 avg=32
	#pragma HLS PIPELINE II=1
	#pragma HLS dependence variable=BNW inter false
		for(int14 too=0;too<DMAwidth;too++){
			for(int14 tii=0;tii<BNbank;tii++){
		//#pragma HLS unroll
				BNW[too][tii][too_index]=0;
			}
		}
		too_index++;
	}
}


void OFM_STORE_bnfor( hls::stream<DMA_DATA_128> &dma_OFM,
		FPGA_DATA BNW[DMAwidth][BNbank][BNindex],
		ap_uint<4> state,
		int8 custom_batch,
		//int state,
		//int13 ti,
		int14 to,
		int14 to_index,
		int14 M,
		//int13 M_index,
		//int13 M_in,
		int8 R,
		int8 C,
		hls::stream<DMA_DATA_128> &dma_Weightsout,
		hls::stream<DMA_DATA_128> &dma_Output){
	DMA_DATA_128 ofm_input_data;
	DMA_DATA_128 weightsout_dma_O_data;
	DMA_DATA_128 output_dma_O_data;

//	//output xhat
//	for(int8 ba=0;ba<custom_batch;ba++){
//		#pragma HLS loop_tripcount min=16 max=16 avg=16
//		for(int8 trr=0;trr<R;trr++){
//		#pragma HLS loop_tripcount min=32 max=32 avg=32
//			for(int8 tcc=0;tcc<C;tcc++){
//		#pragma HLS loop_tripcount min=32 max=32 avg=32
//				for(int14 too=0;too<Tm;too+=4){
//		#pragma HLS loop_tripcount min=6 max=6 avg=6
//					int14 too_current=too+to;
//					//int13 M_current=M_index+too_current;
//		#pragma HLS PIPELINE II=1
//		#pragma HLS dependence variable=WEIGHT intra false
//					//FPGA_DATA ofm_input;
//					if(too_current<M){
//						ofm_input_data=dma_OFM.read();
//						output_dma_O_data.data.data1=(ofm_input_data.data.data1-WEIGHT[too][4][to_index][0][0])
//								*WEIGHT[too][5][to_index][0][0];
//						output_dma_O_data.data.data2=(ofm_input_data.data.data2-WEIGHT[too+1][4][to_index][0][0])
//								*WEIGHT[too+1][5][to_index][0][0];
//						output_dma_O_data.data.data3=(ofm_input_data.data.data3-WEIGHT[too+2][4][to_index][0][0])
//								*WEIGHT[too+2][5][to_index][0][0];
//						output_dma_O_data.data.data4=(ofm_input_data.data.data4-WEIGHT[too+3][4][to_index][0][0])
//								*WEIGHT[too+3][5][to_index][0][0];
//						if(trr+1>=R && tcc+1>=C && too_current+4>=M && too+4>=Tm){
//							output_dma_O_data.last=1;
//						}
//						else{
//							output_dma_O_data.last=0;
//						}
//						dma_Output.write(output_dma_O_data);
//					}
//				}
//			}
//		}
//	}
	int14 too_index=to_index;
	//output result
	//for(int8 ba=0;ba<custom_batch;ba++){
	//	#pragma HLS loop_tripcount min=16 max=16 avg=16
	for(int8 trr=0;trr<R;trr++){
	#pragma HLS loop_tripcount min=32 max=32 avg=32
		for(int8 tcc=0;tcc<C;tcc++){
			too_index=to_index;
	#pragma HLS loop_tripcount min=32 max=32 avg=32
			for(int14 too=0;too<4*12;too+=4){
	#pragma HLS loop_tripcount min=6 max=6 avg=6
				int14 too_current=too+to;
				//int13 M_current=M_index+too_current;
	#pragma HLS PIPELINE II=1
	#pragma HLS dependence variable=BNW intra false
				FPGA_DATA ofm_input1;
				FPGA_DATA ofm_input2;
				FPGA_DATA ofm_input3;
				FPGA_DATA ofm_input4;
				FPGA_DATA xhat1;
				FPGA_DATA xhat2;
				FPGA_DATA xhat3;
				FPGA_DATA xhat4;
				FPGA_DATA y1;
				FPGA_DATA y2;
				FPGA_DATA y3;
				FPGA_DATA y4;
				FPGA_DATA Write_dma_O1;
				FPGA_DATA Write_dma_O2;
				FPGA_DATA Write_dma_O3;
				FPGA_DATA Write_dma_O4;
				if(too_current<M && too<Tm){
					ofm_input_data=dma_OFM.read();
					ofm_input1=ofm_input_data.data.data1;
					ofm_input2=ofm_input_data.data.data2;
					ofm_input3=ofm_input_data.data.data3;
					ofm_input4=ofm_input_data.data.data4;
				}
				else{
					ofm_input1=0;
					ofm_input2=0;
					ofm_input3=0;
					ofm_input4=0;
				}
				xhat1=(ofm_input1-BNW[0][4][too_index])
						*BNW[0][5][too_index];
				xhat2=(ofm_input2-BNW[1][4][too_index])
						*BNW[1][5][too_index];
				xhat3=(ofm_input3-BNW[2][4][too_index])
						*BNW[2][5][too_index];
				xhat4=(ofm_input4-BNW[3][4][too_index])
						*BNW[3][5][too_index];
				y1=xhat1*BNW[0][2][too_index]+BNW[0][3][too_index];
				y2=xhat2*BNW[1][2][too_index]+BNW[1][3][too_index];
				y3=xhat3*BNW[2][2][too_index]+BNW[2][3][too_index];
				y4=xhat4*BNW[3][2][too_index]+BNW[3][3][too_index];
				if(too_current<M && too<Tm){
					if(state[0]==1){//XX1 relu after bn need to be processed
						if (y1> 0){
							Write_dma_O1 =y1;
						}
						else{
							Write_dma_O1 = 0;
						}
						if (y2 > 0){
							Write_dma_O2 =y2;
						}
						else{
							Write_dma_O2 = 0;
						}
						if (y3 > 0){
							Write_dma_O3 =y3;
						}
						else{
							Write_dma_O3 = 0;
						}
						if (y4 > 0){
							Write_dma_O4 =y4;
						}
						else{
							Write_dma_O4 = 0;
						}
					}
					else{
						Write_dma_O1 = y1;//Output1[too - to][trr - row][tcc - col];
						Write_dma_O2 = y2;
						Write_dma_O3 = y3;
						Write_dma_O4 = y4;
					}
					weightsout_dma_O_data.data.data1=xhat1;
					weightsout_dma_O_data.data.data2=xhat2;
					weightsout_dma_O_data.data.data3=xhat3;
					weightsout_dma_O_data.data.data4=xhat4;
					output_dma_O_data.data.data1=Write_dma_O1;
					output_dma_O_data.data.data2=Write_dma_O2;
					output_dma_O_data.data.data3=Write_dma_O3;
					output_dma_O_data.data.data4=Write_dma_O4;
					if(trr+1>=R && tcc+1>=C && too_current+4>=M ){
						weightsout_dma_O_data.last=1;
						output_dma_O_data.last=1;
					}
					else{
						weightsout_dma_O_data.last=0;
						output_dma_O_data.last=0;
					}
					dma_Weightsout.write(weightsout_dma_O_data);
					dma_Output.write(output_dma_O_data);
				}
				too_index++;
			}
		}
	}
	//}
}


void OFM_STORE_bnback( hls::stream<DMA_DATA_128> &dma_IFM,
		hls::stream<DMA_DATA_128> &dma_OFM,
		FPGA_DATA BNW[DMAwidth][BNbank][BNindex],
		//ap_uint<3> state,
		int8 custom_batch,
		//int state,
		//int13 ti,
		int14 to,
		int14 to_index,
		int14 M,
		//int13 M_index,
		//int13 M_in,
		int8 R,
		int8 C,
		hls::stream<DMA_DATA_128> &dma_Output){
	DMA_DATA_128 ifm_input_data;
	DMA_DATA_128 ofm_input_data;
	DMA_DATA_128 output_dma_O_data;
	int14 too_index=to_index;
	//for(int8 ba=0;ba<custom_batch;ba++){
	//	#pragma HLS loop_tripcount min=16 max=16 avg=16
	for(int8 trr=0;trr<R;trr++){
	#pragma HLS loop_tripcount min=32 max=32 avg=32
		for(int8 tcc=0;tcc<C;tcc++){
			too_index=to_index;
	#pragma HLS loop_tripcount min=32 max=32 avg=32
			for(int14 too=0;too<12*4;too+=4){
	#pragma HLS loop_tripcount min=6 max=6 avg=6
				int14 too_current=too+to;
				//int13 M_current=M_index+too_current;
	#pragma HLS PIPELINE II=1
	#pragma HLS dependence variable=BNW intra false
				FPGA_DATA ifm_input1;
				FPGA_DATA ifm_input2;
				FPGA_DATA ifm_input3;
				FPGA_DATA ifm_input4;
				FPGA_DATA ofm_input1;
				FPGA_DATA ofm_input2;
				FPGA_DATA ofm_input3;
				FPGA_DATA ofm_input4;
				FPGA_DATA Write_dma_O1;
				FPGA_DATA Write_dma_O2;
				FPGA_DATA Write_dma_O3;
				FPGA_DATA Write_dma_O4;
				if(too_current<M && too<Tm){// && M_current<M_in){
					ifm_input_data=dma_IFM.read();
					ofm_input_data=dma_OFM.read();
					ifm_input1 = ifm_input_data.data.data1;
					ifm_input2 = ifm_input_data.data.data2;
					ifm_input3 = ifm_input_data.data.data3;
					ifm_input4 = ifm_input_data.data.data4;
					ofm_input1 = ofm_input_data.data.data1;
					ofm_input2 = ofm_input_data.data.data2;
					ofm_input3 = ofm_input_data.data.data3;
					ofm_input4 = ofm_input_data.data.data4;

				}
				else{
					ifm_input1 = 0;
					ifm_input2 = 0;
					ifm_input3 = 0;
					ifm_input4 = 0;
					ofm_input1 = 0;
					ofm_input2 = 0;
					ofm_input3 = 0;
					ofm_input4 = 0;
				}
				Write_dma_O1=(ifm_input1-BNW[0][0][too_index]
					-BNW[0][1][too_index]*ofm_input1)*BNW[0][7][too_index];
				Write_dma_O2=(ifm_input2-BNW[1][0][too_index]
					-BNW[1][1][too_index]*ofm_input2)*BNW[1][7][too_index];
				Write_dma_O3=(ifm_input3-BNW[2][0][too_index]
					-BNW[2][1][too_index]*ofm_input3)*BNW[2][7][too_index];
				Write_dma_O4=(ifm_input4-BNW[3][0][too_index]
					-BNW[3][1][too_index]*ofm_input4)*BNW[3][7][too_index];
				if(too_current<M && too<Tm){// && M_current<M_in){
					output_dma_O_data.data.data1=Write_dma_O1;
					output_dma_O_data.data.data2=Write_dma_O2;
					output_dma_O_data.data.data3=Write_dma_O3;;
					output_dma_O_data.data.data4=Write_dma_O4;
					if(trr+1>=R && tcc+1>=C && too_current+4>=M){

						output_dma_O_data.last=1;
					}
					else{
						output_dma_O_data.last=0;
					}
					dma_Output.write(output_dma_O_data);
				}
				too_index++;
			}
		}
	}
}

void bn(	hls::stream<DMA_DATA_128> &dma_IFM,
			hls::stream<DMA_DATA_128> &dma_Weights,
			hls::stream<DMA_DATA_128> &dma_Weightsout,
			hls::stream<DMA_DATA_128> &dma_OFM,
			//int state,
			ap_uint<4> state,
			//int8 custom_batch,
			int8 batch_size,
			int14 M,
			int8 R,
			int8 C,
			hls::stream<DMA_DATA_128> &dma_Output,
			FPGA_DATA BNW[DMAwidth][BNbank][BNindex]
			//FPGA_DATA BNW_DB[DMAwidth][BNbank][BNindex]
			//int8& flag
			){
	//flag=100;
	FPGA_DATA Fnum=R*C*batch_size;
	FPGA_DATA Fnumdiv=1.0/Fnum;
	if(state>=4){//FP
		int14 to_index=0;
		//for(int14 to=0;to<M;to+=Tm){
		//#pragma HLS loop_tripcount min=3 max=3 avg=3

		LOAD_Weights_bn(dma_Weights,dma_OFM,dma_IFM,BNW,state,M);
			//}

		//}
		to_index=0;
		for(int14 to=0;to<M;to+=Tm){
		#pragma HLS loop_tripcount min=3 max=3 avg=3
			for(int8 ba=0;ba<batch_size;ba++){
		#pragma HLS loop_tripcount min=16 max=16 avg=16
				LOAD_bnfor(dma_IFM,BNW,batch_size,to,to_index,M,R,C);
			}
			//LOAD_Weights_bnback(dma_Weights,dma_OFM,dma_IFM,dma_Output,WEIGHT,to_index,to,M,Fnum);
			//OFM_STORE_bnback(dma_IFM,dma_OFM,WEIGHT,batch_size,to,to_index,M,R,C,dma_Output);
			to_index+=4;
		}
		Weights_bnfor(BNW,M,Fnumdiv);
		for(int8 ba=0;ba<batch_size;ba++){
			to_index=0;
		#pragma HLS loop_tripcount min=16 max=16 avg=16
			for(int14 to=0;to<M;to+=Tm){
			#pragma HLS loop_tripcount min=3 max=3 avg=3
				//LOAD_bnback(dma_IFM,dma_OFM,WEIGHT,batch_size,to,to_index,M,R,C,Fnum);
				//LOAD_Weights_bnback(dma_Weights,dma_OFM,dma_IFM,dma_Output,WEIGHT,to_index,to,M,Fnum);
				OFM_STORE_bnfor(dma_OFM,BNW,state,batch_size,to,to_index,M,R,C,dma_Weightsout,dma_Output);
				to_index+=4;
			}
		}
		//for(int print=0;print<2;print++){
		//	to_index=0;
		//	for(int14 to=0;to<M;to+=Tm){
		//	#pragma HLS loop_tripcount min=3 max=3 avg=3
		Weights_STORE_bn(BNW,state,M,dma_Weightsout,dma_Output);
		//		to_index+=4;
		//	}
		//}
	}
	else{//BP
		int14 to_index=0;
		//for(int14 to=0;to<M;to+=Tm){
		//#pragma HLS loop_tripcount min=3 max=3 avg=3
		LOAD_Weights_bn(dma_Weights,dma_OFM,dma_IFM,BNW,state,M);
		//	to_index+=4;
		//}
		to_index=0;
		for(int14 to=0;to<M;to+=Tm){
		#pragma HLS loop_tripcount min=3 max=3 avg=3
			for(int8 ba=0;ba<batch_size;ba++){
		#pragma HLS loop_tripcount min=16 max=16 avg=16
				LOAD_bnback(dma_IFM,dma_OFM,BNW,batch_size,to,to_index,M,R,C);
			//LOAD_Weights_bnback(dma_Weights,dma_OFM,dma_IFM,dma_Output,WEIGHT,to_index,to,M,Fnum);
			//OFM_STORE_bnback(dma_IFM,dma_OFM,WEIGHT,batch_size,to,to_index,M,R,C,dma_Output);
			}
			to_index+=4;
		}
		Weights_bnback(BNW,M,Fnumdiv);
		for(int8 ba=0;ba<batch_size;ba++){
			to_index=0;
		#pragma HLS loop_tripcount min=16 max=16 avg=16
			for(int14 to=0;to<M;to+=Tm){
		#pragma HLS loop_tripcount min=3 max=3 avg=3
				//LOAD_bnback(dma_IFM,dma_OFM,WEIGHT,batch_size,to,to_index,M,R,C,Fnum);
				//LOAD_Weights_bnback(dma_Weights,dma_OFM,dma_IFM,dma_Output,WEIGHT,to_index,to,M,Fnum);
				OFM_STORE_bnback(dma_IFM,dma_OFM,BNW,batch_size,to,to_index,M,R,C,dma_Output);
				to_index+=4;
			}
		}
//		to_index=0;
//		for(int14 to=0;to<M;to+=Tm){
//		#pragma HLS loop_tripcount min=3 max=3 avg=3
//			Weights_STORE_bn2(WEIGHT,to_index,to,M,dma_Output);
//			to_index++;
//		}
		//to_index=0;
		//for(int14 to=0;to<M;to+=Tm){
		//#pragma HLS loop_tripcount min=3 max=3 avg=3
			//Weights_STORE_bn(BNW,state,to_index,to,M,dma_Weightsout,dma_Output);
		Weights_STORE_bn(BNW,state,M,dma_Weightsout,dma_Output);
		//	to_index+=4;
		//}
	}
}







void LOAD_IFM(hls::stream<DMA_DATA_128> &dma_IFM,
		//Relu_index Reluindex1[M1][R1][C1],
		//int8 Relulayerin,
		FPGA_DATA IFM[Tn][Trtcin],
		ap_uint<1> state,
		//int state,
		int14 ti,
		int8 row,
		//int8 col,
		int14 N,
		int8 R,
		int8 C,
		int4 custom_stride,
		int4 padding,
		int4 custom_k,
		int8 custom_Tr,
		int8 custom_Tc
		//int8& flag
		){
	DMA_DATA_128 ifm_input_data;
	int8 i_upper=custom_Tr-1+custom_k;//(custom_Tr-1)*custom_stride+custom_k;
	int8 j_upper=custom_Tc-1+custom_k;//(custom_Tc-1)*custom_stride+custom_k;
	int8 i_current_sub=R - 1+ custom_k - padding;//(R - 1) * custom_stride + custom_k - padding;
	int8 j_current_sub=C - 1 + custom_k - padding;//(C - 1) * custom_stride + custom_k - padding;
	//int14 ij_upper=ij_upper;
	if(N<=4){// 1st layer
//		for(int14 i=0;i<ij_upper;i++){
//		#pragma HLS loop_tripcount min=240 max=240 avg=240
//		#pragma HLS PIPELINE II=1
//			for(int14 tii=0;tii<Tn;tii++){
//				IFM[tii][i] = 0;
//			}
//		}
		for(int8 i=0;i<i_upper;i++){
		#pragma HLS loop_tripcount min=10 max=10 avg=10
			int8 i_current=i+row;//i+row*custom_stride;
			for(int8 j=0;j<j_upper;j++){
		#pragma HLS loop_tripcount min=24 max=24 avg=24
				//int6 j_current=j;
				//flag=121;
		#pragma HLS PIPELINE II=1
		#pragma HLS dependence variable=IFM intra false
				int14 trtcindex=i*j_upper+j;
					//FPGA_DATA ifm_input;
				if(i_current >= padding && i_current <i_current_sub
				&& j >= padding && j< j_current_sub){
					ifm_input_data=dma_IFM.read();
					IFM[0][trtcindex] = ifm_input_data.data.data1;
					IFM[1][trtcindex] = ifm_input_data.data.data2;
					IFM[2][trtcindex] = ifm_input_data.data.data3;
					IFM[3][trtcindex] = ifm_input_data.data.data4;
				}
				else{
					IFM[0][trtcindex] = 0;
					IFM[1][trtcindex] = 0;
					IFM[2][trtcindex] = 0;
					IFM[3][trtcindex] = 0;
				}
			}
		}
	}
	else{
		for(int8 i=0;i<i_upper;i++){
		#pragma HLS loop_tripcount min=10 max=10 avg=10
			int8 i_current=i+row;//i+row*custom_stride;
			for(int8 j=0;j<j_upper;j++){
		#pragma HLS loop_tripcount min=24 max=24 avg=24
				//int8 j_current=j;
				//flag=121;
				for(int14 tii=0;tii<Tn;tii+=4){
					int14 tii_current = tii+ti;
		#pragma HLS loop_tripcount min=6 max=6 avg=6
		#pragma HLS PIPELINE II=1
		#pragma HLS dependence variable=IFM intra false
					//FPGA_DATA ifm_input;
					int14 trtcindex=i*j_upper+j;
					if(i_current >= padding && i_current <i_current_sub
					&& j >= padding && j< j_current_sub
					&& tii_current<N  ){
						ifm_input_data=dma_IFM.read();
						IFM[tii][trtcindex] = ifm_input_data.data.data1;
						IFM[tii+1][trtcindex] = ifm_input_data.data.data2;
						IFM[tii+2][trtcindex] = ifm_input_data.data.data3;
						IFM[tii+3][trtcindex] = ifm_input_data.data.data4;
						//flag=IFM[tii][i][j];

					}
					else{
						IFM[tii][trtcindex] = 0;
						IFM[tii+1][trtcindex] = 0;
						IFM[tii+2][trtcindex] = 0;
						IFM[tii+3][trtcindex] = 0;
						//flag=122;
					}
					//IFM[tii][i][j] = ifm_input;
				}
			}
		}
	}
}


void LOAD_Weights(hls::stream<DMA_DATA_128> &dma_Weights,
					 FPGA_DATA WEIGHT[Tm][Tn][TiTo][K][K],
					 ap_uint<1> state,
					 //int state,
					 int8 custom_Tib,
					 int8 to_index,
					 int8 ti_index,
					 int4 ki_index,
					 int4 kj_index,
					 int14 ti,
					 int14 to,
					 int14 N,
					 int14 M,
					 int14 M_index,
					 int14 M_in,
					 int4 custom_k
					 //int8& flag
					 ){

	DMA_DATA_128 weight_input_dma;
	int8 weights_index_base;
	#pragma HLS RESOURCE variable=weights_index_base core=MUL_LUT
	weights_index_base=to_index*custom_Tib+ti_index;
	int4 ki_index1=ki_index;
	int4 kj_index1 = kj_index;
	int4 ki = ki_index1;
	int4 kj = kj_index1;
	//int4 ti_count;
	if (state==0){//0 backward propagation the weights need to flip
		for(int14 to=0;to<M;to+=Tm){
		#pragma HLS loop_tripcount min=3 max=3 avg=3
			for (int14 tii=0;tii<Tn;tii+=4){
		#pragma HLS loop_tripcount min=6 max=6 avg=6
				int14 tii_current =tii+ti;
				for (int14 too=0;too<Tm;too++){
			#pragma HLS loop_tripcount min=3 max=3 avg=3
				int14 too_current =too+to;
				int14 M_current=too_current+M_index;
				for(int4 i=0;i<custom_k;i++){
				#pragma HLS loop_tripcount min=3 max=3 avg=3
						int4 flapi =custom_k-1-i;
						int4 ki=ki_index1+flapi;
						for(int4 j=0;j<custom_k;j++){
				#pragma HLS loop_tripcount min=3 max=3 avg=3
							int4 flapj =custom_k-1-j;
							int4 kj=kj_index1+flapj;
				#pragma HLS PIPELINE II=1
				#pragma HLS dependence variable=WEIGHT intra false
//							int14 weights_index;
							//FPGA_DATA weights_input;
//							#pragma HLS RESOURCE variable=weights_index core=MUL_LUT
//							weights_index=weights_index_base+flapi*custom_k+flapj;
							if(too_current<M && tii_current<N && M_current<M_in){
								//flag=140;
								weight_input_dma=dma_Weights.read();
								WEIGHT[too][tii][weights_index_base][ki][kj] = weight_input_dma.data.data1;
								WEIGHT[too][tii+1][weights_index_base][ki][kj] = weight_input_dma.data.data2;
								WEIGHT[too][tii+2][weights_index_base][ki][kj] = weight_input_dma.data.data3;
								WEIGHT[too][tii+3][weights_index_base][ki][kj] = weight_input_dma.data.data4;
								//flag=140;
							}
							else{
								WEIGHT[too][tii][weights_index_base][ki][kj] =0;
								WEIGHT[too][tii+1][weights_index_base][ki][kj] =0;
								WEIGHT[too][tii+2][weights_index_base][ki][kj] =0;
								WEIGHT[too][tii+3][weights_index_base][ki][kj] =0;
							}
							//WEIGHT[too][tii][weights_index_base][ki][kj] =weights_input;
						}
					}
				}
			}
//			if(N==1000 && custom_k<=1){//last FC for BP
//				ki_index1++;
//				if(ki_index1>=K){
//					weights_index_base+=custom_Tib;
//					ki_index1=0;
//					//ti_count=0;
//				}
//			}
//			else{
				weights_index_base+=custom_Tib;
//			}
		}
	}

	else{//1 forward propagation remain same
		for (int14 too=0;too<Tm;too+=4){
	#pragma HLS loop_tripcount min=6 max=6 avg=6
			int14 too_current =too+to;
			int14 M_current=too_current+M_index;
			for (int14 tii=0;tii<Tn;tii++){
	#pragma HLS loop_tripcount min=3 max=3 avg=3
				int14 tii_current =tii+ti;
				for(int4 i=0;i<custom_k;i++){
				#pragma HLS loop_tripcount min=3 max=3 avg=3
					int4 ki=ki_index1+i;
					//int4 flapi = custom_k-1-i;
					for(int4 j=0;j<custom_k;j++){
				#pragma HLS loop_tripcount min=3 max=3 avg=3
						int4 kj=kj_index1+j;
						//int4 flapj = custom_k-1-j;
		#pragma HLS PIPELINE II=1
		#pragma HLS dependence variable=WEIGHT intra false

//						int14 weights_index;
//						#pragma HLS RESOURCE variable=weights_index core=MUL_LUT
//						weights_index=weights_index_base+i*custom_k+j;
						//flag=to_index;
						//FPGA_DATA weights_input;
//							#pragma HLS RESOURCE variable=weights_index core=MUL_LUT
//							weights_index=weights_index_base+flapi*custom_k+flapj;
						if(too_current<M && tii_current<N && M_current<M_in){
							//flag=140;
							weight_input_dma=dma_Weights.read();
							WEIGHT[too][tii][weights_index_base][ki][kj] = weight_input_dma.data.data1;
							WEIGHT[too+1][tii][weights_index_base][ki][kj] = weight_input_dma.data.data2;
							WEIGHT[too+2][tii][weights_index_base][ki][kj] = weight_input_dma.data.data3;
							WEIGHT[too+3][tii][weights_index_base][ki][kj] = weight_input_dma.data.data4;
							 //flag=140;
						}
						else{
							WEIGHT[too][tii][weights_index_base][ki][kj] =0;
							WEIGHT[too+1][tii][weights_index_base][ki][kj] =0;
							WEIGHT[too+2][tii][weights_index_base][ki][kj] =0;
							WEIGHT[too+3][tii][weights_index_base][ki][kj] =0;
						}
						//WEIGHT[too][tii][weights_index_base][ki][kj] =weights_input;
					}
				}
			}
		}
	}
}




void LOAD_OFM(hls::stream<DMA_DATA_128> &dma_OFM,
//		Relu_index Reluindex1[M1][R1][C1],
//		int Relulayerin,
		FPGA_DATA OFM[Tm][Trtc],
		//ap_uint<5> state,
		//int state,
		//int14 ti,
		int14 to,
		int8 row,
		//int8 col,
		int14 M,
		int14 M_index,
		int14 M_in,
		int8 R,
		int8 C,
		int8 custom_Tr,
		int8 custom_Tc){
	DMA_DATA_128 ofm_input_data;
	//if((custom_Tr>=R && ti==0)||//when whole FM can be placed, OFM do not need to repeat [N/Tn] times)
	//		custom_Tr<R	){//FM is bigger than BRAM can hold, custom_Tr<R
		for(int8 trr=0;trr<custom_Tr;trr++){
		#pragma HLS loop_tripcount min=8 max=8 avg=8
			int8 trr_current=trr+row;
			for(int8 tcc=0;tcc<custom_Tc;tcc++){
		#pragma HLS loop_tripcount min=24 max=24 avg=24
				//int8 tcc_current=tcc+col;
				for(int14 too=0;too<Tm;too+=4){
		#pragma HLS loop_tripcount min=6 max=6 avg=6
					int14 too_current=too+to;
					int14 M_current=M_index+too_current;
		#pragma HLS PIPELINE II=1
		#pragma HLS dependence variable=OFM intra false
					//FPGA_DATA ofm_input;
					int14 trtcindex=trr*custom_Tc+tcc;
					if(trr_current<R && tcc<C && too_current<M && M_current<M_in){
						ofm_input_data=dma_OFM.read();
						OFM[too][trtcindex] = ofm_input_data.data.data1;
						OFM[too+1][trtcindex] = ofm_input_data.data.data2;
						OFM[too+2][trtcindex] = ofm_input_data.data.data3;
						OFM[too+3][trtcindex] = ofm_input_data.data.data4;
					}
					else{
						OFM[too][trtcindex] = 0;
						OFM[too+1][trtcindex] = 0;
						OFM[too+2][trtcindex] = 0;
						OFM[too+3][trtcindex] = 0;
					}
					//OFM[too][trr][tcc] = ofm_input;
				}
			}
		}
	//}
}
//
//void FIRE(  FPGA_DATA IFM[Tn][Trtcin],
//			FPGA_DATA WEIGHT[Tm][Tn][TiTo][K][K],
//			FPGA_DATA OFM[Tm][Trtc],
//			ap_uint<1> state,
//			//int state,
//			int8 custom_Tib,
//			int8 to_index,
//			int8 ti_index,
//			int4 ki_index,
//			int4 kj_index,
//			//int14 tito_index,
//			int14 ti,
//			int14 to,
//			int8 row,
//			//int8 col,
//			int14 N,
//			int14 M,
//			int8 R,
//			int8 C,
//			int4 custom_stride,
//			int4 padding,
//			int4 custom_k,
//			int8 custom_Tr,
//			int8 custom_Tc
//			//FPGA_DATA Output1[Tm][Tr][Tc],
//			//FPGA_DATA Output2[Tm][Tn][K2],
//			//int8& flag
//			){
//
//	int8 weights_index_base;
//	#pragma HLS RESOURCE variable=weights_index_base core=MUL_LUT
//	weights_index_base=to_index*custom_Tib+ti_index;
//	int8 j_upper=custom_Tc-1+custom_k;//(custom_Tc-1)*custom_stride+custom_k;
////	int4 kib=ki_index+custom_k;
////	int4 kjb=kj_index+custom_k;
//	FPGA_DATA Weights_reg[Tm][Tn][K][K];
//	#pragma HLS RESOURCE variable= Weights_reg core=RAM_S2P_BRAM
//	#pragma HLS ARRAY_PARTITION variable=Weights_reg complete dim=1
//	#pragma HLS ARRAY_PARTITION variable=Weights_reg complete dim=2
//	for(int4 i=0;i<custom_k;i++){
//	#pragma HLS loop_tripcount min=3 max=3 avg=3
//		int4 ki=ki_index+i;
//		for(int4 j=0;j<custom_k;j++){
//	#pragma HLS loop_tripcount min=3 max=3 avg=3
//			int4 kj=kj_index+j;
//			//int8 k_index=i*custom_k+j;
//	#pragma HLS PIPELINE II=1
//			for(int14 too=0;too<Tm;too++){
//				for(int14 tii=0;tii<Tn;tii++){
//					 Weights_reg[too][tii][i][j]=WEIGHT[too][tii][weights_index_base][ki][kj];
//				}
//			}
//		}
//	}
//	if(state == 1){//1 forward or backward propagation
//		for(int8 i=0;i<custom_k;i++){
//		#pragma HLS loop_tripcount min=3 max=3 avg=3
//			for(int8 j=0;j<custom_k;j++){
//		#pragma HLS loop_tripcount min=3 max=3 avg=3
//				//int8 k_index = i*custom_k+j;
//				for(int8 trr=0;trr<custom_Tr;trr++){
//		#pragma HLS loop_tripcount min=8 max=8 avg=8
//					//int8 ifm_trr=custom_stride*trr+i;
//					for(int8 tcc=0;tcc<custom_Tc;tcc++){
//						//flag=130;
//		#pragma HLS loop_tripcount min=24 max=24 avg=24
//						//int8 ifm_tcc=custom_stride*tcc+j;
//
//		#pragma HLS PIPELINE II=1
//						#pragma HLS dependence variable=OFM inter false
//						int14 trtcindex=trr*custom_Tc+tcc;
//						int14 trtcindexifm=(trr+i)*j_upper+tcc+j;//(custom_stride*trr+i)*j_upper+custom_stride*tcc+j;
//
//						for(int14 too=0;too<Tm; too++){
//							FPGA_DATA add_res1[Tn];
//							FPGA_DATA add_res2[8];
//							FPGA_DATA add_res3[4];
//							FPGA_DATA add_res4[2];
//							FPGA_DATA add_res5;
//							for(int14 tii=0;tii<Tn;tii++){
//								add_res1[tii] = Weights_reg[too][tii][i][j]*IFM[tii][trtcindexifm];
//							}
//							add_res2[0]=add_res1[0]+add_res1[1];
//							add_res2[1]=add_res1[2]+add_res1[3];
//							add_res2[2]=add_res1[4]+add_res1[5];
//							add_res2[3]=add_res1[6]+add_res1[7];
//							add_res2[4]=add_res1[8]+add_res1[9];
//							add_res2[5]=add_res1[10]+add_res1[11];
//							add_res2[6]=add_res1[12]+add_res1[13];
//							add_res2[7]=add_res1[14]+add_res1[15];
//							add_res3[0]=add_res2[0]+add_res2[1];
//							add_res3[1]=add_res2[2]+add_res2[3];
//							add_res3[2]=add_res2[4]+add_res2[5];
//							add_res3[3]=add_res2[6]+add_res2[7];
//							add_res4[0]=add_res3[0]+add_res3[1];
//							add_res4[1]=add_res3[2]+add_res3[3];
//							add_res5=add_res4[0]+add_res4[1];
//							OFM[too][trtcindex] = OFM[too][trtcindex] + add_res5;
//						}
//					}
//				}
//			}
//		}
//	}
//	else {//0update weights
//		//if(custom_k>1){//common conv
//
//			for(int8 trr=0;trr<custom_Tr;trr++){
//			#pragma HLS loop_tripcount min=8 max=8 avg=8
//				//int8 ifm_trr=custom_stride*trr+i;
//				for(int8 tcc=0;tcc<custom_Tc;tcc++){
//					//flag=130;
//			#pragma HLS loop_tripcount min=24 max=24 avg=24
//					//int8 ifm_tcc=custom_stride*tcc+j;
//					for(int8 i=0;i<custom_k;i++){
//			#pragma HLS loop_tripcount min=3 max=3 avg=3
//						for(int8 j=0;j<custom_k;j++){
//			#pragma HLS loop_tripcount min=3 max=3 avg=3
//							//int8 k_index = i*custom_k+j;
//			#pragma HLS PIPELINE II=1
//
//
//			#pragma HLS dependence variable=Weights_reg inter false
//			int14 trtcindex=trr*custom_Tc+tcc;
//			int14 trtcindexifm=(trr+i)*j_upper+tcc+j;//(custom_stride*trr+i)*j_upper+custom_stride*tcc+j;
//							for(int14 too=0;too<Tm; too++){
//								for(int14 tii=0;tii<Tn;tii++){
//
//									FPGA_DATA add_res1;
//									add_res1 = OFM[too][trtcindex]*IFM[tii][trtcindexifm];
//									Weights_reg[too][tii][i][j]=add_res1+Weights_reg[too][tii][i][j];
//								}
//							}
//						}
//					}
//				}
//			}
//			for(int4 i=0;i<custom_k;i++){
//			#pragma HLS loop_tripcount min=3 max=3 avg=3
//				int4 ki=ki_index+i;
//				for(int4 j=0;j<custom_k;j++){
//			#pragma HLS loop_tripcount min=3 max=3 avg=3
//					int4 kj=kj_index+j;
//					//int8 k_index=i*custom_k+j;
//			#pragma HLS PIPELINE II=1
//					for(int14 too=0;too<Tm;too++){
//						for(int14 tii=0;tii<Tn;tii++){
//							WEIGHT[too][tii][weights_index_base][ki][kj] =Weights_reg[too][tii][i][j];
//						}
//					}
//				}
//			}
//		}
//
//}
//



void FIRE(  FPGA_DATA IFM[Tn][Trtcin],
			FPGA_DATA WEIGHT[Tm][Tn][TiTo][K][K],
			FPGA_DATA OFM[Tm][Trtc],
			ap_uint<1> state,
			//int state,
			int8 custom_Tib,
			int8 to_index,
			int8 ti_index,
			int4 ki_index,
			int4 kj_index,
			//int14 tito_index,
			int14 ti,
			int14 to,
			int8 row,
			//int8 col,
			int14 N,
			int14 M,
			int8 R,
			int8 C,
			int4 custom_stride,
			int4 padding,
			int4 custom_k,
			int8 custom_Tr,
			int8 custom_Tc
			//FPGA_DATA Output1[Tm][Tr][Tc],
			//FPGA_DATA Output2[Tm][Tn][K2],
			//int8& flag
			){

	int8 weights_index_base;
	#pragma HLS RESOURCE variable=weights_index_base core=MUL_LUT
	weights_index_base=to_index*custom_Tib+ti_index;
	int8 j_upper=custom_Tc-1+custom_k;//(custom_Tc-1)*custom_stride+custom_k;
	if(custom_k>1){//common conv
		if(state == 1){//1 forward or backward propagation
			for(int4 i=0;i<K;i++){
			#pragma HLS loop_tripcount min=3 max=3 avg=3
				for(int4 j=0;j<K;j++){
			#pragma HLS loop_tripcount min=3 max=3 avg=3
					//int8 k_index = i*custom_k+j;
					for(int8 trr=0;trr<custom_Tr;trr++){
			#pragma HLS loop_tripcount min=8 max=8 avg=8
						//int8 ifm_trr=custom_stride*trr+i;
						for(int8 tcc=0;tcc<custom_Tc;tcc++){
							//flag=130;
			#pragma HLS loop_tripcount min=24 max=24 avg=24
							//int8 ifm_tcc=custom_stride*tcc+j;

			#pragma HLS PIPELINE II=1
			#pragma HLS dependence variable=OFM inter false
							int14 trtcindex=trr*custom_Tc+tcc;
							int14 trtcindexifm=(trr+i)*j_upper+tcc+j;//(custom_stride*trr+i)*j_upper+custom_stride*tcc+j;

							for(int14 too=0;too<Tm; too++){
								FPGA_DATA add_res1[Tn];
								FPGA_DATA add_res2[8];
								FPGA_DATA add_res3[4];
								FPGA_DATA add_res4[2];
								FPGA_DATA add_res5;
								for(int14 tii=0;tii<Tn;tii++){
									add_res1[tii] =WEIGHT[too][tii][weights_index_base][i][j]*IFM[tii][trtcindexifm];
								}
								add_res2[0]=add_res1[0]+add_res1[1];
								add_res2[1]=add_res1[2]+add_res1[3];
								add_res2[2]=add_res1[4]+add_res1[5];
								add_res2[3]=add_res1[6]+add_res1[7];
								add_res2[4]=add_res1[8]+add_res1[9];
								add_res2[5]=add_res1[10]+add_res1[11];
								add_res2[6]=add_res1[12]+add_res1[13];
								add_res2[7]=add_res1[14]+add_res1[15];
								add_res3[0]=add_res2[0]+add_res2[1];
								add_res3[1]=add_res2[2]+add_res2[3];
								add_res3[2]=add_res2[4]+add_res2[5];
								add_res3[3]=add_res2[6]+add_res2[7];
								add_res4[0]=add_res3[0]+add_res3[1];
								add_res4[1]=add_res3[2]+add_res3[3];
								add_res5=add_res4[0]+add_res4[1];
								OFM[too][trtcindex] = OFM[too][trtcindex] + add_res5;
							}
						}
					}
				}
			}
		}
		else {//0update weights
			for(int8 trr=0;trr<custom_Tr;trr++){
			#pragma HLS loop_tripcount min=8 max=8 avg=8
				//int8 ifm_trr=custom_stride*trr+i;
				for(int8 tcc=0;tcc<custom_Tc;tcc++){
					//flag=130;
			#pragma HLS loop_tripcount min=24 max=24 avg=24
					//int8 ifm_tcc=custom_stride*tcc+j;
					for(int4 i=0;i<K;i++){
			#pragma HLS loop_tripcount min=3 max=3 avg=3
						for(int4 j=0;j<K;j++){
			#pragma HLS loop_tripcount min=3 max=3 avg=3
							//int8 k_index = i*custom_k+j;
			#pragma HLS PIPELINE II=1
			#pragma HLS dependence variable=WEIGHT inter false
							int14 trtcindex=trr*custom_Tc+tcc;
							int14 trtcindexifm=(trr+i)*j_upper+tcc+j;//(custom_stride*trr+i)*j_upper+custom_stride*tcc+j;
							for(int14 too=0;too<Tm; too++){
								for(int14 tii=0;tii<Tn;tii++){
									FPGA_DATA add_res1;
									add_res1 = OFM[too][trtcindex]*IFM[tii][trtcindexifm];
									WEIGHT[too][tii][weights_index_base][i][j]=add_res1+WEIGHT[too][tii][weights_index_base][i][j];
								}
							}
						}
					}
				}
			}
		}
	}
	else{//FC
		int4 i=ki_index;
		int4 j=kj_index;
		if(state == 1){//1 forward or backward propagation
			#pragma HLS dependence variable=OFM inter false
			for(int14 too=0;too<Tm; too++){
			#pragma HLS unroll
				FPGA_DATA add_res1[Tn];
				FPGA_DATA add_res2[8];
				FPGA_DATA add_res3[4];
				FPGA_DATA add_res4[2];
				FPGA_DATA add_res5;
				for(int14 tii=0;tii<Tn;tii++){
			#pragma HLS unroll
					add_res1[tii] =WEIGHT[too][tii][weights_index_base][i][j]*IFM[tii][0];
				}
				add_res2[0]=add_res1[0]+add_res1[1];
				add_res2[1]=add_res1[2]+add_res1[3];
				add_res2[2]=add_res1[4]+add_res1[5];
				add_res2[3]=add_res1[6]+add_res1[7];
				add_res2[4]=add_res1[8]+add_res1[9];
				add_res2[5]=add_res1[10]+add_res1[11];
				add_res2[6]=add_res1[12]+add_res1[13];
				add_res2[7]=add_res1[14]+add_res1[15];
				add_res3[0]=add_res2[0]+add_res2[1];
				add_res3[1]=add_res2[2]+add_res2[3];
				add_res3[2]=add_res2[4]+add_res2[5];
				add_res3[3]=add_res2[6]+add_res2[7];
				add_res4[0]=add_res3[0]+add_res3[1];
				add_res4[1]=add_res3[2]+add_res3[3];
				add_res5=add_res4[0]+add_res4[1];
				OFM[too][0] = OFM[too][0] + add_res5;
			}
		}
		else{//update weights
		#pragma HLS dependence variable=WEIGHT inter false
			for(int14 too=0;too<Tm; too++){
		#pragma HLS unroll
				for(int14 tii=0;tii<Tn;tii++){
		#pragma HLS unroll
					FPGA_DATA add_res1;
					add_res1 = OFM[too][0]*IFM[tii][0];
					WEIGHT[too][tii][weights_index_base][i][j]=add_res1+WEIGHT[too][tii][weights_index_base][i][j];
				}
			}
		}
	}
}



void OFM_STORE( FPGA_DATA Output[Tm][Trtc],
				//FPGA_DATA OFM[Tm][Tr][Tc],
//				Relu_index Reluindex1[4][M1][R1][C1],
//				Relu_index Reluindex3[4][M3][R3][C3],
//				Relu_index Reluindex5[4][M5][R5][C5],
//				Relu_index Reluindex6[4][M6][R6][C6],
//				int4 Relulayerin,
//				int4 Relulayerout,
				//int state,
				ap_uint<2> state,
				int8 custom_batch,
				int8 batch_size,
				int8 ba,
				int14 to,
				int8 row,
				//int8 col,
				int14 M,
				int14 M_index,
				int14 M_in,
				int8 R,
				int8 C,
				int8 custom_Tr,
				int8 custom_Tc,
				hls::stream<DMA_DATA_128> &dma_OFM,
				hls::stream<DMA_DATA_128> &dma_Output){
				//int8& flag){
	DMA_DATA_128 output_dma_O_data;
	DMA_DATA_128 ofm_dma_O_data;
	//Relu_index Reluindex_data;
	for(int8 trr=0;trr<custom_Tr;trr++){
	#pragma HLS loop_tripcount min=8 max=8 avg=8
		int8 trr_current=trr+row;
		for(int8 tcc=0;tcc<custom_Tc;tcc++){
	#pragma HLS loop_tripcount min=24 max=24 avg=24
			//flag=150;
			//int8 tcc_current=tcc+col;
			for(int14 too=0;too<Tm;too+=4){
	#pragma HLS loop_tripcount min=3 max=3 avg=3
				int14 too_current=too+to;
				int14 M_current=too_current+M_index;
	#pragma HLS PIPELINE II=1

	//#pragma HLS dependence variable=Output1 intra false
	#pragma HLS dependence variable=Output intra false
				int14 trtcindex=trr*custom_Tc+tcc;
				//Output[too][trr][tcc]=0;
				if( trr_current<R && tcc<C && too_current<M && M_current<M_in){
					FPGA_DATA Read_Output1;
					FPGA_DATA Read_Output2;
					FPGA_DATA Read_Output3;
					FPGA_DATA Read_Output4;
					FPGA_DATA Write_dma_O1;
					FPGA_DATA Write_dma_O2;
					FPGA_DATA Write_dma_O3;
					FPGA_DATA Write_dma_O4;
					Read_Output1 = Output[too][trtcindex];
					Read_Output2 = Output[too+1][trtcindex];
					Read_Output3 = Output[too+2][trtcindex];
					Read_Output4 = Output[too+3][trtcindex];
					if(state[0]==1){//XX1 relu after conv need to be processed
						if(state[1]==1){//forward, calculate relu and store into activation
							 //if (Output1[too - to][trr - row][tcc - col] > 0){
							if (Read_Output1 > 0){
								//flag=OFM[too][trr][tcc];
								Write_dma_O1 =Read_Output1;// Output1[too - to][trr - row][tcc - col];
								//Reluindex_data = 1;
							}
							else{
								//flag=151;
								Write_dma_O1 = 0;//Output1[too - to][trr - row][tcc - col];
								//Reluindex_data = 0;
							}
							if (Read_Output2 > 0){
								//flag=OFM[too][trr][tcc];
								Write_dma_O2 =Read_Output2;// Output1[too - to][trr - row][tcc - col];
								//Reluindex_data = 1;
							}
							else{
								//flag=151;
								Write_dma_O2 = 0;//Output1[too - to][trr - row][tcc - col];
								//Reluindex_data = 0;
							}
							if (Read_Output3 > 0){
								//flag=OFM[too][trr][tcc];
								Write_dma_O3 =Read_Output3;// Output1[too - to][trr - row][tcc - col];
								//Reluindex_data = 1;
							}
							else{
								//flag=151;
								Write_dma_O3 = 0;//Output1[too - to][trr - row][tcc - col];
								//Reluindex_data = 0;
							}
							if (Read_Output4 > 0){
								//flag=OFM[too][trr][tcc];
								Write_dma_O4 =Read_Output4;// Output1[too - to][trr - row][tcc - col];
								//Reluindex_data = 1;
							}
							else{
								//flag=151;
								Write_dma_O4 = 0;//Output1[too - to][trr - row][tcc - col];
								//Reluindex_data = 0;
							}
						}
						else{//backward, use reluindexin from last layer update loss
							FPGA_DATA Relu_activ1;
							FPGA_DATA Relu_activ2;
							FPGA_DATA Relu_activ3;
							FPGA_DATA Relu_activ4;
							ofm_dma_O_data=dma_OFM.read();
							Relu_activ1=ofm_dma_O_data.data.data1;
							Relu_activ2=ofm_dma_O_data.data.data2;
							Relu_activ3=ofm_dma_O_data.data.data3;
							Relu_activ4=ofm_dma_O_data.data.data4;
							if (Relu_activ1 <= 0){
								Write_dma_O1 = 0;
							 }
							else{
								//flag=156;
								Write_dma_O1 = Read_Output1;//Output1[too - to][trr - row][tcc - col];
							}
							if (Relu_activ2 <= 0){
								Write_dma_O2 = 0;
							 }
							else{
								//flag=156;
								Write_dma_O2 = Read_Output2;//Output1[too - to][trr - row][tcc - col];
							}
							if (Relu_activ3 <= 0){
								Write_dma_O3 = 0;
							 }
							else{
								//flag=156;
								Write_dma_O3 = Read_Output3;//Output1[too - to][trr - row][tcc - col];
							}
							if (Relu_activ4 <= 0){
								Write_dma_O4 = 0;
							 }
							else{
								//flag=156;
								Write_dma_O4 = Read_Output4;//Output1[too - to][trr - row][tcc - col];
							}
						}
					}
					else{
						//flag=153;
						Write_dma_O1 = Read_Output1;//Output1[too - to][trr - row][tcc - col];
						Write_dma_O2 = Read_Output2;
						Write_dma_O3 = Read_Output3;
						Write_dma_O4 = Read_Output4;
					}
					output_dma_O_data.data.data1=Write_dma_O1;
					output_dma_O_data.data.data2=Write_dma_O2;
					output_dma_O_data.data.data3=Write_dma_O3;
					output_dma_O_data.data.data4=Write_dma_O4;
					if(trr_current+1>=R && tcc+1>=C &&
							((M<M_in && (too_current+4>=M || M_current+4>=M_in))
							||(M>=M_in && ((ba+1>=batch_size)|| ((ba%custom_batch)==(custom_batch-1))) && too_current+4>=M))){
						output_dma_O_data.last=1;
					}
					else{
						output_dma_O_data.last=0;
					}

					dma_Output.write(output_dma_O_data);
				}

			}
		}
	}
	for(int8 trr=0;trr<custom_Tr;trr++){
	#pragma HLS loop_tripcount min=8 max=8 avg=8
		for(int8 tcc=0;tcc<custom_Tc;tcc++){
	#pragma HLS loop_tripcount min=24 max=24 avg=24
	#pragma HLS PIPELINE II=1
			int14 trtcindex=trr*custom_Tc+tcc;
			for(int14 too=0;too<Tm;too++){
				//Output1[too][trr - row][tcc - col]=0;
				Output[too][trtcindex]=0;
			}
		}
	}
}


void Weights_STORE( hls::stream<DMA_DATA_128> &dma_Weights,
					FPGA_DATA Output[Tm][Tn][TiTo][K][K],
					//FPGA_DATA WEIGHT[Tm][Tn][K][K],
					ap_uint<1> state,
					//int8 batch_size,
					//int8 custom_batch,
					//int8 ba,
					int8 custom_Tib,
					int8 to_index,
					int8 ti_index,
					int4 ki_index,
					int4 kj_index,
					int14 ti,
					int14 to,
					int14 N,
					int14 M,
					int14 M_index,
					int14 M_in,
					int4 custom_k,
					hls::stream<DMA_DATA_128> &dma_Output){

	DMA_DATA_128 output_dma_O_data;
	DMA_DATA_128 weight_input_dma;
	FPGA_DATA coeff=learn_rate;
	int8 weights_index_base;
	#pragma HLS RESOURCE variable=weights_index_base core=MUL_LUT
	weights_index_base=to_index*custom_Tib+ti_index;
	//int4 ki=ki_index;
	//int4 kj=kj_index;

	for(int14 too=0;too<Tm;too+=4){
	#pragma HLS loop_tripcount min=3 max=3 avg=3
		int14 too_current=too+to;
		int14 M_current=too_current+M_index;
		for(int14 tii=0;tii<Tn;tii++){
	#pragma HLS loop_tripcount min=6 max=6 avg=6
			int14 tii_current=tii+ti;
			for(int4 i=0;i<custom_k;i++){
			#pragma HLS loop_tripcount min=3 max=3 avg=3
				int4 ki=ki_index+i;
				for(int4 j=0;j<custom_k;j++){
			#pragma HLS loop_tripcount min=3 max=3 avg=3
					int4 kj=kj_index+j;
					//int8 k_index=i*custom_k+j;

	#pragma HLS PIPELINE II=1

	//#pragma HLS dependence variable=WEIGHT intra false
	#pragma HLS dependence variable=Output intra false

//					int14 weights_index;
//					#pragma HLS RESOURCE variable=weights_index core=MUL_LUT
//					weights_index=weights_index_base+i*custom_k+j;

					if (too_current<M && tii_current<N && M_current<M_in){
						FPGA_DATA weights1;
						FPGA_DATA weights2;
						FPGA_DATA weights3;
						FPGA_DATA weights4;
						FPGA_DATA Weights_Output1;
						FPGA_DATA Weights_Output2;
						FPGA_DATA Weights_Output3;
						FPGA_DATA Weights_Output4;
						Weights_Output1=Output[too][tii][weights_index_base][ki][kj];
						Weights_Output2=Output[too+1][tii][weights_index_base][ki][kj];
						Weights_Output3=Output[too+2][tii][weights_index_base][ki][kj];
						Weights_Output4=Output[too+3][tii][weights_index_base][ki][kj];
						weight_input_dma=dma_Weights.read();
						weights1 = weight_input_dma.data.data1;
						weights2 = weight_input_dma.data.data2;
						weights3 = weight_input_dma.data.data3;
						weights4 = weight_input_dma.data.data4;
						output_dma_O_data.data.data1 =weights1 - Weights_Output1*coeff;//Output[too][tii][i][j]; //
						output_dma_O_data.data.data2 =weights2 - Weights_Output2*coeff;
						output_dma_O_data.data.data3 =weights3 - Weights_Output3*coeff;
						output_dma_O_data.data.data4 =weights4 - Weights_Output4*coeff;
						if(i+1>=custom_k && j+1>=custom_k && (M_current+4>=M_in ||(state==1 && too_current+4>=M)) &&  tii_current+1>=N){
							output_dma_O_data.last=1;
						}
						else{
							output_dma_O_data.last=0;
						}
						dma_Output.write(output_dma_O_data);
					}
				}
			}
		}
	}

	for(int4 i=0;i<custom_k;i++){
	#pragma HLS loop_tripcount min=3 max=3 avg=3
		int4 ki=ki_index+i;
		for(int4 j=0;j<custom_k;j++){
	#pragma HLS loop_tripcount min=3 max=3 avg=3
			int4 kj=kj_index+j;
			//int8 k_index=i*custom_k+j;
	#pragma HLS PIPELINE II=1
			for(int14 too=0;too<Tm;too++){
				for(int14 tii=0;tii<Tn;tii++){
					 Output[too][tii][weights_index_base][ki][kj]=0;
				}
			}
		}

	}
}




void acc(	hls::stream<DMA_DATA_128> &dma_IFM,
//			Relu_index Reluindex1[4][M1][R1][C1],
//			Relu_index Reluindex3[4][M3][R3][C3],
//			Relu_index Reluindex5[4][M5][R5][C5],
//			Relu_index Reluindex6[4][M6][R6][C6],
//			int4 Relulayerin,
//			int4 Relulayerout,
			hls::stream<DMA_DATA_128> &dma_Weights,
			hls::stream<DMA_DATA_128> &dma_OFM,
			//int state,
			ap_uint<3> state,
			//int8& flag,
			//int to,
			int8 custom_batch,
			int8 batch_size,
			int14 M_in,
			int8 custom_Tib,
			//int14 custom_k2,
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
//			int custom_Tm,
//			int custom_Tn,
			hls::stream<DMA_DATA_128> &dma_Output,
//			FPGA_DATA Output1[Tm][Tr][Tc],
//			FPGA_DATA Output1_DB[Tm][Tr][Tc],
//			FPGA_DATA Output2[Tm][Tn][K2],
//			FPGA_DATA Output2_DB[Tm][Tn][K2],
////			FPGA_DATA Output2[Tm][Tn][K][K],
////			FPGA_DATA Output2_DB[Tm][Tn][K][K],
			FPGA_DATA IFM[Tn][Trtcin],
			FPGA_DATA IFM_DB[Tn][Trtcin],
			FPGA_DATA WEIGHT[Tm][Tn][TiTo][K][K],
			FPGA_DATA WEIGHT_DB[Tm][Tn][TiTo][K][K],
			FPGA_DATA OFM[Tm][Trtc],
			FPGA_DATA OFM_DB[Tm][Trtc]
			//int8& flag
			){
	//flag=100;
	if(N<=4){
		int8 r_clear=custom_Tr-1+custom_k;//(custom_Tc-1)*custom_stride+custom_k;
		int8 c_clear=custom_Tc-1+custom_k;//(custom_Tc-1)*custom_stride+custom_k;
		Initialize_IFM(IFM,IFM_DB,r_clear,c_clear);
	}
	if(state>=4){//11X-10Xforward or backward propagation
		//int14 to_index_bound=0;
		for(int14 M_index=0;M_index<M_in;M_index+=M){
		#pragma HLS loop_tripcount min=2 max=2 avg=2
			for(int8 ba=0;ba<batch_size;ba++){
		#pragma HLS loop_tripcount min=4 max=4 avg=4
				//flag=101;
				int num =0;
				//int8 colp;
				int8 rowp=0;
				int14 topre=0;
				int8 to_index=0;
				//int4 ki_indexout=0;
				//int4 kj_indexout=0;
				for(int14 to=0;to<M;to+=Tm){
				#pragma HLS loop_tripcount min=3 max=3 avg=3
					if(to+M_index<M_in){
						for(int8 row=0;row<R;row+=custom_Tr){
					#pragma HLS loop_tripcount min=3 max=3 avg=3
							if(ba==0 && row==0 && (state[1]==1 || (state[1]==0 && to==0))){
								//only for the 1st row of the 1st image of the batch
								//forward or the 1st Tm for backward needs to load weights)

								if(num==0){
									int idxp = 0;
									int14 tip=0;
									int8 ti_index=0;
									int8 ti_indexp=0;
									int4 ki_index=0;
									int4 ki_count=0;
									//int4 ki_count=ki_indexout;
									int4 kj_index=0;
									int4 kj_count=0;
									int4 ki_indexp=0;
									//int4 ki_indexp=ki_indexout;
									int4 kj_indexp=0;
									for(int14 ti=0;ti<N;ti+=Tn){
										//flag=120;
									#pragma HLS loop_tripcount min=6 max=6 avg=6
										if(idxp==0){
											LOAD_Weights(dma_Weights,WEIGHT,state[1],custom_Tib,
											to_index,ti_index,ki_index,kj_index,ti,to,N,M,M_index,M_in,custom_k);
											LOAD_IFM(dma_IFM,IFM,state[2],ti,row,N,R,C,
											custom_stride,padding,custom_k,custom_Tr,custom_Tc);
										}
										else {
											if(idxp%2==0){
												LOAD_Weights(dma_Weights,WEIGHT,state[1],custom_Tib,
												to_index,ti_index,ki_index,kj_index,ti,to,N,M,M_index,M_in,custom_k);
												LOAD_IFM(dma_IFM,IFM,state[2],ti,row,N,R,C,
												custom_stride,padding,custom_k,custom_Tr,custom_Tc);
												FIRE(IFM_DB,WEIGHT_DB,OFM,state[2],custom_Tib,
												to_index,ti_indexp,ki_indexp,kj_indexp,tip,to,row,N,M,R,C,custom_stride,
												padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
												ti_indexp=ti_index;//for fire from weights buffer,ti_indexp increase after the two Tn fired
												ki_indexp=ki_index;
												kj_indexp=kj_index;
											}
											else{
												LOAD_Weights(dma_Weights,WEIGHT_DB,state[1],custom_Tib,
												to_index,ti_index,ki_index,kj_index,ti,to,N,M,M_index,M_in,custom_k);
												LOAD_IFM(dma_IFM,IFM_DB,state[2],ti,row,N,R,C,
												custom_stride,padding,custom_k,custom_Tr,custom_Tc);
												FIRE(IFM,WEIGHT,OFM,state[2],custom_Tib,
												to_index,ti_indexp,ki_indexp,kj_indexp,tip,to,row,N,M,R,C,custom_stride,
												padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
												//for load weights buffer,ti_index increase after the two Tn loaded
//												if(N==1000 && custom_k<=1){//last FC for BP
//													kj_index++;
//													if(kj_index>=K){
//														ti_index++;
//														kj_index=0;
//													}
//													//ki_index=ki_indexout;
//												}
//												else{//others
													kj_count++;
													kj_index+=custom_k;
													if(kj_count>=custom_kb){
														ki_count++;
														ki_index+=custom_k;
														kj_count=0;
														kj_index=0;
													}
													if(ki_count>=custom_kb){
														ti_index++;
														ki_count=0;
														ki_index=0;
													}
//												}
											}
										}
										tip=ti;
										idxp+=1;
									}
									if(idxp%2==0){
										FIRE(IFM_DB,WEIGHT_DB,OFM,state[2],custom_Tib,
										to_index,ti_indexp,ki_indexp,kj_indexp,tip,to,row,N,M,R,C,custom_stride,
										padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
									}else{
										FIRE(IFM,WEIGHT,OFM,state[2],custom_Tib,
										to_index,ti_indexp,ki_indexp,kj_indexp,tip,to,row,N,M,R,C,custom_stride,
										padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
									}
								}
								else {
									if(num%2==0){
										int idxp = 0;
										int14 tip;
										int14 ti_index=0;
										int14 ti_indexp=0;
										int4 ki_index=0;//ki_indexout;
										int4 ki_count=0;
										int4 kj_index=0;
										int4 kj_count=0;
										int4 ki_indexp=0;//ki_indexout;
										int4 kj_indexp=0;
										for(int14 ti=0;ti<N;ti+=Tn){
											//flag=120;
										#pragma HLS loop_tripcount min=6 max=6 avg=6
											if(idxp==0){
												LOAD_Weights(dma_Weights,WEIGHT,state[1],custom_Tib,
												to_index,ti_index,ki_index,kj_index,ti,to,N,M,M_index,M_in,custom_k);
												LOAD_IFM(dma_IFM,IFM,state[2],ti,row,N,R,C,
												custom_stride,padding,custom_k,custom_Tr,custom_Tc);

											}
											else {
												if(idxp%2==0){
													LOAD_Weights(dma_Weights,WEIGHT,state[1],custom_Tib,
													to_index,ti_index,ki_index,kj_index,ti,to,N,M,M_index,M_in,custom_k);
													LOAD_IFM(dma_IFM,IFM,state[2],ti,row,N,R,C,
													custom_stride,padding,custom_k,custom_Tr,custom_Tc);
													FIRE(IFM_DB,WEIGHT_DB,OFM,state[2],custom_Tib,
													to_index,ti_indexp,ki_indexp,kj_indexp,tip,to,row,N,M,R,C,custom_stride,
													padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
													ti_indexp=ti_index;//for fire from weights buffer,ti_indexp increase after the two Tn fired
													ki_indexp=ki_index;
													kj_indexp=kj_index;
												}
												else{
													LOAD_Weights(dma_Weights,WEIGHT_DB,state[1],custom_Tib,
													to_index,ti_index,ki_index,kj_index,ti,to,N,M,M_index,M_in,custom_k);
													LOAD_IFM(dma_IFM,IFM_DB,state[2],ti,row,N,R,C,
													custom_stride,padding,custom_k,custom_Tr,custom_Tc);
													FIRE(IFM,WEIGHT,OFM,state[2],custom_Tib,
													to_index,ti_indexp,ki_indexp,kj_indexp,tip,to,row,N,M,R,C,custom_stride,
													padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
													//for load weights buffer,ti_index increase after the two Tn loaded
//													if(N==1000 && custom_k<=1){//last FC for BP
//														kj_index++;
//														if(kj_index>=K){
//															ti_index++;
//															kj_index=0;
//														}
//														//ki_index=ki_indexout;
//													}
//													else{//others
														kj_count++;
														kj_index+=custom_k;
														if(kj_count>=custom_kb){
															ki_count++;
															ki_index+=custom_k;
															kj_count=0;
															kj_index=0;
														}
														if(ki_count>=custom_kb){
															ti_index++;
															ki_count=0;
															ki_index=0;
														}
//													}
												}
											}
											tip=ti;
											idxp+=1;
										}
										if(idxp%2==0){
											FIRE(IFM_DB,WEIGHT_DB,OFM,state[2],custom_Tib,
											to_index,ti_indexp,ki_indexp,kj_indexp,tip,to,row,N,M,R,C,custom_stride,
											padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
											OFM_STORE(OFM_DB,state(1,0),custom_batch,batch_size,ba,topre,rowp,M,M_index,M_in,
											R,C,custom_Tr,custom_Tc,dma_OFM,dma_Output);
										}else{
											FIRE(IFM,WEIGHT,OFM,state[2],custom_Tib,
											to_index,ti_indexp,ki_indexp,kj_indexp,tip,to,row,N,M,R,C,custom_stride,
											padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
											OFM_STORE(OFM_DB,state(1,0),custom_batch,batch_size,ba,topre,rowp,M,M_index,M_in,
											R,C,custom_Tr,custom_Tc,dma_OFM,dma_Output);
										}

					//					OFM_STORE(Output1_DB,Reluindex1,Relulayerin,Relulayerout,state,to,row,col,
					//					M,R,C,custom_Tr,custom_Tc,dma_Output);

									}
									else{
										int idxp = 0;
										int14 tip=0;
										int8 ti_index=0;
										int8 ti_indexp=0;
										int4 ki_index=0;//ki_indexout;
										int4 ki_count=0;
										int4 kj_index=0;
										int4 kj_count=0;
										int4 ki_indexp=0;//ki_indexout;
										int4 kj_indexp=0;
										for(int14 ti=0;ti<N;ti+=Tn){
											//flag=120;
										#pragma HLS loop_tripcount min=6 max=6 avg=6
											if(idxp==0){
												LOAD_Weights(dma_Weights,WEIGHT,state[1],custom_Tib,
												to_index,ti_index,ki_index,kj_index,ti,to,N,M,M_index,M_in,custom_k);
												LOAD_IFM(dma_IFM,IFM,state[2],ti,row,N,R,C,
												custom_stride,padding,custom_k,custom_Tr,custom_Tc);
											}
											else {
												if(idxp%2==0){
													LOAD_Weights(dma_Weights,WEIGHT,state[1],custom_Tib,
													to_index,ti_index,ki_index,kj_index,ti,to,N,M,M_index,M_in,custom_k);
													LOAD_IFM(dma_IFM,IFM,state[2],ti,row,N,R,C,
													custom_stride,padding,custom_k,custom_Tr,custom_Tc);
													FIRE(IFM_DB,WEIGHT_DB,OFM_DB,state[2],custom_Tib,
													to_index,ti_indexp,ki_indexp,kj_indexp,tip,to,row,N,M,R,C,custom_stride,
													padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
													ti_indexp=ti_index;//for fire from weights buffer,ti_indexp increase after the two Tn fired
													ki_indexp=ki_index;
													kj_indexp=kj_index;
												}
												else{
													LOAD_Weights(dma_Weights,WEIGHT_DB,state[1],custom_Tib,
													to_index,ti_index,ki_index,kj_index,ti,to,N,M,M_index,M_in,custom_k);
													LOAD_IFM(dma_IFM,IFM_DB,state[2],ti,row,N,R,C,
													custom_stride,padding,custom_k,custom_Tr,custom_Tc);
													FIRE(IFM,WEIGHT,OFM_DB,state[2],custom_Tib,
													to_index,ti_indexp,ki_indexp,kj_indexp,tip,to,row,N,M,R,C,custom_stride,
													padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
													//for load weights buffer,ti_index increase after the two Tn loaded
//													if(N==1000 && custom_k<=1){//last FC for BP
//														kj_index++;
//														if(kj_index>=K){
//															ti_index++;
//															kj_index=0;
//														}
//														//ki_index=ki_indexout;
//													}
//													else{//others
														kj_count++;
														kj_index+=custom_k;
														if(kj_count>=custom_kb){
															ki_count++;
															ki_index+=custom_k;
															kj_count=0;
															kj_index=0;
														}
														if(ki_count>=custom_kb){
															ti_index++;
															ki_count=0;
															ki_index=0;
														}
//													}
												}
											}
											tip=ti;
											idxp+=1;
										}
										if(idxp%2==0){
											FIRE(IFM_DB,WEIGHT_DB,OFM_DB,state[2],custom_Tib,
											to_index,ti_indexp,ki_indexp,kj_indexp,tip,to,row,N,M,R,C,custom_stride,
											padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
											OFM_STORE(OFM,state(1,0),custom_batch,batch_size,ba,topre,rowp,M,M_index,M_in,
											R,C,custom_Tr,custom_Tc,dma_OFM,dma_Output);
										}else{
											FIRE(IFM,WEIGHT,OFM_DB,state[2],custom_Tib,
											to_index,ti_indexp,ki_indexp,kj_indexp,tip,to,row,N,M,R,C,custom_stride,
											padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
											OFM_STORE(OFM,state(1,0),custom_batch,batch_size,ba,topre,rowp,M,M_index,M_in,
											R,C,custom_Tr,custom_Tc,dma_OFM,dma_Output);
										}
					//					OFM_STORE(Output1,Reluindex1,Relulayerin,Relulayerout,state,to,row,col,
					//					M,R,C,custom_Tr,custom_Tc,dma_Output);

									}
								}
//								rowp=row;
//								//colp=col;
//								topre=to;
//								num++;

							}

							//}
							else{//if it is not the 1st image of the batch
								//or in the backward of the 1st image, except 1st Tm do not need to load weights)
								if(num==0){
									int idxp = 0;
									int14 tip=0;
									int8 ti_index=0;
									int8 ti_indexp=0;
									int4 ki_index=0;//ki_indexout;
									int4 ki_count=0;
									int4 kj_index=0;
									int4 kj_count=0;
									int4 ki_indexp=0;//ki_indexout;
									int4 kj_indexp=0;
									for(int14 ti=0;ti<N;ti+=Tn){
										//flag=120;
									#pragma HLS loop_tripcount min=6 max=6 avg=6
										if(idxp==0){
											LOAD_IFM(dma_IFM,IFM,state[2],ti,row,N,R,C,
											custom_stride,padding,custom_k,custom_Tr,custom_Tc);
										}
										else {
											if(idxp%2==0){
												LOAD_IFM(dma_IFM,IFM,state[2],ti,row,N,R,C,
												custom_stride,padding,custom_k,custom_Tr,custom_Tc);
												FIRE(IFM_DB,WEIGHT_DB,OFM,state[2],custom_Tib,
												to_index,ti_indexp,ki_indexp,kj_indexp,tip,to,row,N,M,R,C,custom_stride,
												padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
												ti_indexp=ti_index;//for fire from weights buffer,ti_indexp increase after the two Tn fired
												ki_indexp=ki_index;
												kj_indexp=kj_index;
											}
											else{
												LOAD_IFM(dma_IFM,IFM_DB,state[2],ti,row,N,R,C,
												custom_stride,padding,custom_k,custom_Tr,custom_Tc);
												FIRE(IFM,WEIGHT,OFM,state[2],custom_Tib,
												to_index,ti_indexp,ki_indexp,kj_indexp,tip,to,row,N,M,R,C,custom_stride,
												padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
												//for load weights buffer,ti_index increase after the two Tn loaded
//												if(N==1000 && custom_k<=1){//last FC for BP
//													kj_index++;
//													if(kj_index>=K){
//														ti_index++;
//														kj_index=0;
//													}
//													//ki_index=ki_indexout;
//												}
//												else{//others
													kj_count++;
													kj_index+=custom_k;
													if(kj_count>=custom_kb){
														ki_count++;
														ki_index+=custom_k;
														kj_count=0;
														kj_index=0;
													}
													if(ki_count>=custom_kb){
														ti_index++;
														ki_count=0;
														ki_index=0;
													}
//												}
											}
										}
										tip=ti;
										idxp+=1;
									}
									if(idxp%2==0){
										FIRE(IFM_DB,WEIGHT_DB,OFM,state[2],custom_Tib,
										to_index,ti_indexp,ki_indexp,kj_indexp,tip,to,row,N,M,R,C,custom_stride,
										padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
									}else{
										FIRE(IFM,WEIGHT,OFM,state[2],custom_Tib,
										to_index,ti_indexp,ki_indexp,kj_indexp,tip,to,row,N,M,R,C,custom_stride,
										padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
									}
								}
								else {
									if(num%2==0){
										int idxp = 0;
										int14 tip=0;
										int8 ti_index=0;
										int8 ti_indexp=0;
										int4 ki_index=0;//ki_indexout;
										int4 ki_count=0;
										int4 kj_index=0;
										int4 kj_count=0;
										int4 ki_indexp=0;//ki_indexout;
										int4 kj_indexp=0;
										for(int14 ti=0;ti<N;ti+=Tn){
											//flag=120;
										#pragma HLS loop_tripcount min=6 max=6 avg=6
											if(idxp==0){
												LOAD_IFM(dma_IFM,IFM,state[2],ti,row,N,R,C,
												custom_stride,padding,custom_k,custom_Tr,custom_Tc);

											}
											else {
												if(idxp%2==0){
													LOAD_IFM(dma_IFM,IFM,state[2],ti,row,N,R,C,
													custom_stride,padding,custom_k,custom_Tr,custom_Tc);
													FIRE(IFM_DB,WEIGHT_DB,OFM,state[2],custom_Tib,
													to_index,ti_indexp,ki_indexp,kj_indexp,tip,to,row,N,M,R,C,custom_stride,
													padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
													ti_indexp=ti_index;//for fire from weights buffer,ti_indexp increase after the two Tn fired
													ki_indexp=ki_index;
													kj_indexp=kj_index;
												}
												else{
													LOAD_IFM(dma_IFM,IFM_DB,state[2],ti,row,N,R,C,
													custom_stride,padding,custom_k,custom_Tr,custom_Tc);
													FIRE(IFM,WEIGHT,OFM,state[2],custom_Tib,
													to_index,ti_indexp,ki_indexp,kj_indexp,tip,to,row,N,M,R,C,custom_stride,
													padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
													//for load weights buffer,ti_index increase after the two Tn loaded
//													if(N==1000 && custom_k<=1){//last FC for BP
//														kj_index++;
//														if(kj_index>=K){
//															ti_index++;
//															kj_index=0;
//														}
//														//ki_index=ki_indexout;
//													}
//													else{//others
														kj_count++;
														kj_index+=custom_k;
														if(kj_count>=custom_kb){
															ki_count++;
															ki_index+=custom_k;
															kj_count=0;
															kj_index=0;
														}
														if(ki_count>=custom_kb){
															ti_index++;
															ki_count=0;
															ki_index=0;
														}
//													}
												}
											}
											tip=ti;
											idxp+=1;
										}
										if(idxp%2==0){
											FIRE(IFM_DB,WEIGHT_DB,OFM,state[2],custom_Tib,
											to_index,ti_indexp,ki_indexp,kj_indexp,tip,to,row,N,M,R,C,custom_stride,
											padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
											OFM_STORE(OFM_DB,state(1,0),custom_batch,batch_size,ba,topre,rowp,M,M_index,M_in,
											R,C,custom_Tr,custom_Tc,dma_OFM,dma_Output);
										}else{
											FIRE(IFM,WEIGHT,OFM,state[2],custom_Tib,
											to_index,ti_indexp,ki_indexp,kj_indexp,tip,to,row,N,M,R,C,custom_stride,
											padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
											OFM_STORE(OFM_DB,state(1,0),custom_batch,batch_size,ba,topre,rowp,M,M_index,M_in,
											R,C,custom_Tr,custom_Tc,dma_OFM,dma_Output);
										}

					//					OFM_STORE(Output1_DB,Reluindex1,Relulayerin,Relulayerout,state,to,row,col,
					//					M,R,C,custom_Tr,custom_Tc,dma_Output);

									}
									else{
										int idxp = 0;
										int14 tip=0;
										int8 ti_index=0;
										int8 ti_indexp=0;
										int4 ki_index=0;//ki_indexout;
										int4 ki_count=0;
										int4 kj_index=0;
										int4 kj_count=0;
										int4 ki_indexp=0;//ki_indexout;
										int4 kj_indexp=0;
										for(int14 ti=0;ti<N;ti+=Tn){
											//flag=120;
										#pragma HLS loop_tripcount min=6 max=6 avg=6
											if(idxp==0){
												LOAD_IFM(dma_IFM,IFM,state[2],ti,row,N,R,C,
												custom_stride,padding,custom_k,custom_Tr,custom_Tc);
											}
											else {
												if(idxp%2==0){
													LOAD_IFM(dma_IFM,IFM,state[2],ti,row,N,R,C,
													custom_stride,padding,custom_k,custom_Tr,custom_Tc);
													FIRE(IFM_DB,WEIGHT_DB,OFM_DB,state[2],custom_Tib,
													to_index,ti_indexp,ki_indexp,kj_indexp,tip,to,row,N,M,R,C,custom_stride,
													padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
													ti_indexp=ti_index;//for fire from weights buffer,ti_indexp increase after the two Tn fired
													ki_indexp=ki_index;
													kj_indexp=kj_index;
												}
												else{
													LOAD_IFM(dma_IFM,IFM_DB,state[2],ti,row,N,R,C,
													custom_stride,padding,custom_k,custom_Tr,custom_Tc);
													FIRE(IFM,WEIGHT,OFM_DB,state[2],custom_Tib,
													to_index,ti_indexp,ki_indexp,kj_indexp,tip,to,row,N,M,R,C,custom_stride,
													padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
													//for load weights buffer,ti_index increase after the two Tn loaded
//													if(N==1000 && custom_k<=1){//last FC for BP
//														kj_index++;
//														if(kj_index>=K){
//															ti_index++;
//															kj_index=0;
//														}
//														//ki_index=ki_indexout;
//													}
//													else{//others
														kj_count++;
														kj_index+=custom_k;
														if(kj_count>=custom_kb){
															ki_count++;
															ki_index+=custom_k;
															kj_count=0;
															kj_index=0;
														}
														if(ki_count>=custom_kb){
															ti_index++;
															ki_count=0;
															ki_index=0;
														}
//													}
												}
											}
											tip=ti;
											idxp++;
										}
										if(idxp%2==0){
											FIRE(IFM_DB,WEIGHT_DB,OFM_DB,state[2],custom_Tib,
											to_index,ti_indexp,ki_indexp,kj_indexp,tip,to,row,N,M,R,C,custom_stride,
											padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
											OFM_STORE(OFM,state(1,0),custom_batch,batch_size,ba,topre,rowp,M,M_index,M_in,
											R,C,custom_Tr,custom_Tc,dma_OFM,dma_Output);
										}else{
											FIRE(IFM,WEIGHT,OFM_DB,state[2],custom_Tib,
											to_index,ti_indexp,ki_indexp,kj_indexp,tip,to,row,N,M,R,C,custom_stride,
											padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
											OFM_STORE(OFM,state(1,0),custom_batch,batch_size,ba,topre,rowp,M,M_index,M_in,
											R,C,custom_Tr,custom_Tc,dma_OFM,dma_Output);
										}
					//					OFM_STORE(Output1,Reluindex1,Relulayerin,Relulayerout,state,to,row,col,
					//					M,R,C,custom_Tr,custom_Tc,dma_Output);

									}
								}

							}

							rowp=row;
							//colp=col;
							topre=to;
							num++;
						}//for(row=0;row<R;row+=custom_Tr)
					}//if(to+M_index<M_in)
//					if(N==1000 && custom_k<=1){//last FC for BP
//						ki_indexout++;
//						if(ki_indexout>=K){
//							to_index++;
//							ki_indexout=0;
//						}
//					}
//					else{
						to_index++;
//					}
				}//for(to=0;to<M;to+=too)
				//to_index_bound=to_index;
				if(num%2==0){
					//Initialize_Out2(Output2_DB);
					OFM_STORE(OFM_DB,state(1,0),custom_batch,batch_size,ba,topre,rowp,M,M_index,M_in,
					R,C,custom_Tr,custom_Tc,dma_OFM,dma_Output);
				}
				else{
					//Initialize_Out2(Output2);
					OFM_STORE(OFM,state(1,0),custom_batch,batch_size,ba,topre,rowp,M,M_index,M_in,
					R,C,custom_Tr,custom_Tc,dma_OFM,dma_Output);
				}
			}
		}

		//flag =102;
		Initialize_WEIGHTS(WEIGHT,WEIGHT_DB);//,custom_Tib,custom_k,to_index_bound);
		Initialize_OFM(OFM,OFM_DB,custom_Tr,custom_Tc);
	}

	else {//01Xupdate weights
		if(custom_Tr>=R){//when whole FM can be placed, OFM do not need to repeat [N/Tn] times
			//flag=102;
			//int14 to_index_bound=0;
			for(int14 M_index=0;M_index<M_in;M_index+=M){
			#pragma HLS loop_tripcount min=2 max=2 avg=2
				for(int8 ba=0;ba<batch_size;ba++){
			#pragma HLS loop_tripcount min=4 max=4 avg=4

					//int14 to_index=0;

					if(ba==batch_size-1){//for the last batch, weights-all dw/batch and update weights
						int8 to_index=0;
						for(int14 to=0;to<M;to+=Tm){
						#pragma HLS loop_tripcount min=3 max=3 avg=3
							if(M_index+to<M_in){
								int num=0;
								int14 tip=0;
								int8 ti_index=0;
								int4 ki_index=0;
								int4 ki_count=0;
								int4 kj_index=0;
								int4 kj_count=0;

								for(int14 ti=0;ti<N;ti+=Tn){
								#pragma HLS loop_tripcount min=6 max=6 avg=6
									if(num==0){
										LOAD_IFM(dma_IFM,IFM,state[2],ti,0,N,R,C,
										custom_stride,padding,custom_k,custom_Tr,custom_Tc);
										LOAD_OFM(dma_OFM,OFM,to,0,M,M_index,M_in,R,C,custom_Tr,custom_Tc);
									}
									else{
										if(num%2==0){
											LOAD_IFM(dma_IFM,IFM,state[2],ti,0,N,R,C,
											custom_stride,padding,custom_k,custom_Tr,custom_Tc);
											FIRE(IFM_DB,WEIGHT,OFM,state[2],custom_Tib,
											to_index,ti_index,ki_index,kj_index,tip,to,0,N,M,R,C,custom_stride,
											padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
											Weights_STORE(dma_Weights,WEIGHT,state[0],custom_Tib,
											to_index,ti_index,ki_index,kj_index,tip,to,N,M,M_index,M_in,custom_k,dma_Output);
											//for fire and store for weights buffer,ti_index increase after the two Tn processed
											kj_count++;
											kj_index+=custom_k;
											if(kj_count>=custom_kb){
												ki_count++;
												ki_index+=custom_k;
												kj_count=0;
												kj_index=0;
											}
											if(ki_count>=custom_kb){
												ti_index++;
												ki_count=0;
												ki_index=0;
											}
										}
										else{
											LOAD_IFM(dma_IFM,IFM_DB,state[2],ti,0,N,R,C,
											custom_stride,padding,custom_k,custom_Tr,custom_Tc);
											FIRE(IFM,WEIGHT_DB,OFM,state[2],custom_Tib,
											to_index,ti_index,ki_index,kj_index,tip,to,0,N,M,R,C,custom_stride,
											padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
											Weights_STORE(dma_Weights,WEIGHT_DB,state[0],custom_Tib,
											to_index,ti_index,ki_index,kj_index,tip,to,N,M,M_index,M_in,custom_k,dma_Output);

										}
									}
									num++;
									tip=ti;
								}
								if(num%2==0){
									//Initialize_Out1(Output1_DB);
									FIRE(IFM_DB,WEIGHT,OFM,state[2],custom_Tib,
									to_index,ti_index,ki_index,kj_index,tip,to,0,N,M,R,C,custom_stride,
									padding,custom_k,custom_Tr,custom_Tc);
									Weights_STORE(dma_Weights,WEIGHT,state[0],custom_Tib,
									to_index,ti_index,ki_index,kj_index,tip,to,N,M,M_index,M_in,custom_k,dma_Output);
								}else{
									//Initialize_Out1(Output1);
									FIRE(IFM,WEIGHT_DB,OFM,state[2],custom_Tib,
									to_index,ti_index,ki_index,kj_index,tip,to,0,N,M,R,C,custom_stride,
									padding,custom_k,custom_Tr,custom_Tc);
									Weights_STORE(dma_Weights,WEIGHT_DB,state[0],custom_Tib,
									to_index,ti_index,ki_index,kj_index,tip,to,N,M,M_index,M_in,custom_k,dma_Output);
								}
								to_index++;
							}
						}
						//to_index_bound=to_index;
					}
					else{//for other images in the batch, just fire the dw into the weights buffer
						int8 to_index=0;
						for(int14 to=0;to<M;to+=Tm){
						#pragma HLS loop_tripcount min=3 max=3 avg=3
							if(M_index+to<M_in){
								int num=0;
								int14 tip=0;
								int8 ti_index=0;
								int4 ki_index=0;
								int4 ki_count=0;
								int4 kj_index=0;
								int4 kj_count=0;
								for(int14 ti=0;ti<N;ti+=Tn){
								#pragma HLS loop_tripcount min=6 max=6 avg=6
									if(num==0){
										LOAD_IFM(dma_IFM,IFM,state[2],ti,0,N,R,C,
										custom_stride,padding,custom_k,custom_Tr,custom_Tc);
										LOAD_OFM(dma_OFM,OFM,to,0,M,M_index,M_in,R,C,custom_Tr,custom_Tc);
									}
									else{
										if(num%2==0){
											LOAD_IFM(dma_IFM,IFM,state[2],ti,0,N,R,C,
											custom_stride,padding,custom_k,custom_Tr,custom_Tc);
											FIRE(IFM_DB,WEIGHT,OFM,state[2],custom_Tib,
											to_index,ti_index,ki_index,kj_index,tip,to,0,N,M,R,C,custom_stride,
											padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
											//for fire and store for weights buffer,ti_index increase after the two Tn processed
											kj_count++;
											kj_index+=custom_k;
											if(kj_count>=custom_kb){
												ki_count++;
												ki_index+=custom_k;
												kj_count=0;
												kj_index=0;
											}
											if(ki_count>=custom_kb){
												ti_index++;
												ki_count=0;
												ki_index=0;
											}
										}
										else{
											LOAD_IFM(dma_IFM,IFM_DB,state[2],ti,0,N,R,C,
											custom_stride,padding,custom_k,custom_Tr,custom_Tc);
											FIRE(IFM,WEIGHT_DB,OFM,state[2],custom_Tib,
											to_index,ti_index,ki_index,kj_index,tip,to,0,N,M,R,C,custom_stride,
											padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);

										}
									}
									num++;
									tip=ti;
								}
								if(num%2==0){
									//Initialize_Out1(Output1_DB);
									FIRE(IFM_DB,WEIGHT,OFM,state[2],custom_Tib,
									to_index,ti_index,ki_index,kj_index,tip,to,0,N,M,R,C,custom_stride,
									padding,custom_k,custom_Tr,custom_Tc);

								}else{
									//Initialize_Out1(Output1);
									FIRE(IFM,WEIGHT_DB,OFM,state[2],custom_Tib,
									to_index,ti_index,ki_index,kj_index,tip,to,0,N,M,R,C,custom_stride,
									padding,custom_k,custom_Tr,custom_Tc);

								}
								to_index++;
							}
						}
						//to_index_bound=to_index;
					}
				}
			}
			Initialize_WEIGHTS(WEIGHT,WEIGHT_DB);//,custom_Tib,custom_k,to_index_bound);
			Initialize_OFM(OFM,OFM_DB,custom_Tr,custom_Tc);
		}
		else{//FM is bigger than BRAM can hold, custom_Tr<R
			//flag=102;
			//int14 to_index_bound=0;
			//int8 row_sub=R-1;
			for(int14 M_index=0;M_index<M_in;M_index+=M){
			#pragma HLS loop_tripcount min=2 max=2 avg=2
				for(int8 ba=0;ba<batch_size;ba++){
			#pragma HLS loop_tripcount min=4 max=4 avg=4
					int8 to_index=0;
					if(ba==batch_size-1){//for the image in the, weights-all dw and update weights
						//int14 to_index=0;
						//int14 topre=0;

						for(int14 to=0;to<M;to+=Tm){
						#pragma HLS loop_tripcount min=3 max=3 avg=3
							if(M_index+to<M_in){
								int8 ti_index=0;
								int8 ti_indexp=0;
								int4 ki_index=0;
								int4 ki_count=0;
								int4 kj_index=0;
								int4 kj_count=0;
								int4 ki_indexp=0;
								int4 kj_indexp=0;
								int num=0;
								int14 tip=0;

								for(int14 ti=0;ti<N;ti+=Tn){
								#pragma HLS loop_tripcount min=6 max=6 avg=6
									if(num==0){
										int idxu = 0;
										int8 rowp=0;
										//int8 colp;
										for(int8 row=0;row<R;row+=custom_Tr){
										#pragma HLS loop_tripcount min=2 max=2 avg=2
											if(idxu==0){
												LOAD_IFM(dma_IFM,IFM,state[2],ti,row,N,R,C,
												custom_stride,padding,custom_k,custom_Tr,custom_Tc);
												LOAD_OFM(dma_OFM,OFM,to,row,M,M_index,M_in,R,C,custom_Tr,custom_Tc);
											}
											else {
												if(idxu%2==0){
													LOAD_IFM(dma_IFM,IFM,state[2],ti,row,N,R,C,
													custom_stride,padding,custom_k,custom_Tr,custom_Tc);
													LOAD_OFM(dma_OFM,OFM,to,row,M,M_index,M_in,R,C,custom_Tr,custom_Tc);
													FIRE(IFM_DB,WEIGHT,OFM_DB,state[2],custom_Tib,
													to_index,ti_index,ki_index,kj_index,ti,to,rowp,N,M,R,C,custom_stride,
													padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
//													if(row>0){//except 1st row, other trr stored in originall IFM buffer should move to new DB buffer for right lines
//														Move_IFM(IFM_DB,IFM,state[2],ti,row,N,R,C,
//														custom_stride,padding,custom_k,custom_Tr,custom_Tc);
//													}
												}
												else{
													LOAD_IFM(dma_IFM,IFM_DB,state[2],ti,row,N,R,C,
													custom_stride,padding,custom_k,custom_Tr,custom_Tc);
													LOAD_OFM(dma_OFM,OFM_DB,to,row,M,M_index,M_in,R,C,custom_Tr,custom_Tc);
													FIRE(IFM,WEIGHT,OFM,state[2],custom_Tib,
													to_index,ti_index,ki_index,kj_index,ti,to,rowp,N,M,R,C,custom_stride,
													padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
//													if(row>0){//except 1st row, other trr stored in originall IFM buffer should move to new DB buffer for right lines
//														Move_IFM(IFM,IFM_DB,state[2],ti,row,N,R,C,
//														custom_stride,padding,custom_k,custom_Tr,custom_Tc);
//													}
												}
											}
											rowp=row;
											//colp=col;
											idxu+=1;
										}
										if(idxu%2==0){
											FIRE(IFM_DB,WEIGHT,OFM_DB,state[2],custom_Tib,
											to_index,ti_index,ki_index,kj_index,ti,to,rowp,N,M,R,C,custom_stride,
											padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
										}else{
											FIRE(IFM,WEIGHT,OFM,state[2],custom_Tib,
											to_index,ti_index,ki_index,kj_index,ti,to,rowp,N,M,R,C,custom_stride,
											padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
										}

									//LOAD_Weights(dma_Weights,WEIGHT,state,ti,to,N,M,custom_k,flag);
									}
									else{
										if(num%2==0){
											int idxu = 0;
											int8 rowp=0;
											for(int8 row=0;row<R;row+=custom_Tr){
											#pragma HLS loop_tripcount min=3 max=3 avg=3
												if(idxu==0){
													LOAD_IFM(dma_IFM,IFM,state[2],ti,row,N,R,C,
													custom_stride,padding,custom_k,custom_Tr,custom_Tc);
													LOAD_OFM(dma_OFM,OFM,to,row,M,M_index,M_in,R,C,custom_Tr,custom_Tc);
												}
												else{
													if(idxu%2==0){
														LOAD_IFM(dma_IFM,IFM,state[2],ti,row,N,R,C,
														custom_stride,padding,custom_k,custom_Tr,custom_Tc);
														LOAD_OFM(dma_OFM,OFM,to,row,M,M_index,M_in,R,C,custom_Tr,custom_Tc);
														FIRE(IFM_DB,WEIGHT,OFM_DB,state[2],custom_Tib,
														to_index,ti_index,ki_index,kj_index,ti,to,rowp,N,M,R,C,custom_stride,
														padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
//														if(row>0){//except 1st row, other trr stored in originall IFM buffer should move to new DB buffer for right lines
//															Move_IFM(IFM_DB,IFM,state[2],ti,row,N,R,C,
//															custom_stride,padding,custom_k,custom_Tr,custom_Tc);
//														}
													}
													else{
														LOAD_IFM(dma_IFM,IFM_DB,state[2],ti,row,N,R,C,
														custom_stride,padding,custom_k,custom_Tr,custom_Tc);
														LOAD_OFM(dma_OFM,OFM_DB,to,row,M,M_index,M_in,R,C,custom_Tr,custom_Tc);
														FIRE(IFM,WEIGHT,OFM,state[2],custom_Tib,
														to_index,ti_index,ki_index,kj_index,ti,to,rowp,N,M,R,C,custom_stride,
														padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
//														if(row>0){//except 1st row, other trr stored in originall IFM buffer should move to new DB buffer for right lines
//															Move_IFM(IFM,IFM_DB,state[2],ti,row,N,R,C,
//															custom_stride,padding,custom_k,custom_Tr,custom_Tc);
//														}
													}
												}
												rowp=row;
												//colp=col;
												idxu++;
											}
//
											if(idxu%2==0){
												FIRE(IFM_DB,WEIGHT,OFM_DB,state[2],custom_Tib,
												to_index,ti_index,ki_index,kj_index,ti,to,rowp,N,M,R,C,custom_stride,
												padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
												Weights_STORE(dma_Weights,WEIGHT_DB,state[0],custom_Tib,
												to_index,ti_indexp,ki_indexp,kj_indexp,tip,to,N,M,M_index,M_in,custom_k,dma_Output);
											}else{
												FIRE(IFM,WEIGHT,OFM,state[2],custom_Tib,
												to_index,ti_index,ki_index,kj_index,ti,to,rowp,N,M,R,C,custom_stride,
												padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
												Weights_STORE(dma_Weights,WEIGHT_DB,state[0],custom_Tib,
												to_index,ti_indexp,ki_indexp,kj_indexp,tip,to,N,M,M_index,M_in,custom_k,dma_Output);

											}
											ti_indexp=ti_index;//for fire from weights buffer,ti_indexp increase after the two Tn fired
											ki_indexp=ki_index;
											kj_indexp=kj_index;
											//LOAD_Weights(dma_Weights,WEIGHT,state,ti,to,N,M,custom_k,flag);
											//for fire and store for weights buffer,ti_index increase after the two Tn processed
										}
										else{//num%2=1
											int idxu = 0;
											int8 rowp =0;
											for(int8 row=0;row<R;row+=custom_Tr){
											#pragma HLS loop_tripcount min=3 max=3 avg=3
												if(idxu==0){
													LOAD_IFM(dma_IFM,IFM,state[2],ti,row,N,R,C,
													custom_stride,padding,custom_k,custom_Tr,custom_Tc);
													LOAD_OFM(dma_OFM,OFM,to,row,M,M_index,M_in,R,C,custom_Tr,custom_Tc);
												}
												else {
													if(idxu%2==0){
														LOAD_IFM(dma_IFM,IFM,state[2],ti,row,N,R,C,
														custom_stride,padding,custom_k,custom_Tr,custom_Tc);
														LOAD_OFM(dma_OFM,OFM,to,row,M,M_index,M_in,R,C,custom_Tr,custom_Tc);
														FIRE(IFM_DB,WEIGHT_DB,OFM_DB,state[2],custom_Tib,
														to_index,ti_index,ki_index,kj_index,ti,to,rowp,N,M,R,C,custom_stride,
														padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
//														if(row>0){//except 1st row, other trr stored in originall IFM buffer should move to new DB buffer for right lines
//															Move_IFM(IFM_DB,IFM,state[2],ti,row,N,R,C,
//															custom_stride,padding,custom_k,custom_Tr,custom_Tc);
//														}
													}
													else{
														LOAD_IFM(dma_IFM,IFM_DB,state[2],ti,row,N,R,C,
														custom_stride,padding,custom_k,custom_Tr,custom_Tc);
														LOAD_OFM(dma_OFM,OFM_DB,to,row,M,M_index,M_in,R,C,custom_Tr,custom_Tc);
														FIRE(IFM,WEIGHT_DB,OFM,state[2],custom_Tib,
														to_index,ti_index,ki_index,kj_index,ti,to,rowp,N,M,R,C,custom_stride,
														padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
//														if(row>0){//except 1st row, other trr stored in originall IFM buffer should move to new DB buffer for right lines
//															Move_IFM(IFM,IFM_DB,state[2],ti,row,N,R,C,
//															custom_stride,padding,custom_k,custom_Tr,custom_Tc);
//														}
													}
												}
												rowp=row;
												idxu++;
											}
											//if(rowp>=row_sub){//only the last row need to store weights
											if(idxu%2==0){
												FIRE(IFM_DB,WEIGHT_DB,OFM_DB,state[2],custom_Tib,
												to_index,ti_index,ki_index,kj_index,ti,to,rowp,N,M,R,C,custom_stride,
												padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
												Weights_STORE(dma_Weights,WEIGHT,state[0],custom_Tib,
												to_index,ti_indexp,ki_indexp,kj_indexp,tip,to,N,M,M_index,M_in,custom_k,dma_Output);
											}else{
												FIRE(IFM,WEIGHT_DB,OFM,state[2],custom_Tib,
												to_index,ti_index,ki_index,kj_index,ti,to,rowp,N,M,R,C,custom_stride,
												padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
												Weights_STORE(dma_Weights,WEIGHT,state[0],custom_Tib,
												to_index,ti_indexp,ki_indexp,kj_indexp,tip,to,N,M,M_index,M_in,custom_k,dma_Output);

											}
//
											kj_count++;
											kj_index+=custom_k;
											if(kj_count>=custom_kb){
												ki_count++;
												ki_index+=custom_k;
												kj_count=0;
												kj_index=0;
											}
											if(ki_count>=custom_kb){
												ti_index++;
												ki_count=0;
												ki_index=0;
											}

										}
									}// if(num!=0)
									num++;
									tip=ti;
									//topre=to;
								}
								if(num%2==0){
									//Initialize_Out1(Output1_DB);
									Weights_STORE(dma_Weights,WEIGHT_DB,state[0],custom_Tib,
									to_index,ti_indexp,ki_indexp,kj_indexp,tip,to,N,M,M_index,M_in,custom_k,dma_Output);
								}else{
									//Initialize_Out1(Output1);
									Weights_STORE(dma_Weights,WEIGHT,state[0],custom_Tib,
									to_index,ti_indexp,ki_indexp,kj_indexp,tip,to,N,M,M_index,M_in,custom_k,dma_Output);
								}
								to_index++;
							}
						}
						//to_index_bound=to_index;
					}
					else{//for other images in the batch, just fire the dw into the weights buffer
						//int14 to_index=0;
						//int14 topre=0;
						for(int14 to=0;to<M;to+=Tm){
						#pragma HLS loop_tripcount min=3 max=3 avg=3
							if(M_index+to<M_in){
								int8 ti_index=0;
								//int8 ti_indexp=0;
								int4 ki_index=0;
								int4 ki_count=0;
								int4 kj_index=0;
								int4 kj_count=0;
								//int4 ki_indexp=0;
								//int4 kj_indexp=0;
								int num=0;
								int14 tip=0;
								for(int14 ti=0;ti<N;ti+=Tn){
								#pragma HLS loop_tripcount min=6 max=6 avg=6
									if(num==0){
										int idxu = 0;
										int8 rowp=0;
										//int8 colp;
										for(int8 row=0;row<R;row+=custom_Tr){
										#pragma HLS loop_tripcount min=3 max=3 avg=3
											if(idxu==0){
												LOAD_IFM(dma_IFM,IFM,state[2],ti,row,N,R,C,
												custom_stride,padding,custom_k,custom_Tr,custom_Tc);
												LOAD_OFM(dma_OFM,OFM,to,row,M,M_index,M_in,R,C,custom_Tr,custom_Tc);
											}
											else {
												if(idxu%2==0){
													LOAD_IFM(dma_IFM,IFM,state[2],ti,row,N,R,C,
													custom_stride,padding,custom_k,custom_Tr,custom_Tc);
													LOAD_OFM(dma_OFM,OFM,to,row,M,M_index,M_in,R,C,custom_Tr,custom_Tc);
													FIRE(IFM_DB,WEIGHT,OFM_DB,state[2],custom_Tib,
													to_index,ti_index,ki_index,kj_index,ti,to,rowp,N,M,R,C,custom_stride,
													padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
//													if(row>0){//except 1st row, other trr stored in originall IFM buffer should move to new DB buffer for right lines
//														Move_IFM(IFM_DB,IFM,state[2],ti,row,N,R,C,
//														custom_stride,padding,custom_k,custom_Tr,custom_Tc);
//													}
												}
												else{
													LOAD_IFM(dma_IFM,IFM_DB,state[2],ti,row,N,R,C,
													custom_stride,padding,custom_k,custom_Tr,custom_Tc);
													LOAD_OFM(dma_OFM,OFM_DB,to,row,M,M_index,M_in,R,C,custom_Tr,custom_Tc);
													FIRE(IFM,WEIGHT,OFM,state[2],custom_Tib,
													to_index,ti_index,ki_index,kj_index,ti,to,rowp,N,M,R,C,custom_stride,
													padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
//													if(row>0){//except 1st row, other trr stored in originall IFM buffer should move to new DB buffer for right lines
//														Move_IFM(IFM,IFM_DB,state[2],ti,row,N,R,C,
//														custom_stride,padding,custom_k,custom_Tr,custom_Tc);
//													}
												}
											}
											rowp=row;
											//colp=col;
											idxu++;
										}
										if(idxu%2==0){
											FIRE(IFM_DB,WEIGHT,OFM_DB,state[2],custom_Tib,
											to_index,ti_index,ki_index,kj_index,ti,to,rowp,N,M,R,C,custom_stride,
											padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
										}else{
											FIRE(IFM,WEIGHT,OFM,state[2],custom_Tib,
											to_index,ti_index,ki_index,kj_index,ti,to,rowp,N,M,R,C,custom_stride,
											padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
										}

									//LOAD_Weights(dma_Weights,WEIGHT,state,ti,to,N,M,custom_k,flag);
									}
									else{
										if(num%2==0){
											int idxu = 0;
											int8 rowp=0;
											for(int8 row=0;row<R;row+=custom_Tr){
											#pragma HLS loop_tripcount min=3 max=3 avg=3
												if(idxu==0){
													LOAD_IFM(dma_IFM,IFM,state[2],ti,row,N,R,C,
													custom_stride,padding,custom_k,custom_Tr,custom_Tc);
													LOAD_OFM(dma_OFM,OFM,to,row,M,M_index,M_in,R,C,custom_Tr,custom_Tc);
												}
												else{
													if(idxu%2==0){
														LOAD_IFM(dma_IFM,IFM,state[2],ti,row,N,R,C,
														custom_stride,padding,custom_k,custom_Tr,custom_Tc);
														LOAD_OFM(dma_OFM,OFM,to,row,M,M_index,M_in,R,C,custom_Tr,custom_Tc);
														FIRE(IFM_DB,WEIGHT,OFM_DB,state[2],custom_Tib,
														to_index,ti_index,ki_index,kj_index,ti,to,rowp,N,M,R,C,custom_stride,
														padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
//														if(row>0){//except 1st row, other trr stored in originall IFM buffer should move to new DB buffer for right lines
//															Move_IFM(IFM_DB,IFM,state[2],ti,row,N,R,C,
//															custom_stride,padding,custom_k,custom_Tr,custom_Tc);
//														}
													}
													else{
														LOAD_IFM(dma_IFM,IFM_DB,state[2],ti,row,N,R,C,
														custom_stride,padding,custom_k,custom_Tr,custom_Tc);
														LOAD_OFM(dma_OFM,OFM_DB,to,row,M,M_index,M_in,R,C,custom_Tr,custom_Tc);
														FIRE(IFM,WEIGHT,OFM,state[2],custom_Tib,
														to_index,ti_index,ki_index,kj_index,ti,to,rowp,N,M,R,C,custom_stride,
														padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
//														if(row>0){//except 1st row, other trr stored in originall IFM buffer should move to new DB buffer for right lines
//															Move_IFM(IFM,IFM_DB,state[2],ti,row,N,R,C,
//															custom_stride,padding,custom_k,custom_Tr,custom_Tc);
//														}
													}
												}
												rowp=row;
												//colp=col;
												idxu++;
											}
											if(idxu%2==0){
												FIRE(IFM_DB,WEIGHT,OFM_DB,state[2],custom_Tib,
												to_index,ti_index,ki_index,kj_index,ti,to,rowp,N,M,R,C,custom_stride,
												padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);

											}else{
												FIRE(IFM,WEIGHT,OFM,state[2],custom_Tib,
												to_index,ti_index,ki_index,kj_index,ti,to,rowp,N,M,R,C,custom_stride,
												padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);

											}

											//LOAD_Weights(dma_Weights,WEIGHT,state,ti,to,N,M,custom_k,flag);
											//for fire and store for weights buffer,ti_index increase after the two Tn processed
										}
										else{//num%2=1
											int idxu = 0;
											int8 rowp=0;
											for(int8 row=0;row<R;row+=custom_Tr){
											#pragma HLS loop_tripcount min=3 max=3 avg=3
												if(idxu==0){
													LOAD_IFM(dma_IFM,IFM,state[2],ti,row,N,R,C,
													custom_stride,padding,custom_k,custom_Tr,custom_Tc);
													LOAD_OFM(dma_OFM,OFM,to,row,M,M_index,M_in,R,C,custom_Tr,custom_Tc);
												}
												else {
													if(idxu%2==0){
														LOAD_IFM(dma_IFM,IFM,state[2],ti,row,N,R,C,
														custom_stride,padding,custom_k,custom_Tr,custom_Tc);
														LOAD_OFM(dma_OFM,OFM,to,row,M,M_index,M_in,R,C,custom_Tr,custom_Tc);
														FIRE(IFM_DB,WEIGHT_DB,OFM_DB,state[2],custom_Tib,
														to_index,ti_index,ki_index,kj_index,ti,to,rowp,N,M,R,C,custom_stride,
														padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
//														if(row>0){//except 1st row, other trr stored in originall IFM buffer should move to new DB buffer for right lines
//															Move_IFM(IFM_DB,IFM,state[2],ti,row,N,R,C,
//															custom_stride,padding,custom_k,custom_Tr,custom_Tc);
//														}
													}
													else{
														LOAD_IFM(dma_IFM,IFM_DB,state[2],ti,row,N,R,C,
														custom_stride,padding,custom_k,custom_Tr,custom_Tc);
														LOAD_OFM(dma_OFM,OFM_DB,to,row,M,M_index,M_in,R,C,custom_Tr,custom_Tc);
														FIRE(IFM,WEIGHT_DB,OFM,state[2],custom_Tib,
														to_index,ti_index,ki_index,kj_index,ti,to,rowp,N,M,R,C,custom_stride,
														padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);
//														if(row>0){//except 1st row, other trr stored in originall IFM buffer should move to new DB buffer for right lines
//															Move_IFM(IFM,IFM_DB,state[2],ti,row,N,R,C,
//															custom_stride,padding,custom_k,custom_Tr,custom_Tc);
//														}
													}
												}
												rowp=row;
												idxu+=1;
											}
											if(idxu%2==0){
												FIRE(IFM_DB,WEIGHT_DB,OFM_DB,state[2],custom_Tib,
												to_index,ti_index,ki_index,kj_index,ti,to,rowp,N,M,R,C,custom_stride,
												padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);

											}else{
												FIRE(IFM,WEIGHT_DB,OFM,state[2],custom_Tib,
												to_index,ti_index,ki_index,kj_index,ti,to,rowp,N,M,R,C,custom_stride,
												padding,custom_k,custom_Tr,custom_Tc);//Output1,Output2);

											}
											kj_count++;
											kj_index+=custom_k;
											if(kj_count>=custom_kb){
												ki_count++;
												ki_index+=custom_k;
												kj_count=0;
												kj_index=0;
											}
											if(ki_count>=custom_kb){
												ti_index++;
												ki_count=0;
												ki_index=0;
											}

										}
									}//num!=0
									num++;
									tip=ti;
									//topre=to;
								}
								to_index++;
							}
						}//for(int14 to=0;to<M;to+=Tm)
						//to_index_bound=to_index;
					}
				}
			}
		}
		Initialize_WEIGHTS(WEIGHT,WEIGHT_DB);//,custom_Tib,custom_k,to_index_bound);
		Initialize_OFM(OFM,OFM_DB,custom_Tr,custom_Tc);
	}
}


void top(hls::stream<DMA_DATA_128> &dma_IFM,
			hls::stream<DMA_DATA_128> &dma_Weights,
			hls::stream<DMA_DATA_128> &dma_Weightsout,
			hls::stream<DMA_DATA_128> &dma_OFM,
			hls::stream<DMA_DATA_128> &dma_Output,
			//hls::stream<DMA_index> &dma_Indexin,
			//hls::stream<DMA_DATA_128> &dma_Indexout,
			//int statec,
			//int statep,
			ap_uint<3> statec,
			ap_uint<4> statep,
			ap_uint<4> stateb,
//			int4 Relulayerin,
//			int4 Relulayerout,
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
//			int8 viewTrc,
//			int14 viewTm,
			int8 R_in,
			int8 C_in
			//int8& flag
			){

#pragma HLS INTERFACE s_axilite port=return bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=statec bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=statep bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=stateb bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Relulayerin bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Relulayerout bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Poollayerin bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=Poollayerout bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=to bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=custom_batch bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=batch_size bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=M_in bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=custom_Tib bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=N bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=M bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=R bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=C bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=custom_stride bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=padding bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=custom_kb bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=custom_k bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=custom_Tr bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=custom_Tc bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=R_in bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=C_in bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=flag bundle=CRTL_BUS
//#pragma HLS INTERFACE s_axilite port=flag bundle=CRTL_BUS

//#pragma HLS INTERFACE s_axilite port=num bundle=CRTL_BUS


#pragma HLS INTERFACE axis port=dma_IFM
#pragma HLS INTERFACE axis port=dma_Weights
#pragma HLS INTERFACE axis port=dma_Weightsout
#pragma HLS INTERFACE axis port=dma_OFM
#pragma HLS INTERFACE axis port=dma_Output
//#pragma HLS INTERFACE axis port=dma_Indexin
//#pragma HLS INTERFACE axis port=dma_Indexout


	static FPGA_DATA WEIGHT[Tm][Tn][TiTo][K][K];
#pragma HLS RESOURCE variable=WEIGHT core=RAM_S2P_BRAM
#pragma HLS ARRAY_PARTITION variable=WEIGHT complete dim=1
#pragma HLS ARRAY_PARTITION variable=WEIGHT complete dim=2
//#pragma HLS ARRAY_PARTITION variable=WEIGHT block factor=3 dim=3//make a bank size near 18K as much as possible, or HLS will divide by 2

	static FPGA_DATA WEIGHT_DB[Tm][Tn][TiTo][K][K];
#pragma HLS RESOURCE variable=WEIGHT_DB core=RAM_S2P_BRAM
#pragma HLS ARRAY_PARTITION variable=WEIGHT_DB complete dim=1
#pragma HLS ARRAY_PARTITION variable=WEIGHT_DB complete dim=2
//#pragma HLS ARRAY_PARTITION variable=WEIGHT_DB block factor=3 dim=3


	static FPGA_DATA  IFM[Tn][Trtcin];//[Tr_bound][Tc_bound];
#pragma HLS RESOURCE variable=IFM core=RAM_S2P_BRAM
#pragma HLS ARRAY_PARTITION variable=IFM complete dim=1
//#pragma HLS ARRAY_PARTITION variable=IFM block factor=3 dim=2

	static FPGA_DATA IFM_DB[Tn][Trtcin];//[Tr_bound][Tc_bound];
#pragma HLS RESOURCE variable=IFM_DB core=RAM_S2P_BRAM
#pragma HLS ARRAY_PARTITION variable=IFM_DB complete dim=1
//#pragma HLS ARRAY_PARTITION variable=IFM_DB block factor=3 dim=2
	//flag=13;
	static FPGA_DATA OFM[Tm][Trtc];//[Tr][Tc];
#pragma HLS RESOURCE variable=OFM core=RAM_S2P_BRAM
#pragma HLS ARRAY_PARTITION variable=OFM complete dim=1

	static FPGA_DATA OFM_DB[Tm][Trtc];//[Tr][Tc];
#pragma HLS RESOURCE variable=OFM_DB core=RAM_S2P_BRAM
#pragma HLS ARRAY_PARTITION variable=OFM_DB complete dim=1


	//flag=14;
	static FPGA_DATA BNW[DMAwidth][BNbank][BNindex];
#pragma HLS RESOURCE variable=BNW core=RAM_S2P_BRAM
#pragma HLS ARRAY_PARTITION variable=BNW complete dim=1
#pragma HLS ARRAY_PARTITION variable=BNW complete dim=2
//#pragma HLS ARRAY_PARTITION variable=WEIGHT block factor=3 dim=3//make a bank size near 18K as much as possible, or HLS will divide by 2


	static ufix16 Poolindex[Mp];


	if(statep!=0){//pool
		//if(statep[3]==0){//nomal pool
			pool(dma_IFM,dma_Weights,IFM,IFM_DB,OFM,OFM_DB,Poolindex,statep,custom_batch,batch_size,
			N,R,C,custom_stride,custom_k,custom_Tr,custom_Tc,dma_OFM,dma_Output,R_in,C_in);
//		}
//		else{
//			Avgpool(dma_IFM,IFM,IFM_DB,OFM,OFM_DB,statep,custom_batch,batch_size,
//			N,R,C,custom_stride,custom_k,custom_Tr,custom_Tc,dma_OFM,dma_Output,R_in,C_in);
//		}
	}
	else{//conv accelerator
//		acc(dma_IFM,Reluindex1,Relulayerin,Relulayerout,dma_Weights,dma_OFM,statec,num,to,
//		N,M,R,C,custom_stride,padding,custom_k,custom_Tr,custom_Tc,dma_Output,Output1,Output1_DB,
//		Output2,Output2_DB,IFM,IFM_DB,WEIGHT,WEIGHT_DB,OFM,OFM_DB);
	//flag=20;
		if(stateb!=0){
			bn(dma_IFM,dma_Weights,dma_Weightsout,dma_OFM,stateb,batch_size,M,R,C,dma_Output,BNW);
		}
		else{
			acc(dma_IFM,
			dma_Weights,dma_OFM,statec,custom_batch,batch_size,M_in,custom_Tib,
			N,M,R,C,custom_stride,padding,custom_kb,custom_k,custom_Tr,custom_Tc,
			dma_Output,IFM,IFM_DB,WEIGHT,WEIGHT_DB,OFM,OFM_DB);
		}
	}
}







