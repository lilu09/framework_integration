
#include <iostream>
#include <assert.h>

#include <bits/stdc++.h>

/*using namespace std;*/

/*#include <lib/vectorpu.h>*/
#include <lib/matrix_mul_tunable.h>
#include <lib/tuneit.h>
#include <lib/meterpu.h>

#define OPT
#define BASE


//#include <cuda_runtime.h>
//#include <cublas_v2.h>

/*void matrix_mul_cublas(float const * const a,float const * const b,float * const c,size_t const ha,size_t const wa,size_t const wb)*/
/*{*/

/*const float alf = 1.0f;*/
/*const float bet = 0.0f;*/
/*const float *alpha = &alf;*/
/*const float *beta = &bet;*/

/*cublasStatus_t stat;*/
/*cublasHandle_t handle;*/

/*stat = cublasCreate(&handle);*/
/*assert(stat == CUBLAS_STATUS_SUCCESS);*/

/*cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(wb), static_cast<int>(ha), static_cast<int>(wa), alpha, b, static_cast<int>(wb), a, static_cast<int>(wa), beta, c, static_cast<int>(wb));*/

/*cublasDestroy(handle);*/
/*}*/

//8
int main(int argc, char* argv[])
{

	assert( cudaDeviceReset() == cudaSuccess );

	assert( argc == 2 );

	unsigned int const problem_size=std::atoi(argv[1]);


	/*const size_t ha=200, wa=200, wb=200;*/
	const size_t ha=problem_size, wa=problem_size, wb=problem_size;

#ifdef BASE
	{
		float *a_h, *b_h, *c_h;
		float *a_d, *b_d, *c_d;

		const size_t size_a=ha*wa;
		const size_t size_b=wa*wb;
		/*const size_t size_c=ha*wb;*/

		const size_t raw_size_a=ha*wa*sizeof(float);
		const size_t raw_size_b=wa*wb*sizeof(float);
		const size_t raw_size_c=ha*wb*sizeof(float);

		a_h=(float *)malloc( raw_size_a );
		b_h=(float *)malloc( raw_size_b );
		c_h=(float *)malloc( raw_size_c);

		cudaMalloc(&a_d, raw_size_a );
		cudaMalloc(&b_d, raw_size_b );
		cudaMalloc(&c_d, raw_size_c );

		std::fill(a_h, a_h+size_a, 1.0f);
		std::fill(b_h, b_h+size_b, 1.0f);

		cudaMemcpy(a_d, a_h, raw_size_a, cudaMemcpyHostToDevice);
		cudaMemcpy(b_d, b_h, raw_size_b, cudaMemcpyHostToDevice);

		//dummy call to inialize
		matrix_mul_cublas(a_d, b_d, c_d, ha, wa, wb);

		using namespace meterpu;
		meterpu::meter<meterpu::CPU_Time> my_meter;
		/*meterpu::meter<meterpu::CUDA_Time> my_meter;*/

		for(size_t i=0;i<100;++i){
			std::cerr<<"Testing baseline, problem size: "<<problem_size<<", No."<<i<<std::endl;

			my_meter.start();

			matrix_mul_cublas(a_d, b_d, c_d, ha, wa, wb);

			my_meter.stop();
			my_meter.calc();
			std::cout<<my_meter.get_value()<<","<<std::endl;
		}

		cudaMemcpy(c_h, c_d, raw_size_c, cudaMemcpyDeviceToHost);

		/*std::for_each(c_h, c_h+size_c, [problem_size](float const i){assert(i==float(problem_size));}) ;*/


		free(a_h);
		free(b_h);
		free(c_h);

		cudaFree(a_d);
		cudaFree(b_d);
		cudaFree(c_d);


	}

#endif

#ifdef OPT
	{



		tuneit::tuneit_settings<MATRIX_MUL_NUM_DIM, MATRIX_MUL_NUM_VARIANTS> st{2, std::vector<bool>(4,true), true, false, true, 40, {{1,1000}, {1,1000}, {1,1000}} };

		tuneit::tuneit< MATRIX_MUL_NUM_VARIANTS, 8, matrix_mul_tunable<float, size_t, size_t, size_t>, float, size_t, size_t, size_t> mytuner(st);

		mytuner.train();

		vectorpu::vector<float> a(wa*ha,1), b(wa*wb,1), c(ha*wb,0);

		mytuner.run(mytuner.predict(ha,wa,wb), a, b, c, ha, wa, wb);

		using namespace meterpu;
		meterpu::meter<meterpu::CPU_Time> my_meter;

		for(size_t i=0;i<100;++i){
			std::cerr<<"Testing opt, problem size: "<<problem_size<<", No."<<i<<std::endl;
			my_meter.start();

			mytuner.run(mytuner.predict(ha,wa,wb), a, b, c, ha, wa, wb);

			my_meter.stop();
			my_meter.calc();
			std::cout<<my_meter.get_value()<<std::endl;
		}

		/*std::cout<<mytuner.predict(ha,wa,wb)<<std::endl;*/


		/*std::for_each(RI(c), REI(c), [](float const i){assert(i==200.0f);}) ;*/
		/*std::for_each(RI(c), REI(c), [](float const i){assert(i==10.0f);}) ;*/
		/*std::for_each(RI(c), REI(c), [problem_size](float const i){assert(i==float(problem_size));}) ;*/

	}
#endif




	return EXIT_SUCCESS;
}
