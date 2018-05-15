

#undef BOOST_PP_VARIADICS
#define BOOST_PP_VARIADICS 1
#include <boost/preprocessor.hpp>


//#include <lib/xpdl.h>


#define XPDL_GSL_CBLAS 1
#define XPDL_OPENMP 1
#define XPDL_CUBLAS 1

//CPU, from cublas example code

//#define matrix_mul_cpu_flow (R)(R)(W)(NA)(NA)(NA)
void matrix_mul_cpu(float const * const a,float const * const b,float * const c,size_t const ha,size_t const wa,size_t const wb)
{
	for (size_t i = 0; i < ha; ++i)
		for (size_t j = 0; j < wb; ++j)
		{
			double sum = 0;

			for (size_t k = 0; k < wa; ++k)
			{
				double d = a[i * wa + k];
				double e = b[k * wb + j];
				sum += d * e;
			}

			c[i * wb + j] = (float)sum;
		}
}
//LOC: 9


#if XPDL_GSL_CBLAS == 1

//Blas

#include <gsl/gsl_cblas.h>

//#define matrix_mul_blas_flow (R)(R)(W)(NA)(NA)(NA)
void matrix_mul_blas(float const * const a,float const * const b,float * const c,size_t const ha,size_t const wa,size_t const wb)
{
	cblas_sgemm (CblasRowMajor,
			CblasNoTrans, CblasNoTrans, static_cast<int>(ha), static_cast<int>(wb), static_cast<int>(wa),
			1.0f, a, static_cast<int>(wa), b, static_cast<int>(wb), 0.0f, c, static_cast<int>(wb));
}

#endif
//LOC: 4

#if XPDL_OPENMP == 1


//OpenMP

#include <omp.h>

//#define matrix_mul_openmp_flow (R)(R)(W)(NA)(NA)(NA)
void matrix_mul_openmp (float const * const a,float const * const b,float * const c,size_t const ha,size_t const wa,size_t const wb)
{

	omp_set_dynamic(0);
	//omp_set_num_threads(XPDL_NUM_OF_HW_THREADS);

#pragma omp parallel for num_threads(XPDL_NUM_OF_HW_THREADS)
	for (size_t i = 0; i < ha; ++i)
		for (size_t j = 0; j < wb; ++j)
		{
			double sum = 0;

			for (size_t k = 0; k < wa; ++k)
			{
				double d = a[i * wa + k];
				double e = b[k * wb + j];
				sum += d * e;
			}

			c[i * wb + j] = (float)sum;
		}
}

#endif
//LOC: 13


#if XPDL_CUBLAS == 1

//Cublas

#include <cuda_runtime.h>
#include <cublas_v2.h>


//https://solarianprogrammer.com/2012/05/31/matrix-multiplication-cuda-cublas-curand-thrust/
//cublasHandle_t handle;


//#define matrix_mul_cublas_flow (GR)(GR)(GW)(NA)(NA)(NA)
void matrix_mul_cublas(float const * const a,float const * const b,float * const c,size_t const ha,size_t const wa,size_t const wb)
{

	const float alf = 1.0f;
	const float bet = 0.0f;
	const float *alpha = &alf;
	const float *beta = &bet;

	cublasStatus_t stat;
	cublasHandle_t handle;

	//If you need to do more than one matrix multiplication in your code it
	//is advisable to move the create/destroy handle code to the main function
	//, and use the same handle for all multiplications.
	stat = cublasCreate(&handle);
	assert(stat == CUBLAS_STATUS_SUCCESS);

	//suprising cublas use column major order, just for Fortran
	//reversed order of A and B passed to cublasSgemm
	//a good explanation:
	//http://mccormickml.com/2015/08/29/matrix-multiplication-with-cublas-example/
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(wb), static_cast<int>(ha), static_cast<int>(wa), alpha, b, static_cast<int>(wb), a, static_cast<int>(wa), beta, c, static_cast<int>(wb));

	//can not work, but it is not critical to make destroy work
	//cublasDestory(handle);
	//stat = cublasDestory(handle);
	cublasDestroy(handle);
	//assert(stat == CUBLAS_STATUS_SUCCESS);
}

#endif

//LOC: 13

typedef void (*matrix_mul_func)(float const * const a,float const * const b,float * const c,size_t const ha,size_t const wa,size_t const wb);
const matrix_mul_func matrix_mul_dispatch_table[]{matrix_mul_cpu,matrix_mul_blas,matrix_mul_openmp,matrix_mul_cublas};

//#include <string>
//std::string matrix_mul_names[]{"matrix_mul_cpu","matrix_mul_blas","matrix_mul_openmp","matrix_mul_cublas"};


#define DEPENDENCE (1)(XPDL_GSL_CBLAS)(XPDL_OPENMP)(XPDL_CUBLAS)

#define SUM(s, state, x) BOOST_PP_ADD(state, x)

//#define MATRIX_MUL_NUM_VARIANTS 4
#define MATRIX_MUL_NUM_VARIANTS BOOST_PP_SEQ_FOLD_LEFT(SUM, BOOST_PP_SEQ_HEAD(DEPENDENCE), BOOST_PP_SEQ_TAIL(DEPENDENCE))
#define MATRIX_MUL_NUM_DIM 3

//LOC: 5

#include <lib/tunable.h>
#include <lib/vectorpu.h>
#include <lib/meterpu.h>

//Two questions:
//how is the result of measurement for a point stored
//	return a tuple, not responsible to build a perf database
//	and use proportion to avoid outliers
//how to pass the parameter to run from sampling? by tuple or unpacked list?
//	it seems you can easily unpack a tuple to parameter list
//	so technically it is possible to use both
//	it seems I prefer to use the named parameter in the run(),
//	and use a packed tuple passed from adaptive sampling
//	this will save users from code that unpack a tuple



template <class MeasureType, class ...Tunable_Args>
class matrix_mul_tunable : public tunable<MeasureType, matrix_mul_func, Tunable_Args...>{

	public:
		matrix_mul_tunable():tunable<MeasureType, matrix_mul_func, Tunable_Args...>(MATRIX_MUL_NUM_DIM, MATRIX_MUL_NUM_VARIANTS, {matrix_mul_cpu
#if XPDL_GSL_CBLAS == 1
				,matrix_mul_blas
#endif
#if XPDL_OPENMP == 1
				,matrix_mul_openmp
#endif
#if XPDL_CUBLAS == 1
				,matrix_mul_cublas
#endif
				}){
			//cublasStatus_t stat;
			//stat = cublasCreate(&handle);
			//assert(stat == CUBLAS_STATUS_SUCCESS);
		}


		//matrix_mul_tunable():Tunable<MeasureType, matrix_mul_func, Tunable_Args...>(MATRIX_MUL_NUM_DIM, MATRIX_MUL_NUM_VARIANTS){}
		//tuple format: seq_id, vid, val, size1, size2, size3
		//parameter: variant mask, repeat size, size1, size2, ...
		std::vector<std::tuple<size_t, size_t, MeasureType, Tunable_Args...> > training_run(std::vector<bool> const& variant_mask, size_t const repeat_size, size_t const HA, size_t const WA, size_t const WB) const{


			using namespace meterpu;
			//encapsulate the difference of correct meter
			meter<CPU_Time> my_meter;
#if defined(XPDL_CUBLAS) && XPDL_CUBLAS == 1
			meter<CUDA_Time> cuda_meter;
#endif

			//meter<System_Energy> my_meter;

			//System_Energy::ResultType val;
			CPU_Time::ResultType val;

			//encapsulate the difference of problem instance generation
			vectorpu::vector<float> A(WA*HA,1), B(WA*WB,1), C(HA*WB,0), C_ref(HA*WB,WA);

			size_t const num_variant_run=count(variant_mask.cbegin(),variant_mask.cend(),true);
			std::vector<std::tuple<size_t, size_t, MeasureType, Tunable_Args...> > results;
			results.reserve(num_variant_run*repeat_size);

			//LOC: 24

			//support by high performance sampling by reuse
			//the problem instance allocated
			//for(size_t i=0; i<this->num_variants;
			//++i){
			//if(variant_mask[i]){
			//for(size_t r=0; r<repeat_size; ++r){
			//if(i<this->num_variants-1){
			//cpu_meter.start();
			////encapsulate the difference of correct invoke
			//(*this->dispatch_table[i])(R(A), R(B), W(C), HA, WA, WB);
			//cpu_meter.stop();
			//cpu_meter.calc();
			//val=cpu_meter.get_value();
			//}
			//else{
			//cuda_meter.start();
			//(*this->dispatch_table[i])(GR(A), GR(B), GW(C), HA, WA, WB);
			//cuda_meter.stop();
			//cuda_meter.calc();
			//val=cuda_meter.get_value();
			//}
			//results.emplace_back(r,i,val,HA,WA,WB);
			//}
			//}
			//}

			size_t i=0;

			if(variant_mask[i]){
				for(size_t r=0; r<repeat_size; ++r){
					my_meter.start();
					//encapsulate the difference of correct invoke
					(*this->dispatch_table[i])(R(A), R(B), W(C), HA, WA, WB);
					my_meter.stop();
					my_meter.calc();
					val=my_meter.get_value();
					results.emplace_back(r,i,val,HA,WA,WB);
				}
			}
			++i;
			//LOC: 10

#if XPDL_GSL_CBLAS == 1
			if(variant_mask[i]){
				for(size_t r=0; r<repeat_size; ++r){
					my_meter.start();
					(*this->dispatch_table[i])(R(A), R(B), W(C), HA, WA, WB);
					my_meter.stop();
					my_meter.calc();
					val=my_meter.get_value();
					results.emplace_back(r,i,val,HA,WA,WB);
				}
			}
			++i;
#endif
			//LOC: 11

#if XPDL_OPENMP == 1
			if(variant_mask[i]){
				for(size_t r=0; r<repeat_size; ++r){
					my_meter.start();
					(*this->dispatch_table[i])(R(A), R(B), W(C), HA, WA, WB);
					my_meter.stop();
					my_meter.calc();
					val=my_meter.get_value();
					results.emplace_back(r,i,val,HA,WA,WB);
				}
			}
			++i;
#endif
			//LOC: 11

#if XPDL_CUBLAS == 1
			if(variant_mask[i]){
				for(size_t r=0; r<repeat_size; ++r){
					cuda_meter.start();
					(*this->dispatch_table[i])(GR(A), GR(B), GW(C), HA, WA, WB);
					cuda_meter.stop();
					cuda_meter.calc();
					val=cuda_meter.get_value();
					results.emplace_back(r,i,val,HA,WA,WB);
				}
			}
			++i;
#endif
			//LOC: 11


			//encapsulate the difference of correctness check
			assert( equal(RI(C),REI(C), RI(C_ref)) );


			//unify the return value
			return results;
			//LOC: 2
		}

		template <class T>
		void run(size_t predicted_index, vectorpu::vector<T> &a, vectorpu::vector<T> &b, vectorpu::vector<T> &c, size_t const ha,size_t const wa,size_t const wb) const
		{
			size_t i=0;
			if(predicted_index==i)
				(*this->dispatch_table[i])(R(a), R(b), W(c), ha, wa, wb);
			i++;

#if XPDL_GSL_CBLAS == 1
			if(predicted_index==i)
				(*this->dispatch_table[i])(R(a), R(b), W(c), ha, wa, wb);
			i++;
#endif
		
#if XPDL_OPENMP == 1
			if(predicted_index==i)
				(*this->dispatch_table[i])(R(a), R(b), W(c), ha, wa, wb);
			i++;
#endif

#if XPDL_CUBLAS == 1
			if(predicted_index==i)
				(*this->dispatch_table[i])(GR(a), GR(b), GW(c), ha, wa, wb);
#endif
		}

		//LOC: 20



};
