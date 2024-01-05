
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "pressio_options.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"
#include "std_compat/memory.h"
#include <cmath>

namespace libpressio { namespace qcatsobolevp2_metrics_ns {
#define QCAT_FLOAT 0
#define QCAT_DOUBLE 1
#define QCAT_INT32 2
#define QCAT_INT16 3
#define QCAT_UINT32 4
#define QCAT_UINT16 5
size_t computeDataLength(size_t r5, size_t r4, size_t r3, size_t r2, size_t r1)
{
	size_t dataLength;
	if(r1==0) 
	{
		dataLength = 0;
	}
	else if(r2==0) 
	{
		dataLength = r1;
	}
	else if(r3==0) 
	{
		dataLength = r1*r2;
	}
	else if(r4==0) 
	{
		dataLength = r1*r2*r3;
	}
	else if(r5==0) 
	{
		dataLength = r1*r2*r3*r4;
	}
	else 
	{
		dataLength = r1*r2*r3*r4*r5;
	}
	return dataLength;
}
int computeDimension(size_t r5, size_t r4, size_t r3, size_t r2, size_t r1)
{
	if(r1==0)
		return 0;
	else if(r2 == 0)
		return 1;
	else if(r3 == 0)
		return 2;
	else if(r4 == 0)
		return 3;
	else if(r5 == 0)
		return 4;
	return 5;
}

//the sum of square / nbEle
double calculateSobolevNorm_s0_p2_float(float *data, size_t r5, size_t r4, size_t r3, size_t r2, size_t r1)
{
	double result = 0;
	size_t i;
	size_t nbEle = computeDataLength(r5, r4, r3, r2, r1);
	double sum = 0;
	for(i=0;i<nbEle;i++)
		sum += data[i]*data[i];
	result = sqrt(sum/(double)nbEle);
	return result;
}

//sqrt((||f(0)||^2+||f(1)||^2)/nbEle), where f(i) means i-th derivative
double calculateSobolevNorm_s1_p2_float(float *data, size_t r5, size_t r4, size_t r3, size_t r2, size_t r1)
{
	size_t i, j, k;
	int dim = computeDimension(r5, r4, r3, r2, r1);
	//size_t nbEle = computeDataLength(r5, r4, r3, r2, r1);
	double sum = 0;
	size_t counter =  0;
	if(dim==1)
	{
		for(i=1;i<r1-1;i++)
		{
			sum += data[i]*data[i];
			sum += (data[i+1]-data[i-1])*(data[i+1]-data[i-1])/4;
			counter ++;
		}
		return sqrt(sum/counter);
	}
	else if(dim==2)
	{
		for(i=1;i<r2-1;i++)
		{	
			for(j=1;j<r1-1;j++)
			{
				size_t index = i*r1+j;
				sum += data[index]*data[index];
				sum += (data[index+1]-data[index-1])*(data[index+1]-data[index-1])/4;
				sum += (data[index+r1]-data[index-r1])*(data[index+r1]-data[index-r1])/4;
				counter ++;
			}
		}				
		
		return sqrt(sum/counter);
	}
	else if(dim==3)
	{
		size_t r2r1 = r2*r1;
		for(i=1;i<r3-1;i++)
		{	
			for(j=1;j<r2-1;j++)
			{
				for(k=1;k<r1-1;k++)
				{
					size_t index = i*r2r1+j*r1+k;
					sum += data[index]*data[index];
					sum += (data[index+1]-data[index-1])*(data[index+1]-data[index-1])/4;
					sum += (data[index+r1]-data[index-r1])*(data[index+r1]-data[index-r1])/4;
					sum += (data[index+r2r1]-data[index-r2r1])*(data[index+r2r1]-data[index-r2r1])/4;		
					counter ++;			
				}
			}
		}			
		
		return sqrt(sum/counter);
	}
	
	return -1; //error or does not support dim>3
}

 //sqrt((||f(0)||^2+||f(1)||^2+||f(2)||^2)/nbEle), where f(i) means i-th derivative (including mixed partial if possible)
double calculateSobolevNorm_s2_p2_float(float *data, size_t r5, size_t r4, size_t r3, size_t r2, size_t r1)
{
	size_t i, j, k;
	int dim = computeDimension(r5, r4, r3, r2, r1);
	//size_t nbEle = computeDataLength(r5, r4, r3, r2, r1);
	size_t counter = 0;
	double sum = 0;
	if(dim==1)
	{
		for(i=1;i<r1-1;i++)
		{
			float x1_dev = (data[i+1]-data[i-1])/2; //x1_dev1 means along x, 1st-order partial
			float x2_dev = (data[i+1]-data[i])-(data[i]-data[i-1]); //second order partial
		
			sum += data[i]*data[i];
			sum += x1_dev*x1_dev;
			sum += x2_dev*x2_dev;
			counter ++;
		}
		
		return sqrt(sum/counter);
	}	
	else if(dim==2)
	{
		for(i=1;i<r2-1;i++)
			for(j=1;j<r1-1;j++)
			{
				size_t index = i*r1+j;
				float x1_dev = (data[index+1]-data[index-1])/2;
				float x2_dev = (data[index+1]-data[index])-(data[index]-data[index-1]);
				float y1_dev = (data[index+r1]-data[index-r1])/2;
				float y2_dev = (data[index+r1]-data[index])-(data[index]-data[index-r1]);
				float xy_dev = (data[index-r1-1]+data[index+r1+1]-data[index-r1+1]-data[index+r1-1])/4;
				
				sum += data[index]*data[index];
				sum += x1_dev*x1_dev;
				sum += x2_dev*x2_dev;
				sum += y1_dev*y1_dev;
				sum += y2_dev*y2_dev;
				sum += xy_dev*xy_dev;
				counter ++;
			}
			
		return sqrt(sum/counter);
	}
	else if(dim==3)
	{
		size_t r2r1 = r2*r1;
		for(i=1;i<r3-1;i++)
			for(j=1;j<r2-1;j++)
				for(k=1;k<r1-1;k++)
				{
					size_t index = i*r2r1+j*r1+k;
					float x1_dev = (data[index+1]-data[index-1])/2;
					float x2_dev = (data[index+1]-data[index])-(data[index]-data[index-1]);
					float y1_dev = (data[index+r1]-data[index-r1])/2;
					float y2_dev = (data[index+r1]-data[index])-(data[index]-data[index-r1]);
					float z1_dev = (data[index+r2r1]-data[index-r2r1])/2;
					float z2_dev = (data[index+r2r1]-data[index])-(data[index]-data[index-r2r1]);										
					float xy_dev = (data[index-r1-1]+data[index+r1+1] - data[index-r1+1] - data[index+r1-1])/4;
					float yz_dev = (data[index-r2r1-r1]+data[index+r2r1+r1] - data[index-r2r1+r1] - data[index+r2r1-r1])/4;
					float xz_dev = (data[index-r2r1-1]+data[index+r2r1+1]-data[index-r2r1+1]-data[index+r2r1-1])/4;
					sum += data[index]*data[index];
					sum += x1_dev*x1_dev;
					sum += x2_dev*x2_dev;
					sum += y1_dev*y1_dev;
					sum += y2_dev*y2_dev;
					sum += z1_dev*z1_dev;
					sum += z2_dev*z2_dev;
					sum += xy_dev*xy_dev;
					sum += yz_dev*yz_dev;
					sum += xz_dev*xz_dev;
					counter ++;
				}
				
		return sqrt(sum/counter);
	}
	
	return -1;
}

//the sum of square / nbEle
double calculateSobolevNorm_s0_p2_double(double *data, size_t r5, size_t r4, size_t r3, size_t r2, size_t r1)
{
	double result = 0;
	size_t i;
	size_t nbEle = computeDataLength(r5, r4, r3, r2, r1);
	double sum = 0;
	for(i=0;i<nbEle;i++)
		sum += data[i]*data[i];
	result = sqrt(sum/nbEle);
	return result;
}

//sqrt((||f(0)||^2+||f(1)||^2)/nbEle), where f(i) means i-th derivative
double calculateSobolevNorm_s1_p2_double(double *data, size_t r5, size_t r4, size_t r3, size_t r2, size_t r1)
{
	size_t i, j, k;
	int dim = computeDimension(r5, r4, r3, r2, r1);
	//size_t nbEle = computeDataLength(r5, r4, r3, r2, r1);
	size_t counter = 0;
	double sum = 0;
	if(dim==1)
	{
		for(i=1;i<r1-1;i++)
		{
			sum += data[i]*data[i];
			sum += (data[i+1]-data[i-1])*(data[i+1]-data[i-1])/4;
			counter ++;
		}
		
		return sqrt(sum/counter);
	}
	else if(dim==2)
	{
		for(i=1;i<r2-1;i++)
		{	
			for(j=1;j<r1-1;j++)
			{
				size_t index = i*r1+j;
				sum += data[index]*data[index];
				sum += (data[index+1]-data[index-1])*(data[index+1]-data[index-1])/4;
				sum += (data[index+r1]-data[index-r1])*(data[index+r1]-data[index-r1])/4;
				counter ++;
			}
		}				
		
		return sqrt(sum/counter);
	}
	else if(dim==3)
	{
		size_t r2r1 = r2*r1;
		for(i=1;i<r3-1;i++)
		{	
			for(j=1;j<r2-1;j++)
			{
				for(k=1;k<r1-1;k++)
				{
					size_t index = i*r2r1+j*r1+k;
					sum += data[index]*data[index];
					sum += (data[index+1]-data[index-1])*(data[index+1]-data[index-1])/4;
					sum += (data[index+r1]-data[index-r1])*(data[index+r1]-data[index-r1])/4;
					sum += (data[index+r2r1]-data[index-r2r1])*(data[index+r2r1]-data[index-r2r1])/4;				
					counter ++;	
				}
			}
		}			
		
		return sqrt(sum/counter);
	}
	
	return -1; //error or does not support dim>3
}

 //sqrt((||f(0)||^2+||f(1)||^2+||f(2)||^2)/nbEle), where f(i) means i-th derivative (including mixed partial if possible)
double calculateSobolevNorm_s2_p2_double(double *data, size_t r5, size_t r4, size_t r3, size_t r2, size_t r1)
{
	size_t i, j, k;
	int dim = computeDimension(r5, r4, r3, r2, r1);
	//size_t nbEle = computeDataLength(r5, r4, r3, r2, r1);
	size_t counter = 0;
	double sum = 0;
	if(dim==1)
	{
		for(i=1;i<r1-1;i++)
		{
			float x1_dev = (data[i+1]-data[i-1])/2; //x1_dev1 means along x, 1st-order partial
			float x2_dev = (data[i+1]-data[i])-(data[i]-data[i-1]); //second order partial
		
			sum += data[i]*data[i];
			sum += x1_dev*x1_dev;
			sum += x2_dev*x2_dev;
			counter ++;
		}
		
		return sqrt(sum/counter);
	}	
	else if(dim==2)
	{
		for(i=1;i<r2-1;i++)
			for(j=1;j<r1-1;j++)
			{
				size_t index = i*r1+j;
				float x1_dev = (data[index+1]-data[index-1])/2;
				float x2_dev = (data[index+1]-data[index])-(data[index]-data[index-1]);
				float y1_dev = (data[index+r1]-data[index-r1])/2;
				float y2_dev = (data[index+r1]-data[index])-(data[index]-data[index-r1]);
				float xy_dev = (data[index-r1-1]+data[index+r1+1]-data[index-r1+1]-data[index+r1-1])/4;
				
				sum += data[index]*data[index];
				sum += x1_dev*x1_dev;
				sum += x2_dev*x2_dev;
				sum += y1_dev*y1_dev;
				sum += y2_dev*y2_dev;
				sum += xy_dev*xy_dev;
				counter ++;
			}
			
		return sqrt(sum/counter);
	}
	else if(dim==3)
	{
		size_t r2r1 = r2*r1;
		for(i=1;i<r3-1;i++)
			for(j=1;j<r2-1;j++)
				for(k=1;k<r1-1;k++)
				{
					size_t index = i*r2r1+j*r1+k;
					float x1_dev = (data[index+1]-data[index-1])/2;
					float x2_dev = (data[index+1]-data[index])-(data[index]-data[index-1]);
					float y1_dev = (data[index+r1]-data[index-r1])/2;
					float y2_dev = (data[index+r1]-data[index])-(data[index]-data[index-r1]);
					float z1_dev = (data[index+r2r1]-data[index-r2r1])/2;
					float z2_dev = (data[index+r2r1]-data[index])-(data[index]-data[index-r2r1]);										
					float xy_dev = (data[index-r1-1]+data[index+r1+1] - data[index-r1+1] - data[index+r1-1])/4;
					float yz_dev = (data[index-r2r1-r1]+data[index+r2r1+r1] - data[index-r2r1+r1] - data[index+r2r1-r1])/4;
					float xz_dev = (data[index-r2r1-1]+data[index+r2r1+1]-data[index-r2r1+1]-data[index+r2r1-1])/4;
					sum += data[index]*data[index];
					sum += x1_dev*x1_dev;
					sum += x2_dev*x2_dev;
					sum += y1_dev*y1_dev;
					sum += y2_dev*y2_dev;
					sum += z1_dev*z1_dev;
					sum += z2_dev*z2_dev;
					sum += xy_dev*xy_dev;
					sum += yz_dev*yz_dev;
					sum += xz_dev*xz_dev;
					counter ++;
				}
				
		return sqrt(sum/counter);
	}
	
	return -1;
}


double calculateSobolevNorm_p2(void *data, int dataType, int order, size_t , size_t r4, size_t r3, size_t r2, size_t r1)
{
	double result = 0;
	if(dataType==QCAT_FLOAT)
	{
		float* d = (float*) data;
		switch(order)
		{
			case 0: 
				result = calculateSobolevNorm_s0_p2_float(d, 0, r4, r3, r2, r1);
				break;
			case 1:
				result = calculateSobolevNorm_s1_p2_float(d, 0, r4, r3, r2, r1);
				break;
			case 2:
				result = calculateSobolevNorm_s2_p2_float(d, 0, r4, r3, r2, r1);
				break;				
			default:
				printf("Error: wrong order: %d\n", order);
		}
	}
	else
	{
		double* d = (double*) data;
		switch(order)
		{
			case 0: 
				result = calculateSobolevNorm_s0_p2_double(d, 0, r4, r3, r2, r1);
				break;
			case 1:
				result = calculateSobolevNorm_s1_p2_double(d, 0, r4, r3, r2, r1);
				break;
			case 2:
				result = calculateSobolevNorm_s2_p2_double(d, 0, r4, r3, r2, r1);
				break;				
			default:
				printf("Error: wrong order: %d\n", order);
		}	
	}
	
	return result;
}

class qcatsobolevp2_plugin : public libpressio_metrics_plugin {
  public:
    int begin_compress_impl(struct pressio_data const* input, pressio_data const*) override {
      if(run_input) {
          int datatype;
          if(input == nullptr) return 0;
          if(input->dtype() == pressio_float_dtype) datatype = QCAT_FLOAT;
          else if(input->dtype() == pressio_double_dtype) datatype = QCAT_DOUBLE;
          else {
              return 0;
          }
          auto dims = input->normalized_dims(4);

          uncompressed_result = calculateSobolevNorm_p2(input->data(), datatype, order, 0, dims[3], dims[2], dims[1], dims[0]);
      }
      return 0;
    }

    int end_decompress_impl(struct pressio_data const* , pressio_data const* output, int rc) override {
      if(run_output) {
          int datatype;
          if(rc > 0 || output == nullptr) return 0;
          if(output->dtype() == pressio_float_dtype) datatype = QCAT_FLOAT;
          else if(output->dtype() == pressio_double_dtype) datatype = QCAT_DOUBLE;
          else {
              return 0;
          }
          auto dims = output->normalized_dims(4);

          decompressed_result = calculateSobolevNorm_p2(output->data(), datatype, order, 0, dims[3], dims[2], dims[1], dims[0]);
      }
      return 0;
    }

  struct pressio_options get_configuration_impl() const override {
    pressio_options opts;
    set(opts, "pressio:stability", "stable");
    set(opts, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(opts, "predictors:requires_decompress", true);
    set(opts, "predictors:invalidate", std::vector<std::string>{"predictors:error_dependent"});
    return opts;
  }

  struct pressio_options get_options() const override {
      pressio_options opt;
      set(opt, "qcatsobolevp2:run_uncompressed", run_input);
      set(opt, "qcatsobolevp2:run_decompressed", run_output);
      set(opt, "qcatsobolevp2:order", order);
      return opt;
  }
  int set_options(pressio_options const& opt) override {
      get(opt, "qcatsobolevp2:run_uncompressed", &run_input);
      get(opt, "qcatsobolevp2:run_decompressed", &run_output);
      get(opt, "qcatsobolevp2:order", &order);
      return 0;
  }


  struct pressio_options get_documentation_impl() const override {
    pressio_options opt;
    set(opt, "pressio:description", R"(Run the sobolev p2 norm from qcat

    https://github.com/szcompressor/qcat)");
    set(opt, "qcatsobolevp2:input", "sobolev norm for the input data");
    set(opt, "qcatsobolevp2:decompressed", "sobolev norm for the decompresesd data");
    set(opt, "qcatsobolevp2:diff", "diff of the sobolev norm");
    set(opt, "qcatsobolevp2:run_uncompressed", "run sobolev norm on input");
    set(opt, "qcatsobolevp2:run_decompressed", "run sobolev norm on output");
    set(opt, "qcatsobolevp2:order", "order of the norm {0,1,2}");
    return opt;
  }

  pressio_options get_metrics_results(pressio_options const &) override {
    pressio_options opt;
    set(opt, "qcatsobolevp2:input", uncompressed_result);
    set(opt, "qcatsobolevp2:decompressed", decompressed_result);
    if(uncompressed_result && decompressed_result) {
        set(opt, "qcatsobolevp2:diff", *uncompressed_result - *decompressed_result);
    } else {
        set_type(opt, "qcatsobolevp2:diff", pressio_option_double_type);
    }
    return opt;
  }

  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<qcatsobolevp2_plugin>(*this);
  }
  const char* prefix() const override {
    return "qcatsobolevp2";
  }

  private:
  compat::optional<double> uncompressed_result;
  compat::optional<double> decompressed_result;
  int32_t order = 0;
  bool run_input = true;
  bool run_output = true;

};

static pressio_register metrics_qcatsobolevp2_plugin(metrics_plugins(), "qcatsobolevp2", [](){ return compat::make_unique<qcatsobolevp2_plugin>(); });
}}

