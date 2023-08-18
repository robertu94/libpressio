
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "pressio_options.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"
#include "std_compat/memory.h"
#include <cmath>

namespace libpressio { namespace gradlength_metrics_ns {
#define QCAT_FLOAT 0
#define QCAT_DOUBLE 1
#define QCAT_INT32 2
#define QCAT_INT16 3
#define QCAT_UINT32 4
#define QCAT_UINT16 5


void computeGradientLength_float(float* data, float*gradMag, size_t , size_t r4, size_t r3, size_t r2, size_t r1)
{
	size_t i, j, k, index;
	double gradx, grady, gradz;
	
	if(r2==0)
	{
		gradMag[0] = data[1] - data[0];
		gradMag[r1-1] = data[r1-1]-data[r1-2];
		for(i=1;i<r1-1;i++)
			gradMag[i] = (data[i+1]-data[i-1])/2;
	}
	else if(r3==0)
	{
		//process four corners
		gradx = data[1]-data[0];
		grady = data[r1]-data[0];
		gradMag[0] = sqrt(gradx*gradx+grady*grady);
		index = r1-1;
		gradx = data[index]-data[index-1];
		grady = data[index+r1]-data[index];
		gradMag[index] = sqrt(gradx*gradx+grady*grady);		
		index = (r2-1)*r1;
		gradx = data[index+1]-data[index];
		grady = data[index]-data[index-r1];
		gradMag[index] = sqrt(gradx*gradx+grady*grady);	
		index = (r2-1)*r1 + r1 - 1;			
		gradx = data[index]-data[index-1];
		grady = data[index]-data[index-r1];
		gradMag[index] = sqrt(gradx*gradx+grady*grady);								
		
		//process four edges
		for(i=1;i<r1-1;i++)
		{
			index = i;
			gradx = (data[index+1]-data[index-1])/2;
			grady = data[index+r1]-data[index];
			gradMag[index] = sqrt(gradx*gradx+grady*grady);			
		}
		for(i=1;i<r1-1;i++)
		{
			index = (r2-1)*r1 + i;
			gradx = (data[index+1]-data[index-1])/2;
			grady = data[index]-data[index-r1];
			gradMag[index] = sqrt(gradx*gradx+grady*grady);			
		}
		
		for(i=1;i<r2-1;i++)
		{
			index = i*r1;
			gradx = (data[index+1] - data[index]);
			grady = (data[index+r1]-data[index-r1])/2;
			gradMag[index] = sqrt(gradx*gradx+grady*grady);						
		}
		for(i=1;i<r2-1;i++)
		{
			index = i*r1+r1-1;
			gradx = (data[index] - data[index-1]);
			grady = (data[index+r1]-data[index-r1])/2;
			gradMag[index] = sqrt(gradx*gradx+grady*grady);						
		}		
		
		//process all interior points
		for(i=1;i<r2-1;i++)
			for(j=1;j<r1-1;j++)
			{
				index = i*r1+j;
				gradx = (data[index+1] - data[index-1])/2;
				grady = (data[index+r1] - data[index-r1])/2;
				gradMag[index] = sqrt(gradx*gradx+grady*grady);
			}
		
	}else if(r4==0) //3D
	{
		size_t r2r1 = r2*r1;
		//process all 8 corners
		gradx = data[1]-data[0];
		grady = data[r1]-data[0];
		gradz = data[r2r1]-data[0];		
		gradMag[0] = sqrt(gradx*gradx+grady*grady+gradz*gradz);
		
		index = r1-1;
		gradx = data[index]-data[index-1];
		grady = data[index+r1]-data[index];		
		gradz = data[index+r2r1]-data[index];		
		gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);
		
		index = (r2-1)*r1;
		gradx = data[index+1]-data[index];
		grady = data[index]-data[index-r1];		
		gradz = data[index+r2r1]-data[index];		
		gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);

		index = (r2-1)*r1 + r1 - 1;
		gradx = data[index]-data[index-1];
		grady = data[index]-data[index-r1];		
		gradz = data[index+r2r1]-data[index];		
		gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);
		
		index = (r3-1)*r2r1;
		gradx = data[index+1]-data[index];
		grady = data[index+r1]-data[index];
		gradz = data[index]-data[index-r2r1];		
		gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);
		
		index = (r3-1)*r2r1+r1-1;
		gradx = data[index]-data[index-1];
		grady = data[index+r1]-data[index];		
		gradz = data[index]-data[index-r2r1];		
		gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);
		
		index = (r3-1)*r2r1 + (r2-1)*r1;
		gradx = data[index+1]-data[index];
		grady = data[index]-data[index-r1];		
		gradz = data[index]-data[index-r2r1];		
		gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);

		index = (r3-1)*r2r1 + (r2-1)*r1 + r1 - 1;
		gradx = data[index]-data[index-1];
		grady = data[index]-data[index-r1];		
		gradz = data[index]-data[index-r2r1];		
		gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);		
				
		//process all 8 edges
		for(i=1;i<r1-1;i++)
		{
			index = i;
			gradx = (data[index+1]-data[index-1])/2;
			grady = data[index+r1]-data[index];
			gradz = data[index+r2r1] - data[index];
			gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);			
		}

		for(i=1;i<r1-1;i++)
		{
			index = (r2-1)*r1 + i;
			gradx = (data[index+1]-data[index-1])/2;
			grady = data[index]-data[index-r1];
			gradz = data[index+r2r1] - data[index];
			gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);			
		}
		
		for(i=1;i<r2-1;i++)
		{
			index = i*r1;
			gradx = (data[index+1] - data[index]);
			grady = (data[index+r1]-data[index-r1])/2;
			gradz = data[index+r2r1] - data[index];
			gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);						
		}
		for(i=1;i<r2-1;i++)
		{
			index = i*r1+r1-1;
			gradx = (data[index] - data[index-1]);
			grady = (data[index+r1]-data[index-r1])/2;
			gradz = data[index+r2r1] - data[index];
			gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);						
		}
		
		for(i=1;i<r1-1;i++)
		{
			index = (r3-1)*r2r1 + i;
			gradx = (data[index+1]-data[index-1])/2;
			grady = data[index+r1]-data[index];
			gradz = data[index] - data[index-r2r1];
			gradMag[i] = sqrt(gradx*gradx+grady*grady+gradz*gradz);			
		}

		for(i=1;i<r1-1;i++)
		{
			index = (r3-1)*r2r1 + (r2-1)*r1 + i;
			gradx = (data[index+1]-data[index-1])/2;
			grady = data[index]-data[index-r1];
			gradz = data[index] - data[index-r2r1];
			gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);			
		}
		
		for(i=1;i<r2-1;i++)
		{
			index = (r3-1)*r2r1+i*r1;
			gradx = (data[index+1] - data[index]);
			grady = (data[index+r1]-data[index-r1])/2;
			gradz = data[index] - data[index-r2r1];
			gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);						
		}
		
		for(i=1;i<r2-1;i++)
		{
			index = (r3-1)*r2r1+i*r1+r1-1;
			gradx = (data[index] - data[index-1]);
			grady = (data[index+r1]-data[index-r1])/2;
			gradz = data[index] - data[index-r2r1];
			gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);						
		}
		
		//process all 6 sides
		for(i=1;i<r2-1;i++)
			for(j=1;j<r1-1;j++)
			{
				index = i*r1+j;
				gradx = (data[index+1] - data[index-1])/2;
				grady = (data[index+r1] - data[index-r1])/2;
				gradz = data[index+r2r1] - data[index];
				gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);
			}		

		for(i=1;i<r2-1;i++)
			for(j=1;j<r1-1;j++)
			{
				index = (r3-1)*r2r1 + i*r1 + j;
				gradx = (data[index+1] - data[index-1])/2;
				grady = (data[index+r1] - data[index-r1])/2;
				gradz = data[index] - data[index-r2r1];
				gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);
			}				
		
		for(i=1;i<r3-1;i++)
			for(k=1;k<r1-1;k++)
			{
				index = i*r2r1 + k; //j is always 0
				gradx = (data[index+1] - data[index-1])/2;
				grady = data[index+r1] - data[index];
				gradz = (data[index+r2r1] - data[index-r2r1])/2;
				gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);
			}
			
		for(i=1;i<r3-1;i++)
			for(k=1;k<r1-1;k++)
			{
				j = r2-1;
				index = i*r2r1 + j*r1 + k;
				gradx = (data[index+1] - data[index-1])/2;
				grady = data[index] - data[index-r1];
				gradz = (data[index+r2r1] - data[index-r2r1])/2;
				gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);
			}

		for(i=1;i<r3-1;i++)
			for(j=1;j<r2-1;j++)
			{
				index = i*r2r1 + j*r1;
				gradx = data[index+1] - data[index];
				grady = (data[index+r1] - data[index-r1])/2;
				gradz = (data[index+r2r1] - data[index-r2r1])/2;
				gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);
			}

		for(i=1;i<r3-1;i++)
			for(j=1;j<r2-1;j++)
			{
				k = r1-1;
				index = i*r2r1 + j*r1 + k; 
				gradx = data[index] - data[index-1];
				grady = (data[index+r1] - data[index-r1])/2;
				gradz = (data[index+r2r1] - data[index-r2r1])/2;
				gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);
			}
		
		//process interior points
		for(i=1;i<r3-1;i++)
			for(j=1;j<r2-1;j++)
				for(k=1;k<r1-1;k++)
				{
					size_t index = i*r2r1+j*r1+k;
					gradx = (data[index+1] - data[index-1])/2;
					grady = (data[index+r1] - data[index-r1])/2;
					gradz = (data[index+r2r1] - data[index-r2r1])/2;
					gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);
				}		
	}
	
}

void computeGradientLength_double(double* data, double*gradMag, size_t , size_t r4, size_t r3, size_t r2, size_t r1)
{
	size_t i, j, k, index;
	double gradx, grady, gradz;
	
	if(r2==0)
	{
		gradMag[0] = data[1] - data[0];
		gradMag[r1-1] = data[r1-1]-data[r1-2];
		for(i=1;i<r1-1;i++)
			gradMag[i] = (data[i+1]-data[i-1])/2;
	}
	else if(r3==0)
	{
		//process four corners
		gradx = data[1]-data[0];
		grady = data[r1]-data[0];
		gradMag[0] = sqrt(gradx*gradx+grady*grady);
		index = r1-1;
		gradx = data[index]-data[index-1];
		grady = data[index+r1]-data[index];
		gradMag[index] = sqrt(gradx*gradx+grady*grady);		
		index = (r2-1)*r1;
		gradx = data[index+1]-data[index];
		grady = data[index]-data[index-r1];
		gradMag[index] = sqrt(gradx*gradx+grady*grady);	
		index = (r2-1)*r1 + r1 - 1;			
		gradx = data[index]-data[index-1];
		grady = data[index]-data[index-r1];
		gradMag[index] = sqrt(gradx*gradx+grady*grady);								
		
		//process four edges
		for(i=1;i<r1-1;i++)
		{
			index = i;
			gradx = (data[index+1]-data[index-1])/2;
			grady = data[index+r1]-data[index];
			gradMag[index] = sqrt(gradx*gradx+grady*grady);			
		}
		for(i=1;i<r1-1;i++)
		{
			index = (r2-1)*r1 + i;
			gradx = (data[index+1]-data[index-1])/2;
			grady = data[index]-data[index-r1];
			gradMag[index] = sqrt(gradx*gradx+grady*grady);			
		}
		
		for(i=1;i<r2-1;i++)
		{
			index = i*r1;
			gradx = (data[index+1] - data[index]);
			grady = (data[index+r1]-data[index-r1])/2;
			gradMag[index] = sqrt(gradx*gradx+grady*grady);						
		}
		for(i=1;i<r2-1;i++)
		{
			index = i*r1+r1-1;
			gradx = (data[index] - data[index-1]);
			grady = (data[index+r1]-data[index-r1])/2;
			gradMag[index] = sqrt(gradx*gradx+grady*grady);						
		}		
		
		//process all interior points
		for(i=1;i<r2-1;i++)
			for(j=1;j<r1-1;j++)
			{
				index = i*r1+j;
				gradx = (data[index+1] - data[index-1])/2;
				grady = (data[index+r1] - data[index-r1])/2;
				gradMag[index] = sqrt(gradx*gradx+grady*grady);
			}
		
	}else if(r4==0) //3D
	{
		size_t r2r1 = r2*r1;
		//process all 8 corners
		gradx = data[1]-data[0];
		grady = data[r1]-data[0];
		gradz = data[r2r1]-data[0];		
		gradMag[0] = sqrt(gradx*gradx+grady*grady+gradz*gradz);
		
		index = r1-1;
		gradx = data[index]-data[index-1];
		grady = data[index+r1]-data[index];		
		gradz = data[index+r2r1]-data[index];		
		gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);
		
		index = (r2-1)*r1;
		gradx = data[index+1]-data[index];
		grady = data[index]-data[index-r1];		
		gradz = data[index+r2r1]-data[index];		
		gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);

		index = (r2-1)*r1 + r1 - 1;
		gradx = data[index]-data[index-1];
		grady = data[index]-data[index-r1];		
		gradz = data[index+r2r1]-data[index];		
		gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);
		
		index = (r3-1)*r2r1;
		gradx = data[index+1]-data[index];
		grady = data[index+r1]-data[index];
		gradz = data[index]-data[index-r2r1];		
		gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);
		
		index = (r3-1)*r2r1+r1-1;
		gradx = data[index]-data[index-1];
		grady = data[index+r1]-data[index];		
		gradz = data[index]-data[index-r2r1];		
		gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);
		
		index = (r3-1)*r2r1 + (r2-1)*r1;
		gradx = data[index+1]-data[index];
		grady = data[index]-data[index-r1];		
		gradz = data[index]-data[index-r2r1];		
		gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);

		index = (r3-1)*r2r1 + (r2-1)*r1 + r1 - 1;
		gradx = data[index]-data[index-1];
		grady = data[index]-data[index-r1];		
		gradz = data[index]-data[index-r2r1];		
		gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);		
				
		//process all 8 edges
		for(i=1;i<r1-1;i++)
		{
			index = i;
			gradx = (data[index+1]-data[index-1])/2;
			grady = data[index+r1]-data[index];
			gradz = data[index+r2r1] - data[index];
			gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);			
		}

		for(i=1;i<r1-1;i++)
		{
			index = (r2-1)*r1 + i;
			gradx = (data[index+1]-data[index-1])/2;
			grady = data[index]-data[index-r1];
			gradz = data[index+r2r1] - data[index];
			gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);			
		}
		
		for(i=1;i<r2-1;i++)
		{
			index = i*r1;
			gradx = (data[index+1] - data[index]);
			grady = (data[index+r1]-data[index-r1])/2;
			gradz = data[index+r2r1] - data[index];
			gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);						
		}
		for(i=1;i<r2-1;i++)
		{
			index = i*r1+r1-1;
			gradx = (data[index] - data[index-1]);
			grady = (data[index+r1]-data[index-r1])/2;
			gradz = data[index+r2r1] - data[index];
			gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);						
		}
		
		for(i=1;i<r1-1;i++)
		{
			index = (r3-1)*r2r1 + i;
			gradx = (data[index+1]-data[index-1])/2;
			grady = data[index+r1]-data[index];
			gradz = data[index] - data[index-r2r1];
			gradMag[i] = sqrt(gradx*gradx+grady*grady+gradz*gradz);			
		}

		for(i=1;i<r1-1;i++)
		{
			index = (r3-1)*r2r1 + (r2-1)*r1 + i;
			gradx = (data[index+1]-data[index-1])/2;
			grady = data[index]-data[index-r1];
			gradz = data[index] - data[index-r2r1];
			gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);			
		}
		
		for(i=1;i<r2-1;i++)
		{
			index = (r3-1)*r2r1+i*r1;
			gradx = (data[index+1] - data[index]);
			grady = (data[index+r1]-data[index-r1])/2;
			gradz = data[index] - data[index-r2r1];
			gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);						
		}
		
		for(i=1;i<r2-1;i++)
		{
			index = (r3-1)*r2r1+i*r1+r1-1;
			gradx = (data[index] - data[index-1]);
			grady = (data[index+r1]-data[index-r1])/2;
			gradz = data[index] - data[index-r2r1];
			gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);						
		}
		
		//process all 6 sides
		for(i=1;i<r2-1;i++)
			for(j=1;j<r1-1;j++)
			{
				index = i*r1+j;
				gradx = (data[index+1] - data[index-1])/2;
				grady = (data[index+r1] - data[index-r1])/2;
				gradz = data[index+r2r1] - data[index];
				gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);
			}		

		for(i=1;i<r2-1;i++)
			for(j=1;j<r1-1;j++)
			{
				index = (r3-1)*r2r1 + i*r1 + j;
				gradx = (data[index+1] - data[index-1])/2;
				grady = (data[index+r1] - data[index-r1])/2;
				gradz = data[index] - data[index-r2r1];
				gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);
			}				
		
		for(i=1;i<r3-1;i++)
			for(k=1;k<r1-1;k++)
			{
				index = i*r2r1 + k; //j is always 0
				gradx = (data[index+1] - data[index-1])/2;
				grady = data[index+r1] - data[index];
				gradz = (data[index+r2r1] - data[index-r2r1])/2;
				gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);
			}
			
		for(i=1;i<r3-1;i++)
			for(k=1;k<r1-1;k++)
			{
				j = r2-1;
				index = i*r2r1 + j*r1 + k;
				gradx = (data[index+1] - data[index-1])/2;
				grady = data[index] - data[index-r1];
				gradz = (data[index+r2r1] - data[index-r2r1])/2;
				gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);
			}

		for(i=1;i<r3-1;i++)
			for(j=1;j<r2-1;j++)
			{
				index = i*r2r1 + j*r1;
				gradx = data[index+1] - data[index];
				grady = (data[index+r1] - data[index-r1])/2;
				gradz = (data[index+r2r1] - data[index-r2r1])/2;
				gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);
			}

		for(i=1;i<r3-1;i++)
			for(j=1;j<r2-1;j++)
			{
				k = r1-1;
				index = i*r2r1 + j*r1 + k; 
				gradx = data[index] - data[index-1];
				grady = (data[index+r1] - data[index-r1])/2;
				gradz = (data[index+r2r1] - data[index-r2r1])/2;
				gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);
			}
		
		//process interior points
		for(i=1;i<r3-1;i++)
			for(j=1;j<r2-1;j++)
				for(k=1;k<r1-1;k++)
				{
					size_t index = i*r2r1+j*r1+k;
					gradx = (data[index+1] - data[index-1])/2;
					grady = (data[index+r1] - data[index-r1])/2;
					gradz = (data[index+r2r1] - data[index-r2r1])/2;
					gradMag[index] = sqrt(gradx*gradx+grady*grady+gradz*gradz);
				}		
	}
	
}

int computeGradientLength(void* data, void*gradMag, int dataType, size_t r5, size_t r4, size_t r3, size_t r2, size_t r1)
{
	if(dataType==QCAT_FLOAT)
		computeGradientLength_float((float*)data, (float*)gradMag, r5, r4, r3, r2, r1);
	else if(dataType==QCAT_DOUBLE)
		computeGradientLength_double((double*)data, (double*)gradMag, r5, r4, r3, r2, r1);
	else
	{
		printf("Error: support only float or double.\n");
		return -1;	
	}
	return 0;
}

class gradlength_plugin : public libpressio_metrics_plugin {
    void evaluate(pressio_data const& input, pressio_data& output) {
        int datatype;
        output = pressio_data::owning(input.dtype(), input.dimensions());
        memset(output.data(), 0, output.size_in_bytes());
        if(input.dtype() == pressio_float_dtype) datatype=QCAT_FLOAT;
        else if(input.dtype() == pressio_double_dtype) datatype=QCAT_DOUBLE;
        else return;
        auto dims = input.normalized_dims(4);

        computeGradientLength(input.data(), output.data(), datatype, 0, dims[3], dims[2], dims[1], dims[0]);
    }
  public:
    int begin_compress_impl(struct pressio_data const* input, pressio_data const*) override {
      if(run_input) evaluate(*input, input_gradmag);
      return 0;
    }

    int end_decompress_impl(struct pressio_data const* , pressio_data const* output, int) override {
      if(run_output) evaluate(*output, decompressed_gradmag);
      return 0;
    }


  pressio_options get_options() const override {
      pressio_options opt;
      set(opt, "gradlength:run_input", run_input);
      set(opt, "gradlength:run_decompressed", run_output);
      return opt;
  }
  int set_options(pressio_options const& opt) override {
      get(opt, "gradlength:run_input", &run_input);
      get(opt, "gradlength:run_decompressed", &run_output);
      return 0;
  }

  struct pressio_options get_configuration_impl() const override {
    pressio_options opts;
    set(opts, "pressio:stability", "stable");
    set(opts, "pressio:thread_safe", pressio_thread_safety_multiple);
    return opts;
  }

  struct pressio_options get_documentation_impl() const override {
    pressio_options opt;
    set(opt, "pressio:description", R"(run the gradient length metric from qcat

    https:://github.com/szcompressor/qcat
    )");
    set(opt, "gradlength:input", "gradient magnitude of the input data");
    set(opt, "gradlength:decompressed", "gradient magnitude of the decompressed data");
    set(opt, "gradlength:run_input", "run gradient on input");
    set(opt, "gradlength:run_decompressed", "run gradient on output");
    return opt;
  }

  pressio_options get_metrics_results(pressio_options const &) override {
    pressio_options opt;
    set(opt, "gradlength:input", input_gradmag);
    set(opt, "gradlength:decompressed", decompressed_gradmag);
    return opt;
  }

  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<gradlength_plugin>(*this);
  }
  const char* prefix() const override {
    return "gradlength";
  }

  private:
  bool run_input = true;
  bool run_output = true;
  pressio_data input_gradmag;
  pressio_data decompressed_gradmag;

};

static pressio_register metrics_gradlength_plugin(metrics_plugins(), "gradlength", [](){ return compat::make_unique<gradlength_plugin>(); });
}}

