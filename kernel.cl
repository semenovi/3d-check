__kernel void shorter_than(__global const float *matrix_in,
					__global float *matrix_out,
					__global const float *max_length)
{
	int gid = get_global_id(0);
	if (matrix_in[gid] < *max_length)
	{
		matrix_out[gid] = matrix_in[gid];
	}
}