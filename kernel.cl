__kernel void set_is_small(__global uint4 *triangles_array,
    __global const float3 *verticles_array, __global const float *min)
{
    int gid = get_global_id(0);

    bool is_small = false;

    float3 x1 = { verticles_array[triangles_array[gid].x].x, verticles_array[triangles_array[gid].x].y, verticles_array[triangles_array[gid].x].z };
    float3 x2 = { verticles_array[triangles_array[gid].y].x, verticles_array[triangles_array[gid].y].y, verticles_array[triangles_array[gid].y].z };
    float3 x3 = { verticles_array[triangles_array[gid].z].x, verticles_array[triangles_array[gid].z].y, verticles_array[triangles_array[gid].z].z };

    is_small = (distance(x1, x2) < *min) || (distance(x2, x3) < *min) || (distance(x1, x3) < *min);

    if (is_small)
        triangles_array[gid].w = 1;
    //printf("triangle %d: %d\ndistance: %f\nmin: %f", gid, triangles_array[gid].w, distance(x1, x2), *min);
}

__kernel void sort_distances(__global float3 *distances,
    __global uint3 *triangles_array,
    __global const uint *triangles_size)
{
    
}