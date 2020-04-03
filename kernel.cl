__kernel void find_distances(__global const uint3 *triangles_array,
    __global const float3 *verticles_array, __global const uint *triangles_size,
    __global const uint *verticles_size,
    __global float3 *distances)
{
    int gid = get_global_id(0);

    float3 x1 = { verticles_array[triangles_array[gid].x].x, verticles_array[triangles_array[gid].x].y, verticles_array[triangles_array[gid].x].z };
    float3 x2 = { verticles_array[triangles_array[gid].y].x, verticles_array[triangles_array[gid].y].y, verticles_array[triangles_array[gid].y].z };
    float3 x3 = { verticles_array[triangles_array[gid].z].x, verticles_array[triangles_array[gid].z].y, verticles_array[triangles_array[gid].z].z };

    distances[gid].x = distance(x1, x2);
    distances[gid].y = distance(x2, x3);
    distances[gid].z = distance(x1, x3);
    /*
    printf("triangle %d: %d, %d, %d\n  vertex_1: %f, %f, %f\n  vertex_2: %f, %f, %f\n  vertex_3: %f, %f, %f\n  distance: %f, %f, %f", gid,
    triangles_array[gid].x, triangles_array[gid].y, triangles_array[gid].z,
    verticles_array[triangles_array[gid].x].x,
    verticles_array[triangles_array[gid].x].y,
    verticles_array[triangles_array[gid].x].z,
    verticles_array[triangles_array[gid].y].x,
    verticles_array[triangles_array[gid].y].y,
    verticles_array[triangles_array[gid].y].z,
    verticles_array[triangles_array[gid].z].x,
    verticles_array[triangles_array[gid].z].y,
    verticles_array[triangles_array[gid].z].z,
    distances[gid].x, distances[gid].y, distances[gid].z);
    */
}

__kernel void sort_distances(__global float3 *distances,
    __global uint3 *triangles_array,
    __global const uint *triangles_size)
{
    
}