#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "OBJ_Loader.h"

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

///
//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
//
cl_context CreateContext()
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = NULL;

    // First, select an OpenCL platform to run on.  For this example, we
    // simply choose the first available platform.  Normally, you would
    // query for all available platforms and select the most appropriate one.
    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return NULL;
    }

    // Next, create an OpenCL context on the platform.  Attempt to
    // create a GPU-based context, and if that fails, try to create
    // a CPU-based context.
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)firstPlatformId,
        0
    };
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
                                      NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS)
    {
        std::cout << "Could not create GPU context, trying CPU..." << std::endl;
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                                          NULL, NULL, &errNum);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
            return NULL;
        }
    }

    return context;
}

///
//  Create a command queue on the first device available on the
//  context
//
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device)
{
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    // First get the size of the devices buffer
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
        return NULL;
    }

    if (deviceBufferSize <= 0)
    {
        std::cerr << "No devices available.";
        return NULL;
    }

    // Allocate memory for the devices buffer
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    if (errNum != CL_SUCCESS)
    {
        delete [] devices;
        std::cerr << "Failed to get device IDs";
        return NULL;
    }

    // In this example, we just choose the first available device.  In a
    // real program, you would likely use all available devices or choose
    // the highest performance device based on OpenCL device queries
    commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
    if (commandQueue == NULL)
    {
        delete [] devices;
        std::cerr << "Failed to create commandQueue for device 0";
        return NULL;
    }

    *device = devices[0];
    delete [] devices;
    return commandQueue;
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
    cl_int errNum;
    cl_program program;

    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open())
    {
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
                                        (const char**)&srcStr,
                                        NULL, NULL);
    if (program == NULL)
    {
        std::cerr << "Failed to create CL program from source." << std::endl;
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

///
//  Create memory objects used as the arguments to the kernel
//  The kernel takes four arguments: triangles_array, verticles_array,
//  triangles_size, verticles_size - input;
//  distances - output
//
bool create_mem_objects(cl_context context, cl_mem mem_objects[5],
    cl_uint3* triangles_array, cl_float3* verticles_array,
    size_t *triangles_size, size_t *verticles_size,
    cl_float3* distances)
{
    mem_objects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(cl_uint3) * (*triangles_size), triangles_array, NULL);
    mem_objects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(cl_float3) * (*verticles_size), verticles_array, NULL);
    mem_objects[2] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(size_t), triangles_size, NULL);
    mem_objects[3] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(size_t), verticles_size, NULL);
    mem_objects[4] = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(cl_float3) * (*triangles_size), distances, NULL);

    bool check = true;
    for (int i = 0; i < 5; i++)
        if (mem_objects[i] == NULL)
            check = false;
    if (!check)
    {
        std::cerr << "Error creating memory objects" << std::endl;
        return false;
    }

    return true;
}

///
//  Cleanup any created OpenCL resources
//
void Cleanup(cl_context context, cl_command_queue commandQueue,
             cl_program program, cl_kernel kernel, cl_mem *memObjects)
{
    for (int i = 0; i < 5; i++)
    {
        if (memObjects[i] != 0)
            clReleaseMemObject(memObjects[i]);
    }
    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);

    if (kernel != 0)
        clReleaseKernel(kernel);

    if (program != 0)
        clReleaseProgram(program);

    if (context != 0)
        clReleaseContext(context);

}

int main(int argc, char** argv)
{
    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernel = 0;
    cl_mem mem_objects[5] = { 0, 0, 0, 0, 0 };
    cl_int errNum;

    // Create an OpenCL context on first available platform
    context = CreateContext();
    if (context == NULL)
    {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        return 1;
    }

    // Create a command-queue on the first device available
    // on the created context
    commandQueue = CreateCommandQueue(context, &device);
    if (commandQueue == NULL)
    {
        Cleanup(context, commandQueue, program, kernel, mem_objects);
        return 1;
    }

    // Create OpenCL program from kernel.cl kernel source
    program = CreateProgram(context, device, "kernel.cl");
    if (program == NULL)
    {
        Cleanup(context, commandQueue, program, kernel, mem_objects);
        return 1;
    }

    // Create OpenCL kernel
    kernel = clCreateKernel(program, "find_distances", NULL);
    if (kernel == NULL)
    {
        std::cerr << "Failed to create kernel" << std::endl;
        Cleanup(context, commandQueue, program, kernel, mem_objects);
        return 1;
    }

    // Initialize Loader
    objl::Loader Loader;

    // Load .obj File
    bool loadout = Loader.LoadFile("box_stack.obj");
    if (!loadout)
    {
        std::cerr << "Failed to load File. May have failed to find it or it was not an .obj file." << std::endl;
        return 1;
    }

    size_t triangles_number = Loader.LoadedIndices.size() / 3;
    size_t verticles_number = Loader.LoadedVertices.size();
    cl_uint3* triangles_array = new cl_uint3[triangles_number];
    cl_int idx = 0;
    for (int i = 0; i < Loader.LoadedIndices.size(); i += 3)
    {
        idx = i / 3;
        triangles_array[idx] =
        {
            Loader.LoadedIndices[i],
            Loader.LoadedIndices[i + 1],
            Loader.LoadedIndices[i + 2]
        };
    }
    cl_float3* verticles_array = new cl_float3[verticles_number * 3];
    for (int i = 0; i < verticles_number; i++)
    {
        verticles_array[i] =
        {
            Loader.LoadedVertices[i].Position.X,
            Loader.LoadedVertices[i].Position.Y,
            Loader.LoadedVertices[i].Position.Z
        };
    }
    cl_float3* distances = new cl_float3[triangles_number];
    
    // Create memory objects that will be used as arguments to
    // kernel.  First create host memory arrays that will be
    // used to store the arguments to the kernel
    if (!create_mem_objects(context, mem_objects, triangles_array, verticles_array,
        &triangles_number, &verticles_number, distances))
    {
        Cleanup(context, commandQueue, program, kernel, mem_objects);
        return 1;
    }

    // Set the kernel arguments
    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_objects[0]);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &mem_objects[1]);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &mem_objects[2]);
    errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &mem_objects[3]);
    errNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &mem_objects[4]);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error setting kernel arguments." << std::endl;
        Cleanup(context, commandQueue, program, kernel, mem_objects);
        return 1;
    }

    size_t globalWorkSize[1] = { triangles_number };
    size_t localWorkSize[1] = { 1 };

    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
        globalWorkSize, localWorkSize,
        0, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error queuing kernel for execution." << std::endl;
        Cleanup(context, commandQueue, program, kernel, mem_objects);
        return 1;
    }

    // Read the output buffer back to the Host
    errNum = clEnqueueReadBuffer(commandQueue, mem_objects[4], CL_TRUE,
        0, triangles_number * sizeof(cl_float3), distances,
        0, NULL, NULL);

    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error reading result buffer." << std::endl;
        Cleanup(context, commandQueue, program, kernel, mem_objects);
        return 1;
    }

    // Output the result buffer
    std::cout << "Size of an edges:" << std::endl;
    for (int i = 0; i < triangles_number; i++)
        std::cout << distances[i].x << ' ' << distances[i].y << ' ' << distances[i].z << ' ' << " " << std::endl;
    std::cout << std::endl;
    std::cout << "Executed program succesfully." << std::endl;
    Cleanup(context, commandQueue, program, kernel, mem_objects);
    delete[] triangles_array;
    delete[] verticles_array;
    
    return 0;
}
