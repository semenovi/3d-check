#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

void write_model(std::string file_name, cl_int n, cl_float*adjacency_matrix)
{
    std::ofstream output_stream(file_name, std::ios::binary);

    if (!output_stream.is_open()) {
        std::cerr << "Failed to open file to write: " << file_name << std::endl;
        return;
    }

    output_stream << n << ' ';

    for (cl_int i = 0; i < n; i++)
    {
        output_stream << adjacency_matrix[i] << ' ';
    }
}

cl_float *read_model(std::string file_name)
{
    std::ifstream input_stream(file_name, std::ios::binary);

    if (!input_stream.is_open())
    {
        std::cerr << "Failed to open file for reading: " << file_name << std::endl;
        return NULL;
    }

    cl_int n = 0;
    input_stream >> n;
        
    cl_float* adjacency_matrix = new cl_float[n];
    cl_int i = 0;
    while (input_stream.good()) {
        input_stream >> adjacency_matrix[i];
        i++;
    }
    
    return adjacency_matrix;
}

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
//  The kernel takes three arguments: matrix_out (output), matrix_in (input)
//  and max_length (input)
//
bool create_mem_objects(cl_context context, cl_mem mem_objects[3],
                        cl_float* matrix_in, cl_float* max_length, cl_int n)
{
    cl_int size = n * n;
    mem_objects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(cl_float) * size, matrix_in, NULL);
    mem_objects[1] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(cl_float) * size, NULL, NULL);
    mem_objects[2] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(cl_int), max_length, NULL);

    if (mem_objects[0] == NULL || mem_objects[1] == NULL || mem_objects[2] == NULL)
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
             cl_program program, cl_kernel kernel, cl_mem memObjects[2])
{
    for (int i = 0; i < 3; i++)
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
    // uncomment to write file
    /*
    std::string input_file("3d_model.in");
    cl_int n = 8;
    cl_float* adjacency_matrix = new cl_float[n * n]
    {
            0.0, 3.0, 5.0, 4.0, 3.0, 4.0, 0.0, 0.0,
            3.0, 0.0, 4.0, 0.0, 0.0, 3.0, 4.0, 0.0,
            5.0, 4.0, 0.0, 3.0, 0.0, 0.0, 3.0, 4.0,
            4.0, 0.0, 3.0, 0.0, 4.0, 0.0, 0.0, 3.0,
            3.0, 0.0, 0.0, 4.0, 0.0, 3.0, 5.0, 4.0,
            4.0, 3.0, 0.0, 0.0, 3.0, 0.0, 4.0, 0.0,
            0.0, 4.0, 3.0, 0.0, 5.0, 4.0, 0.0, 3.0,
            0.0, 0.0, 4.0, 3.0, 4.0, 0.0, 3.0, 0.0
    };
    write_model("3d_model.in", n * n, adjacency_matrix);
    */

    // uncomment to execute
    
    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernel = 0;
    cl_mem mem_objects[3] = { 0, 0, 0 };
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
    kernel = clCreateKernel(program, "shorter_than", NULL);
    if (kernel == NULL)
    {
        std::cerr << "Failed to create kernel" << std::endl;
        Cleanup(context, commandQueue, program, kernel, mem_objects);
        return 1;
    }

    // Create memory objects that will be used as arguments to
    // kernel.  First create host memory arrays that will be
    // used to store the arguments to the kernel
    
    cl_int n = 8;
    cl_float* adjacency_matrix = new cl_float[n * n];
    adjacency_matrix = read_model("3d_model.in");
    cl_float* matrix_out = new cl_float[n * n];


    // Write input
    std::cout << "Input adjacency matrix:" << std::endl;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::cout << adjacency_matrix[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Ask the max_length
    cl_float max_length = 0.0;
    std::cout << "Enter max_length:" << std::endl << "> ";
    std::cin >> max_length;
    std::cout << std::endl;

    if (!create_mem_objects(context, mem_objects, adjacency_matrix, &max_length, n))
    {
        Cleanup(context, commandQueue, program, kernel, mem_objects);
        return 1;
    }

    // Set the kernel arguments (matrix_out, matrix_in)
    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_objects[0]);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &mem_objects[1]);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &mem_objects[2]);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error setting kernel arguments." << std::endl;
        Cleanup(context, commandQueue, program, kernel, mem_objects);
        return 1;
    }

    size_t globalWorkSize[1] = { (size_t)n * (size_t)n };
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
    errNum = clEnqueueReadBuffer(commandQueue, mem_objects[1], CL_TRUE,
                                 0, 64 * sizeof(float), matrix_out,
                                 0, NULL, NULL);
    
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error reading result buffer." << std::endl;
        Cleanup(context, commandQueue, program, kernel, mem_objects);
        return 1;
    }

    // Output the result buffer
    std::cout << "Model with removed edges longer than max_length:" << std::endl;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::cout << matrix_out[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << "Executed program succesfully." << std::endl;
    Cleanup(context, commandQueue, program, kernel, mem_objects);
    

    return 0;
}
