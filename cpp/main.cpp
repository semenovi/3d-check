#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "OBJ_Loader.h"

#include <CL/cl.h>
#include <D:/Program Files (x86)/Common Files/MSVC/freeglut/include/GL/glut.h>

size_t triangles_number = 0, verticles_number = 0;
cl_uint4* triangles_array = new cl_uint4[1];
cl_float3* verticles_array = new cl_float3[1];

const struct OBJ_COLOR {
    GLfloat red, green, blue;
    OBJ_COLOR() : red(1.0), green(1.0), blue(1.0) {}
} OBJ_COLOR;

bool render_mode; // true = solid body, false = wireframe

const float ZOOM_SPEED = 0.1f;
const float ROTATE_SPEED = 0.1f;
float       DISTANCE = 4.0f;

struct camera
{
    GLfloat x, y, z, phi, theta;
    camera() : x(-4.0f), y(2.0f), z(0.0f), phi(0), theta(0) {}
} camera;

void init()
{
    glShadeModel(GL_SMOOTH);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClearDepth(1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_COLOR);
    glEnable(GL_COLOR_MATERIAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

    glEnable(GL_LIGHTING);
    glEnable(GL_NORMALIZE);
    glEnable(GL_LIGHT1);
    GLfloat lightAmbient1[4] = { 0.2, 0.2, 0.2, 1.0 };
    GLfloat lightPos1[4] = { 0.5, 0.5, 0.5, 1.0 };
    GLfloat lightDiffuse1[4] = { 0.8, 0.8, 0.8, 1.0 };
    GLfloat lightSpec1[4] = { 1.0, 1.0, 1.0, 1.0 };
    GLfloat lightLinAtten = 0.0f;
    GLfloat lightQuadAtten = 1.0f;
    glLightfv(GL_LIGHT1, GL_POSITION, (GLfloat*)&lightPos1);
    glLightfv(GL_LIGHT1, GL_AMBIENT, (GLfloat*)&lightAmbient1);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, (GLfloat*)&lightDiffuse1);
    glLightfv(GL_LIGHT1, GL_SPECULAR, (GLfloat*)&lightSpec1);
    glLightfv(GL_LIGHT1, GL_LINEAR_ATTENUATION, &lightLinAtten);
    glLightfv(GL_LIGHT1, GL_QUADRATIC_ATTENUATION, &lightQuadAtten);
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);

}

void calculate_normal(cl_uint3 f, GLdouble* normal, cl_float3* vertices)
{
    // x
    normal[0] = (vertices[f.y].y - vertices[f.x].y) * (vertices[f.z].z - vertices[f.x].z)
        - (vertices[f.z].y - vertices[f.x].y) * (vertices[f.y].z - vertices[f.x].z);
    // y
    normal[1] = (vertices[f.y].z - vertices[f.x].z) * (vertices[f.z].x - vertices[f.x].x)
        - (vertices[f.z].x - vertices[f.x].x) * (vertices[f.z].z - vertices[f.x].z);
    // z
    normal[2] = (vertices[f.y].x - vertices[f.x].x) * (vertices[f.z].y - vertices[f.x].y)
        - (vertices[f.z].x - vertices[f.x].x) * (vertices[f.y].y - vertices[f.x].y);
}

void draw_obj(cl_uint4* triangles, cl_float3* vertices,
    size_t triangles_size, size_t verticles_size)
{
    if (render_mode)
    {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    else
    {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    }
    for (int i = 0; i < triangles_size; i++)
    {
        GLdouble normal[3];
        calculate_normal(triangles[i], normal, vertices);
        glBegin(GL_TRIANGLES);
        if (triangles[i].w == 1)
        {
            glColor3f(OBJ_COLOR.red, 0.0, 0.0);
        }
        else
        {
            glColor3f(OBJ_COLOR.red, OBJ_COLOR.green, OBJ_COLOR.blue);
        }
        glNormal3dv(normal);
        glVertex3d(vertices[triangles[i].x].x, vertices[triangles[i].x].y, vertices[triangles[i].x].z);
        glVertex3d(vertices[triangles[i].y].x, vertices[triangles[i].y].y, vertices[triangles[i].y].z);
        glVertex3d(vertices[triangles[i].z].x, vertices[triangles[i].z].y, vertices[triangles[i].z].z);
        glEnd();
    }
    glFlush();
}

void reshape(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    if (h == 0) {
        gluPerspective(80, (float)w, 1.0, 5000.0);
    }
    else {
        gluPerspective(80, (float)w / (float)h, 1.0, 5000.0);
    }
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void arrow_keys(int key, int x, int y) {
    switch (key) {
    case GLUT_KEY_UP: {
        DISTANCE -= ZOOM_SPEED;
        break;
    }
    case GLUT_KEY_DOWN: {
        DISTANCE += ZOOM_SPEED;
        break;
    }
    case GLUT_KEY_LEFT: {
        camera.theta -= ROTATE_SPEED;
        break;
    }
    case GLUT_KEY_RIGHT:
        camera.theta += ROTATE_SPEED;
        break;
    default:
        break;
    }
}

void keyboard(unsigned char key, int x, int y) {
    switch (key) {
    case 27:
        exit(0);
    case 's':
        render_mode = true;
        break;
    case 'w':
        render_mode = false;
        break;
    default:
        break;
    }
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    camera.x = DISTANCE * cos(camera.phi) * sin(camera.theta);
    camera.y = 2.0f + DISTANCE * sin(camera.phi) * sin(camera.theta);
    camera.z = DISTANCE * cos(camera.theta);

    gluLookAt(camera.x, camera.y, camera.z, 0, 2.0f, 0, 0.0f, 1.0f, 0.0f);
    draw_obj(triangles_array, verticles_array, triangles_number, verticles_number);
    glutSwapBuffers();
    glutPostRedisplay();
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
//  The kernel takes four arguments: triangles_array - input and output;
//  verticles_array, triangles_size, verticles_size, min - input;
//
bool create_mem_objects(cl_context context, cl_mem mem_objects[3],
    cl_uint4* triangles_array, cl_float3* verticles_array,
    size_t *triangles_size, size_t *verticles_size, cl_float *min)
{
    mem_objects[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(cl_uint4) * (*triangles_size), triangles_array, NULL);
    mem_objects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(cl_float3) * (*verticles_size), verticles_array, NULL);
    mem_objects[2] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(cl_float), min, NULL);

    bool check = true;
    for (int i = 0; i < 3; i++)
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

int main(int argc, char* argv[])
{
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
    kernel = clCreateKernel(program, "set_is_small", NULL);
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

    triangles_number = Loader.LoadedIndices.size() / 3;
    verticles_number = Loader.LoadedVertices.size();
    cl_uint4* _triangles_array = new cl_uint4[triangles_number];
    triangles_array = _triangles_array;
    cl_int idx = 0;
    for (int i = 0; i < Loader.LoadedIndices.size(); i += 3)
    {
        idx = i / 3;
        triangles_array[idx] =
        {
            Loader.LoadedIndices[i],
            Loader.LoadedIndices[i + 1],
            Loader.LoadedIndices[i + 2],
            0
        };
    }
    cl_float3* _verticles_array = new cl_float3[verticles_number * 3];
    verticles_array = _verticles_array;
    for (int i = 0; i < verticles_number; i++)
    {
        verticles_array[i] =
        {
            Loader.LoadedVertices[i].Position.X,
            Loader.LoadedVertices[i].Position.Y,
            Loader.LoadedVertices[i].Position.Z
        };
    }

    cl_float min = 0.05;
    
    // Create memory objects that will be used as arguments to
    // kernel.  First create host memory arrays that will be
    // used to store the arguments to the kernel
    if (!create_mem_objects(context, mem_objects, triangles_array, verticles_array,
        &triangles_number, &verticles_number, &min))
    {
        Cleanup(context, commandQueue, program, kernel, mem_objects);
        return 1;
    }

    // Set the kernel arguments
    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_objects[0]);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &mem_objects[1]);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &mem_objects[2]);
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
    errNum = clEnqueueReadBuffer(commandQueue, mem_objects[0], CL_TRUE,
        0, triangles_number * sizeof(cl_float4), triangles_array,
        0, NULL, NULL);

    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error reading result buffer." << std::endl;
        Cleanup(context, commandQueue, program, kernel, mem_objects);
        return 1;
    }

    std::cout << "Executed program succesfully." << std::endl;

    // initialize rendering with solid body
    render_mode = true;

    int window;
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH);
    glutInitWindowSize(960, 720);
    glutInitWindowPosition(0, 0);
    window = glutCreateWindow("3d_check");
    init();
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutSpecialFunc(arrow_keys);
    glutKeyboardFunc(keyboard);
    glutMainLoop();

    Cleanup(context, commandQueue, program, kernel, mem_objects);
    delete[] triangles_array;
    delete[] verticles_array;
    
    return 0;
}
