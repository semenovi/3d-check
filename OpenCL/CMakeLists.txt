cmake_minimum_required(VERSION 3.8)

# Project name
project(OpenCL LANGUAGES CSharp)

# Include CMake utilities for CSharp, for WinForm and WPF application support.
include(CSharpUtilities)

set(Project_SOURCES
	App.config
	Properties/AssemblyInfo.cs
	Program.cs
)

# Define the executable, including any .cs files. 
# The .resx and other Properties files are optional here, but including them makes them visible in the VS solution for easy editing. 
add_executable(OpenCL ${Project_SOURCES})

# Define resources folder
source_group(Resources FILES Resources/Performance.cl)

# Set the .NET Framework version for the executable.
set_property(TARGET OpenCL PROPERTY VS_DOTNET_TARGET_FRAMEWORK_VERSION "v4.6.1")

set_property(TARGET OpenCL PROPERTY WIN32_EXECUTABLE TRUE)

set(CMAKE_CSharp_FLAGS "/langversion:latest")

include_directories(${CMAKE_SOURCE_DIR}/OpenCLlib)

target_link_libraries(OpenCL OpenCLlib)

set_property(TARGET OpenCL
    PROPERTY VS_PACKAGE_REFERENCES "Sourceforge.Cloo_0.9.1;ClooN_1.1.2.0"
)

# Add in the .NET reference libraries.
set_property(TARGET OpenCL PROPERTY VS_DOTNET_REFERENCES
	"Microsoft.CSharp"
	"System"
	"System.Core"
	"System.Data"
	"System.Data.DataSetExtensions"
	"System.Net.Http"
	"System.Xml"
	"System.Xml.Linq")
	
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)