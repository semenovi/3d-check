cmake_minimum_required(VERSION 3.8)

# Project name
project(OpenCLlib LANGUAGES CSharp)

# Include CMake utilities for CSharp, for WinForm and WPF application support.
include(CSharpUtilities)

set(Project_SOURCES
	Properties/AssemblyInfo.cs
	AcceleratorDevice.cs
	Awaitable.cs
	ComputeMethod.cs
	EasyCL.cs
	Kernels.Designer.cs
	MultiCL.cs
	OpenCL.cs
	Kernels.resx
)

# Define the library, including any .cs files. 
# The .resx and other Properties files are optional here, but including them makes them visible in the VS solution for easy editing. 
add_library(OpenCLlib SHARED ${Project_SOURCES})

set_property(SOURCE App.xaml PROPERTY VS_XAML_TYPE "ApplicationDefinition")

set_property(TARGET OpenCLlib PROPERTY VS_DOTNET_TARGET_FRAMEWORK_VERSION "v4.6.1")

set_property(TARGET OpenCLlib PROPERTY WIN32_EXECUTABLE TRUE)

set(CMAKE_CSharp_FLAGS "/langversion:latest")

set_property(TARGET OpenCLlib
    PROPERTY VS_PACKAGE_REFERENCES "Sourceforge.Cloo_0.9.1;ClooN_1.1.2.0"
)

# Add in the .NET reference libraries.
set_property(TARGET OpenCLlib PROPERTY VS_DOTNET_REFERENCES
	"Microsoft.CSharp"
	"System"
	"System.Core"
	"System.Data"
	"System.Data.DataSetExtensions"
	"System.Net.Http"
	"System.Xml"
	"System.Xml.Linq")
	
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)