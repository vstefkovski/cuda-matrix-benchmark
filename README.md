# CUDA Matrix Multiplication Benchmark

A CUDA C/C++ implementation comparing different loop unrolling strategies for matrix multiplication on GPU. This project demonstrates the performance impact of various loop unrolling factors (2, 4, 8, and 16) compared to basic implementation.

## Overview

This benchmark suite implements and compares five different versions of matrix multiplication:
- Basic implementation (no unrolling)
- Loop unrolling with factor 2
- Loop unrolling with factor 4
- Loop unrolling with factor 8
- Loop unrolling with factor 16

The program tests these implementations across different matrix sizes (128x128 to 4096x4096) and block sizes (8x8, 16x16, 32x32) to provide comprehensive performance metrics.

## Features

- Multiple matrix multiplication implementations with different optimization levels
- Automatic benchmarking across various matrix sizes
- Support for different CUDA block sizes
- Performance measurement using CUDA events
- Error checking and handling for CUDA operations
- Automated memory management for both host and device

## Requirements

- Windows 10/11
- Visual Studio 2022 with CUDA development workload installed
- NVIDIA GPU with CUDA support
- CUDA Toolkit (tested with CUDA 11.0+)

## Project Structure

- `kernel.cu` - Main CUDA source file containing kernel implementations and benchmark code
- `LoopUnrolling.sln` - Visual Studio solution file
- `LoopUnrolling.vcxproj` - Visual Studio project file

## Building and Running

1. Open `LoopUnrolling.sln` in Visual Studio 2022
2. Select your preferred build configuration (Release recommended for benchmarking)
3. Build the solution (F7 or Build > Build Solution)
4. Run the program (F5 or Debug > Start Debugging)

## Output Format

The program outputs results in the following format:
```
Method: [Implementation], Matrix Size: [Size], Block Size: [Block], Time: [Time] ms
```

## Performance Notes

- Performance varies significantly based on GPU architecture
- Larger matrices generally benefit more from loop unrolling
- Optimal block size depends on your specific GPU model
- Memory transfer times are not included in the kernel execution measurements

## Development Notes

- The project is configured for CUDA development in Visual Studio 2022
- Debug configuration includes additional error checking
- Release configuration is optimized for performance benchmarking
- The solution includes proper CUDA toolkit integration settings

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
