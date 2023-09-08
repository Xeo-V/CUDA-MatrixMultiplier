# CUDA-MatrixMultiplier
Optimized CUDA-based matrix multiplication leveraging tiling and shared memory techniques for enhanced performance. Ideal for both academic and high-performance computing applications.

# CUDA Matrix Multiplication 🚀

## Table of Contents
1. [Introduction](#Introduction)
2. [Tech Stack](#Tech-Stack)
3. [How to Setup](#How-To-Setup)
4. [How to Use](#How-To-Use)
5. [Special Note for Visual Studio Users](#Special-Note-For-Visual-Studio-Users)
6. [For Students](#For-Students)
7. [License](#License)

## Introduction 📚
This repository contains an optimized CUDA-based Matrix Multiplication code written in C++. The code leverages the power of GPU parallel computing to speed up matrix multiplication tasks. With detailed comments and explanations, this code is ideal for anyone looking to dive into the realm of CUDA programming.

## Tech Stack 💻
- CUDA
- C++
- Visual Studio (Optional)

## How to Setup 🛠️
1. Make sure you have a CUDA-compatible GPU installed in your system.
2. Install the CUDA toolkit and SDK from NVIDIA's official site.
3. Clone this repository.
4. Open the `.cu` file in an editor that supports CUDA, such as Visual Studio or any text editor of your choice.
5. Compile and run the program.

## How to Use 🖱️
Run the compiled executable. The program will prompt you to enter the size of the matrix and whether you'd like to populate the matrix with integers or floating-point numbers. The program will then generate two random matrices, perform the matrix multiplication using the GPU, and display the result.

## Special Note for Visual Studio Users ⚠️
If you are using Visual Studio, you may encounter some Intellisense errors where certain CUDA functions are marked as undefined. Don't worry! This is a known Intellisense issue, and the code should compile and run correctly.

## For Students 🎓
If you are a student and you intend to present this project as your own, please make sure to read the `Explanation.md` file included in this repository. It contains a detailed breakdown of each function, its parameters, how the code interacts with the operating system, and much more. Diagrams are also included for better understanding. Your understanding of the code will be enhanced, and you'll be better prepared to present it.

## License 📝
This project is licensed under the MIT License. Feel free to clone, modify, and distribute as per the terms of the license.

