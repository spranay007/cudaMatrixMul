# projectOne.cu

Implement a tiled dense matrix multiplication routine (C=A*B) using shared memory in 
CUDA. A and B matrix elements can be randomly generated on the host.

## Platform
Project created on Windows X64

## IDE Used
Visual Studio 2022

## Build the code 
-> Extract the zip folder on a Windows machine 
-> Build using Visual Studio 2022 [use the 2022 version]
-> Now open the terminal and traverse to the Debug directory.

## Usage
Sample command to create 
Matrix A = 320 X 320
Matrix B = 555 X 320
******************************************
```bash
cd .\x64\
cd .\Debug\

# Now in the Debug directory
./Project_one -i 320 320 555
```
******************************************
## NOTE
Please make sure that the project settings are also imported, if not then follow this
******************************************
--> Open Solution Explorer 
--> Navigate to projectOne properties page 
--> Under VC++ Directories 
--> Find External Included Directories 
--> Add the Project_one\Common folder to the list(Already present in the Zipped folder)
