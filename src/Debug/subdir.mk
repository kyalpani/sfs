################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../height.cpp \
../lfs.cpp \
../lfs1d.cpp \
../point14d.cpp \
../point20d.cpp \
../point6d.cpp \
../sfs.cpp \
../sfs_expand.cpp \
../sfsimg.cpp \
../sfslib.cpp \
../solution.cpp 

OBJS += \
./height.o \
./lfs.o \
./lfs1d.o \
./point14d.o \
./point20d.o \
./point6d.o \
./sfs.o \
./sfs_expand.o \
./sfsimg.o \
./sfslib.o \
./solution.o 

CPP_DEPS += \
./height.d \
./lfs.d \
./lfs1d.d \
./point14d.d \
./point20d.d \
./point6d.d \
./sfs.d \
./sfs_expand.d \
./sfsimg.d \
./sfslib.d \
./solution.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/usr/include/pcl-1.10 -I/usr/local/include/opencv4 -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


