
# Create a library called "Filterbank" which includes the sourcefile "Filterbank.cpp"
add_library(Filterbank filterbank.cpp)

# Make sure top level CMakeLists file can find created libraries
target_include_directories(Filterbank PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
