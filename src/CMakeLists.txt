# This is the root ITK CMakeLists file.
cmake_minimum_required(VERSION 2.8.12)
if(COMMAND CMAKE_POLICY)
  cmake_policy(SET CMP0003 NEW)
endif()


# This project is designed to be built outside the Insight source tree.
project(HelloWorld)

# Find ITK.
find_package(ITK)
include(${ITK_USE_FILE})

add_executable(registration registration.cxx )

target_link_libraries(registration ${ITK_LIBRARIES})
