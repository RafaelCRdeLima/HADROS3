cmake_minimum_required(VERSION 3.16)
project(HADROS3Geant4Transport VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Geant4 11.4.2 REQUIRED)
find_package(HepMC3 3.3 REQUIRED)

add_executable(hadros3_geant4_transport ../apps/hadros3_geant4_transport.cpp)
target_link_libraries(hadros3_geant4_transport PRIVATE ${Geant4_LIBRARIES} HepMC3::HepMC3)
target_include_directories(hadros3_geant4_transport PRIVATE ${Geant4_INCLUDE_DIRS})
target_compile_options(hadros3_geant4_transport PRIVATE -Wall -Wextra -Wpedantic)

install(TARGETS hadros3_geant4_transport RUNTIME DESTINATION bin)
