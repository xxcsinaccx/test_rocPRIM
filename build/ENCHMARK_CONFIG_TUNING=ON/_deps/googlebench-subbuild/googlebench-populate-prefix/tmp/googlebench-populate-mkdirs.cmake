# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/workspace/rocPRIM/build/ENCHMARK_CONFIG_TUNING=ON/_deps/googlebench-src")
  file(MAKE_DIRECTORY "/workspace/rocPRIM/build/ENCHMARK_CONFIG_TUNING=ON/_deps/googlebench-src")
endif()
file(MAKE_DIRECTORY
  "/workspace/rocPRIM/build/ENCHMARK_CONFIG_TUNING=ON/_deps/googlebench-build"
  "/workspace/rocPRIM/build/ENCHMARK_CONFIG_TUNING=ON/_deps/googlebench-subbuild/googlebench-populate-prefix"
  "/workspace/rocPRIM/build/ENCHMARK_CONFIG_TUNING=ON/_deps/googlebench-subbuild/googlebench-populate-prefix/tmp"
  "/workspace/rocPRIM/build/ENCHMARK_CONFIG_TUNING=ON/_deps/googlebench-subbuild/googlebench-populate-prefix/src/googlebench-populate-stamp"
  "/workspace/rocPRIM/build/ENCHMARK_CONFIG_TUNING=ON/_deps/googlebench-subbuild/googlebench-populate-prefix/src"
  "/workspace/rocPRIM/build/ENCHMARK_CONFIG_TUNING=ON/_deps/googlebench-subbuild/googlebench-populate-prefix/src/googlebench-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/workspace/rocPRIM/build/ENCHMARK_CONFIG_TUNING=ON/_deps/googlebench-subbuild/googlebench-populate-prefix/src/googlebench-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/workspace/rocPRIM/build/ENCHMARK_CONFIG_TUNING=ON/_deps/googlebench-subbuild/googlebench-populate-prefix/src/googlebench-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
