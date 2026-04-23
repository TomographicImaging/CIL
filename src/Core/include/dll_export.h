//  Copyright 2019 United Kingdom Research and Innovation
//  Copyright 2019 The University of Manchester
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
// Authors:
// CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

#pragma once
#ifndef DLLEXPORT_H
#define DLLEXPORT_H

#if defined(_WIN32) || defined(__WIN32__)
#if defined(dll_EXPORTS)  // add by CMake
#define  DLL_EXPORT __declspec(dllexport)
#define EXPIMP_TEMPLATE
#else
#define  DLL_EXPORT __declspec(dllexport)
#define EXPIMP_TEMPLATE extern
#endif
#elif defined(linux) || defined(__linux) || defined(__APPLE__)
#define DLL_EXPORT
#ifndef __cdecl
#define __cdecl
#endif
#endif

#endif
//define int64_t
#if defined(dll_EXPORTS)
using int64_t = __int64;
#endif

#ifdef _MSC_VER
//typedef __int64 int64; // 64-bit unsigned integer
  using int64 = __int64;
#else
//typedef long long int64; //64-bit unsigned integer
  using int64 = long long;
#endif
