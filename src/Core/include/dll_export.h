#pragma once
#ifndef DLLEXPORT_H
#define DLLEXPORT_H

#if defined(_WIN32) || defined(__WIN32__)
#if defined(dll_EXPORTS)  // add by CMake 
#define  DLL_EXPORT __declspec(dllexport)
#define EXPIMP_TEMPLATE
#else
#define  DLL_EXPORT __declspec(dllimport)
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
#if defined(DLL_EXPORT)
using int64_t = __int64;
#endif