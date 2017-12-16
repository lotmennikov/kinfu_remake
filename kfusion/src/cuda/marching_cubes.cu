/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2011, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "device.hpp"
#include "../precomp.hpp"
#include "../safe_call.hpp"
#include "kfusion/marching_cubes.h"

#include "thrust/device_ptr.h"
#include "thrust/scan.h"

namespace kfusion
{
  namespace device
  {
    //texture<int, 1, cudaReadModeElementType> edgeTex;
    texture<int, 1, cudaReadModeElementType> triTex;
    texture<int, 1, cudaReadModeElementType> numVertsTex;
  }
}

void
kfusion::device::bindTextures (const int */*edgeBuf*/, const int *triBuf, const int *numVertsBuf)
{
  cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
  //cudaSafeCall(cudaBindTexture(0, edgeTex, edgeBuf, desc) );
  cudaSafeCall (cudaBindTexture (0, triTex, triBuf, desc) );
  cudaSafeCall (cudaBindTexture (0, numVertsTex, numVertsBuf, desc) );
}


void
kfusion::device::unbindTextures ()
{
  //cudaSafeCall( cudaUnbindTexture(edgeTex) );
  cudaSafeCall ( cudaUnbindTexture (numVertsTex) );
  cudaSafeCall ( cudaUnbindTexture (triTex) );
}

namespace kfusion
{
  namespace device
  {
    __device__ int global_count = 0;
    __device__ int output_count;
    __device__ unsigned int blocks_done = 0;

    struct CubeIndexEstimator
    {
      TsdfVolume volume;

	  CubeIndexEstimator(const TsdfVolume& v) : volume(v) {};

	  static __device__ __forceinline__ float isoValue() { return 0.f; }

      __device__ __forceinline__ void
      readTsdf (int x, int y, int z, float& tsdf, int& weight) const
      {
        unpack_tsdf (*volume(x,y,z), tsdf, weight);
      }

	  __device__ __forceinline__ int
		  computeCubeIndex(int x, int y, int z, float f[8]) const
	  {
		  int weight;
		  readTsdf(x	, y, z,			f[0], weight); if (weight == 0) return 0;
		  readTsdf(x + 1, y, z,			f[1], weight); if (weight == 0) return 0;
		  readTsdf(x + 1, y + 1, z,		f[2], weight); if (weight == 0) return 0;
		  readTsdf(x	, y + 1, z,		f[3], weight); if (weight == 0) return 0;
		  readTsdf(x	, y, z + 1,		f[4], weight); if (weight == 0) return 0;
		  readTsdf(x + 1, y, z + 1,		f[5], weight); if (weight == 0) return 0;
		  readTsdf(x + 1, y + 1, z + 1, f[6], weight); if (weight == 0) return 0;
		  readTsdf(x	, y + 1, z + 1,	f[7], weight); if (weight == 0) return 0;

		  // calculate flag indicating if each vertex is inside or outside isosurface
		  int cubeindex;
		  cubeindex = int(f[0] < isoValue());
		  cubeindex += int(f[1] < isoValue()) * 2;
		  cubeindex += int(f[2] < isoValue()) * 4;
		  cubeindex += int(f[3] < isoValue()) * 8;
		  cubeindex += int(f[4] < isoValue()) * 16;
		  cubeindex += int(f[5] < isoValue()) * 32;
		  cubeindex += int(f[6] < isoValue()) * 64;
		  cubeindex += int(f[7] < isoValue()) * 128;

		  return cubeindex;
	  }

	  __device__ __forceinline__ int
		  computeCubeIndex(int x, int y, int z) const
	  {
		  float f[8];
		  int weight;
		  readTsdf(x, y, z, f[0], weight); if (weight == 0) return 0;
		  readTsdf(x + 1, y, z, f[1], weight); if (weight == 0) return 0;
		  readTsdf(x + 1, y + 1, z, f[2], weight); if (weight == 0) return 0;
		  readTsdf(x, y + 1, z, f[3], weight); if (weight == 0) return 0;
		  readTsdf(x, y, z + 1, f[4], weight); if (weight == 0) return 0;
		  readTsdf(x + 1, y, z + 1, f[5], weight); if (weight == 0) return 0;
		  readTsdf(x + 1, y + 1, z + 1, f[6], weight); if (weight == 0) return 0;
		  readTsdf(x, y + 1, z + 1, f[7], weight); if (weight == 0) return 0;

		  // calculate flag indicating if each vertex is inside or outside isosurface
		  int cubeindex;
		  cubeindex = int(f[0] < isoValue());
		  cubeindex += int(f[1] < isoValue()) * 2;
		  cubeindex += int(f[2] < isoValue()) * 4;
		  cubeindex += int(f[3] < isoValue()) * 8;
		  cubeindex += int(f[4] < isoValue()) * 16;
		  cubeindex += int(f[5] < isoValue()) * 32;
		  cubeindex += int(f[6] < isoValue()) * 64;
		  cubeindex += int(f[7] < isoValue()) * 128;

		  return cubeindex;
	  }
    };

    struct OccupiedVoxels : public CubeIndexEstimator
    {
		OccupiedVoxels(const TsdfVolume& v) : CubeIndexEstimator(v) {};

      enum
      {        
        CTA_SIZE_X = 32,
        CTA_SIZE_Y = 8,
        CTA_SIZE = CTA_SIZE_X * CTA_SIZE_Y,

        WARPS_COUNT = CTA_SIZE / Warp::WARP_SIZE
      };

      mutable int* voxels_indeces;
      mutable int* vetexes_number;
      int max_size;

      __device__ __forceinline__ void
      operator () () const
      {
        int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
        int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

#if __CUDA_ARCH__ < 200
        __shared__ int cta_buffer[CTA_SIZE];
#endif


#if __CUDA_ARCH__ >= 120
        if (__all (x >= volume.dims.x) || __all (y >= volume.dims.y))
          return;
#else        
        if (Emulation::All(x >= volume.dims.x, cta_buffer) || 
            Emulation::All(y >= volume.dims.y, cta_buffer))
            return;
#endif

        int ftid = Block::flattenedThreadId ();
		int warp_id = Warp::id();
		int lane_id = Warp::laneId();

        volatile __shared__ int warps_buffer[WARPS_COUNT];

        for (int z = 0; z < volume.dims.z - 1; z++)
        {
          int numVerts = 0;;
          if (x + 1 < volume.dims.x && y + 1 < volume.dims.y)
          {
            float field[8];
            int cubeindex = computeCubeIndex (x, y, z, field);

            // read number of vertices from texture
            numVerts = (cubeindex == 0 || cubeindex == 255) ? 0 : tex1Dfetch (numVertsTex, cubeindex);
          }
#if __CUDA_ARCH__ >= 200
          int total = __popc (__ballot (numVerts > 0));
#else
          int total = __popc (Emulation::Ballot(numVerts > 0, cta_buffer));
#endif
		  if (total == 0)
			continue;

          if (lane_id == 0)
          {
            int old = atomicAdd (&global_count, total);
            warps_buffer[warp_id] = old;
          }
          int old_global_voxels_count = warps_buffer[warp_id];

#if __CUDA_ARCH__ >= 200
          int offs = Warp::binaryExclScan (__ballot (numVerts > 0));
#else          
          int offs = Warp::binaryExclScan(Emulation::Ballot(numVerts > 0, cta_buffer));
#endif

          if (old_global_voxels_count + offs < max_size && numVerts > 0)
          {
            voxels_indeces[old_global_voxels_count + offs] = volume.dims.y * volume.dims.x * z + volume.dims.x * y + x;
            vetexes_number[old_global_voxels_count + offs] = numVerts;
          }

          bool full = old_global_voxels_count + total >= max_size;

          if (full)
            break;

        } /* for(int z = 0; z < VOLUME_Z - 1; z++) */


        /////////////////////////
        // prepare for future scans
        if (ftid == 0)
        {
          unsigned int total_blocks = gridDim.x * gridDim.y * gridDim.z;
          unsigned int value = atomicInc (&blocks_done, total_blocks);

          //last block
          if (value == total_blocks - 1)
          {
            output_count = min (max_size, global_count);
            blocks_done = 0;
            global_count = 0;
          }
        } 
      } /* operator () */
    };

    __global__ void getOccupiedVoxelsKernel (const OccupiedVoxels ov) { ov (); }
  }
}

int
kfusion::device::getOccupiedVoxels (const TsdfVolume& volume, DeviceArray2D<int>& occupied_voxels)
{
  OccupiedVoxels ov(volume);

  ov.voxels_indeces = occupied_voxels.ptr (0);
  ov.vetexes_number = occupied_voxels.ptr (1);
  ov.max_size = occupied_voxels.cols ();

  dim3 block (OccupiedVoxels::CTA_SIZE_X, OccupiedVoxels::CTA_SIZE_Y);
  dim3 grid (divUp (volume.dims.x, block.x), divUp (volume.dims.y, block.y));

  //cudaFuncSetCacheConfig(getOccupiedVoxelsKernel, cudaFuncCachePreferL1);
  //printFuncAttrib(getOccupiedVoxelsKernel);

  getOccupiedVoxelsKernel<<<grid, block>>>(ov);
  cudaSafeCall ( cudaGetLastError () );
  cudaSafeCall (cudaDeviceSynchronize ());

  int size;
  cudaSafeCall ( cudaMemcpyFromSymbol (&size, output_count, sizeof(size)) );
  return size;
}

int
kfusion::device::computeOffsetsAndTotalVertexes (DeviceArray2D<int>& occupied_voxels)
{
  thrust::device_ptr<int> beg (occupied_voxels.ptr (1));
  thrust::device_ptr<int> end = beg + occupied_voxels.cols ();

  thrust::device_ptr<int> out (occupied_voxels.ptr (2));
  thrust::exclusive_scan (beg, end, out);

  int lastElement, lastScanElement;

  DeviceArray<int> last_elem (occupied_voxels.ptr(1) + occupied_voxels.cols () - 1, 1);
  DeviceArray<int> last_scan (occupied_voxels.ptr(2) + occupied_voxels.cols () - 1, 1);

  last_elem.download (&lastElement);
  last_scan.download (&lastScanElement);

  return lastElement + lastScanElement;
}


namespace kfusion
{
	namespace device
	{
		struct TrianglesGenerator : public CubeIndexEstimator
		{
			TrianglesGenerator(const TsdfVolume& v) : CubeIndexEstimator(v) {};

#if __CUDA_ARCH__ >= 200
			enum { CTA_SIZE = 256, MAX_GRID_SIZE_X = 65536 };
#else
			enum { CTA_SIZE = 96, MAX_GRID_SIZE_X = 65536 };
#endif

			const int* occupied_voxels;
			const int* vertex_ofssets;
			int voxels_count;
			float3 cell_size;

			mutable Point *output;

			__device__ __forceinline__ float3
				getNodeCoo(int x, int y, int z) const
			{
				float3 coo = make_float3(x, y, z);
				coo += 0.5f;                 //shift to volume cell center;

				coo.x *= cell_size.x;
				coo.y *= cell_size.y;
				coo.z *= cell_size.z;

				return coo;
			}

			__device__ __forceinline__ float3
				vertex_interp(float3 p0, float3 p1, float f0, float f1) const
			{
				float t = (isoValue() - f0) / (f1 - f0 + 1e-15f);
				float x = p0.x + t * (p1.x - p0.x);
				float y = p0.y + t * (p1.y - p0.y);
				float z = p0.z + t * (p1.z - p0.z);
				return make_float3(x, y, z);
			}

			__device__ __forceinline__ void
				operator () () const
			{
				int tid = threadIdx.x;
				int idx = (blockIdx.y * MAX_GRID_SIZE_X + blockIdx.x) * CTA_SIZE + tid;


				if (idx >= voxels_count)
					return;

				int voxel = occupied_voxels[idx];

				int z = voxel / (volume.dims.x * volume.dims.y);
				int y = (voxel - z * volume.dims.x * volume.dims.y) / volume.dims.x;
				int x = (voxel - z * volume.dims.x * volume.dims.y) - y * volume.dims.x;

				float f[8];
				int cubeindex = computeCubeIndex(x, y, z, f);

				// calculate cell vertex positions
				float3 v[8];
				v[0] = getNodeCoo(x, y, z);
				v[1] = getNodeCoo(x + 1, y, z);
				v[2] = getNodeCoo(x + 1, y + 1, z);
				v[3] = getNodeCoo(x, y + 1, z);
				v[4] = getNodeCoo(x, y, z + 1);
				v[5] = getNodeCoo(x + 1, y, z + 1);
				v[6] = getNodeCoo(x + 1, y + 1, z + 1);
				v[7] = getNodeCoo(x, y + 1, z + 1);

				// find the vertices where the surface intersects the cube
				// use shared memory to avoid using local
				__shared__ float3 vertlist[12][CTA_SIZE];

				vertlist[0][tid] = vertex_interp(v[0], v[1], f[0], f[1]);
				vertlist[1][tid] = vertex_interp(v[1], v[2], f[1], f[2]);
				vertlist[2][tid] = vertex_interp(v[2], v[3], f[2], f[3]);
				vertlist[3][tid] = vertex_interp(v[3], v[0], f[3], f[0]);
				vertlist[4][tid] = vertex_interp(v[4], v[5], f[4], f[5]);
				vertlist[5][tid] = vertex_interp(v[5], v[6], f[5], f[6]);
				vertlist[6][tid] = vertex_interp(v[6], v[7], f[6], f[7]);
				vertlist[7][tid] = vertex_interp(v[7], v[4], f[7], f[4]);
				vertlist[8][tid] = vertex_interp(v[0], v[4], f[0], f[4]);
				vertlist[9][tid] = vertex_interp(v[1], v[5], f[1], f[5]);
				vertlist[10][tid] = vertex_interp(v[2], v[6], f[2], f[6]);
				vertlist[11][tid] = vertex_interp(v[3], v[7], f[3], f[7]);
				__syncthreads();

				// output triangle vertices
				int numVerts = tex1Dfetch(numVertsTex, cubeindex);

				for (int i = 0; i < numVerts; i += 3)
				{
					int index = vertex_ofssets[idx] + i;

					int v1 = tex1Dfetch(triTex, (cubeindex * 16) + i + 0);
					int v2 = tex1Dfetch(triTex, (cubeindex * 16) + i + 1);
					int v3 = tex1Dfetch(triTex, (cubeindex * 16) + i + 2);

					store_point(output, index + 2, vertlist[v1][tid]);
					store_point(output, index + 1, vertlist[v2][tid]);
					store_point(output, index + 0, vertlist[v3][tid]);
				}
			}

			__device__ __forceinline__ void
				store_point(float4 *ptr, int index, const float3& point) const {
				ptr[index] = make_float4(point.x, point.y, point.z, 1.0f);
			}
		};
		__global__ void
			trianglesGeneratorKernel(const TrianglesGenerator tg) { tg(); }

		struct IndicesGenerator : public CubeIndexEstimator
		{
			enum { CTA_SIZE = 256, MAX_GRID_SIZE_X = 65536 };

			const int* occupied_voxels;
			const int* vertex_ofssets;
			int voxels_count;
			float3 cell_size;

			mutable int2 *double_inds;

			IndicesGenerator(const TsdfVolume& v) : CubeIndexEstimator(v) {};

			__device__ __forceinline__ float3
				getNodeCoo(int x, int y, int z) const
			{
				float3 coo = make_float3(x, y, z);
				coo += 0.5f;                 //shift to volume cell center;

				coo.x *= cell_size.x;
				coo.y *= cell_size.y;
				coo.z *= cell_size.z;

				return coo;
			}

			__device__ __forceinline__ int
				getNodeCo(int x, int y, int z) const
			{
				return volume.dims.y * volume.dims.x * z + volume.dims.x * y + x;
			}

			__device__ __forceinline__ float3
				vertex_interp(float3 p0, float3 p1, float f0, float f1) const
			{
				float t = (isoValue() - f0) / (f1 - f0 + 1e-15f);
				float x = p0.x + t * (p1.x - p0.x);
				float y = p0.y + t * (p1.y - p0.y);
				float z = p0.z + t * (p1.z - p0.z);
				return make_float3(x, y, z);
			}

			__device__ __forceinline__ int2  vertex_int2(int v1, int v2) const {
				if (v1 < v2) return make_int2(v1, v2);
				else return make_int2(v2, v1);
			}

			__device__ __forceinline__ void
				operator () () const
			{
				int tid = threadIdx.x;
				int idx = (blockIdx.y * MAX_GRID_SIZE_X + blockIdx.x) * CTA_SIZE + tid;


				if (idx >= voxels_count)
					return;

				int voxel = occupied_voxels[idx];

				int z = voxel / (volume.dims.x * volume.dims.y);
				int y = (voxel - z * volume.dims.x * volume.dims.y) / volume.dims.x;
				int x = (voxel - z * volume.dims.x * volume.dims.y) - y * volume.dims.x;

				//float f[8];
				int cubeindex = computeCubeIndex(x, y, z);

				// calculate cell vertex positions
				int v[8];
				v[0] = getNodeCo(x, y, z);
				v[1] = getNodeCo(x + 1, y, z);
				v[2] = getNodeCo(x + 1, y + 1, z);
				v[3] = getNodeCo(x, y + 1, z);
				v[4] = getNodeCo(x, y, z + 1);
				v[5] = getNodeCo(x + 1, y, z + 1);
				v[6] = getNodeCo(x + 1, y + 1, z + 1);
				v[7] = getNodeCo(x, y + 1, z + 1);

				// find the vertices where the surface intersects the cube
				// use shared memory to avoid using local
				__shared__ int2 vertlist[12][CTA_SIZE];
				//__shared__ int vertlist_usage[12][CTA_SIZE];

				//vertlist_usage[0][tid] = vertlist_usage[1][tid] =vertlist_usage[2][tid] = vertlist_usage[3][tid] =vertlist_usage[4][tid] = vertlist_usage[5][tid] =
				//vertlist_usage[6][tid] = vertlist_usage[7][tid] =vertlist_usage[8][tid] = vertlist_usage[9][tid] = vertlist_usage[10][tid] = vertlist_usage[11][tid] = -1;

				vertlist[0][tid] = vertex_int2(v[0], v[1]);// vertex_interp (v[0], v[1], f[0], f[1]);
				vertlist[1][tid] = vertex_int2(v[1], v[2]);//vertex_interp (v[1], v[2], f[1], f[2]);
				vertlist[2][tid] = vertex_int2(v[2], v[3]);//vertex_interp (v[2], v[3], f[2], f[3]);
				vertlist[3][tid] = vertex_int2(v[3], v[0]);//vertex_interp (v[3], v[0], f[3], f[0]);
				vertlist[4][tid] = vertex_int2(v[4], v[5]);//vertex_interp (v[4], v[5], f[4], f[5]);
				vertlist[5][tid] = vertex_int2(v[5], v[6]);//vertex_interp (v[5], v[6], f[5], f[6]);
				vertlist[6][tid] = vertex_int2(v[6], v[7]);//vertex_interp (v[6], v[7], f[6], f[7]);
				vertlist[7][tid] = vertex_int2(v[7], v[4]);//vertex_interp (v[7], v[4], f[7], f[4]);
				vertlist[8][tid] = vertex_int2(v[0], v[4]);//vertex_interp (v[0], v[4], f[0], f[4]);
				vertlist[9][tid] = vertex_int2(v[1], v[5]);//vertex_interp (v[1], v[5], f[1], f[5]);
				vertlist[10][tid] = vertex_int2(v[2], v[6]);//vertex_interp (v[2], v[6], f[2], f[6]);
				vertlist[11][tid] = vertex_int2(v[3], v[7]);//vertex_interp (v[3], v[7], f[3], f[7]);
				__syncthreads();

				// output triangle vertices
				int numVerts = tex1Dfetch(numVertsTex, cubeindex);

				for (int i = 0; i < numVerts; i += 3)
				{
					int index = vertex_ofssets[idx] + i;

					int v1 = tex1Dfetch(triTex, (cubeindex * 16) + i + 0);
					int v2 = tex1Dfetch(triTex, (cubeindex * 16) + i + 1);
					int v3 = tex1Dfetch(triTex, (cubeindex * 16) + i + 2);

					store_point(double_inds, index + 2, vertlist[v1][tid]);
					store_point(double_inds, index + 1, vertlist[v2][tid]);
					store_point(double_inds, index + 0, vertlist[v3][tid]);
				}
			}

			__device__ __forceinline__ void
				store_point(int2 *ptr, int index, const int2& double_ind) const {
				ptr[index] = double_ind;//make_float4 (point.x, point.y, point.z, 1.0f);
			}
		};

		__global__ void
			indicesGeneratorKernel(const IndicesGenerator ig) { ig(); }

		struct VerticesGenerator : public CubeIndexEstimator
		{
			enum { CTA_SIZE = 256, MAX_GRID_SIZE_X = 65536 };

			int vertices_count;
			float3 cell_size;

			int2 *double_inds;
			mutable Point* vertices;

			VerticesGenerator(const TsdfVolume& v) : CubeIndexEstimator(v) {};

			__device__ __forceinline__ float3
				getNodeCoo(int x, int y, int z) const
			{
				float3 coo = make_float3(x, y, z);
				coo += 0.5f;                 //shift to volume cell center;

				coo.x *= cell_size.x;
				coo.y *= cell_size.y;
				coo.z *= cell_size.z;

				return coo;
			}

			__device__ __forceinline__ float4
				vertex_interp(float3 p0, float3 p1, float f0, float f1) const
			{
				float t = (isoValue() - f0) / (f1 - f0 + 1e-15f);
				float x = p0.x + t * (p1.x - p0.x);
				float y = p0.y + t * (p1.y - p0.y);
				float z = p0.z + t * (p1.z - p0.z);
				return make_float4(x, y, z, 1.0f);
			}

			__device__ __forceinline__ void
				operator () () const
			{
				int tid = threadIdx.x;
				int idx = (blockIdx.y * MAX_GRID_SIZE_X + blockIdx.x) * CTA_SIZE + tid;

				if (idx >= vertices_count)
					return;

				int2 di = double_inds[idx];

				int z1 = di.x / (volume.dims.x * volume.dims.y);
				int y1 = (di.x - z1 * volume.dims.x * volume.dims.y) / volume.dims.x;
				int x1 = (di.x - z1 * volume.dims.x * volume.dims.y) - y1 * volume.dims.x;

				int z2 = di.y / (volume.dims.x * volume.dims.y);
				int y2 = (di.y - z2 * volume.dims.x * volume.dims.y) / volume.dims.x;
				int x2 = (di.y - z2 * volume.dims.x * volume.dims.y) - y2 * volume.dims.x;

				//float f[8];
				float f1, f2; int w;
				readTsdf(x1, y1, z1, f1, w);
				readTsdf(x2, y2, z2, f2, w);

				float3 v1, v2;
				v1 = getNodeCoo(x1, y1, z1);
				v2 = getNodeCoo(x2, y2, z2);

				vertices[idx] = vertex_interp(v1, v2, f1, f2);
			}
		};

		__global__ void
			verticesGeneratorKernel(const VerticesGenerator vg) { vg(); }
	}
}


void
kfusion::device::generateTriangles (const TsdfVolume& volume, const DeviceArray2D<int>& occupied_voxels, DeviceArray<Point>& output)
{   
  int device;
  cudaSafeCall( cudaGetDevice(&device) );

  cudaDeviceProp prop;
  cudaSafeCall( cudaGetDeviceProperties(&prop, device) );
  
  int block_size = prop.major < 2 ? 96 : 256; // please see TrianglesGenerator::CTA_SIZE

  typedef TrianglesGenerator Tg;
  Tg tg(volume);

  tg.occupied_voxels = occupied_voxels.ptr (0);
  tg.vertex_ofssets = occupied_voxels.ptr (2);
  tg.voxels_count = occupied_voxels.cols ();

  tg.cell_size = volume.voxel_size;
  tg.output = output;

  int blocks_num = divUp (tg.voxels_count, block_size);

  dim3 block (block_size);
  dim3 grid(min(blocks_num, Tg::MAX_GRID_SIZE_X), divUp(blocks_num, Tg::MAX_GRID_SIZE_X));

  trianglesGeneratorKernel<<<grid, block>>>(tg);
  
  cudaSafeCall ( cudaGetLastError () );
  cudaSafeCall (cudaDeviceSynchronize ());
}

void
kfusion::device::generateDoubleInds(const TsdfVolume& volume, const DeviceArray2D<int>& occupied_voxels, int2* double_inds)
{
	int device;
	cudaGetDevice(&device);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);

	int block_size = prop.major < 2 ? 96 : 256; // please see TrianglesGenerator::CTA_SIZE

	typedef IndicesGenerator Ig;
	Ig ig(volume);

	ig.occupied_voxels = occupied_voxels.ptr(0);
	ig.vertex_ofssets = occupied_voxels.ptr(2);
	ig.voxels_count = occupied_voxels.cols();
	ig.cell_size = volume.voxel_size;
	ig.double_inds = double_inds;

	int blocks_num = divUp(ig.voxels_count, block_size);

	dim3 block(block_size);
	dim3 grid(min(blocks_num, Ig::MAX_GRID_SIZE_X), divUp(blocks_num, Ig::MAX_GRID_SIZE_X));

	indicesGeneratorKernel << <grid, block >> >(ig);

	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}

void
kfusion::device::computePoints(const TsdfVolume& volume, int2* double_inds, Point* points, int vertices_size) {

	int block_size = 256;

	typedef VerticesGenerator Vg;
	Vg vg(volume);

	vg.vertices_count = vertices_size;
	vg.cell_size = volume.voxel_size;
	vg.double_inds = double_inds;
	vg.vertices = points;

	int blocks_num = divUp(vg.vertices_count, block_size);

	dim3 block(block_size);
	dim3 grid(min(blocks_num, Vg::MAX_GRID_SIZE_X), divUp(blocks_num, Vg::MAX_GRID_SIZE_X));

	verticesGeneratorKernel << <grid, block >> >(vg);

	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}
