use std::sync::Arc;
use std::sync::Mutex;

extern crate ndarray;
extern crate libc;

use ndarray::Array2;

pub enum GpuAllocations {
    Dense(NpmmvDenseGpuAllocations),
    Sparse(NpmmvCsrGpuAllocations)
}

// TODO: It may be better to keep move semantics on this type and call npmmv_gpu_free() on its destruction
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NpmmvDenseGpuAllocations {
    matrix: GpuFloatArray,
    in_vector: GpuFloatArray,
    out_vector: GpuFloatArray
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NpmmvCsrGpuAllocations {
    mat_cum_row_indexes: GpuUIntArray,
    mat_column_indexes: GpuUIntArray,
    mat_values: GpuFloatArray,
    in_vector: GpuFloatArray,
    out_vector: GpuFloatArray
}

pub struct BfsCsrGpuAllocations {
    mat_cum_row_indexes: GpuUIntArray,
    mat_column_indexes: GpuUIntArray,
    mat_values: GpuFloatArray,
    in_infections: GpuUIntArray,
    out_infections: GpuUIntArray
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GpuFloatArray {
    start: *mut f32,
    end: *mut f32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GpuUIntArray {
    start: *mut usize,
    end: *mut usize
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CsrMatrixPtrs {
    pub cum_row_indexes: *const usize,
    pub column_indexes: *const usize,
    pub values: *const f32
}

#[link(name = "concurrentgraph_cuda")]
extern {
    fn negative_prob_multiply_dense_matrix_vector_cpu(iters: isize, matrix: *const f32, in_vector: *const f32, out_vector: *mut f32, outerdim: usize, innerdim: usize);
    fn negative_prob_multiply_dense_matrix_vector_gpu(iters: isize, matrix: *const f32, in_vector: *const f32, out_vector: *mut f32, outerdim: usize, innerdim: usize);

    fn npmmv_gpu_set_float_array(src: *const f32, size: usize, dst: GpuFloatArray);
    fn npmmv_gpu_get_float_array(src: GpuFloatArray, dst: *mut f32, size: usize);

    fn npmmv_dense_gpu_allocate(outerdim: usize, innerdim: usize) -> NpmmvDenseGpuAllocations;
    fn npmmv_dense_gpu_free(gpu_allocations: NpmmvDenseGpuAllocations);
    fn npmmv_gpu_set_dense_matrix(matrix_cpu: *const f32, matrix_gpu: GpuFloatArray, outerdim: usize, innerdim: usize);
    fn npmmv_dense_gpu_compute(gpu_allocations: NpmmvDenseGpuAllocations, outerdim: usize, innerdim: usize);

    fn npmmv_csr_gpu_allocate(outerdim: usize, innerdim: usize, values: usize) -> NpmmvCsrGpuAllocations;
    fn npmmv_csr_gpu_free(gpu_allocations: NpmmvCsrGpuAllocations);
    fn npmmv_gpu_set_csr_matrix(matrix_cpu: CsrMatrixPtrs, gpu_allocations: NpmmvCsrGpuAllocations, outerdim: usize, values: usize);
    fn npmmv_csr_gpu_compute(gpu_allocations: NpmmvCsrGpuAllocations, outerdim: usize, computation_restriction_factor: usize);

    fn bfs_csr_gpu_allocate(rows: usize, values: usize) -> BfsCsrGpuAllocations;
    fn bfs_csr_gpu_free(gpu_allocations: BfsCsrGpuAllocations);
    fn bfs_gpu_set_csr_matrix(matrix_cpu: CsrMatrixPtrs, gpu_allocations: BfsCsrGpuAllocations, rows: usize, values: usize);
    fn bfs_csr_gpu_compute(gpu_allocations: BfsCsrGpuAllocations, rows: usize);
}

pub fn negative_prob_multiply_dense_matrix_vector_cpu_safe<'a>(iters: isize, mat: Arc<Array2<f32>>, vector: Vec<f32>)
        -> Result<Vec<f32>, &'a str> {
    if mat.shape()[1] != vector.len() {
        return Err("Incompatible dimensions");
    } else {
        let mut result: Vec<f32> = Vec::with_capacity(mat.shape()[0]);
        result.resize(mat.shape()[0], 1.0);
        unsafe {
            //result.set_len(mat.shape()[0]);
            negative_prob_multiply_dense_matrix_vector_cpu(iters, mat.as_ptr(), vector.as_ptr(), result.as_mut_ptr(), mat.shape()[0], mat.shape()[1]);
        }
        return Ok(result);
    }
}

pub fn negative_prob_multiply_dense_matrix_vector_gpu_safe<'a>(iters: isize, mat: Arc<Array2<f32>>, vector: Vec<f32>)
        -> Result<Vec<f32>, &'a str> {
    if mat.shape()[1] != vector.len() {
        return Err("Incompatible dimensions");
    } else {
        let mut result: Vec<f32> = Vec::with_capacity(mat.shape()[0]);
        unsafe {
            result.set_len(mat.shape()[0]);
            negative_prob_multiply_dense_matrix_vector_gpu(iters, mat.as_ptr(), vector.as_ptr(), result.as_mut_ptr(), mat.shape()[0], mat.shape()[1]);
        }
        return Ok(result);
    }
}


pub fn npmmv_dense_gpu_allocate_safe(outerdim: usize, innerdim: usize) -> NpmmvDenseGpuAllocations {
    unsafe { npmmv_dense_gpu_allocate(outerdim, innerdim) }
}

pub fn npmmv_dense_gpu_free_safe(gpu_allocations: NpmmvDenseGpuAllocations) {
    unsafe { npmmv_dense_gpu_free(gpu_allocations) }
}

pub fn npmmv_gpu_set_in_vector_safe(vector: Vec<f32>, gpu_allocations: GpuAllocations) {
    let iv_ptr = match gpu_allocations { GpuAllocations::Dense(ga) => ga.in_vector, GpuAllocations::Sparse(ga) => ga.in_vector };
    unsafe { npmmv_gpu_set_float_array(vector.as_ptr(), vector.len(), iv_ptr) }
}

pub fn npmmv_gpu_set_dense_matrix_safe(mat: &Array2<f32>, gpu_allocations: NpmmvDenseGpuAllocations) {
    unsafe { npmmv_gpu_set_dense_matrix(mat.as_ptr(), gpu_allocations.matrix, mat.shape()[0], mat.shape()[1]) }
}

pub fn npmmv_gpu_get_out_vector_safe(gpu_allocations: GpuAllocations, size: usize) -> Vec<f32> {
    let ov_ptr = match gpu_allocations { GpuAllocations::Dense(ga) => ga.out_vector, GpuAllocations::Sparse(ga) => ga.out_vector };
    let mut result: Vec<f32> = Vec::with_capacity(size);
    unsafe {
        result.set_len(size);
        npmmv_gpu_get_float_array(ov_ptr, result.as_mut_ptr(), size);
    }
    result
}

pub fn npmmv_dense_gpu_compute_safe(gpu_allocations: NpmmvDenseGpuAllocations, outerdim: usize, innerdim: usize) {
    unsafe { npmmv_dense_gpu_compute(gpu_allocations, outerdim, innerdim) }
}


pub fn npmmv_csr_gpu_allocate_safe(outerdim: usize, innerdim: usize, values: usize) -> NpmmvCsrGpuAllocations {
    unsafe { npmmv_csr_gpu_allocate(outerdim, innerdim, values) }
}

pub fn npmmv_csr_gpu_free_safe(gpu_allocations: NpmmvCsrGpuAllocations) {
    unsafe { npmmv_csr_gpu_free(gpu_allocations) }
}

pub fn npmmv_gpu_set_csr_matrix_safe(mat: CsrMatrixPtrs, gpu_allocations: NpmmvCsrGpuAllocations, outerdim: usize, values: usize) {
    unsafe { npmmv_gpu_set_csr_matrix(mat, gpu_allocations, outerdim, values) }
}

pub fn npmmv_csr_gpu_compute_safe(gpu_allocations: NpmmvCsrGpuAllocations, outerdim: usize, computation_restriction_factor: usize) {
    unsafe { npmmv_csr_gpu_compute(gpu_allocations, outerdim, computation_restriction_factor) }
}

pub fn bfs_csr_gpu_allocate_safe(rows: usize, values: usize) -> BfsCsrGpuAllocations {
    unsafe { bfs_csr_gpu_allocate(rows, values) }
}

pub fn bfs_csr_gpu_free_safe(gpu_allocations: BfsCsrGpuAllocations) {
    unsafe { bfs_csr_gpu_free(gpu_allocations) }
}

pub fn bfs_gpu_set_csr_matrix_safe(matrix_cpu: CsrMatrixPtrs, gpu_allocations: BfsCsrGpuAllocations, rows: usize, values: usize) {
    unsafe { bfs_gpu_set_csr_matrix(matrix_cpu, gpu_allocations, rows, values) }
}

pub fn bfs_csr_gpu_compute_safe(gpu_allocations: BfsCsrGpuAllocations, rows: usize) {
    unsafe { bfs_csr_gpu_compute(gpu_allocations, rows) }
}