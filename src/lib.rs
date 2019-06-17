use std::sync::Arc;
use std::sync::Mutex;

extern crate ndarray;
extern crate libc;

use ndarray::Array2;

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
    pub cum_row_indexes: *mut usize,
    pub column_indexes: *mut usize,
    pub values: *mut f32
}

#[link(name = "concurrentgraph_cuda")]
extern {
    fn negative_prob_multiply_dense_matrix_vector_cpu(iters: isize, matrix: *const f32, in_vector: *const f32, out_vector: *mut f32, outerdim: usize, innerdim: usize);
    fn negative_prob_multiply_dense_matrix_vector_gpu(iters: isize, matrix: *const f32, in_vector: *const f32, out_vector: *mut f32, outerdim: usize, innerdim: usize);
    
    fn npmmv_gpu_set_float_array(src: *const f32, size: usize, dst: GpuFloatArray);
    fn npmmv_gpu_get_float_array(src: GpuFloatArray, dst: *mut f32, size: usize);
    fn npmmv_gpu_set_uint_array(src: *const usize, size: usize, dst: GpuUIntArray);
    fn npmmv_gpu_get_uint_array(src: GpuUIntArray, dst: *mut usize, size: usize);

    fn npmmv_dense_gpu_allocate(outerdim: usize, innerdim: usize) -> NpmmvDenseGpuAllocations;
    fn npmmv_dense_gpu_free(gpu_allocations: NpmmvDenseGpuAllocations);
    fn npmmv_gpu_set_dense_matrix(matrix_cpu: *const f32, matrix_gpu: GpuFloatArray, outerdim: usize, innerdim: usize);
    fn npmmv_dense_gpu_compute(gpu_allocations: NpmmvDenseGpuAllocations, outerdim: usize, innerdim: usize);

    fn npmmv_csr_gpu_allocate(outerdim: usize, innerdim: usize, values: usize) -> NpmmvCsrGpuAllocations;
    fn npmmv_csr_gpu_free(gpu_allocations: NpmmvCsrGpuAllocations);
    fn npmmv_gpu_set_csr_matrix(matrix_cpu: CsrMatrixPtrs, gpu_allocations: NpmmvCsrGpuAllocations, outerdim: usize, values: usize);
    fn npmmv_csr_gpu_compute(gpu_allocations: NpmmvCsrGpuAllocations, outerdim: usize);
}

pub fn negative_prob_multiply_dense_matrix_vector_cpu_safe<'a>(iters: isize, mat: Arc<Array2<f32>>, vector: Vec<f32>)
        -> Result<Vec<f32>, &'a str> {
    if mat.shape()[1] != vector.len() {
        return Err("Incompatible dimensions");
    } else {
        let mut result: Vec<f32> = Vec::with_capacity(mat.shape()[0]);
        unsafe {
            result.set_len(mat.shape()[0]);
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

pub fn npmmv_gpu_set_in_vector_safe(vector: Vec<f32>, gpu_allocations: NpmmvDenseGpuAllocations) {
    unsafe { npmmv_gpu_set_float_array(vector.as_ptr(), vector.len(), gpu_allocations.in_vector) }
}

pub fn npmmv_gpu_set_dense_matrix_safe(mat: &Array2<f32>, gpu_allocations: NpmmvDenseGpuAllocations) {
    unsafe { npmmv_gpu_set_dense_matrix(mat.as_ptr(), gpu_allocations.matrix, mat.shape()[0], mat.shape()[1]) }
}

pub fn npmmv_gpu_get_out_vector_safe(gpu_allocations: NpmmvDenseGpuAllocations, size: usize) -> Vec<f32> {
    let mut result: Vec<f32> = Vec::with_capacity(size);
    unsafe {
        result.set_len(size);
        npmmv_gpu_get_float_array(gpu_allocations.out_vector, result.as_mut_ptr(), size);
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

pub fn npmmv_csr_gpu_compute_safe(gpu_allocations: NpmmvCsrGpuAllocations, outerdim: usize) {
    unsafe { npmmv_csr_gpu_compute(gpu_allocations, outerdim) }
}