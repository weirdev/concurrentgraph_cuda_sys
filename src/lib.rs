use std::sync::Arc;

extern crate ndarray;
extern crate libc;

use ndarray::Array2;

pub enum NpmmvAllocations {
    Dense(NpmmvDenseGpuAllocations),
    Sparse(NpmmvCsrGpuAllocations)
}

pub enum BfsAllocations {
    Dense,
    Sparse(BfsCsrGpuAllocations)
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

#[repr(C)]
#[derive(Clone, Copy)]
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
pub struct CsrFloatMatrixPtrs {
    pub cum_row_indexes: *const usize,
    pub column_indexes: *const usize,
    pub values: *const f32
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CsrIntMatrixPtrs {
    pub cum_row_indexes: *const usize,
    pub column_indexes: *const usize,
    pub values: *const isize
}

#[link(name = "concurrentgraph_cuda")]
extern {
    fn negative_prob_multiply_dense_matrix_vector_cpu(iters: isize, matrix: *const f32, in_vector: *const f32, out_vector: *mut f32, outerdim: usize, innerdim: usize);
    fn negative_prob_multiply_dense_matrix_vector_gpu(iters: isize, matrix: *const f32, in_vector: *const f32, out_vector: *mut f32, outerdim: usize, innerdim: usize);

    fn set_gpu_float_array(src: *const f32, size: usize, dst: GpuFloatArray);
    fn get_gpu_float_array(src: GpuFloatArray, dst: *mut f32, size: usize);
    fn set_gpu_uint_array(src: *const usize, size: usize, dst: GpuUIntArray);
    fn get_gpu_uint_array(src: GpuUIntArray, dst: *mut usize, size: usize);

    fn npmmv_dense_gpu_allocate(outerdim: usize, innerdim: usize) -> NpmmvDenseGpuAllocations;
    fn npmmv_dense_gpu_free(gpu_allocations: NpmmvDenseGpuAllocations);
    fn npmmv_gpu_set_dense_matrix(matrix_cpu: *const f32, matrix_gpu: GpuFloatArray, outerdim: usize, innerdim: usize);
    fn npmmv_dense_gpu_compute(gpu_allocations: NpmmvDenseGpuAllocations, outerdim: usize, innerdim: usize);

    fn npmmv_csr_gpu_allocate(outerdim: usize, innerdim: usize, values: usize) -> NpmmvCsrGpuAllocations;
    fn npmmv_csr_gpu_free(gpu_allocations: NpmmvCsrGpuAllocations);
    fn npmmv_gpu_set_csr_matrix(matrix_cpu: CsrFloatMatrixPtrs, gpu_allocations: NpmmvCsrGpuAllocations, outerdim: usize, values: usize);
    fn npmmv_csr_gpu_compute(gpu_allocations: NpmmvCsrGpuAllocations, outerdim: usize, computation_restriction_factor: usize);

    fn bfs_csr_gpu_allocate(rows: usize, values: usize) -> BfsCsrGpuAllocations;
    fn bfs_csr_gpu_free(gpu_allocations: BfsCsrGpuAllocations);
    fn bfs_gpu_set_csr_matrix(matrix_cpu: CsrIntMatrixPtrs, gpu_allocations: BfsCsrGpuAllocations, rows: usize, values: usize);
    fn bfs_csr_gpu_compute(gpu_allocations: BfsCsrGpuAllocations, rows: usize);

    fn graph_deterministic_weights(matrix_cpu: CsrFloatMatrixPtrs, rows: usize, values: usize, immunities: *const f32, shedding_curve: *const f32, infection_length: usize, transmission_rate: f32) -> *mut isize;

    // From sssp.cpp
    fn sssp(cum_col_indexes: *const isize, row_indexes: *const isize, values: *const f32, nodes_i: usize, edges_i: usize, output: *mut f32);
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

pub fn npmmv_gpu_set_in_vector_safe(vector: Vec<f32>, gpu_allocations: NpmmvAllocations) {
    let iv_ptr = match gpu_allocations { NpmmvAllocations::Dense(ga) => ga.in_vector, NpmmvAllocations::Sparse(ga) => ga.in_vector };
    unsafe { set_gpu_float_array(vector.as_ptr(), vector.len(), iv_ptr) }
}

pub fn npmmv_gpu_set_dense_matrix_safe(mat: &Array2<f32>, gpu_allocations: NpmmvDenseGpuAllocations) {
    unsafe { npmmv_gpu_set_dense_matrix(mat.as_ptr(), gpu_allocations.matrix, mat.shape()[0], mat.shape()[1]) }
}

pub fn npmmv_gpu_get_out_vector_safe(gpu_allocations: NpmmvAllocations, size: usize) -> Vec<f32> {
    let ov_ptr = match gpu_allocations { NpmmvAllocations::Dense(ga) => ga.out_vector, NpmmvAllocations::Sparse(ga) => ga.out_vector };
    let mut result: Vec<f32> = Vec::with_capacity(size);
    unsafe {
        result.set_len(size);
        get_gpu_float_array(ov_ptr, result.as_mut_ptr(), size);
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

pub fn npmmv_gpu_set_csr_matrix_safe(mat: CsrFloatMatrixPtrs, gpu_allocations: NpmmvCsrGpuAllocations, outerdim: usize, values: usize) {
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

pub fn bfs_gpu_set_csr_matrix_safe(matrix_cpu: CsrIntMatrixPtrs, gpu_allocations: BfsCsrGpuAllocations, rows: usize, values: usize) {
    unsafe { bfs_gpu_set_csr_matrix(matrix_cpu, gpu_allocations, rows, values) }
}

pub fn bfs_csr_gpu_compute_safe(gpu_allocations: BfsCsrGpuAllocations, rows: usize) {
    unsafe { bfs_csr_gpu_compute(gpu_allocations, rows) }
}

pub fn bfs_gpu_set_in_vector_safe(vector: Vec<usize>, gpu_allocations: BfsAllocations) {
    let iv_ptr = match gpu_allocations { BfsAllocations::Dense => panic!("dense not implemented"), BfsAllocations::Sparse(ga) => ga.in_infections };
    unsafe { set_gpu_uint_array(vector.as_ptr(), vector.len(), iv_ptr) }
}

pub fn bfs_gpu_swap_in_out_vector_refs(gpu_allocations: BfsAllocations) -> BfsAllocations {
    match gpu_allocations {
        BfsAllocations::Dense => panic!("dense not implemented"), 
        BfsAllocations::Sparse(mut ga) => {
            let iv_ptr = ga.in_infections;
            ga.in_infections = ga.out_infections;
            ga.out_infections = iv_ptr;
            BfsAllocations::Sparse(ga)
        }
    }
}

pub fn bfs_gpu_get_out_vector_safe(gpu_allocations: BfsAllocations, size: usize) -> Vec<usize> {
    let ov_ptr = match gpu_allocations { BfsAllocations::Dense => panic!("Dense not implemented yet"), BfsAllocations::Sparse(ga) => ga.out_infections };
    let mut result: Vec<usize> = Vec::with_capacity(size);
    unsafe {
        result.set_len(size);
        get_gpu_uint_array(ov_ptr, result.as_mut_ptr(), size);
    }
    result
}

pub fn graph_deterministic_weights_gpu_safe(matrix_cpu: CsrFloatMatrixPtrs, rows: usize, values: usize,
        immunities: Vec<f32>, shedding_curve: Vec<f32>, infection_length: usize, transmission_rate: f32) 
        -> Vec<isize> {

    unsafe {
        let imm = immunities.as_ptr();
        let shedd = shedding_curve.as_ptr();
        let ar_ptr: *mut isize = graph_deterministic_weights(matrix_cpu, rows, values, imm, shedd, infection_length, transmission_rate);
        Vec::from_raw_parts(ar_ptr,
                values, values)
    }
}

pub fn sssp_safe(matrix_cpu: CsrFloatMatrixPtrs, nodes: usize, edges: usize) -> Vec<f32> {
    let mut output: Vec<f32> = Vec::with_capacity(nodes);
    unsafe {
        output.set_len(nodes);
        sssp(matrix_cpu.cum_row_indexes as *const isize, matrix_cpu.column_indexes as *const isize, matrix_cpu.values, nodes, edges, output.as_mut_ptr());
    }
    output
}