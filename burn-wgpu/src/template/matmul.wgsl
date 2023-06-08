@group(0)
@binding(0)
var<storage, read> lhs: array<elem>;

@group(0)
@binding(1)
var<storage, read> rhs: array<elem>;

@group(0)
@binding(2)
var<storage, read_write> output: array<elem>;

@group(0)
@binding(3)
var<storage, read> info: array<u32>;

@compute
@workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Indexes
    let batch = global_id.x;
    let row = global_id.y;
    let col = global_id.z;

    // Basic information
    let dim = info[0];
    let n_rows = info[3u * dim - 1u];
    let n_cols = info[4u * dim];
    let K = info[3u * dim];

    // Returns if outside the output dimension
    if row >= n_rows || col >= n_cols {
        return;
    }

    // Calculate the corresponding offsets with support for broadcasting.
    let offset_output = batch * n_rows * n_cols;
    var offset_lhs: u32 = 0u;
    var offset_rhs: u32 = 0u;

    let batch_dims = dim - 2u;
    for (var b: u32 = 0u; b < batch_dims; b++) {
        let stride_lhs = info[b + 1u];
        let stride_rhs = info[b + 1u * dim + 1u];
        let shape_lhs = info[b + 2u * dim + 1u];
        let shape_rhs = info[b + 3u * dim + 1u];

        offset_lhs += offset_output / stride_lhs % shape_lhs * stride_lhs;
        offset_rhs += offset_output / stride_rhs % shape_rhs * stride_rhs;
    }

    // Basic matmul implementation
    var sum = 0.0;
    for (var k: u32 = 0u; k < K; k++) {
        let lhs_index = row * K + k;
        let rhs_index = k * n_cols + col;

        sum += lhs[offset_lhs + lhs_index] * rhs[offset_rhs + rhs_index];
    }

    let output_index = row * n_rows + col;
    output[offset_output + output_index] = sum;
}