use cudarc::driver::CudaContext;

fn test() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let mut a = stream.alloc_zeros::<f32>(10).unwrap();
    
    unsafe {
        ctx.memset_d8(&mut a, 0).unwrap();
    }
}
