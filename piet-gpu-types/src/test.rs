use piet_gpu_derive::piet_gpu;

piet_gpu! {
    #[rust_encode]
    #[gpu_write]
    mod test {
        struct StructA {
            a: f16,
            b: f16,
        }

        struct StructB {
            a: f16,
            b: u16,
            c: f16,
        }

        struct StructC {
            a: f16,
            b: u16,
            c: u16,
            d: f16,
        }

        struct StructD {
            a: [f16; 2],
        }

        struct StructE {
            a: [f16; 3],
        }
    }
}
