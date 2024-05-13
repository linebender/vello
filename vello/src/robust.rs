//! A discussion of Vello's robust dynamic memory support
//!
//! When running the Vello pipeline, there are several buffers which:
//! 1) Need to be large enough to store
//! 2) Have a size which is non-trivial to calculate before running the pipeline
//!
//! When using wgpu (and most GPU apis), it is not possible for the GPU to synchronously
//! request a larger buffer, so we have to provide a best-effort buffer for this purpose.
//!
//! ## Handling failures
//!
//! If the buffer which was provided was too small, we have an issue.
