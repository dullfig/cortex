//! CubeCL backend scaffold for the wgpu → CubeCL migration.
//!
//! **Status (M1):** scaffold only. Depends on `cubecl-core` (DSL macros
//! + IR, no runtime). The runtime addition + `BlockScratch` swap happens
//! in M2; M1's purpose is to prove cortex compiles alongside CubeCL and
//! to land the migration plan + feature flag plumbing.
//!
//! **Why core-only for M1:** adding cubecl's `wgpu` feature pulls
//! wgpu 29, which collides with cortex's existing wgpu 24 in
//! wgpu-hal's DX12 backend on Windows (incompatible windows-rs
//! subcrate versions). M2 will resolve this either by vendoring
//! cubecl as a separate workspace or by upgrading cortex's wgpu
//! during the same milestone.
//!
//! **M2 next:** wrap `BlockScratch::allocate` to allocate via CubeCL's
//! `MemoryManagement` instead of raw `wgpu::Buffer`. That single change
//! kills the ~17s `vkFreeMemory` cliff cortex hits on chat_completions
//! request 2+ (validated empirically in `pinky/cubecl-poolbench/`).
//!
//! See `pinky/cubecl-migration-plan-2026-05-18.md` for the full plan.

#![allow(dead_code)] // M1 scaffold; populated in M2+.

/// Sentinel: re-export to prove `cubecl-core` is a real dependency
/// resolved by Cargo. Replaced in M2 with actual `MemoryManagement`
/// wiring and a `CubeClBackend<R: Runtime>` struct.
pub use cubecl_core as core;
