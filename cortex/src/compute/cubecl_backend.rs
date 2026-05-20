//! CubeCL backend integration for the wgpu → CubeCL migration.
//!
//! **Status (M2 partial):** wgpu upgraded 24→29 (M2 step 1 done).
//! Runtime addition blocked on cubecl-wgpu's dx12 transitive feature
//! that won't disable from cortex's side (Cargo feature union); needs
//! either a cubecl fork with backend selectors or CUDA backend pivot.
//!
//! See `pinky/cubecl-migration-plan-2026-05-18.md` for the full plan.

#![allow(dead_code)] // populated incrementally through M2-M7.

pub use cubecl_core as core;
