// Corvus Chess Engine
// Copyright (C) 2026 Avi
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

//! Compiles the vendored Fathom Syzygy tablebase prober.
//!
//! Fathom is C, so this crate has a build script after all.  The compile is
//! deliberately *non-fatal*: tablebases are an optional endgame aid, and a
//! machine without a C compiler should still be able to build Corvus.  If the
//! translation unit cannot be compiled we simply do not define `corvus_fathom`
//! and `src/syzygy.rs` falls back to its no-op implementation.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(corvus_fathom)");
    println!("cargo:rerun-if-changed=vendor/fathom/tbprobe.c");
    println!("cargo:rerun-if-changed=vendor/fathom/tbchess.c");
    println!("cargo:rerun-if-changed=vendor/fathom/tbprobe.h");
    println!("cargo:rerun-if-changed=vendor/fathom/tbconfig.h");
    println!("cargo:rerun-if-changed=vendor/fathom/stdendian.h");

    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let include = PathBuf::from("vendor/fathom");

    let cc = env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let mut cmd = std::process::Command::new(&cc);
    cmd.arg("-O2")
        .arg("-fPIC")
        .arg("-std=gnu11")
        .arg("-c")
        .arg("vendor/fathom/tbprobe.c")
        .arg("-o")
        .arg(out.join("tbprobe.o"))
        .arg("-I")
        .arg(&include);

    match cmd.status() {
        Ok(s) if s.success() => {}
        other => {
            // No C compiler (or the compile failed): build without tablebases.
            println!(
                "cargo:warning=corvus: Fathom (Syzygy) not compiled ({other:?}); \
                 building without tablebase support"
            );
            return;
        }
    }

    let ar = env::var("AR").unwrap_or_else(|_| "ar".to_string());
    let lib = out.join("libfathom.a");
    let ok = std::process::Command::new(&ar)
        .arg("crus")
        .arg(&lib)
        .arg(out.join("tbprobe.o"))
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    if !ok {
        println!("cargo:warning=corvus: could not archive Fathom; building without tablebase support");
        return;
    }

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=fathom");
    println!("cargo:rustc-cfg=corvus_fathom");
}
