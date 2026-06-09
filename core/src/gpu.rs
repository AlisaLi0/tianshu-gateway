//! Best-effort local GPU detection for one-click serving.
//!
//! Vendor CLIs are the most accurate source, so we try them first:
//!   * NVIDIA → `nvidia-smi --query-gpu=index,name,memory.total,memory.free ...`
//!   * AMD    → `rocm-smi --showproductname --showmeminfo vram --csv`
//!
//! Returns an empty vec on a CPU-only host (no vendor tooling found). This is
//! advisory only — the serving orchestration logs it and warns when no GPU is
//! present, but never blocks; engines may still run on CPU.

use serde::{Deserialize, Serialize};
use tokio::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuVendor {
    #[serde(rename = "nvidia")]
    Nvidia,
    #[serde(rename = "amd")]
    Amd,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub index: u32,
    pub vendor: GpuVendor,
    pub name: String,
    pub mem_total_mib: Option<u64>,
    pub mem_free_mib: Option<u64>,
}

impl std::fmt::Display for GpuInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let vendor = match self.vendor {
            GpuVendor::Nvidia => "NVIDIA",
            GpuVendor::Amd => "AMD",
        };
        write!(f, "[{}] {} {}", self.index, vendor, self.name)?;
        if let Some(total) = self.mem_total_mib {
            match self.mem_free_mib {
                Some(free) => write!(f, " ({free}/{total} MiB free)")?,
                None => write!(f, " ({total} MiB)")?,
            }
        }
        Ok(())
    }
}

/// Detect GPUs, NVIDIA first then AMD. Empty vec = none found (CPU-only).
pub async fn detect() -> Vec<GpuInfo> {
    if let Some(g) = nvidia().await {
        if !g.is_empty() {
            return g;
        }
    }
    if let Some(g) = amd().await {
        if !g.is_empty() {
            return g;
        }
    }
    Vec::new()
}

async fn nvidia() -> Option<Vec<GpuInfo>> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .await
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut v = Vec::new();
    for line in text.lines() {
        let cols: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if cols.len() < 4 {
            continue;
        }
        v.push(GpuInfo {
            index: cols[0].parse().unwrap_or(0),
            vendor: GpuVendor::Nvidia,
            name: cols[1].to_string(),
            mem_total_mib: cols[2].parse().ok(),
            mem_free_mib: cols[3].parse().ok(),
        });
    }
    Some(v)
}

/// AMD via `rocm-smi --csv`. Column layout varies across ROCm versions, so we
/// parse by header name rather than fixed position, and degrade gracefully:
/// a card with no parseable memory still yields a `GpuInfo` (mem = `None`).
async fn amd() -> Option<Vec<GpuInfo>> {
    let out = Command::new("rocm-smi")
        .args(["--showproductname", "--showmeminfo", "vram", "--csv"])
        .output()
        .await
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header = lines.next()?;
    let cols: Vec<String> = header.split(',').map(|s| s.trim().to_lowercase()).collect();

    let find = |needle: &str| cols.iter().position(|c| c.contains(needle));
    let name_idx = find("card series")
        .or_else(|| find("product name"))
        .or_else(|| find("name"));
    let total_idx = find("vram total memory").or_else(|| find("total memory"));
    let used_idx = find("vram total used memory").or_else(|| find("used memory"));

    let mut v = Vec::new();
    for (i, line) in lines.enumerate() {
        let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        // First column is the card id like "card0".
        let id_raw = f.first().copied().unwrap_or("");
        if !id_raw.to_lowercase().starts_with("card") {
            continue;
        }
        let index = id_raw
            .trim_start_matches(|c: char| !c.is_ascii_digit())
            .parse()
            .unwrap_or(i as u32);
        let name = name_idx
            .and_then(|n| f.get(n))
            .map(|s| s.to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "AMD GPU".to_string());
        let total_bytes = total_idx.and_then(|n| f.get(n)).and_then(|s| s.parse::<u64>().ok());
        let used_bytes = used_idx.and_then(|n| f.get(n)).and_then(|s| s.parse::<u64>().ok());
        let mem_total_mib = total_bytes.map(|b| b / (1024 * 1024));
        let mem_free_mib = match (total_bytes, used_bytes) {
            (Some(t), Some(u)) => Some(t.saturating_sub(u) / (1024 * 1024)),
            _ => None,
        };
        v.push(GpuInfo {
            index,
            vendor: GpuVendor::Amd,
            name,
            mem_total_mib,
            mem_free_mib,
        });
    }
    Some(v)
}
