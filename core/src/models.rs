//! Local model repository.
//!
//! Tracks files under a user-chosen models directory. Provides:
//!   * `list_local`     — walk the dir, group by repo (`org/repo`).
//!   * `disk_usage`     — `du -sb` equivalent.
//!   * `download_hf`    — git-free HF download via `https://huggingface.co/{repo}/resolve/{rev}/{file}`.
//!   * `download_ms`    — ModelScope analog.
//!   * `delete_local`   — remove a model directory.
//!
//! Downloads stream to a temp file, then atomic rename. Supports HF token via
//! `Authorization: Bearer`.

use anyhow::{anyhow, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalModel {
    pub repo: String,        // "Qwen/Qwen3.6-35B-A3B" (relative path)
    pub abs_path: PathBuf,
    pub size_bytes: u64,
    pub file_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadRequest {
    pub repo_id: String,         // "Qwen/Qwen3-8B"
    pub revision: Option<String>, // default "main"
    pub files: Vec<String>,      // explicit files; if empty download common set
    pub dest_root: PathBuf,
    pub source: DownloadSource,
    pub token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DownloadSource {
    HuggingFace,
    ModelScope,
}

pub fn list_local(root: &Path) -> Result<Vec<LocalModel>> {
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    // 2-level scan: <root>/<org>/<repo>/...
    for org in std::fs::read_dir(root)? {
        let org = org?;
        if !org.file_type()?.is_dir() {
            continue;
        }
        let org_name = org.file_name().to_string_lossy().to_string();
        let p1 = org.path();
        for repo in std::fs::read_dir(&p1)? {
            let repo = repo?;
            if !repo.file_type()?.is_dir() {
                continue;
            }
            let repo_name = repo.file_name().to_string_lossy().to_string();
            let abs = repo.path();
            let (size, count) = walk_size(&abs)?;
            out.push(LocalModel {
                repo: format!("{org_name}/{repo_name}"),
                abs_path: abs,
                size_bytes: size,
                file_count: count,
            });
        }
    }
    Ok(out)
}

pub fn delete_local(p: &Path) -> Result<()> {
    if !p.exists() {
        return Ok(());
    }
    if !p.is_dir() {
        return Err(anyhow!("not a directory: {}", p.display()));
    }
    std::fs::remove_dir_all(p)?;
    Ok(())
}

pub fn disk_usage(p: &Path) -> Result<u64> {
    walk_size(p).map(|(s, _)| s)
}

fn walk_size(p: &Path) -> Result<(u64, u64)> {
    let mut total = 0u64;
    let mut count = 0u64;
    let mut stack = vec![p.to_path_buf()];
    while let Some(d) = stack.pop() {
        let rd = match std::fs::read_dir(&d) {
            Ok(rd) => rd,
            Err(_) => continue,
        };
        for entry in rd.flatten() {
            let ft = match entry.file_type() {
                Ok(ft) => ft,
                Err(_) => continue,
            };
            if ft.is_dir() {
                stack.push(entry.path());
            } else if ft.is_file() {
                if let Ok(meta) = entry.metadata() {
                    total += meta.len();
                    count += 1;
                }
            }
        }
    }
    Ok((total, count))
}

// ─── download ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    pub repo_id: String,
    pub file: String,
    pub downloaded: u64,
    pub total: Option<u64>,
    pub done: bool,
    pub error: Option<String>,
}

pub async fn download(req: DownloadRequest, mut on_progress: impl FnMut(DownloadProgress)) -> Result<()> {
    let revision = req.revision.unwrap_or_else(|| "main".into());
    let dest = req.dest_root.join(&req.repo_id);
    tokio::fs::create_dir_all(&dest).await?;

    if req.files.is_empty() {
        return Err(anyhow!("`files` list is empty; pre-resolve via gateway/HF API"));
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(0))
        .build()?;

    for file in &req.files {
        let url = match req.source {
            DownloadSource::HuggingFace => format!(
                "https://huggingface.co/{}/resolve/{}/{}",
                req.repo_id, revision, file
            ),
            DownloadSource::ModelScope => format!(
                "https://www.modelscope.cn/api/v1/models/{}/repo?Revision={}&FilePath={}",
                req.repo_id, revision, file
            ),
        };

        let mut rb = client.get(&url);
        if let Some(t) = req.token.as_ref() {
            rb = rb.bearer_auth(t);
        }
        let resp = rb.send().await?;
        if !resp.status().is_success() {
            let code = resp.status();
            let body = resp.text().await.unwrap_or_default();
            on_progress(DownloadProgress {
                repo_id: req.repo_id.clone(),
                file: file.clone(),
                downloaded: 0,
                total: None,
                done: false,
                error: Some(format!("HTTP {code}: {body}")),
            });
            return Err(anyhow!("download {} failed: HTTP {code}", file));
        }
        let total = resp.content_length();

        let target = dest.join(file);
        if let Some(parent) = target.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let tmp = target.with_extension("part");
        let mut f = tokio::fs::File::create(&tmp).await?;
        let mut downloaded = 0u64;
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            f.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;
            on_progress(DownloadProgress {
                repo_id: req.repo_id.clone(),
                file: file.clone(),
                downloaded,
                total,
                done: false,
                error: None,
            });
        }
        f.flush().await?;
        drop(f);
        tokio::fs::rename(&tmp, &target).await?;
        on_progress(DownloadProgress {
            repo_id: req.repo_id.clone(),
            file: file.clone(),
            downloaded,
            total,
            done: true,
            error: None,
        });
    }
    Ok(())
}
