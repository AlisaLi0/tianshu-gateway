//! Runtime provisioning: detect what's available so serving works without the
//! user installing an engine by hand.
//!
//! On Windows the realistic paths are Docker Desktop (`docker` on the Windows
//! PATH) or Docker inside a WSL2 distro (`wsl -- docker`). We probe both and
//! recommend a `serving::Runtime` accordingly.

use std::time::Duration;
use tokio::process::Command;

use crate::serving::Runtime;

/// How docker can be reached on this host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DockerAccess {
    /// `docker` works directly (Docker Desktop on Windows, native on Linux).
    Native,
    /// Docker is reachable via `wsl [-d <distro>] -- docker`.
    Wsl(Option<String>),
    /// No docker found by either path.
    None,
}

impl DockerAccess {
    /// The serving runtime that matches this access (Native docker access maps
    /// to `Runtime::Docker`; WSL access to `Runtime::WslDocker`).
    pub fn runtime(&self) -> Option<Runtime> {
        match self {
            DockerAccess::Native => Some(Runtime::Docker),
            DockerAccess::Wsl(_) => Some(Runtime::WslDocker),
            DockerAccess::None => None,
        }
    }

    pub fn describe(&self) -> String {
        match self {
            DockerAccess::Native => "docker (direct)".into(),
            DockerAccess::Wsl(Some(d)) => format!("docker via WSL distro '{d}'"),
            DockerAccess::Wsl(None) => "docker via WSL (default distro)".into(),
            DockerAccess::None => "not found".into(),
        }
    }
}

async fn ok(program: &str, args: &[&str]) -> bool {
    let fut = Command::new(program)
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    matches!(
        tokio::time::timeout(Duration::from_secs(12), fut).await,
        Ok(Ok(s)) if s.success()
    )
}

/// Probe docker: direct first, then through WSL.
pub async fn detect_docker() -> DockerAccess {
    if ok("docker", &["version"]).await {
        return DockerAccess::Native;
    }
    // Windows: docker may live inside the default WSL distro.
    if ok("wsl", &["--", "docker", "version"]).await {
        return DockerAccess::Wsl(None);
    }
    DockerAccess::None
}

/// Is `wsl` present at all (Windows)?
pub async fn wsl_present() -> bool {
    ok("wsl", &["--status"]).await || ok("wsl", &["-l", "-q"]).await
}

/// Pick the best runtime automatically:
/// prefer docker (so the user installs nothing), fall back to native.
pub async fn auto_runtime() -> Runtime {
    detect_docker()
        .await
        .runtime()
        .unwrap_or(Runtime::Native)
}

/// A snapshot for the `setup` command.
pub struct SetupReport {
    pub gpus: Vec<crate::gpu::GpuInfo>,
    pub docker: DockerAccess,
    pub wsl: bool,
}

pub async fn setup_report() -> SetupReport {
    let gpus = crate::gpu::detect().await;
    let docker = detect_docker().await;
    // Only bother probing wsl separately if docker isn't already native.
    let wsl = if docker == DockerAccess::Native {
        false
    } else {
        wsl_present().await
    };
    SetupReport { gpus, docker, wsl }
}
