use std::path::Path;
use anyhow::Result;
use tokio::io::{AsyncBufReadExt, BufReader};

pub async fn tail_file(p: &Path, max_lines: usize) -> Result<String> {
    if !p.exists() {
        return Ok(String::new());
    }
    let f = tokio::fs::File::open(p).await?;
    let mut rdr = BufReader::new(f).lines();
    let mut buf: std::collections::VecDeque<String> = std::collections::VecDeque::with_capacity(max_lines);
    while let Some(line) = rdr.next_line().await? {
        if buf.len() == max_lines {
            buf.pop_front();
        }
        buf.push_back(line);
    }
    Ok(buf.into_iter().collect::<Vec<_>>().join("\n"))
}
