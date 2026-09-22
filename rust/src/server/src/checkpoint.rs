// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::future::Future;
use std::os::fd::FromRawFd;

use anyhow::{Context as _, Result, bail};
use serde::Deserialize;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use vllm_engine_core_client::EngineCoreClient;

#[derive(Deserialize)]
struct Command {
    method: String,
    args: Vec<Value>,
}

/// Join the Python launcher's barrier after engine connection, before serving.
pub(crate) async fn join(fd: i32, client: &EngineCoreClient) -> Result<()> {
    // The launcher passes exclusive ownership of this descriptor to the frontend.
    let stream = unsafe { std::os::unix::net::UnixStream::from_raw_fd(fd) };
    stream.set_nonblocking(true)?;
    participate(UnixStream::from_std(stream)?, |command| async move {
        client.call_utility::<Value, _>(&command.method, command.args).await?;
        Ok(())
    })
    .await
}

async fn participate<F, Fut>(stream: UnixStream, mut call: F) -> Result<()>
where
    F: FnMut(Command) -> Fut,
    Fut: Future<Output = Result<()>>,
{
    let (reader, mut writer) = stream.into_split();
    let mut lines = BufReader::new(reader).lines();
    writer.write_all(b"\"ready\"\n").await?;
    while let Some(line) = lines.next_line().await? {
        let command: Command = serde_json::from_str(&line).context("invalid checkpoint command")?;
        if command.method == "serve" {
            return Ok(());
        }
        call(command).await?;
        writer.write_all(b"\"ok\"\n").await?;
    }
    bail!("checkpoint coordinator exited before releasing frontend")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn waits_for_explicit_serve_after_resource_recovery() {
        let (launcher, frontend) = UnixStream::pair().unwrap();
        let (reader, mut writer) = launcher.into_split();
        let mut lines = BufReader::new(reader).lines();
        let participant = tokio::spawn(async move {
            let mut methods = Vec::new();
            participate(frontend, |command| {
                methods.push(command.method);
                std::future::ready(Ok(()))
            })
            .await?;
            Ok::<_, anyhow::Error>(methods)
        });
        assert_eq!(
            lines.next_line().await.unwrap().as_deref(),
            Some("\"ready\"")
        );
        for method in [
            "pause_scheduler",
            "checkpoint_prepare",
            "checkpoint_restore",
            "resume_scheduler",
        ] {
            let command = serde_json::json!({"method": method, "args": []});
            writer.write_all(format!("{command}\n").as_bytes()).await.unwrap();
            assert_eq!(lines.next_line().await.unwrap().as_deref(), Some("\"ok\""));
            assert!(!participant.is_finished());
        }
        writer.write_all(b"{\"method\":\"serve\",\"args\":[]}\n").await.unwrap();
        expect_test::expect![[r#"
            [
                "pause_scheduler",
                "checkpoint_prepare",
                "checkpoint_restore",
                "resume_scheduler",
            ]
        "#]]
        .assert_debug_eq(&participant.await.unwrap().unwrap());
    }

    #[tokio::test]
    async fn coordinator_exit_is_not_permission_to_serve() {
        let (launcher, frontend) = UnixStream::pair().unwrap();
        let participant = tokio::spawn(participate(frontend, |_| std::future::ready(Ok(()))));
        let mut reader = BufReader::new(launcher);
        let mut line = String::new();
        reader.read_line(&mut line).await.unwrap();
        drop(reader);
        assert!(participant.await.unwrap().is_err());
    }

    #[tokio::test]
    async fn failed_policy_is_not_acknowledged() {
        let (mut launcher, frontend) = UnixStream::pair().unwrap();
        let participant = tokio::spawn(participate(frontend, |_| {
            std::future::ready(Err(anyhow::anyhow!("policy failed")))
        }));
        launcher
            .write_all(b"{\"method\":\"checkpoint_prepare\",\"args\":[]}\n")
            .await
            .unwrap();
        let mut lines = BufReader::new(launcher).lines();
        assert_eq!(
            lines.next_line().await.unwrap().as_deref(),
            Some("\"ready\"")
        );
        assert!(participant.await.unwrap().is_err());
        assert!(lines.next_line().await.unwrap().is_none());
    }
}
