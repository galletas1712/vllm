// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::HashMap;
use std::os::fd::FromRawFd;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use anyhow::{Context as _, Result, bail};
use axum::body::{Body, Bytes, HttpBody};
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::{Next, from_fn_with_state};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::future::BoxFuture;
use http_body::{Frame, SizeHint};
use serde_json::{Value, json};
use thiserror_ext::AsReport;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::sync::{Notify, mpsc, oneshot};
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;
use tokio_util::task::AbortOnDropHandle;

use crate::state::AppState;

type UtilityCall = Arc<dyn Fn(String, Vec<Value>) -> BoxFuture<'static, Result<()>> + Send + Sync>;

#[derive(Default)]
struct Admission {
    state: Mutex<(bool, usize)>,
    idle: Notify,
}

impl Admission {
    fn set_accepting(&self, accepting: bool) {
        self.state.lock().unwrap().0 = accepting;
    }

    fn enter(self: &Arc<Self>) -> Option<RequestGuard> {
        let mut state = self.state.lock().unwrap();
        if !state.0 {
            return None;
        }
        state.1 += 1;
        Some(RequestGuard(self.clone()))
    }

    async fn drain(&self) {
        loop {
            let notified = self.idle.notified();
            if self.state.lock().unwrap().1 == 0 {
                return;
            }
            notified.await;
        }
    }
}

struct RequestGuard(Arc<Admission>);

impl Drop for RequestGuard {
    fn drop(&mut self) {
        let mut state = self.0.state.lock().unwrap();
        state.1 -= 1;
        if state.1 == 0 {
            self.0.idle.notify_one();
        }
    }
}

struct TrackedBody {
    inner: Body,
    _guard: RequestGuard,
}

impl HttpBody for TrackedBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        Pin::new(&mut self.inner).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

pub(crate) struct Checkpoint {
    admission: Arc<Admission>,
    outgoing: mpsc::UnboundedSender<Value>,
    pending: Mutex<HashMap<u64, oneshot::Sender<Value>>>,
    sequence: AtomicU64,
}

impl Checkpoint {
    async fn request(&self, method: &str) -> Result<Value> {
        let id = self.sequence.fetch_add(1, Ordering::Relaxed);
        let (sender, receiver) = oneshot::channel();
        self.pending.lock().unwrap().insert(id, sender);
        self.outgoing.send(json!({"id": id, "method": method, "args": []}))?;
        let response = receiver.await.context("checkpoint coordinator disconnected")?;
        if let Some(error) = response.get("error") {
            bail!("checkpoint operation failed: {error}");
        }
        Ok(response["result"].clone())
    }

    async fn command(&self, message: &Value, call: &UtilityCall) -> Result<()> {
        match message["method"].as_str() {
            Some("quiesce") => self.admission.set_accepting(false),
            Some("drain") => self.admission.drain().await,
            Some("admit") => self.admission.set_accepting(true),
            // Rust currently has no frontend multimodal cache to clear.
            Some("clear_cache") => {}
            Some("utility") => {
                let args = message["args"].as_array().context("missing utility args")?;
                let method =
                    args.first().and_then(Value::as_str).context("missing utility method")?;
                call(method.to_owned(), args[1..].to_vec()).await?;
            }
            _ => bail!("unknown checkpoint command"),
        }
        Ok(())
    }

    async fn run(
        self: Arc<Self>,
        stream: UnixStream,
        mut outgoing: mpsc::UnboundedReceiver<Value>,
        call: UtilityCall,
    ) -> Result<()> {
        let (reader, mut writer) = stream.into_split();
        let mut lines = BufReader::new(reader).lines();
        let mut commands = JoinSet::new();
        loop {
            tokio::select! {
                line = lines.next_line() => {
                    let line = line?.context("checkpoint coordinator disconnected")?;
                    let message: Value = serde_json::from_str(&line)?;
                    let id = message["id"].as_u64().context("missing checkpoint request id")?;
                    if message.get("method").is_some() {
                        let checkpoint = self.clone();
                        let call = call.clone();
                        commands.spawn(async move {
                            let response = match checkpoint.command(&message, &call).await {
                                Ok(()) => json!({"id": id, "result": null}),
                                Err(error) => json!({"id": id, "error": format!("{}", error.as_report())}),
                            };
                            let _ = checkpoint.outgoing.send(response);
                        });
                    } else if let Some(sender) = self.pending.lock().unwrap().remove(&id) {
                        let _ = sender.send(message);
                    }
                }
                Some(message) = outgoing.recv() => {
                    writer.write_all(format!("{message}\n").as_bytes()).await?;
                }
                Some(result) = commands.join_next(), if !commands.is_empty() => { result?; }
            }
        }
    }
}

async fn start(
    stream: UnixStream,
    call: UtilityCall,
    shutdown: CancellationToken,
) -> Result<(Arc<Checkpoint>, AbortOnDropHandle<()>)> {
    let (outgoing, receiver) = mpsc::unbounded_channel();
    let checkpoint = Arc::new(Checkpoint {
        admission: Arc::default(),
        outgoing,
        pending: Mutex::default(),
        sequence: AtomicU64::new(1),
    });
    let participant = checkpoint.clone();
    let task = AbortOnDropHandle::new(tokio::spawn(async move {
        if let Err(error) = participant.clone().run(stream, receiver, call).await {
            tracing::error!(error = %error.as_report(), "checkpoint control failed");
        }
        participant.admission.set_accepting(false);
        participant.pending.lock().unwrap().clear();
        shutdown.cancel();
    }));
    checkpoint.request("join").await?;
    checkpoint.admission.set_accepting(true);
    Ok((checkpoint, task))
}

/// Connect to the Python launcher's service-wide lifecycle coordinator.
pub(crate) async fn connect(
    fd: i32,
    state: Arc<AppState>,
    shutdown: CancellationToken,
) -> Result<(Arc<Checkpoint>, AbortOnDropHandle<()>)> {
    // The launcher passes exclusive ownership of this descriptor to the frontend.
    let stream = unsafe { std::os::unix::net::UnixStream::from_raw_fd(fd) };
    stream.set_nonblocking(true)?;
    start(
        UnixStream::from_std(stream)?,
        Arc::new(move |method, args| {
            let state = state.clone();
            Box::pin(async move {
                state.engine_core_client().call_utility::<Value, _>(&method, args).await?;
                Ok(())
            })
        }),
        shutdown,
    )
    .await
}

async fn admission(
    State(checkpoint): State<Arc<Checkpoint>>,
    req: Request,
    next: Next,
) -> Response {
    if req.uri().path().starts_with("/checkpoint/") {
        return next.run(req).await;
    }
    let Some(guard) = checkpoint.admission.enter() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "detail": "Service is quiesced for checkpointing"
            })),
        )
            .into_response();
    };
    let (parts, body) = next.run(req).await.into_parts();
    Response::from_parts(
        parts,
        Body::new(TrackedBody {
            inner: body,
            _guard: guard,
        }),
    )
}

pub(crate) fn attach(router: Router, checkpoint: Arc<Checkpoint>, state: Arc<AppState>) -> Router {
    let handler = |method: &'static str| {
        let checkpoint = checkpoint.clone();
        move || async move {
            match checkpoint.request(method).await {
                Ok(value) => Json(value).into_response(),
                Err(error) => (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(json!({
                        "detail": format!("{}", error.as_report())
                    })),
                )
                    .into_response(),
            }
        }
    };
    let mut control = Router::new()
        .route("/checkpoint/prepare", post(handler("prepare")))
        .route("/checkpoint/resume", post(handler("resume")))
        .route("/checkpoint/status", get(handler("status")));
    if state.has_api_keys() {
        control = control.layer(from_fn_with_state(
            state,
            crate::middleware::authenticate_api_key,
        ));
    }
    router.merge(control).layer(from_fn_with_state(checkpoint, admission))
}

#[cfg(test)]
mod tests {
    use axum::body::to_bytes;
    use tokio::io::{AsyncRead, AsyncWrite};
    use tower::Service as _;

    use super::*;

    async fn read(reader: &mut (impl AsyncBufReadExt + Unpin)) -> Value {
        let mut line = String::new();
        reader.read_line(&mut line).await.unwrap();
        serde_json::from_str(&line).unwrap()
    }

    async fn send(writer: &mut (impl AsyncWrite + Unpin), message: Value) {
        writer.write_all(format!("{message}\n").as_bytes()).await.unwrap();
    }

    async fn command(
        reader: &mut BufReader<impl AsyncRead + Unpin>,
        writer: &mut (impl AsyncWrite + Unpin),
        method: &str,
    ) {
        send(writer, json!({"id": 100, "method": method, "args": []})).await;
        assert_eq!(read(reader).await, json!({"id": 100, "result": null}));
    }

    #[tokio::test]
    async fn drain_waits_for_response_body_without_blocking_control_requests() {
        let (launcher, frontend) = UnixStream::pair().unwrap();
        let (reader, mut writer) = launcher.into_split();
        let mut reader = BufReader::new(reader);
        let shutdown = CancellationToken::new();
        let participant = tokio::spawn(start(
            frontend,
            Arc::new(|_, _| Box::pin(async { Ok(()) })),
            shutdown.clone(),
        ));
        let join = read(&mut reader).await;
        assert_eq!(join["method"], "join");
        send(
            &mut writer,
            json!({"id": join["id"], "result": {"state": "running"}}),
        )
        .await;
        let (checkpoint, _task) = participant.await.unwrap().unwrap();
        let mut router = Router::new()
            .route("/inference", get(|| async { "generated output" }))
            .layer(from_fn_with_state(checkpoint.clone(), admission));
        let request = || Request::builder().uri("/inference").body(Body::empty()).unwrap();
        let response = router.call(request()).await.unwrap();
        command(&mut reader, &mut writer, "quiesce").await;
        assert_eq!(
            router.call(request()).await.unwrap().status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        send(
            &mut writer,
            json!({"id": 101, "method": "drain", "args": []}),
        )
        .await;
        let status = tokio::spawn({
            let checkpoint = checkpoint.clone();
            async move { checkpoint.request("status").await }
        });
        let query = read(&mut reader).await;
        assert_eq!(
            query["method"], "status",
            "drain must not finish while the body is held"
        );
        send(
            &mut writer,
            json!({"id": query["id"], "result": {"state": "preparing"}}),
        )
        .await;
        assert_eq!(
            status.await.unwrap().unwrap(),
            json!({"state": "preparing"})
        );
        assert_eq!(
            to_bytes(response.into_body(), usize::MAX).await.unwrap(),
            "generated output"
        );
        assert_eq!(read(&mut reader).await, json!({"id": 101, "result": null}));
        command(&mut reader, &mut writer, "admit").await;
        assert_eq!(
            router.call(request()).await.unwrap().status(),
            StatusCode::OK
        );
        drop(reader);
        drop(writer);
        shutdown.cancelled().await;
        assert_eq!(
            router.call(request()).await.unwrap().status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[tokio::test]
    async fn utility_failure_is_reported_without_reopening_admission() {
        let (launcher, frontend) = UnixStream::pair().unwrap();
        let (reader, mut writer) = launcher.into_split();
        let mut reader = BufReader::new(reader);
        let participant = tokio::spawn(start(
            frontend,
            Arc::new(|_, _| Box::pin(async { bail!("policy failed") })),
            CancellationToken::new(),
        ));
        let join = read(&mut reader).await;
        send(
            &mut writer,
            json!({"id": join["id"], "result": {"state": "running"}}),
        )
        .await;
        let (checkpoint, _task) = participant.await.unwrap().unwrap();
        command(&mut reader, &mut writer, "quiesce").await;
        send(
            &mut writer,
            json!({"id": 102, "method": "utility", "args": ["checkpoint_restore"]}),
        )
        .await;
        let response = read(&mut reader).await;
        assert_eq!(response["id"], 102);
        assert!(response["error"].as_str().unwrap().contains("policy failed"));
        assert!(checkpoint.admission.enter().is_none());
    }
}
