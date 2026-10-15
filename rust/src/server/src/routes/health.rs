// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;
use std::sync::atomic::Ordering;

use axum::extract::State;
use axum::http::StatusCode;
use vllm_engine_core_client::protocol::utility::CheckpointState;

use crate::state::AppState;

pub async fn health(State(state): State<Arc<AppState>>) -> StatusCode {
    let client = state.engine_core_client();
    if !client.is_healthy() {
        return StatusCode::SERVICE_UNAVAILABLE;
    }
    // Engines started with `--checkpoint-prepare-on-startup` serve only after
    // `/checkpoint/resume`. Once serving, they stay serving, so the engines are
    // asked only until then.
    if !state.engine_serving.load(Ordering::Relaxed) {
        if !matches!(
            client.checkpoint_state().await,
            Ok((CheckpointState::Running, _))
        ) {
            return StatusCode::SERVICE_UNAVAILABLE;
        }
        state.engine_serving.store(true, Ordering::Relaxed);
    }
    StatusCode::OK
}
