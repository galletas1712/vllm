// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use vllm_engine_core_client::protocol::utility::CheckpointState;

use crate::error::ApiError;
use crate::state::AppState;
use crate::utils::utility_call_error;

#[derive(Serialize)]
pub(crate) struct CheckpointStateResponse {
    state: CheckpointState,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

type CheckpointResult = Result<(StatusCode, Json<CheckpointStateResponse>), ApiError>;

fn state_response(
    status: StatusCode,
    state: CheckpointState,
    error: Option<String>,
) -> CheckpointResult {
    Ok((status, Json(CheckpointStateResponse { state, error })))
}

#[derive(Default, Deserialize)]
pub(crate) struct CheckpointResumeRequest {
    /// Environment of the pod restored into. A null value unsets a variable.
    #[serde(default)]
    env: BTreeMap<String, Option<String>>,
    /// Engine arguments of the pod restored into.
    #[serde(default)]
    flags: BTreeMap<String, JsonValue>,
}

/// Bring back what preparation released and serve. Repeating is a no-op.
pub async fn resume(
    State(state): State<Arc<AppState>>,
    body: Result<Option<Json<CheckpointResumeRequest>>, JsonRejection>,
) -> CheckpointResult {
    let request = body?.map(|Json(request)| request).unwrap_or_default();
    let client = state.engine_core_client();
    let (checkpoint_state, error) = client
        .checkpoint_state()
        .await
        .map_err(|error| utility_call_error("checkpoint_state", error))?;
    if matches!(
        checkpoint_state,
        CheckpointState::Preparing | CheckpointState::Resuming | CheckpointState::Failed
    ) {
        return state_response(StatusCode::CONFLICT, checkpoint_state, error);
    }
    if checkpoint_state != CheckpointState::Running {
        client
            .checkpoint_restore(&request.env, &request.flags)
            .await
            .map_err(|error| utility_call_error("checkpoint_restore", error))?;
        client
            .wake_up(Some(vec!["kv_cache".to_string()]))
            .await
            .map_err(|error| utility_call_error("wake_up", error))?;
    }
    state_response(StatusCode::OK, CheckpointState::Running, None)
}

/// Return 200 only once every engine is prepared, so orchestrators can poll it
/// to learn when the process is safe to capture.
pub async fn status(State(state): State<Arc<AppState>>) -> CheckpointResult {
    let (checkpoint_state, error) = state
        .engine_core_client()
        .checkpoint_state()
        .await
        .map_err(|error| utility_call_error("checkpoint_state", error))?;
    let status = if checkpoint_state == CheckpointState::Prepared {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    state_response(status, checkpoint_state, error)
}
