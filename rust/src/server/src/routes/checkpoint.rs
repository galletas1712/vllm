// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use serde::Serialize;

use crate::error::ApiError;
use crate::state::AppState;
use crate::utils::utility_call_error;

#[derive(Serialize)]
pub(crate) struct CheckpointStatusResponse {
    prepared: bool,
}

/// Prepare the engines for an external process checkpoint.
pub async fn prepare(State(state): State<Arc<AppState>>) -> Result<StatusCode, ApiError> {
    state
        .engine_core_client()
        .checkpoint_prepare()
        .await
        .map_err(|error| utility_call_error("checkpoint_prepare", error))?;

    Ok(StatusCode::OK)
}

/// Rebuild released state and resume serving after a checkpoint restore.
pub async fn resume(State(state): State<Arc<AppState>>) -> Result<StatusCode, ApiError> {
    state
        .engine_core_client()
        .checkpoint_restore()
        .await
        .map_err(|error| utility_call_error("checkpoint_restore", error))?;

    Ok(StatusCode::OK)
}

/// Return 200 only once every engine is prepared, so orchestrators can poll it.
pub async fn status(
    State(state): State<Arc<AppState>>,
) -> Result<(StatusCode, Json<CheckpointStatusResponse>), ApiError> {
    let prepared = state
        .engine_core_client()
        .is_checkpoint_prepared()
        .await
        .map_err(|error| utility_call_error("is_checkpoint_prepared", error))?;
    let status = if prepared {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    Ok((status, Json(CheckpointStatusResponse { prepared })))
}
