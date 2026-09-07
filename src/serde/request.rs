//! Request and Response (serde.py `request_*`, `response_*`).

use serde_json::Value;

use super::config::config_from_parent;
use super::helpers::{Obj, Reader, VResult};
use super::message::{system_from_json, system_to_json, tools_from_json, tools_to_json};
use super::stream::{
    logprobs_from_json, logprobs_to_json, provider_data_from_json, usage_from_parent,
    usage_to_json_opt,
};
use super::{impl_serde_via_canonical, Canonical};
use crate::types::{FinishReason, Message, Request, Response, ValidationError};

impl Canonical for Request {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "Request")?;
        let request = Request {
            model: r.req_str("model")?,
            messages: match r.req("messages")? {
                Value::Array(items) => items
                    .iter()
                    .map(Message::from_json)
                    .collect::<VResult<Vec<Message>>>()?,
                _ => {
                    return Err(ValidationError::type_error(
                        "Request.messages must contain Message objects",
                    ))
                }
            },
            system: system_from_json(&r)?,
            tools: tools_from_json(&r)?,
            config: config_from_parent(&r)?,
        };
        request.validate()?;
        Ok(request)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::new();
        o.set("model", self.model.as_str());
        o.set(
            "messages",
            Value::Array(self.messages.iter().map(Canonical::to_json).collect()),
        );
        o.omit_empty_opt("system", self.system.as_ref().map(system_to_json));
        o.omit_empty("tools", tools_to_json(&self.tools));
        o.omit_empty("config", self.config.to_json());
        o.finish()
    }
}

impl Response {
    /// The canonical form with `provider_data` included (the default
    /// serializer never emits it; batch entries do).
    pub fn to_json_with_provider_data(&self) -> Value {
        let mut value = self.to_json();
        if let (Value::Object(o), Some(pd)) = (&mut value, &self.provider_data) {
            o.insert("provider_data".to_string(), Value::Object(pd.clone()));
        }
        value
    }
}

impl Canonical for Response {
    fn from_json(value: &Value) -> VResult<Self> {
        let r = Reader::new(value, "Response")?;
        let response = Response {
            id: r.opt_str("id")?,
            model: r.req_str("model")?,
            message: Message::from_json(r.req("message")?)?,
            finish_reason: FinishReason::parse(&r.req_str("finish_reason")?)?,
            usage: usage_from_parent(&r, "usage")?,
            logprobs: logprobs_from_json(&r)?,
            provider_data: provider_data_from_json(&r)?,
        };
        response.validate()?;
        Ok(response)
    }

    fn to_json(&self) -> Value {
        let mut o = Obj::new();
        o.omit_empty_opt("id", self.id.clone());
        o.set("model", self.model.as_str())
            .set("message", self.message.to_json())
            .set("finish_reason", self.finish_reason.as_str());
        o.opt("usage", usage_to_json_opt(&self.usage));
        o.opt(
            "logprobs",
            self.logprobs.as_deref().and_then(logprobs_to_json),
        );
        o.finish()
    }
}

impl_serde_via_canonical!(Request, Response);
