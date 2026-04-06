//! cortex-local — in-process inference provider for AgentOS.
//!
//! Provides [`CortexLocal`], an in-process inference engine that speaks the same
//! types as the OpenAI wire format but without any HTTP overhead. AgentOS can
//! depend on this crate directly and call `complete()` / `complete_with_tools()`
//! as a `LlmClient::Local` variant.
//!
//! ```text
//! // In AgentOS:
//! let provider = CortexLocal::load("model.gguf")?;
//! let response = provider.complete(&request)?;
//! ```

use std::sync::Mutex;

use cortex::layers::kv_cache::ModelKvCache;
use cortex::layers::model::TransformerModel;
use cortex::layers::sampler::SamplerConfig;
use cortex::{ModelConfig, Tokenizer};
use serde::{Deserialize, Serialize};
use tracing::info;

// ---------------------------------------------------------------------------
// Public types — mirror the OpenAI wire format for zero-cost translation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
}

fn default_max_tokens() -> u32 { 2048 }
fn default_temperature() -> f32 { 0.7 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ToolFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum LocalError {
    #[error("model load failed: {0}")]
    Load(String),
    #[error("inference error: {0}")]
    Inference(String),
}

// ---------------------------------------------------------------------------
// CortexLocal — the in-process provider
// ---------------------------------------------------------------------------

/// In-process cortex inference engine.
///
/// Loads a GGUF model once, then serves completions directly — no HTTP, no
/// serialization overhead. Designed to slot into AgentOS as a `LlmClient`
/// variant alongside `OpenAiClient` and `AnthropicClient`.
pub struct CortexLocal {
    model: TransformerModel,
    tokenizer: Tokenizer,
    #[allow(dead_code)] // used when persistent KV cache is wired
    config: ModelConfig,
    model_name: String,
    #[allow(dead_code)]
    cache: Mutex<ModelKvCache>,
    #[allow(dead_code)]
    max_seq_len: usize,
}

impl CortexLocal {
    /// Load a model from a GGUF file.
    pub fn load(path: &str, max_seq_len: usize) -> Result<Self, LocalError> {
        let loaded = cortex::load_model(path)
            .map_err(|e| LocalError::Load(e.to_string()))?;

        let model_name = loaded
            .config
            .model_name
            .clone()
            .unwrap_or_else(|| "cortex-local".to_string());

        let cache = loaded.model.create_kv_cache(max_seq_len);

        info!(model = %model_name, max_seq_len, "cortex-local ready");

        Ok(Self {
            model: loaded.model,
            tokenizer: loaded.tokenizer,
            config: loaded.config,
            model_name,
            cache: Mutex::new(cache),
            max_seq_len,
        })
    }

    /// Model name (from GGUF metadata or default).
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Run a chat completion — the core interface.
    ///
    /// This is the equivalent of `POST /v1/chat/completions` but in-process.
    pub fn complete(&self, req: &ChatRequest) -> Result<ChatResponse, LocalError> {
        let prompt_tokens = self.apply_chat_template(
            &req.messages,
            req.tools.as_deref(),
        );
        let prompt_len = prompt_tokens.len() as u32;

        let sampler_config = if req.temperature <= 0.0 {
            SamplerConfig::greedy()
        } else {
            SamplerConfig {
                temperature: req.temperature,
                top_k: 40,
                top_p: 0.95,
                ..Default::default()
            }
        };

        let eos = self.tokenizer.eos_token_id();
        let seed = rand_seed();
        let max_tokens = req.max_tokens as usize;

        let output_tokens = self.model.generate(
            &prompt_tokens, max_tokens, sampler_config, seed, Some(eos),
        );

        let generated = &output_tokens[prompt_tokens.len()..];
        let completion_len = generated.len() as u32;
        let text = self.tokenizer.decode(generated);

        let (finish_reason, response_msg) = self.build_response_message(
            &text,
            completion_len,
            req.max_tokens,
            req.tools.is_some(),
        );

        Ok(ChatResponse {
            id: format!("cortex-{}", &uuid::Uuid::new_v4().to_string()[..12]),
            model: self.model_name.clone(),
            choices: vec![Choice {
                message: response_msg,
                finish_reason,
            }],
            usage: Usage {
                prompt_tokens: prompt_len,
                completion_tokens: completion_len,
            },
        })
    }

    /// Convenience: complete with tools.
    pub fn complete_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Vec<Tool>,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<ChatResponse, LocalError> {
        self.complete(&ChatRequest {
            model: None,
            messages,
            max_tokens,
            temperature,
            tools: Some(tools),
        })
    }

    // -- Internal helpers --

    fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Vec<u32> {
        let mut prompt = String::new();

        for msg in messages {
            prompt.push_str("<|im_start|>");
            prompt.push_str(&msg.role);
            prompt.push('\n');

            if let Some(ref content) = msg.content {
                prompt.push_str(content);
            }

            if msg.role == "tool" {
                if let Some(ref id) = msg.tool_call_id {
                    prompt.push_str(&format!("\n[tool_call_id: {id}]"));
                }
            }

            prompt.push_str("<|im_end|>\n");
        }

        if let Some(tools) = tools {
            if !tools.is_empty() {
                prompt.push_str("<|im_start|>system\n");
                prompt.push_str("You have access to the following tools. To call a tool, respond with a JSON object in this exact format:\n");
                prompt.push_str("{\"tool_call\": {\"name\": \"<function_name>\", \"arguments\": {<args>}}}\n\n");
                prompt.push_str("Available tools:\n");
                for tool in tools {
                    if let Ok(json) = serde_json::to_string_pretty(&tool.function) {
                        prompt.push_str(&json);
                        prompt.push('\n');
                    }
                }
                prompt.push_str("<|im_end|>\n");
            }
        }

        prompt.push_str("<|im_start|>assistant\n");

        self.tokenizer.encode(&prompt, self.tokenizer.add_bos_default())
    }

    fn build_response_message(
        &self,
        text: &str,
        completion_len: u32,
        max_tokens: u32,
        has_tools: bool,
    ) -> (String, ChatMessage) {
        let mut msg = ChatMessage {
            role: "assistant".to_string(),
            content: None,
            tool_calls: None,
            tool_call_id: None,
        };

        if has_tools {
            if let Some(tool_calls) = parse_tool_calls(text) {
                msg.tool_calls = Some(tool_calls);
                return ("tool_calls".to_string(), msg);
            }
        }

        msg.content = Some(text.to_string());
        let reason = if completion_len >= max_tokens { "length" } else { "stop" };
        (reason.to_string(), msg)
    }
}

// ---------------------------------------------------------------------------
// Tool call parsing (shared logic)
// ---------------------------------------------------------------------------

fn parse_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
    let start = text.find("{\"tool_call\"")?;
    let obj = extract_json_object(&text[start..])?;
    let parsed: serde_json::Value = serde_json::from_str(&obj).ok()?;
    let tc = parsed.get("tool_call")?;
    let name = tc.get("name")?.as_str()?.to_string();
    let arguments = tc
        .get("arguments")
        .map(|a| serde_json::to_string(a).unwrap_or_default())
        .unwrap_or_default();

    Some(vec![ToolCall {
        id: format!("call_{}", &uuid::Uuid::new_v4().to_string()[..8]),
        call_type: "function".to_string(),
        function: ToolCallFunction { name, arguments },
    }])
}

fn extract_json_object(s: &str) -> Option<String> {
    let start = s.find('{')?;
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;

    for (i, ch) in s[start..].char_indices() {
        if escape {
            escape = false;
            continue;
        }
        match ch {
            '\\' if in_string => escape = true,
            '"' => in_string = !in_string,
            '{' if !in_string => depth += 1,
            '}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    return Some(s[start..start + i + 1].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

fn rand_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_json_simple() {
        let input = r#"some text {"tool_call": {"name": "foo", "arguments": {"x": 1}}} more"#;
        let obj = extract_json_object(input).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&obj).unwrap();
        assert_eq!(parsed["tool_call"]["name"], "foo");
    }

    #[test]
    fn extract_json_nested_braces() {
        let input = r#"{"tool_call": {"name": "bar", "arguments": {"nested": {"a": 1}}}}"#;
        let obj = extract_json_object(input).unwrap();
        assert_eq!(obj, input);
    }

    #[test]
    fn extract_json_with_escaped_quotes() {
        let input = r#"{"tool_call": {"name": "baz", "arguments": {"q": "hello \"world\""}}}"#;
        let obj = extract_json_object(input).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&obj).unwrap();
        assert_eq!(parsed["tool_call"]["name"], "baz");
    }

    #[test]
    fn parse_tool_calls_found() {
        let text = r#"Let me search for that. {"tool_call": {"name": "search_events", "arguments": {"query": "tonight"}}}"#;
        let calls = parse_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search_events");
        assert_eq!(calls[0].call_type, "function");
    }

    #[test]
    fn parse_tool_calls_not_found() {
        let text = "Just a regular response with no tool calls.";
        assert!(parse_tool_calls(text).is_none());
    }

    #[test]
    fn chat_message_serialization() {
        let msg = ChatMessage {
            role: "assistant".to_string(),
            content: Some("Hello!".to_string()),
            tool_calls: None,
            tool_call_id: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(!json.contains("tool_calls"));
        assert!(!json.contains("tool_call_id"));
    }
}
