# Polar quality validation harness.
#
# Loads the same tokenized "context" prefix as three shards — f32-only,
# polar (no QJL), and polar+QJL — then sends the same greedy chat request
# against each and compares the resulting token sequences. The greedy
# decoder is deterministic, so f32 vs polar drift is purely quantization
# noise: polar-without-QJL should drift earliest, polar+QJL should track
# f32 the longest.
#
# Requires the server to have been started with:
#     --enable-cache --enable-polar-cache
# (--enable-retrieve too if you want /v1/retrieve to work, but this
# harness doesn't use it.)
#
# Usage:
#     pwsh polar-quality-validation.ps1 [-ServerUrl <url>] [-MaxTokens <n>]
#                                       [-Question <string>] [-Verbose]

param(
    [string]$ServerUrl = "http://127.0.0.1:8084",
    [int]$MaxTokens = 32,
    [string]$Question = "Briefly: what is the kingdom's name and who rules it?",
    [string]$ContextText = $null,
    [switch]$VerboseChat
)

$ErrorActionPreference = "Stop"

# A small "system prompt + persona" context. Long enough to give the
# polar quantization real attention positions to mis-quantize, short
# enough to load quickly. About 80-100 tokens at Qwen tokenizer.
if (-not $ContextText) {
    $ContextText = @"
You are a careful assistant for the kingdom of Vellandar, a small mountain realm
known for its silver mines, its annual harvest festival on the autumnal equinox,
and its long-standing trade alliance with the coastal city of Marshold. The
current monarch is Queen Iliana the Steadfast, who has ruled for twenty-three
years following the abdication of her father, King Aldric. The royal council
meets weekly in the Crystal Hall.
"@
}

function Invoke-Json($Method, $Path, $Body = $null) {
    $args = @{
        Uri = "$ServerUrl$Path"
        Method = $Method
        ContentType = "application/json"
        TimeoutSec = 300
    }
    if ($null -ne $Body) {
        $args.Body = ($Body | ConvertTo-Json -Depth 10 -Compress)
    }
    Invoke-RestMethod @args
}

# Safe DELETE — silently ignore 404 (shard absent on first run).
function Try-Delete($shardId) {
    try {
        $null = Invoke-WebRequest -Uri "$ServerUrl/v1/cache/$shardId" `
            -Method Delete -ErrorAction Stop
    } catch {
        # 404 is fine.
    }
}

# 1. Health probe.
Write-Host "=== Polar quality validation ===" -ForegroundColor Cyan
Write-Host "Server:    $ServerUrl"
$health = Invoke-Json GET "/health"
Write-Host "Model:     $($health.model)"
Write-Host "Max seq:   $($health.memory.max_seq_len)"

# 2. Tokenize the context wrapped as a Qwen-style system message. The
#    cache_shards mechanism prepends these K/V to the model's attention
#    BEFORE the chat handler's user-message turn. For the model to treat
#    the cached state as proper context, the cached tokens MUST already
#    include the <|im_start|>system ... <|im_end|> markers — otherwise
#    the model sees raw text after the 4 BOS sinks and gives gibberish.
$wrappedContext = "<|im_start|>system`n$ContextText<|im_end|>`n"
$tok = Invoke-Json POST "/v1/tokenize" @{ text = $wrappedContext; add_bos = $false }
Write-Host "Context:   $($tok.count) tokens (wrapped as system message)"

# 3. Clean any prior shards from a previous run.
$shardIds = @("qv_f32", "qv_polar", "qv_qjl")
foreach ($id in $shardIds) { Try-Delete $id }

# 4. Load three shards with identical tokens.
function Load-Shard($id, [hashtable]$flags) {
    $body = @{ cache_id = $id; tokens = $tok.tokens } + $flags
    $resp = Invoke-Json POST "/v1/cache/load" $body
    Write-Host ("Loaded {0,-9} seq_len={1}  flags={2}" -f $id, $resp.seq_len, ($flags | ConvertTo-Json -Compress))
}

Load-Shard "qv_f32"   @{}
Load-Shard "qv_polar" @{ polar_chat = $true; polar_only = $true; qjl = $false }
Load-Shard "qv_qjl"   @{ polar_chat = $true; polar_only = $true; qjl = $true  }

# 5. Send the same greedy chat to each. Capture text + re-tokenize the
#    output so we can count token-prefix agreement.
function Greedy-Chat($shard) {
    $body = @{
        model = "cortex"
        messages = @(
            @{ role = "user"; content = $Question }
        )
        temperature = 0
        max_tokens = $MaxTokens
        cache_shards = @($shard)
    }
    $resp = Invoke-Json POST "/v1/chat/completions" $body
    $text = $resp.choices[0].message.content
    $reTok = Invoke-Json POST "/v1/tokenize" @{ text = $text; add_bos = $false }
    @{ text = $text; tokens = $reTok.tokens }
}

Write-Host ""
Write-Host "Question:  $Question" -ForegroundColor Yellow
Write-Host ""

$f32   = Greedy-Chat "qv_f32"
$polar = Greedy-Chat "qv_polar"
$qjl   = Greedy-Chat "qv_qjl"

function Prefix-Agreement($a, $b) {
    $i = 0
    while ($i -lt $a.Count -and $i -lt $b.Count -and $a[$i] -eq $b[$i]) { $i++ }
    $i
}

$polarAgree = Prefix-Agreement $f32.tokens $polar.tokens
$qjlAgree   = Prefix-Agreement $f32.tokens $qjl.tokens

# Also run a stateless chat with the same context as a system message —
# this is what the answer SHOULD look like if the cache_shards pathway
# were healthy.
function Stateless-Chat {
    $body = @{
        model = "cortex"
        messages = @(
            @{ role = "system"; content = $ContextText }
            @{ role = "user";   content = $Question }
        )
        temperature = 0
        max_tokens = $MaxTokens
    }
    $resp = Invoke-Json POST "/v1/chat/completions" $body
    $resp.choices[0].message.content
}
$stateless = Stateless-Chat

Write-Host "=== Outputs ==="
Write-Host "[stateless ref]" -NoNewline -ForegroundColor Gray;   Write-Host " $stateless"
Write-Host "[f32 + shard]  " -NoNewline -ForegroundColor Green;  Write-Host " $($f32.text)"
Write-Host "[polar shard]  " -NoNewline -ForegroundColor Yellow; Write-Host " $($polar.text)"
Write-Host "[polar+QJL]    " -NoNewline -ForegroundColor Cyan;   Write-Host " $($qjl.text)"

Write-Host ""
Write-Host "=== Token-prefix agreement (polar vs f32-shard baseline) ==="
Write-Host ("polar:     {0,3} / {1,3} tokens before divergence" -f $polarAgree, $f32.tokens.Count)
Write-Host ("polar+QJL: {0,3} / {1,3} tokens before divergence" -f $qjlAgree,   $f32.tokens.Count)

# Detect the upstream-cache_shards-broken case: all three shard outputs
# match each other but disagree with the stateless reference. That means
# polar isn't doing anything wrong relative to f32 — but the shard
# attention pathway as a whole is producing nonsense (often `!!!!!`).
$allShardsEqual = ($f32.text -eq $polar.text) -and ($polar.text -eq $qjl.text)
$shardDiffersFromStateless = ($f32.text -ne $stateless)

Write-Host ""
if ($allShardsEqual -and $shardDiffersFromStateless) {
    Write-Host "Verdict: polar/QJL faithful to f32 baseline (all 3 shard outputs match)" -ForegroundColor Green
    Write-Host "         BUT shard output diverges from stateless reference - looks like" -ForegroundColor Yellow
    Write-Host "         the cache_shards chat pathway is broken upstream of polar." -ForegroundColor Yellow
    Write-Host "         Quality of polar specifically cannot be evaluated until that is fixed." -ForegroundColor Yellow
} else {
    $delta = $qjlAgree - $polarAgree
    $verdict = ""
    if ($delta -gt 0) {
        $verdict = "QJL helps  (+" + $delta + " tokens)"
    } elseif ($delta -lt 0) {
        $verdict = "QJL HURTS  (" + $delta + " tokens) <- investigate"
    } else {
        $verdict = "QJL neutral on this prompt"
    }
    Write-Host "Verdict:   $verdict" -ForegroundColor Magenta
}

# 6. Cleanup.
foreach ($id in $shardIds) { Try-Delete $id }

# 7. Exit code: 0 always (this is observational; quality regressions
#    are diagnosed visually, not failed CI-style).
exit 0
