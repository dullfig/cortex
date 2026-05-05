# Retrieval smoke-test harness for cortex-server.
#
# Usage (server must be running with --enable-retrieve):
#   .\retrieval-smoke.ps1 -TextFile C:\src\bhs-corpus\sources\Harmonizer\1941-11_rechordings-vol1-no1.md `
#                         -ShardId harm1941 `
#                         -Queries "Who founded the society?","Where did O.C. Cash grow up?"
#
# Pipeline:
#   1. Read file, strip YAML frontmatter if present.
#   2. POST /v1/tokenize     → token IDs (kept client-side for offset resolution).
#   3. POST /v1/cache/load   → build the shard in the server's cache pool.
#   4. For each query:
#        POST /v1/chat/completions (mode=retrieve) → ranked (offset, score) hits.
#        For each hit: pull ±$ContextWindow tokens from the client-side token list,
#        POST /v1/detokenize → readable context around that hit.

param(
    [Parameter(Mandatory=$true)][string]$TextFile,
    [string]$ShardId = "smoke",
    [Parameter(Mandatory=$true)][string[]]$Queries,
    [int]$TopK = 5,
    [int]$ContextWindow = 12,
    [string]$ServerUrl = "http://127.0.0.1:8080",
    # Server prepends 4 BOS sink tokens at cache_load time.
    # Hit offsets < SinkTokens are sink artifacts (not real content).
    [int]$SinkTokens = 4
)

$ErrorActionPreference = "Stop"

function Strip-Frontmatter([string]$Content) {
    if ($Content -match '(?s)\A---\r?\n.*?\r?\n---\r?\n') {
        return $Content.Substring($matches[0].Length)
    }
    return $Content
}

function Invoke-Retrieval([string]$Query, [int]$TopK) {
    $body = @{
        model        = "qwen"
        mode         = "retrieve"
        cache_shards = @($ShardId)
        messages     = @(@{ role = "user"; content = $Query })
        top_k        = $TopK
    } | ConvertTo-Json -Depth 5
    Invoke-RestMethod -Uri "$ServerUrl/v1/chat/completions" -Method Post `
                      -ContentType "application/json" -Body $body -TimeoutSec 600
}

function Detokenize([int[]]$Tokens) {
    if ($Tokens.Count -eq 0) { return "" }
    $body = @{ tokens = $Tokens } | ConvertTo-Json
    (Invoke-RestMethod -Uri "$ServerUrl/v1/detokenize" -Method Post `
                       -ContentType "application/json" -Body $body -TimeoutSec 60).text
}

# ---- 1. Load and clean text ----
$raw  = Get-Content $TextFile -Raw -Encoding utf8
$body = Strip-Frontmatter $raw
Write-Host ""
Write-Host "[load] $TextFile"
Write-Host "       raw=$($raw.Length) chars, stripped=$($body.Length) chars"

# ---- 2. Tokenize ----
$tokReq  = @{ text = $body } | ConvertTo-Json
$tokResp = Invoke-RestMethod -Uri "$ServerUrl/v1/tokenize" -Method Post `
                             -ContentType "application/json" -Body $tokReq -TimeoutSec 60
$tokens  = $tokResp.tokens
Write-Host "       tokens=$($tokens.Count)"

# ---- 3. Load shard ----
$loadBody = @{ cache_id = $ShardId; tokens = $tokens } | ConvertTo-Json
$t0       = Get-Date
$loadResp = Invoke-RestMethod -Uri "$ServerUrl/v1/cache/load" -Method Post `
                              -ContentType "application/json" -Body $loadBody -TimeoutSec 1800
$loadSec  = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
Write-Host "[ingest] shard=$ShardId seq_len=$($loadResp.seq_len) in ${loadSec}s"

# ---- 4. Run queries ----
foreach ($q in $Queries) {
    Write-Host ""
    Write-Host "==========================================================="
    Write-Host "QUERY: $q"
    Write-Host "==========================================================="
    $t0  = Get-Date
    $res = Invoke-Retrieval $q $TopK
    $ms  = [math]::Round(((Get-Date) - $t0).TotalSeconds * 1000, 0)
    Write-Host "[retrieve] $ms ms, $($res.hits.Count) hits, corpus=$($res.metadata.corpus_tokens) query=$($res.metadata.query_tokens)"

    $rank = 0
    foreach ($hit in $res.hits) {
        $rank++
        $serverOffset = [int]$hit.offset
        $score        = [math]::Round($hit.score, 3)

        # Map server-side offset (includes sink tokens) to client-side token index.
        $clientIdx = $serverOffset - $SinkTokens

        if ($clientIdx -lt 0) {
            Write-Host ("  #{0,-2} offset={1,-4} score={2,-8} [SINK ARTIFACT - ignore]" -f $rank, $serverOffset, $score)
            continue
        }

        $lo = [math]::Max(0, $clientIdx - $ContextWindow)
        $hi = [math]::Min($tokens.Count - 1, $clientIdx + $ContextWindow)
        $window = $tokens[$lo..$hi]

        $ctx = (Detokenize $window) -replace '\s+', ' '
        if ($ctx.Length -gt 240) { $ctx = $ctx.Substring(0, 240) + "..." }

        Write-Host ("  #{0,-2} offset={1,-4} score={2,-8} " -f $rank, $serverOffset, $score)
        Write-Host ("      ...{0}..." -f $ctx)
    }
}

Write-Host ""
Write-Host "[done]"
