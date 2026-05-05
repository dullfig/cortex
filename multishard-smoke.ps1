# Quick multi-shard retrieve smoke test.
# Loads two shards via /v1/cache/load, then issues a multi-shard retrieve
# query and prints the ranked hits. Used to validate the multi-shard
# composition path in cortex-cloud's retrieve handler.

param(
    [string]$ServerUrl = "http://127.0.0.1:8080"
)
$ErrorActionPreference = "Stop"

function Strip-Frontmatter([string]$Content) {
    if ($Content -match '(?s)\A---\r?\n.*?\r?\n---\r?\n') {
        return $Content.Substring($matches[0].Length)
    }
    return $Content
}

function Tokenize-And-Load([string]$File, [string]$ShardId) {
    $raw  = Get-Content $File -Raw -Encoding utf8
    $body = Strip-Frontmatter $raw

    $tokReq  = @{ text = $body } | ConvertTo-Json
    $tokResp = Invoke-RestMethod -Uri "$ServerUrl/v1/tokenize" -Method Post `
                                 -ContentType "application/json" -Body $tokReq -TimeoutSec 60

    Write-Host "[$ShardId] tokens=$($tokResp.tokens.Count) ingesting..."
    $loadBody = @{ cache_id = $ShardId; tokens = $tokResp.tokens } | ConvertTo-Json
    $t0 = Get-Date
    $resp = Invoke-RestMethod -Uri "$ServerUrl/v1/cache/load" -Method Post `
                              -ContentType "application/json" -Body $loadBody -TimeoutSec 1800
    $sec = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    Write-Host "[$ShardId] loaded seq_len=$($resp.seq_len) in ${sec}s"
}

# 1. Ingest two shards
Tokenize-And-Load "C:\src\bhs-corpus\sources\Harmonizer\1941-11_rechordings-vol1-no1.md" "harm1941"
Tokenize-And-Load "C:\src\bhs-corpus\sources\bhs-org\what-is-barbershop.md" "whatis"

# 2. Multi-shard retrieve
$queries = @(
    "Who founded the society?",
    "What is barbershop singing?",
    "Where did O.C. Cash grow up?"
)
foreach ($q in $queries) {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "QUERY: $q"
    Write-Host "============================================================"
    $body = @{
        model        = "qwen"
        mode         = "retrieve"
        cache_shards = @("harm1941", "whatis")
        messages     = @(@{ role = "user"; content = $q })
        top_k        = 6
    } | ConvertTo-Json -Depth 5
    $t0 = Get-Date
    try {
        $res = Invoke-RestMethod -Uri "$ServerUrl/v1/chat/completions" -Method Post `
                                 -ContentType "application/json" -Body $body -TimeoutSec 600
        $ms = [math]::Round(((Get-Date) - $t0).TotalSeconds * 1000, 0)
        Write-Host "[retrieve] $ms ms, $($res.hits.Count) hits, corpus=$($res.metadata.corpus_tokens)"
        $rank = 0
        foreach ($hit in $res.hits) {
            $rank++
            Write-Host ("  #{0,-2} shard={1,-10} offset={2,-5} score={3,-7}" -f $rank, $hit.shard_id, $hit.offset, ([math]::Round($hit.score,3)))
        }
    } catch {
        Write-Host "FAILED: $_"
    }
}

Write-Host ""
Write-Host "[done]"
