<#
.SYNOPSIS
    本地 Qdrant 启停脚本（使用 tools/qdrant/qdrant.exe 官方 Windows 二进制）。

.DESCRIPTION
    数据目录为 tools/qdrant/storage，与 Docker 方式等价，
    监听 REST 6333 / gRPC 6334。

.EXAMPLE
    .\scripts\qdrant.ps1 start
    .\scripts\qdrant.ps1 status
    .\scripts\qdrant.ps1 stop
#>
param(
    [ValidateSet("start", "status", "stop")]
    [string]$Action = "start"
)

$root = Split-Path -Parent $PSScriptRoot
$dir = Join-Path $root "tools\qdrant"
$exe = Join-Path $dir "qdrant.exe"

function Get-QdrantProcess {
    return Get-Process qdrant -ErrorAction SilentlyContinue
}

function Test-QdrantReady {
    try {
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:6333/" -TimeoutSec 2
        return $resp
    } catch {
        return $null
    }
}

switch ($Action) {
    "start" {
        if (-not (Test-Path $exe)) {
            Write-Error "未找到 $exe。请从 https://github.com/qdrant/qdrant/releases 下载 qdrant-x86_64-pc-windows-msvc.zip 并解压到 $dir"
            exit 1
        }

        $proc = Get-QdrantProcess
        if ($proc) {
            Write-Output "Qdrant 已在运行（pid=$($proc.Id)）"
        } else {
            Start-Process -FilePath $exe `
                -WorkingDirectory $dir `
                -WindowStyle Hidden `
                -RedirectStandardOutput (Join-Path $dir "qdrant.out.log") `
                -RedirectStandardError (Join-Path $dir "qdrant.err.log")
            Write-Output "已启动 qdrant.exe，工作目录：$dir"
        }

        for ($i = 0; $i -lt 20; $i++) {
            Start-Sleep -Milliseconds 700
            $resp = Test-QdrantReady
            if ($resp) {
                Write-Output "Qdrant 就绪：version=$($resp.version)，地址=http://127.0.0.1:6333"
                exit 0
            }
        }

        Write-Error "Qdrant 启动超时，请查看 $dir\qdrant.err.log"
        exit 1
    }

    "status" {
        $proc = Get-QdrantProcess
        $resp = Test-QdrantReady
        if ($proc) { Write-Output "进程：运行中（pid=$($proc.Id)）" } else { Write-Output "进程：未运行" }
        if ($resp) { Write-Output "服务：可用（version=$($resp.version)）" } else { Write-Output "服务：不可用（6333 无响应）" }
    }

    "stop" {
        $proc = Get-QdrantProcess
        if (-not $proc) {
            Write-Output "Qdrant 未运行"
            exit 0
        }
        $proc | Stop-Process -Force
        Write-Output "已停止 Qdrant（pid=$($proc.Id)）"
    }
}
