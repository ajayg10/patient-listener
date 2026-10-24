# Deploy Patient Listener to AWS: API on Lambda (Function URL, streaming), frontend on Amplify Hosting.
# Uses only: IAM role, Lambda (+ AWS Lambda Web Adapter public layer), Amplify Hosting.
# Credentials come from your normal AWS CLI profile. Nothing secret is written to the repo.
#
# Usage (PowerShell, repo root):  .\scripts\deploy_aws.ps1 [-Step all|package|lambda|frontend]
param([string]$Step = "all")

$ErrorActionPreference = "Continue"  # native tools write notices to stderr; exit codes are checked explicitly
$Region   = "us-east-1"
$Fn       = "patient-listener-api"
$Role     = "patient-listener-lambda-role"
$AppName  = "patient-listener"
$Branch   = "main"
$Layer    = "arn:aws:lambda:us-east-1:753240598075:layer:LambdaAdapterLayerX86:26"
$Root     = Split-Path -Parent $PSScriptRoot
$Build    = Join-Path $env:TEMP "pl-lambda-build"
$Zip      = Join-Path $env:TEMP "pl-lambda.zip"
$Py       = Join-Path $Root ".venv\Scripts\python.exe"

function Invoke-Aws { param([Parameter(ValueFromRemainingArguments)]$a) $out = & aws.exe @a --region $Region 2>&1; if ($LASTEXITCODE -ne 0) { throw "aws $($a -join ' ') failed: $out" }; $out }

# ---------- Amplify app first: its domain goes into the API's CORS list ----------
function Get-AmplifyApp {
  $id = (& aws amplify list-apps --region $Region --query "apps[?name=='$AppName'].appId | [0]" --output text)
  if (-not $id -or $id -eq "None") {
    $id = Invoke-Aws amplify create-app --name $AppName --platform WEB --query "app.appId" --output text
    Invoke-Aws amplify create-branch --app-id $id --branch-name $Branch | Out-Null
  }
  return $id.Trim()
}

function Build-Package {
  Write-Host "== Building Lambda package"
  $env:PIP_USER = "0"  # --target cannot be combined with a user-level pip config
  if (Test-Path $Build) { Remove-Item -Recurse -Force $Build }
  New-Item -ItemType Directory $Build | Out-Null
  # uv resolves Linux-only environment markers correctly when cross-installing from Windows
  & python -m uv pip install -q --target $Build --python-platform x86_64-manylinux_2_28 --python-version 3.12 `
      --only-binary :all: -r (Join-Path $Root "requirements-lambda.txt")
  if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
  # strands-decider is imported for its types only; the replay never loads the model, so skip torch.
  & python -m uv pip install -q --target $Build --python-platform x86_64-manylinux_2_28 --python-version 3.12 --no-deps "strands-decider==0.2.0"
  Get-ChildItem $Build -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
  # app code and data (no tests, no frontend, no model files)
  foreach ($d in "backend","scripts","patient_listener_pattern") { Copy-Item -Recurse (Join-Path $Root $d) (Join-Path $Build $d) }
  Copy-Item (Join-Path $Root "config.yaml") $Build
  Get-ChildItem $Build -Recurse -Directory | Where-Object { $_.Name -in "tests","__pycache__" -and $_.FullName -like "*\backend\*" } | Remove-Item -Recurse -Force
  Remove-Item -Recurse -Force (Join-Path $Build "backend\app\data") -ErrorAction SilentlyContinue
  Get-ChildItem (Join-Path $Build "backend\eval\data") -Filter "decisions_cache*.jsonl" | Remove-Item -Force  # keep only the .gz
  # Lambda Web Adapter starts this script, then proxies Function URL requests to port 8080
  "#!/bin/sh`nexec python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8080`n" -replace "`r","" |
    Set-Content -NoNewline -Encoding ascii (Join-Path $Build "run.sh")
  if (Test-Path $Zip) { Remove-Item $Zip }
  # zip with a unix exec bit on run.sh
  & $Py -c @"
import os, sys, zipfile
src, dst = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as z:
    for base, _, files in os.walk(src):
        for f in files:
            p = os.path.join(base, f); rel = os.path.relpath(p, src).replace(os.sep, '/')
            info = zipfile.ZipInfo.from_file(p, rel)
            info.external_attr = (0o755 if rel == 'run.sh' else 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            with open(p, 'rb') as fh: z.writestr(info, fh.read())
"@ $Build $Zip
  Write-Host ("   zip: {0:N1} MB" -f ((Get-Item $Zip).Length / 1MB))
}

function Ensure-Role {
  $arn = (& aws iam get-role --role-name $Role --query "Role.Arn" --output text 2>$null)
  if (-not $arn -or $LASTEXITCODE -ne 0) {
    $trust = '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
    $tf = Join-Path $env:TEMP "pl-trust.json"; $trust | Set-Content -Encoding ascii $tf
    $arn = (& aws iam create-role --role-name $Role --assume-role-policy-document "file://$tf" --query "Role.Arn" --output text)
    & aws iam attach-role-policy --role-name $Role --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
    Start-Sleep 10  # IAM propagation
  }
  # Bedrock: invoke only (the reply agent streams one short reply per replay)
  $pol = '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["bedrock:InvokeModel","bedrock:InvokeModelWithResponseStream"],"Resource":"*"}]}'
  $pf = Join-Path $env:TEMP "pl-bedrock.json"; $pol | Set-Content -Encoding ascii $pf
  & aws iam put-role-policy --role-name $Role --policy-name bedrock-invoke --policy-document "file://$pf"
  return $arn.Trim()
}

function Deploy-Lambda([string]$roleArn, [string]$amplifyOrigin) {
  Write-Host "== Deploying Lambda $Fn"
  $envJson = @{ Variables = @{
      AWS_LAMBDA_EXEC_WRAPPER = "/opt/bootstrap"; AWS_LWA_INVOKE_MODE = "response_stream"; PORT = "8080"
      MOCK_BEDROCK = "0"; BEDROCK_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
      REGRET_DB_PATH = "/tmp/regrets.db"; HF_HOME = "/tmp/hf"; DECIDER_BACKEND = "mock"
      CORS_ORIGINS = "[`"$amplifyOrigin`",`"http://localhost:5173`"]"
  } } | ConvertTo-Json -Depth 4 -Compress
  $ef = Join-Path $env:TEMP "pl-env.json"; $envJson | Set-Content -Encoding ascii $ef
  & aws lambda get-function --function-name $Fn --region $Region *> $null
  if ($LASTEXITCODE -ne 0) {
    Invoke-Aws lambda create-function --function-name $Fn --runtime python3.12 --architectures x86_64 `
      --role $roleArn --handler run.sh --zip-file "fileb://$Zip" --timeout 60 --memory-size 1024 `
      --layers $Layer --environment "file://$ef" | Out-Null
  } else {
    Invoke-Aws lambda update-function-code --function-name $Fn --zip-file "fileb://$Zip" | Out-Null
    & aws lambda wait function-updated --function-name $Fn --region $Region
    Invoke-Aws lambda update-function-configuration --function-name $Fn --environment "file://$ef" --layers $Layer --handler run.sh --runtime python3.12 | Out-Null
  }
  & aws lambda wait function-updated --function-name $Fn --region $Region
  # public HTTPS endpoint with response streaming (needed for the replay's event stream)
  $url = (& aws lambda get-function-url-config --function-name $Fn --region $Region --query FunctionUrl --output text 2>$null)
  if (-not $url -or $LASTEXITCODE -ne 0) {
    $url = Invoke-Aws lambda create-function-url-config --function-name $Fn --auth-type NONE --invoke-mode RESPONSE_STREAM --query FunctionUrl --output text
    & aws lambda add-permission --function-name $Fn --region $Region --statement-id public-url --action lambda:InvokeFunctionUrl --principal "*" --function-url-auth-type NONE | Out-Null
    & aws lambda add-permission --function-name $Fn --region $Region --statement-id public-invoke --action lambda:InvokeFunction --principal "*" | Out-Null
  }
  return $url.Trim().TrimEnd("/")
}

function Deploy-Frontend([string]$appId, [string]$apiUrl) {
  Write-Host "== Building and deploying frontend to Amplify"
  Push-Location (Join-Path $Root "frontend")
  $env:VITE_API_URL = $apiUrl
  npm run build | Out-Null
  if ($LASTEXITCODE -ne 0) { Pop-Location; throw "frontend build failed" }
  Remove-Item Env:VITE_API_URL
  $fz = Join-Path $env:TEMP "pl-web.zip"; if (Test-Path $fz) { Remove-Item $fz }
  # zip with forward-slash paths (Compress-Archive on Windows PowerShell 5.1 writes backslashes, which Amplify serves as 404)
  & $Py -c "import os,sys,zipfile; s,d=sys.argv[1],sys.argv[2]; z=zipfile.ZipFile(d,'w',zipfile.ZIP_DEFLATED); [z.write(os.path.join(b,f), os.path.relpath(os.path.join(b,f),s).replace(os.sep,'/')) for b,_,fs in os.walk(s) for f in fs]; z.close()" (Resolve-Path "dist") $fz
  Pop-Location
  $dep = Invoke-Aws amplify create-deployment --app-id $appId --branch-name $Branch --output json | ConvertFrom-Json
  Invoke-WebRequest -Uri $dep.zipUploadUrl -Method Put -InFile $fz -ContentType "application/zip" -UseBasicParsing | Out-Null
  Invoke-Aws amplify start-deployment --app-id $appId --branch-name $Branch --job-id $dep.jobId | Out-Null
  do { Start-Sleep 5; $st = (& aws amplify get-job --app-id $appId --branch-name $Branch --job-id $dep.jobId --region $Region --query "job.summary.status" --output text) } while ($st -in "PENDING","RUNNING")
  Write-Host "   amplify job: $st"
}

$appId  = Get-AmplifyApp
$origin = "https://$Branch.$appId.amplifyapp.com"
if ($Step -in "all","package") { Build-Package }
if ($Step -in "all","lambda")  { $roleArn = Ensure-Role; $api = Deploy-Lambda $roleArn $origin; Write-Host "API: $api" }
if ($Step -in "all","frontend") {
  if (-not $api) { $api = (& aws lambda get-function-url-config --function-name $Fn --region $Region --query FunctionUrl --output text).Trim().TrimEnd("/") }
  Deploy-Frontend $appId $api
}
Write-Host "Frontend: $origin"
