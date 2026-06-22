# ============================================================
# Copy_Matching_Files.ps1
# Finds matching subfolders between two directories and copies
# R1_Trial_Track and R4_Trial_Track files from source to dest.
# ============================================================

$SourceRoot = CResearchEncoder_ModelingEncoder_AnalysisProcessed_EncoderR14
$DestRoot   = CResearchEncoder_ModelingEncoder_AnalysisResults_Final_RemakeAprilResults_Final
$FilesToCopy = @(R1_Trial_Track, R4_Trial_Track)

# --- Validate root folders exist ---
if (-not (Test-Path $SourceRoot)) {
    Write-Error Source folder not found $SourceRoot
    exit 1
}
if (-not (Test-Path $DestRoot)) {
    Write-Error Destination folder not found $DestRoot
    exit 1
}

# --- Get subfolder names from both roots ---
$SourceSubfolders = Get-ChildItem -Path $SourceRoot -Directory  Select-Object -ExpandProperty Name
$DestSubfolders   = Get-ChildItem -Path $DestRoot   -Directory  Select-Object -ExpandProperty Name

# --- Find matches ---
$Matches = $SourceSubfolders  Where-Object { $DestSubfolders -contains $_ }

if ($Matches.Count -eq 0) {
    Write-Host No matching subfolders found between the two directories. -ForegroundColor Yellow
    exit 0
}

Write-Host Found $($Matches.Count) matching subfolder(s) -ForegroundColor Cyan
$Matches  ForEach-Object { Write-Host   - $_ }
Write-Host 

$CopyCount  = 0
$SkipCount  = 0
$ErrorCount = 0

foreach ($Folder in $Matches) {
    $SourceFolder = Join-Path $SourceRoot $Folder
    $DestFolder   = Join-Path $DestRoot   $Folder

    foreach ($Pattern in $FilesToCopy) {
        $Files = Get-ChildItem -Path $SourceFolder -Filter $Pattern -File -ErrorAction SilentlyContinue

        if ($Files.Count -eq 0) {
            Write-Host   [SKIP] No file matching '$Pattern' in $SourceFolder -ForegroundColor Yellow
            $SkipCount++
            continue
        }

        foreach ($File in $Files) {
            $DestFile = Join-Path $DestFolder $File.Name
            try {
                Copy-Item -Path $File.FullName -Destination $DestFile -Force
                Write-Host   [OK]   Copied '$($File.Name)' - $DestFolder -ForegroundColor Green
                $CopyCount++
            } catch {
                Write-Host   [ERR]  Failed to copy '$($File.Name)' $_ -ForegroundColor Red
                $ErrorCount++
            }
        }
    }
}

# --- Summary ---
Write-Host 
Write-Host ============================================================ -ForegroundColor Cyan
Write-Host   Done.  Copied $CopyCount    Skipped $SkipCount    Errors $ErrorCount
Write-Host ============================================================ -ForegroundColor Cyan