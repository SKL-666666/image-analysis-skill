param(
    [Parameter(Mandatory = $true)][string]$ImagePath,
    [string]$OutFile = ""
)

$ErrorActionPreference = 'Stop'

# Load WinRT types (built-in Windows 10/11 OCR engine)
[void][Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
[void][Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics, ContentType = WindowsRuntime]
[void][Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime]
[void][Windows.Globalization.Language, Windows.Globalization, ContentType = WindowsRuntime]

# Convert WinRT IAsyncOperation to .NET Task
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
})[0]

function Await($WinRtTask, $ResultType) {
    $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)
    $netTask = $asTask.Invoke($null, @($WinRtTask))
    $netTask.Wait(-1) | Out-Null
    $netTask.Result
}

try {
    $file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($ImagePath)) ([Windows.Storage.StorageFile])
    $stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
    $decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
    $bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
} catch {
    Write-Error ("Cannot read image: " + $_.Exception.Message)
    exit 1
}

$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
if ($null -eq $engine) {
    # Fallback: force Simplified Chinese
    $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage((New-Object Windows.Globalization.Language('zh-Hans-CN')))
}
if ($null -eq $engine) {
    Write-Error "No OCR language pack available (install one in Settings - Time & Language - Language)"
    exit 2
}

$result = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])

$words = @()
foreach ($line in $result.Lines) {
    foreach ($word in $line.Words) {
        $r = $word.BoundingRect
        $words += [PSCustomObject]@{
            text = $word.Text
            x    = [math]::Round($r.X)
            y    = [math]::Round($r.Y)
            w    = [math]::Round($r.Width)
            h    = [math]::Round($r.Height)
        }
    }
}

$out = [PSCustomObject]@{ words = $words }
if ($OutFile) {
    $out | ConvertTo-Json -Depth 5 -Compress | Out-File -FilePath $OutFile -Encoding utf8
} else {
    $out | ConvertTo-Json -Depth 5 -Compress
}
