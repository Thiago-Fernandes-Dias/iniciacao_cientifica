Get-ChildItem -Filter "*.py" | ForEach-Object {
    Write-Output "Executing $_"
    python $_.FullName
}