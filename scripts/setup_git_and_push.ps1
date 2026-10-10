# ==============================================================================
# Script de Automatización: Inicialización de Git y Publicación en GitHub
# Maestría en Ciencia de Datos e Inteligencia Artificial - UDI
# ==============================================================================

param(
    [string]$GitHubUser = "",
    [string]$RepoName = "U3-ProgamacionIA"
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  CONFIGURACIÓN DE GIT Y SUBIDA A GITHUB (PROYECTO UDI)  " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Verificar si git está instalado
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "[!] Git no fue detectado en el PATH del sistema." -ForegroundColor Yellow
    Write-Host "[*] Puede instalarlo automáticamente ejecutando:" -ForegroundColor White
    Write-Host "    winget install --id Git.Git -e --source winget" -ForegroundColor Green
    Write-Host "    (O descargarlo desde https://git-scm.com/download/win)" -ForegroundColor White
    Write-Host "Una vez instalado Git, vuelva a ejecutar este script." -ForegroundColor Yellow
    exit 1
}

# 2. Inicializar repositorio local si no existe
if (-not (Test-Path ".git")) {
    Write-Host "[*] Inicializando repositorio Git local..." -ForegroundColor Cyan
    git init
    git branch -M main
}

# 3. Añadir archivos y realizar commit inicial
Write-Host "[*] Agregando archivos del proyecto..." -ForegroundColor Cyan
git add .
git commit -m "feat: complete medical AI pipeline with OOP refactoring, Gradio CDSS dashboard, unit tests and academic PDF report"

# 4. Configurar remoto y subir si se especificó usuario
if ($GitHubUser -ne "") {
    $remoteUrl = "https://github.com/$GitHubUser/$RepoName.git"
    Write-Host "[*] Configurando origen remoto: $remoteUrl" -ForegroundColor Cyan
    git remote remove origin -ErrorAction SilentlyContinue
    git remote add origin $remoteUrl
    Write-Host "[*] Subiendo rama main a GitHub..." -ForegroundColor Cyan
    git push -u origin main
    Write-Host "[OK] Repositorio publicado exitosamente en: $remoteUrl" -ForegroundColor Green
} else {
    Write-Host "[*] Commit local creado con éxito." -ForegroundColor Green
    Write-Host "[*] Para subirlo a su repositorio de GitHub, cree un repo vacío en GitHub y ejecute:" -ForegroundColor White
    Write-Host "    git remote add origin https://github.com/SU_USUARIO/$RepoName.git" -ForegroundColor Yellow
    Write-Host "    git push -u origin main" -ForegroundColor Yellow
}

