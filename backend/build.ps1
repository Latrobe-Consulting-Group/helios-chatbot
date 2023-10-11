# PowerShell script to automate Docker image build and push to Azure Container Registry

# Step 1: Login to Azure
az login

# Step 2: Set Azure Subscription (Replace with your Azure Subscription ID)
$subscriptionId = "ca70215a-6390-42f8-a20d-6b566a6f6eb2"
az account set --subscription $subscriptionId

# Step 3: Login to Azure Container Registry
$registryName = "baronbot"
az acr login --name $registryName

# Step 4: Build Docker image
$tagName = "$registryName.azurecr.io/axebot:latest"
docker build -t $tagName .

# Step 5: Push image to Azure Container Registry
docker push $tagName

Write-Host "Image has been pushed to $tagName"