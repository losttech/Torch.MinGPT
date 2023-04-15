param ([Parameter(Mandatory=$true)] $Name)

git clone https://github.com/losttech/Templates.ClassLib.git "$Name"
cd "$Name"
git remote rename origin template
git branch --unset-upstream

Move-Item Templates.ClassLib.sln "$Name.sln"
Move-Item src/ClassLib.csproj "src/$Name.csproj"
Move-Item test/ClassLib.Tests.csproj "test/$Name.Tests.csproj"

Remove-Item README.md
Move-Item README.template.md README.md

function Replace-InFile($path, $what, $with){
    $content = Get-Content "$path" -Raw
    $updated = $content -replace $what,$with
    Set-Content -Path "$path" $updated
}

Replace-InFile "$Name.sln" "ClassLib" "$Name"
Replace-InFile "test/$Name.Tests.csproj" "ClassLib" "$Name"
Replace-InFile "README.md" "ClassLib" "$Name"


Remove-Item New-ClassLib.ps1
