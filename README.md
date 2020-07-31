An enhanced template for a C# class library project

## Features

- class lib + test projects
- solution file referencing common stuff in repository: gitignore, license, readme, CI
- Azure DevOps-based build + test pipeline
- tests: DevOps results and coverage integration
- builds NuGet package with
[Source Link](https://docs.microsoft.com/en-us/dotnet/standard/library-guidance/sourcelink) support
- builds symbols package

## Setting up

- clone this repository into the folder you want your new lib to be
- enter working copy directory
- `git branch --unset-upstream` to detach `master` branch from template; now it belongs to **your** project
- `git remote rename origin template` to preserve the ability to pull template updates
- create a new repository on GitHub/GitLab/etc
- `git remote add origin https://full.url/to/your_new.git`
- `git push --set-upstream origin master` to upload your new project
- go to Azure DevOps and create a new pipeline for your new project, and point it to `CI/Azure-Master.yml`
- remove the sample classes and tests, rename the projects (if needed), and start hacking!

## Updating your project to the latest version of the template

- add the template repository to remotes: `git remote add template https://full.url/to/this_project.git`
- `git pull template master`

## Planned/missing features

- run tests on all platforms
- publish preview versions of NuGet package on successful build
- README status badges
