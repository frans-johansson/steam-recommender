# Editing the page
The content of the page is given by the *./index.md* file in this directory. Write your parts of the report in this file using *Markdown* syntax as presented on [this page](https://guides.github.com/features/mastering-markdown/). Images and other resources should all go in the *./assets* directory.

# Local testing
This part will outline the steps to preview changes to the page in a local testing environment before pushing to GitHub.

## 1. Install Ruby
If you are working in a Linux environment, this should be fairly straightforward. Either install Ruby using your distribution's package manager (e.g. `apt` for Ubuntu-based distributions), or install it via the [Ruby Version Manager](https://rvm.io/) `RVM` tool. The latter is preferred as `RVM` lets you specify and manage multiple versions of Ruby. For this to work you will as of the time of writing require version 2.7.1 of Ruby. You can check your current version using `ruby -v`.

## 2. Install Bundler
With Ruby installed, you should also automatically have the `gem` package manager for Ruby installed. Verify this by running `gem -v`, then install Bundler with `gem install bundler`.

## 3. Update package dependencies
The Gemfile in this directory should specify the dependencies you require to properly run the local testing environment. To make sure you have it all installed, run `bundle update` in this directory (if you get error messages about missing gemfiles, you are most likely not in the correct directory).

## 4. Run the environment
With all these steps completed, you should just be able to run `bundle exec jekyll serve` to start a local server hosting the page. Here you can play around with changes **before** you push them to GitHub.

