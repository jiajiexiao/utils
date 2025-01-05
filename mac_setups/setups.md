# Config a new Mac 

This is a documentation for setups of a new Mac. 


## Terminal
For simplicity, I just use the system's Terminal as my terminal emulator with
the following settings for colors. 

```zsh
echo 'export CLICOLOR=1' >> /Users/$USER/.zprofile
echo 'export LSCOLORS=ExFxBxDxCxegedabagacad' >> /Users/$USER/.zprofile
```

Note that Zsh has been set as the default shell on macOS as of Catalina
(replacing Bash). One may also use [iTerm2](https://www.iterm2.com/) or [Hyper](https://hyper.is/). 

## [Brew](https://brew.sh/)
Brew is a package manager for macOS. It makes it easy to install and manage
software on your computer. One can install it with the following command:

```zsh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

or trying the latest .pkg file in the Assets section from the brew's release
page on GitHub [https://github.com/Homebrew/brew/releases/](https://github.com/Homebrew/brew/releases). 


At the end of the installation, remember to add Homebrew to your PATH by adding
to your shell profile (e.g. `~/.bash_profile` or `~/.zprofile`):

```zsh
echo 'eval $(/opt/homebrew/bin/brew shellenv)' >> /Users/$USER/.zprofile
eval $(/opt/homebrew/bin/brew shellenv)
```

To validate that it is working, run `brew --version` or `brew --help` in the terminal.

## [Git](https://git-scm.com/)
Git is a distributed version control system. It is used to manage source code
and track changes in a project. To use Git, you need to install it on your
computer. If you have installed Homebrew, you should have Git installed already.
If not, you can try to install Git with the following command: `brew install git`. 

After that, you will need to config your name and email in Git. They will be
used to identify you when you commit changes to the repository. You can do it
with the following command: 

```zsh
git config --global user.name "Your Name"
git config --global user.email "Your Email".
```

Note if you have email privacy enabled on GitHub, you will need to use a
different email address when committing changes to the repository. Otherwise,
Git may not allow you to push your changes to the remote repository onGitHub as
your email address will be visible to the public. You can find a corresponding
alternative email address on [GitHub's settings email setting
page](https://github.com/settings/emails) that you can use to commit changes to
your repository. Such an email address should be ending with
`@users.noreply.github.com`. Your activities will be recorded as `Your Name <Your Email>` 
in the commit history and will be recognized by GitHub's activity overview. 
