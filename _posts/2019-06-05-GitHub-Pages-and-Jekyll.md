---
typora-copy-images-to: ../
typora-root-url: ../
categories: [web, ruby]
tags: [jekyll, ruby, web]
---

# GitHub Pages and Jekyll
### May 28, 2019

#### In this blog, I was trying to build own blog website using Jekyll which it hosted by GitHub Pages.

1. First of all, we required to create a folder.
2. start git project - type: "Git init"
3. Make a index file - type: "touch index.html"
4. git add .
5. git commit - m 'int'
6. git status

Create new repositoy on your github account

![9b6f9439](/9b6f9439-9726486.png)

push your repository







We have done all the basic steps to putting our code up on to the github account

next thing is I will create new branch called gh-page

type:
- git branch gh-pages
- git branch

In order to have own domain name, we need to follow couple of more steps.

Type:
- touch CNAME
- nano 
- type the domain name
- git add .
- git commit -m 'cname'
- git checkout gh-pages
- git branch
- git merge master
- git push

#### Time to use Jekyll

type :
- sudo gem install jekyll
- jekyll new blog
- cd blog
- jekyll serve


push your work on your github


