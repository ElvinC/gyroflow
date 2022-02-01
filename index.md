---
title: "Redirecting to new URL..."
layout: splash
date: 2016-03-23T11:48:41-04:00
header:
  overlay_color: "#000"
  overlay_filter: "0.5"
  overlay_image: assets/images/banner.jpg
  actions:
    - label: "Git Repo"
      url: "https://github.com/ElvinC/gyroflow"
    - label: "Download"
      url: "/download/"
      class: "btn--primary"
excerpt: "Go to gyroflow.xyz"
intro: 
  - excerpt: "Ever wanted to be smooth like [insert favourite FPV pilot] without the skills or budget? This tool probably won't do that, but it might do something..."
feature_row:
  - image_path: /assets/images/ui_picture.jpg
    alt: "placeholder image 1"
    title: "Awful user interface"
    excerpt: "Designed by someone with absolutely no UX design experience. Maybe it's awful, maybe it isn't, idk."
  - image_path: /assets/images/sync_picture.jpg
    alt: "placeholder image 2"
    title: "Alright stabilization"
    excerpt: "Might've been almost state of the art back in 2011, still way behind other (commercial) implementations when it comes to how commercial it is."
    # url: "#test-link"
    # btn_label: "Read More"
    # btn_class: "btn--primary"
  - image_path: /assets/images/blackbox_picture.jpg
    title: "Arduous workflow"
    excerpt: '"Why spend so much time and effort with logging, calibration, and messing around with sync when I can just run it through warp stabilizer?" - Potential user'
---


{% include feature_row id="intro" type="center" %}

<iframe width="560" height="315" src="https://www.youtube.com/embed/f4YD5pGmnxM" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<br>
{% include feature_row %}
