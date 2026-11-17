---
permalink: /
title: ""
author_profile: false
classes:
  - home
redirect_from: 
  - /about/
  - /about.html
---

{% include base_path %}

<div class="about">
  <div class="about__photo">
    <img src="{{ base_path }}/images/headshot.jpg" alt="Jeremy Lu">
    <div class="about__contact">
      <div class="about__email">lu1008 (at) purdue (dot) edu</div>
      <div class="about__icons">
        <a href="https://scholar.google.com/citations?user=1lFsfhkAAAAJ&hl=en" title="Google Scholar" aria-label="Google Scholar"><i class="ai ai-google-scholar"></i></a>
        <a href="https://github.com/lu-jeremy" title="GitHub" aria-label="GitHub"><i class="fab fa-github"></i></a>
        <a href="https://www.linkedin.com/in/lu-jeremy" title="LinkedIn" aria-label="LinkedIn"><i class="fab fa-linkedin"></i></a>
      </div>
    </div>
  </div>

  <h1 class="about__name"><b>Jeremy</b> Lu</h1>

  <p>I am a final-year undergraduate student at Purdue University in Computer Science Honors, Statistics, and Mathematics, supervised by Professor <a href="https://ruqizhang.github.io/">Ruqi Zhang</a>.</p>

  <p>My research focuses on probabilistic machine learning (e.g., variational inference, Langevin dynamics, measure transport) and, currently, diffusion language models (see our paper on <a href="https://arxiv.org/abs/2607.10951">Sticky Jump Diffusions</a>, NeurIPS 2026). More recently, I have been interested in dynamical systems for recursive reasoning.</p>

  <p>Previously, I completed internships at Amazon Artificial General Intelligence (AGI) in Summer 2025 and 2026 and worked on multi-agent planning in robotics at Purdue's <a href="https://corallab.net/">CoRAL Lab</a>.</p>
</div>

<h2>News</h2>
<table class="news">
  <tr><th>Sep 2026</th><td><em>Sticky Jump Diffusions</em> was accepted at NeurIPS 2026!</td></tr>
  <tr><th>Jul 2026</th><td>Our preprint <a href="https://arxiv.org/abs/2607.10951"><em>Sticky Jump Diffusions</em></a> is out on arXiv.</td></tr>
  <tr><th>May 2026</th><td>Started another internship at Amazon AGI in Sunnyvale, CA.</td></tr>
  <tr><th>Jan 2026</th><td><em>Graphs-of-Constraints MPC</em> was accepted at ICRA 2026!</td></tr>
  <tr><th>May 2025</th><td>Started an internship at Amazon AGI in Sunnyvale, CA.</td></tr>
</table>

<h2>Publications</h2>
{% assign pubs = site.publications | sort: "date" | reverse %}
<div class="publications">
{% for post in pubs %}
  {% include pub-entry.html pub=post %}
{% endfor %}
</div>
