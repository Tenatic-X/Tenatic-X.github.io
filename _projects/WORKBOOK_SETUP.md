---
layout: project
date: 2002-09-19
category: 
---

&#123;% raw %&#125;
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
</head>

<body>

<h1>HTML is a Bitch — Quick Setup Guide</h1>

<p>Use this guide whenever you add exported HTML notebooks to your site.</p>

<h2>1. Copy this front matter block:</h2>

<pre><code>---
layout: project
date: yyyy-mm-dd
category: 
---
</code></pre>

<p><strong>date</strong> → use format <code>yyyy-mm-dd</code><br>
<strong>IMPORTANT:</strong> Do <em>not</em> use today’s date.  
Because NZ is ahead of the USA, GitHub Pages may treat it as <em>future dated</em> and skip the file when <code>future: false</code> is set (default).</p>

<p><strong>category</strong> → choose one:<br>
<code>personal</code>, <code>study</code>, <code>homework</code></p>

<h2>2. Add this immediately after the front matter:</h2>

<p>An example of what it looks like</p>

<pre><code>{% raw %}Bunch of shitty code{% endraw %}</code></pre>

<p>Yes — it looks doubled here because we must escape Liquid for the guide to render.  
When you paste it into your actual file, you want:</p>

<pre><code>{% raw %}</code></pre>

<h2>3. Add this at the VERY end of your HTML file:</h2>

<pre><code>{% endraw %}</code></pre>

<p>This prevents Jekyll from interpreting anything inside the HTML file  
(like <code>{{ }}</code> or <code>{% %}</code> scripts) which will otherwise break the build.</p>

</body>
</html>
&#123;% endraw %&#125;
