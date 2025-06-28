<h1>Gaming Chatbot & Player Profiler</h1>

A compact Python toolkit for analysing gamers behavioural data and chatting with them through locally‑run Llama 2 models. The project demonstrates end‑to‑end ML: clustering, classification, sentiment analysis and an interactive recommendation bot. Using the K-means and RandomForest pipeline for optimal prediction capabilities.

<h2>Key Features</h2>
<ol>
  <li>Data profiling &mdash; K-Means clustering (<code>TrainProfiler.py</code>) categorizes players into six intuitive segments.</li>

  <li>Engagement prediction &mdash; Random-Forest classifier (<code>RFTrainBot.py</code>) forecasts each player's engagement level.</li>

  <li><strong>Three chatbots</strong>
    <ol>
      <li><strong>Heavy</strong> &mdash; fully automated ML + Llama; uses regex to extract user features (<code>Chatbot_Heavy_Model.py</code>).</li>
      <li><strong>Light</strong> &mdash; semi-automated, keyword-triggered ML for user-feature extraction (<code>Chatbot_Lighter_Model.py</code>).</li>
      <li><strong>Stupid</strong> &mdash; rule-based fallback with canned responses (<code>Chatbot_Stupid.py</code>).</li>
    </ol>
  </li>

  <li>Sentiment and keyword tracking with NLTK.</li>

  <li>Local Llama 2 inference via <em>llama-cpp-python</em> (no external API keys required).</li>
</ol>

<h2>REQUIREMENTS</h2>
In order to run the light and heavy model, you must first install Visual Studio BuildTools for desktop. 
https://visualstudio.microsoft.com/visual-cpp-build-tools/
Under "Workloads" after installing the BuildTools Choose "Desktop Development with C++" and wait for it to install the packages.

Now you should be able to run it.

<h2>Customising the Models</h2>
Wish to try my models?

You can try editing the <strong>n_clusters</strong> in TrainProfiler.py.
There's quite a few other settings to mess around with aswell. Watch out for the context window, the max is 4096, setting it higher results in crashing the program.
