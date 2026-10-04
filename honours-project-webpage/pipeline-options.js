const loopContent = {
  generate: {
    step: 'Stage 01', title: 'Generate a structured animation',
    summary: 'Specialised AI agents collaborate to write the script, plan the animation, generate code and coordinate revisions.',
    detail: ['Script Writer', 'Animation Planner', 'Coder agents', 'Orchestrator'],
    exampleLabel: 'Output', example: 'Editable Manim code, a rendered video and a complete execution trace.'
  },
  learn: {
    step: 'Stage 02', title: 'Convert experience into candidate beliefs',
    summary: 'The framework analyses plans, tool calls, errors, repairs, visual reviews and evaluation results from completed topics.',
    detail: ['Discover candidates', 'Merge overlap', 'Revise', 'Split', 'Exclude or retain'],
    exampleLabel: 'Key distinction', example: 'The system extracts general guidance rather than memorising a topic-specific answer.'
  },
  store: {
    step: 'Stage 03', title: 'Retain evidence-backed guidance',
    summary: 'Supporting and contradicting observations update each belief’s estimated effectiveness and eligibility status.',
    detail: ['Supporting evidence', 'Contradictions', 'Effectiveness', 'Scope', 'Status'],
    exampleLabel: 'Example belief', example: 'Use .animate or an explicit Animation object when changing a mobject inside Scene.play().' 
  },
  select: {
    step: 'Stage 04', title: 'Select beliefs for the current situation',
    summary: 'Eligible beliefs are ranked using the current workflow stage, problem similarity, contextual conditions and exact runtime errors.',
    detail: ['Eligibility', 'Stage match', 'Problem match', 'Context match', 'Exact error'],
    exampleLabel: 'Purpose', example: 'Only the most applicable guidance is passed into the limited prompt context.'
  },
  apply: {
    step: 'Stage 05', title: 'Guide a subsequent generation run',
    summary: 'Selected guidance is injected broadly, into coder agents, or reactively after a concrete runtime failure.',
    detail: ['Broad', 'Coder-wide', 'Reactive repair'],
    exampleLabel: 'Experimental finding', example: 'Reactive repair had the most favourable efficiency profile and reduced token consumption.'
  }
};

const loopButtons = document.querySelectorAll('[data-stage]');
loopButtons.forEach((button) => button.addEventListener('click', () => {
  const data = loopContent[button.dataset.stage];
  loopButtons.forEach((item) => { item.classList.remove('active'); item.setAttribute('aria-pressed', 'false'); });
  button.classList.add('active'); button.setAttribute('aria-pressed', 'true');
  document.querySelector('#loop-step').textContent = data.step;
  document.querySelector('#loop-title').textContent = data.title;
  document.querySelector('#loop-summary').textContent = data.summary;
  document.querySelector('#loop-detail').innerHTML = data.detail.map((item) => `<span>${item}</span>`).join('');
  document.querySelector('#loop-example').innerHTML = `<strong>${data.exampleLabel}</strong><p>${data.example}</p>`;
}));

const linearContent = {
  generate: ['Stage 01 · Generation', 'A team of agents creates the animation', 'The Script Writer, Animation Planner, Coder agents and Orchestrator cooperate across multiple turns, producing an editable video and a structured record of their work.', ['Script', 'Storyboard', 'Python code', 'Rendered video']],
  observe: ['Stage 02 · Observation', 'The complete execution process becomes data', 'Tool calls, agent messages, code versions, runtime diagnostics, repair attempts, visual reviews and final evaluation scores are retained as execution artefacts.', ['Agent messages', 'Errors', 'Repairs', 'Evaluations']],
  learn: ['Stage 03 · Belief learning', 'Specific experiences become general guidance', 'Candidate lessons are discovered topic by topic, consolidated globally and evaluated against retrospective evidence before entering the belief library.', ['Discover', 'Merge', 'Revise', 'Evaluate evidence']],
  select: ['Stage 04 · Selection', 'Relevant guidance is retrieved', 'Beliefs are filtered by status and agent role, scored for contextual applicability and ranked so that only a small relevant set is selected.', ['Eligibility', 'Applicability', 'Ranking', 'Top-k']],
  inject: ['Stage 05 · Injection', 'Selected beliefs guide the next execution', 'Guidance can enter every agent, only code-producing agents, or reactively after a runtime failure. The new execution then creates further evidence.', ['Broad', 'Coder-wide', 'Reactive repair']]
};

const linearButtons = document.querySelectorAll('[data-linear]');
linearButtons.forEach((button) => button.addEventListener('click', () => {
  const [kicker, title, copy, tags] = linearContent[button.dataset.linear];
  linearButtons.forEach((item) => { item.classList.remove('active'); item.setAttribute('aria-selected', 'false'); });
  button.classList.add('active'); button.setAttribute('aria-selected', 'true');
  document.querySelector('#linear-kicker').textContent = kicker;
  document.querySelector('#linear-title').textContent = title;
  document.querySelector('#linear-copy').textContent = copy;
  document.querySelector('#linear-tags').innerHTML = tags.map((tag) => `<span>${tag}</span>`).join('');
}));

const storyContent = {
  failure: {
    file: 'runtime.log', kicker: 'Observed experience', title: 'A concrete failure creates useful evidence',
    copy: 'The system records the error, the surrounding code and the agent’s subsequent debugging attempt instead of discarding them when the run ends.',
    visual: '<div class="terminal-view"><span class="terminal-muted">Traceback (most recent call last):</span><strong>TypeError</strong><code>Passing Mobject to Scene.play is not supported.</code></div>'
  },
  repair: {
    file: 'section_04.py', kicker: 'Successful repair', title: 'The trace preserves what changed',
    copy: 'A later code version is linked to the resolved runtime outcome, giving the framework evidence about which strategy was useful.',
    visual: '<div class="repair-view"><code class="bad">self.play(object.set_color(WHITE))</code><code class="good">self.play(object.animate.set_color(WHITE))</code></div>'
  },
  belief: {
    file: 'belief_library.json', kicker: 'Generalisation', title: 'The repair becomes reusable guidance',
    copy: 'The consolidated belief expresses a strategy that can apply beyond the original animation or topic, while retaining its evidence history.',
    visual: '<div class="belief-view"><span>Evidence-backed belief</span><blockquote>“Use <code>.animate</code> or an explicit Animation object when changing a mobject inside <code>Scene.play()</code>.”</blockquote><small>Supported by successful runtime repair</small></div>'
  },
  reuse: {
    file: 'future_run.log', kicker: 'Contextual reuse', title: 'A later matching failure retrieves the belief',
    copy: 'The system recognises a relevant runtime condition, selects one specialised belief and gives it to the coder responsible for repair.',
    visual: '<div class="reuse-view"><div class="reuse-diagram"><div>New<br><b>TypeError</b></div><span>→</span><div class="active">Relevant belief<br><b>selected</b></div><span>→</span><div>Error<br><b>resolved</b></div></div></div>'
  }
};

const storyButtons = document.querySelectorAll('[data-story]');
storyButtons.forEach((button) => button.addEventListener('click', () => {
  const data = storyContent[button.dataset.story];
  storyButtons.forEach((item) => { item.classList.remove('active'); item.setAttribute('aria-selected', 'false'); });
  button.classList.add('active'); button.setAttribute('aria-selected', 'true');
  document.querySelector('#story-file').textContent = data.file;
  document.querySelector('#story-kicker').textContent = data.kicker;
  document.querySelector('#story-title').textContent = data.title;
  document.querySelector('#story-copy').textContent = data.copy;
  document.querySelector('#story-visual').outerHTML = `<div id="story-visual">${data.visual}</div>`;
}));
