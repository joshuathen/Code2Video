const menuButton = document.querySelector('.menu-button');
const navigation = document.querySelector('#site-nav');

menuButton?.addEventListener('click', () => {
  const isOpen = menuButton.getAttribute('aria-expanded') === 'true';
  menuButton.setAttribute('aria-expanded', String(!isOpen));
  navigation.classList.toggle('open', !isOpen);
});

navigation?.querySelectorAll('a').forEach((link) => {
  link.addEventListener('click', () => {
    menuButton?.setAttribute('aria-expanded', 'false');
    navigation.classList.remove('open');
  });
});

const animatedValues = document.querySelectorAll('.count');
const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

if (!prefersReducedMotion && 'IntersectionObserver' in window) {
  const observer = new IntersectionObserver((entries, currentObserver) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      const element = entry.target;
      const target = Number(element.dataset.count);
      const start = performance.now();
      const duration = 700;

      const tick = (now) => {
        const progress = Math.min((now - start) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        element.textContent = (target * eased).toFixed(1);
        if (progress < 1) requestAnimationFrame(tick);
      };

      requestAnimationFrame(tick);
      currentObserver.unobserve(element);
    });
  }, { threshold: 0.5 });

  animatedValues.forEach((value) => observer.observe(value));
}

// Tell the memory comparison as a strict sequence. Only one action is active at
// a time, making the difference between forgetting and retaining experience
// readable without asking the visitor to decode several animations at once.
const memoryComparison = document.querySelector('.comparison-card');
if (memoryComparison && !prefersReducedMotion) {
  const memoryStory = [
    { action: 'left-solve' },
    { action: 'left-finish' },
    { action: 'left-forget' },
    { action: 'left-repeat' },
    { action: 'right-solve' },
    { action: 'right-extract' },
    { action: 'right-store' },
    { action: 'right-retrieve' },
    { action: 'right-inject' },
    { action: 'right-improve' }
  ];
  const memoryActions = [...memoryComparison.querySelectorAll('[data-memory-action]')];
  const actionOrder = new Map(memoryStory.map((step, index) => [step.action, index]));
  const stepDuration = 1450;
  const finalPause = 1700;
  let storyTimeout;
  let storyRunning = false;

  const resetMemoryStory = () => {
    clearTimeout(storyTimeout);
    storyRunning = false;
    memoryComparison.classList.remove('story-playing', 'story-right-side');
    memoryComparison.querySelectorAll('.story-active, .story-complete, .story-current, .story-past, .story-side-active')
      .forEach((element) => element.classList.remove('story-active', 'story-complete', 'story-current', 'story-past', 'story-side-active'));
  };

  const showMemoryStep = (index) => {
    const step = memoryStory[index];
    const activeAction = memoryActions.find((element) => element.dataset.memoryAction === step.action);

    memoryActions.forEach((element) => {
      const elementIndex = actionOrder.get(element.dataset.memoryAction);
      element.classList.toggle('story-active', elementIndex === index);
      element.classList.toggle('story-complete', elementIndex < index);
    });

    memoryComparison.querySelectorAll('.comparison-run').forEach((run) => run.classList.remove('story-current', 'story-past'));
    memoryComparison.querySelectorAll('.comparison-side').forEach((side) => side.classList.remove('story-side-active'));
    activeAction?.closest('.comparison-run')?.classList.add('story-current');
    activeAction?.closest('.comparison-side')?.classList.add('story-side-active');
    if (index >= actionOrder.get('left-repeat')) {
      memoryComparison.querySelector('.comparison-side.faded .run-one')?.classList.add('story-past');
    }
    if (index >= actionOrder.get('right-retrieve')) {
      memoryComparison.querySelector('.comparison-side.active .run-one')?.classList.add('story-past');
    }
    memoryComparison.classList.toggle('story-right-side', step.action.startsWith('right-'));

    const nextIndex = (index + 1) % memoryStory.length;
    const delay = index === memoryStory.length - 1 ? finalPause : stepDuration;
    storyTimeout = setTimeout(() => showMemoryStep(nextIndex), delay);
  };

  const startMemoryStory = () => {
    if (storyRunning) return;
    resetMemoryStory();
    storyRunning = true;
    memoryComparison.classList.add('story-playing');
    showMemoryStep(0);
  };

  if ('IntersectionObserver' in window) {
    const storyObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting && entry.intersectionRatio >= .35) startMemoryStory();
        else if (!entry.isIntersecting || entry.intersectionRatio < .1) resetMemoryStory();
      });
    }, { threshold: [0, .1, .35] });
    storyObserver.observe(memoryComparison);
  } else {
    startMemoryStory();
  }
}

// Cursor-following explanations for the compact belief-learning loop.
const tooltipTargets = document.querySelectorAll('[data-tooltip-title]');

if (tooltipTargets.length) {
  const tooltip = document.createElement('div');
  tooltip.className = 'cursor-tooltip';
  tooltip.setAttribute('role', 'tooltip');
  tooltip.setAttribute('aria-hidden', 'true');
  tooltip.id = 'pipeline-tooltip';
  tooltip.innerHTML = '<strong></strong><p></p><p class="tooltip-example"></p>';
  document.body.appendChild(tooltip);

  const title = tooltip.querySelector('strong');
  const copy = tooltip.querySelector('p:not(.tooltip-example)');
  const example = tooltip.querySelector('.tooltip-example');
  const coarsePointer = window.matchMedia('(pointer: coarse)').matches;

  const setContent = (target) => {
    title.textContent = target.dataset.tooltipTitle;
    copy.textContent = target.dataset.tooltipCopy;
    if (target.hasAttribute('data-open-demo')) {
      example.innerHTML = `<strong>Click to watch</strong> a generated animation and inspect the output of this stage.`;
    } else {
      example.textContent = target.dataset.tooltipExample;
    }
  };

  const placeTooltip = (x, y) => {
    const gap = 18;
    const bounds = tooltip.getBoundingClientRect();
    let left = x + gap;
    let top = y + gap;
    if (left + bounds.width > window.innerWidth - 12) left = x - bounds.width - gap;
    if (top + bounds.height > window.innerHeight - 12) top = y - bounds.height - gap;
    tooltip.style.left = `${Math.max(12, left)}px`;
    tooltip.style.top = `${Math.max(12, top)}px`;
  };

  const showTooltip = (target, x, y, touch = false) => {
    setContent(target);
    tooltip.classList.toggle('touch-visible', touch);
    tooltip.classList.add('visible');
    tooltip.setAttribute('aria-hidden', 'false');
    if (!touch) placeTooltip(x, y);
  };

  const hideTooltip = () => {
    tooltip.classList.remove('visible', 'touch-visible');
    tooltip.setAttribute('aria-hidden', 'true');
  };

  tooltipTargets.forEach((target) => {
    target.setAttribute('aria-describedby', tooltip.id);
    target.addEventListener('pointerenter', (event) => {
      if (!coarsePointer) showTooltip(target, event.clientX, event.clientY);
    });
    target.addEventListener('pointermove', (event) => {
      if (!coarsePointer && tooltip.classList.contains('visible')) placeTooltip(event.clientX, event.clientY);
    });
    target.addEventListener('pointerleave', () => { if (!coarsePointer) hideTooltip(); });
    target.addEventListener('focus', () => {
      const rect = target.getBoundingClientRect();
      showTooltip(target, rect.right, rect.top + rect.height / 2);
    });
    target.addEventListener('blur', hideTooltip);
    target.addEventListener('click', () => {
      if (coarsePointer) {
        if (target.hasAttribute('data-open-demo')) return;
        const alreadyVisible = tooltip.classList.contains('visible') && title.textContent === target.dataset.tooltipTitle;
        if (alreadyVisible) hideTooltip(); else showTooltip(target, 0, 0, true);
      }
    });
  });

  document.addEventListener('keydown', (event) => { if (event.key === 'Escape') hideTooltip(); });
}

// The Generate stage opens its real output without interrupting the page narrative.
const generationDialog = document.querySelector('#generation-demo');
const generationVideo = generationDialog?.querySelector('video');

document.querySelectorAll('[data-open-demo]').forEach((trigger) => {
  trigger.addEventListener('click', () => {
    if (!generationDialog) return;
    if (typeof generationDialog.showModal === 'function') generationDialog.showModal();
    else generationDialog.setAttribute('open', '');
    generationVideo?.play().catch(() => {});
  });
});

document.querySelector('[data-close-demo]')?.addEventListener('click', () => generationDialog?.close());
generationDialog?.addEventListener('click', (event) => {
  if (event.target === generationDialog) generationDialog.close();
});
generationDialog?.addEventListener('close', () => {
  generationVideo?.pause();
  if (generationVideo) generationVideo.currentTime = 0;
});

// Worked example: one concrete runtime repair becomes a reusable belief.
const workedStages = {
  failure: {
    file: 'runtime.log', badge: 'Failure observed', badgeClass: 'error', kicker: 'Observed experience',
    title: 'The generated program fails at runtime',
    description: 'A coder tries to pass the result of <code>set_color()</code> directly into <code>Scene.play()</code>. Manim expects an animation, so rendering stops with a concrete diagnostic.',
    takeawayTitle: 'Why retain this?', takeaway: 'The error identifies both the failed operation and the animation framework involved. This context helps evaluate the later repair.',
    visual: `<div class="failure-visual"><div class="code-lines"><span><i>214</i><code>self.play(</code></span><span class="fault-line"><i>215</i><code>self.lecture[0].set_color(WHITE)</code></span><span><i>216</i><code>)</code></span></div><div class="runtime-error"><span>TypeError</span><p>Passing Mobject to <code>Scene.play()</code> is not supported.</p></div></div>`
  },
  context: {
    file: 'execution_trace.json', badge: 'Context retained', badgeClass: 'info', kicker: 'Structured execution trace',
    title: 'The system preserves more than the error message',
    description: 'The framework links the diagnostic to the responsible agent, workflow stage, surrounding code and later outcome. These artefacts make retrospective attribution possible.',
    takeawayTitle: 'Why this matters', takeaway: 'A successful result alone cannot explain why it worked. The trace supplies the before-and-after evidence needed to assess a candidate lesson.',
    visual: `<div class="context-visual"><div class="artefact-card"><span>Diagnostic</span><strong>TypeError in Scene.play()</strong><small>Concrete exception family and API</small></div><div class="artefact-card"><span>Responsible role</span><strong>Coder agent</strong><small>Runtime-repair stage</small></div><div class="artefact-card"><span>Before state</span><strong>Failing code retained</strong><small>Operation and object recorded</small></div><div class="artefact-card"><span>Outcome</span><strong>Subsequent render succeeded</strong><small>Repair can be evaluated</small></div></div>`
  },
  repair: {
    file: 'section_04.py · diff', badge: 'Repair successful', badgeClass: 'success', kicker: 'Observed improvement',
    title: 'One syntax change resolves the failure',
    description: 'The coder changes the direct state mutation into Manim’s animation-building syntax. The following render succeeds, providing outcome evidence for the strategy.',
    takeawayTitle: 'Evidence, not assumption', takeaway: 'The repair is associated with a directly observed transition from runtime failure to successful rendering.',
    visual: `<div class="repair-visual"><div class="repair-code before"><span>Before · failed</span><code>self.play(object.set_color(WHITE))</code></div><div class="repair-arrow">→</div><div class="repair-code after"><span>After · rendered</span><code>self.play(object.animate.set_color(WHITE))</code></div></div>`
  },
  belief: {
    file: 'belief_library.json', badge: 'Belief retained', badgeClass: 'belief', kicker: 'Evidence-backed generalisation',
    title: 'The specific repair becomes a reusable instruction',
    description: 'Candidate discovery proposes a lesson, consolidation expresses it atomically, and retrospective evaluation records how strongly the available evidence supports it.',
    takeawayTitle: 'Not a copied solution', takeaway: 'The final belief describes a general Manim strategy that can apply to different objects, animations and educational topics.',
    visual: `<div class="belief-visual"><div class="belief-document"><span>Reusable belief</span><blockquote>“Use <code>.animate</code> or an explicit Animation object when changing a mobject inside <code>Scene.play()</code>.”</blockquote><div class="belief-meta"><span>Role · Coder</span><span>Stage · Runtime repair</span><span>Evidence retained</span><span>Effectiveness updated</span></div></div></div>`
  },
  reuse: {
    file: 'future_generation.log', badge: 'Guidance applied', badgeClass: 'success', kicker: 'Contextual selection',
    title: 'A later matching situation retrieves the belief',
    description: 'When a subsequent coder encounters a relevant runtime failure, the system filters the belief library, scores applicability and injects the highest-ranked specialised belief.',
    takeawayTitle: 'The learning loop closes', takeaway: 'The belief guides a new run, while the new outcome becomes additional evidence that may support or contradict it.',
    visual: `<div class="reuse-visual"><div class="reuse-card"><span>Current situation</span><strong>Scene.play() TypeError</strong></div><div class="reuse-link">→</div><div class="reuse-card selected"><span>Selection</span><strong>Relevant belief ranked first</strong></div><div class="reuse-link">→</div><div class="reuse-card resolved"><span>Outcome</span><strong>Targeted repair applied</strong></div></div>`
  }
};

const workedButtons = [...document.querySelectorAll('[data-worked-stage]')];
const workedStageDuration = 6500;
const workedContentFadeOutDuration = 320;
let workedStageTimer;
let workedTransitionTimer;
let workedHeightTimer;
let workedExampleRunning = false;

const showWorkedStage = (button, restartProgress = true) => {
  const stage = workedStages[button.dataset.workedStage];
  workedButtons.forEach((item) => {
    item.classList.remove('active', 'timer-running');
    item.setAttribute('aria-pressed', 'false');
  });
  button.classList.add('active');
  button.setAttribute('aria-pressed', 'true');

  const workedDisplay = document.querySelector('.worked-display');
  const updateWorkedContent = () => {
    const previousHeight = workedDisplay?.offsetHeight;
    document.querySelector('#worked-file').textContent = stage.file;
    const badge = document.querySelector('#worked-badge');
    badge.textContent = stage.badge;
    badge.className = `worked-badge ${stage.badgeClass}`;
    document.querySelector('#worked-kicker').textContent = stage.kicker;
    document.querySelector('#worked-title').textContent = stage.title;
    document.querySelector('#worked-description').innerHTML = stage.description;
    document.querySelector('#worked-takeaway').innerHTML = `<strong>${stage.takeawayTitle}</strong><p>${stage.takeaway}</p>`;
    document.querySelector('#worked-visual').innerHTML = stage.visual;

    if (!prefersReducedMotion && workedDisplay && previousHeight) {
      clearTimeout(workedHeightTimer);
      workedDisplay.style.height = 'auto';
      const nextHeight = workedDisplay.offsetHeight;
      workedDisplay.style.height = `${previousHeight}px`;
      void workedDisplay.offsetHeight;
      requestAnimationFrame(() => {
        workedDisplay.style.height = `${nextHeight}px`;
      });
      workedHeightTimer = setTimeout(() => {
        workedDisplay.style.height = 'auto';
      }, 850);
    }

    requestAnimationFrame(() => {
      requestAnimationFrame(() => workedDisplay?.classList.remove('stage-changing'));
    });
  };

  clearTimeout(workedTransitionTimer);
  if (prefersReducedMotion || !workedDisplay) updateWorkedContent();
  else {
    workedDisplay.classList.add('stage-changing');
    workedTransitionTimer = setTimeout(updateWorkedContent, workedContentFadeOutDuration);
  }

  if (workedExampleRunning && restartProgress) {
    // Force a fresh progress animation even when the same stage is selected.
    void button.offsetWidth;
    button.classList.add('timer-running');
  }
};

const scheduleNextWorkedStage = () => {
  clearTimeout(workedStageTimer);
  if (!workedExampleRunning) return;
  workedStageTimer = setTimeout(() => {
    const activeIndex = workedButtons.findIndex((button) => button.classList.contains('active'));
    const nextButton = workedButtons[(activeIndex + 1) % workedButtons.length];
    showWorkedStage(nextButton);
    scheduleNextWorkedStage();
  }, workedStageDuration);
};

const startWorkedExample = (restartAtBeginning = false) => {
  if (!workedButtons.length || prefersReducedMotion) return;
  workedExampleRunning = true;
  const selectedButton = restartAtBeginning ? workedButtons[0] : workedButtons.find((button) => button.classList.contains('active')) || workedButtons[0];
  showWorkedStage(selectedButton);
  scheduleNextWorkedStage();
};

const stopWorkedExample = () => {
  workedExampleRunning = false;
  clearTimeout(workedStageTimer);
  clearTimeout(workedTransitionTimer);
  clearTimeout(workedHeightTimer);
  const workedDisplay = document.querySelector('.worked-display');
  workedDisplay?.classList.remove('stage-changing');
  if (workedDisplay) workedDisplay.style.height = 'auto';
  workedButtons.forEach((button) => button.classList.remove('timer-running'));
};

workedButtons.forEach((button) => button.addEventListener('click', () => {
  workedExampleRunning = !prefersReducedMotion;
  showWorkedStage(button);
  scheduleNextWorkedStage();
}));

const workedExampleSection = document.querySelector('.worked-example-section');
if (workedButtons.length && workedExampleSection && !prefersReducedMotion) {
  if ('IntersectionObserver' in window) {
    const workedObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting && entry.intersectionRatio >= .2 && !workedExampleRunning) startWorkedExample(true);
        else if (!entry.isIntersecting || entry.intersectionRatio < .05) stopWorkedExample();
      });
    }, { threshold: [0, .05, .2] });
    workedObserver.observe(workedExampleSection);
  } else {
    startWorkedExample(true);
  }
}

// Sticky experiment narrative: update the persistent summary as each chapter enters view.
const experimentStories = {
  general: {
    index: 0, number: 'Experiment 01', title: 'General improvement',
    question: 'Does belief-informed generation improve on a matched no-belief baseline?',
    value: '23.1k', label: 'fewer tokens per topic',
    note: 'A statistically significant 6.6% reduction after Holm correction.', theme: 'theme-general'
  },
  scope: {
    index: 1, number: 'Experiment 02', title: 'Injection scope',
    question: 'Does performance depend on where and when beliefs are introduced?',
    value: 'Reactive', label: 'most favourable efficiency profile',
    note: 'Coder-wide increased time; broad injection increased token consumption.', theme: 'theme-scope'
  },
  transfer: {
    index: 2, number: 'Experiment 03', title: 'Transferability',
    question: 'Do beliefs remain useful when applied to previously unseen topic batches?',
    value: '19.6k', label: 'fewer tokens per unseen topic',
    note: 'A statistically significant 5.6% reduction after Holm correction.', theme: 'theme-transfer'
  }
};

const experimentChapters = document.querySelectorAll('[data-experiment]');
const stickyExperiment = document.querySelector('.experiment-sticky');

if (experimentChapters.length && stickyExperiment) {
  let activeExperiment = null;
  let experimentFrameRequested = false;

  const playExperimentStory = (chapter) => {
    if (prefersReducedMotion || !chapter || chapter.classList.contains('story-playing')) return;
    chapter.classList.add('story-playing');
    stickyExperiment.classList.remove('story-updating');
    void stickyExperiment.offsetWidth;
    stickyExperiment.classList.add('story-updating');
  };

  const updateExperiment = (key) => {
    const currentChapter = document.querySelector(`.experiment-chapter[data-experiment="${key}"]`);
    if (key === activeExperiment && currentChapter?.classList.contains('active')) {
      if (currentChapter.getBoundingClientRect().top < window.innerHeight * .82) playExperimentStory(currentChapter);
      return;
    }
    activeExperiment = key;
    const experiment = experimentStories[key];
    stickyExperiment.classList.remove('theme-general', 'theme-scope', 'theme-transfer');
    stickyExperiment.classList.add(experiment.theme);
    document.querySelector('#sticky-experiment-number').textContent = experiment.number;
    document.querySelector('#sticky-experiment-title').textContent = experiment.title;
    document.querySelector('#sticky-experiment-question').textContent = experiment.question;
    document.querySelector('#sticky-experiment-result').innerHTML = `<strong>${experiment.value}</strong><span>${experiment.label}</span>`;
    document.querySelector('#sticky-experiment-note').textContent = experiment.note;
    document.querySelectorAll('.sticky-progress span').forEach((dot, index) => dot.classList.toggle('active', index <= experiment.index));
    experimentChapters.forEach((chapter) => {
      const isActive = chapter.dataset.experiment === key;
      chapter.classList.toggle('active', isActive);
      chapter.classList.remove('story-playing');
      if (isActive && chapter.getBoundingClientRect().top < window.innerHeight * .82) playExperimentStory(chapter);
    });

  };

  const selectExperimentAtActivationLine = () => {
    experimentFrameRequested = false;
    const activationLine = window.innerHeight * 0.38;
    let selectedChapter = experimentChapters[0];

    experimentChapters.forEach((chapter) => {
      if (chapter.getBoundingClientRect().top <= activationLine) selectedChapter = chapter;
    });

    updateExperiment(selectedChapter.dataset.experiment);
  };

  const requestExperimentUpdate = () => {
    if (experimentFrameRequested) return;
    experimentFrameRequested = true;
    window.requestAnimationFrame(selectExperimentAtActivationLine);
  };

  window.addEventListener('scroll', requestExperimentUpdate, { passive: true });
  window.addEventListener('resize', requestExperimentUpdate);
  selectExperimentAtActivationLine();
}

// Keep the significance-card glow active only while the section is in view.
const impactSection = document.querySelector('.impact-section');
if (impactSection && !prefersReducedMotion) {
  if ('IntersectionObserver' in window) {
    const impactObserver = new IntersectionObserver(([entry]) => {
      impactSection.classList.toggle('impact-animating', entry.isIntersecting);
    }, { threshold: .18 });
    impactObserver.observe(impactSection);
  } else {
    impactSection.classList.add('impact-animating');
  }
}

// Drive the hero orbit and stage reactions from one clock. Trigger angles are
// derived from the cards' real centres, so unequal visual spacing is preserved.
const cycleOrbit = document.querySelector('.cycle-orbit');
const orbitMarker = document.querySelector('.orbit-marker');
const guidanceForeground = document.querySelector('.guidance-foreground');
const beliefTransferMarker = document.querySelector('.belief-transfer-marker');
const beliefTransferPath = document.querySelector('.belief-transfer-path');
const selectionTransferMarker = document.querySelector('.selection-transfer-marker');
const selectionTransferPath = document.querySelector('.selection-transfer-path');
const applicationTransferMarker = document.querySelector('.application-transfer-marker');
const applicationTransferPath = document.querySelector('.application-transfer-path');
const beliefTokens = [...document.querySelectorAll('.belief-token')];
const selectionTokens = [...document.querySelectorAll('.selection-token')];
const applicationTokens = [...document.querySelectorAll('.application-token')];
const selectStage = document.querySelector('.node-select');
const selectionLiveText = document.querySelector('#selection-live-text');
const beliefLibrary = document.querySelector('.hero-loop-core');
const cycleStages = [...document.querySelectorAll('.hero-loop-node')];
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

if (cycleOrbit && orbitMarker && cycleStages.length && !reduceMotion.matches) {
  const duration = 20000;
  const centre = 220;
  const radius = 180;
  const markerRadius = 6;
  const transferDelay = 900;
  const transferDuration = 2200;
  const selectionDuration = 1900;
  let stageTriggers = [];
  let stageWindows = [];
  let stagePoints = [];
  let transferCurve;
  let selectionCurve;
  let applicationCurve;
  let cycleStart;
  const generateStageIndex = cycleStages.findIndex((stage) => stage.classList.contains('node-generate'));
  const applyStageIndex = cycleStages.findIndex((stage) => stage.classList.contains('node-apply'));

  const pointOnOrbit = (progress) => {
    const angle = -Math.PI / 2 + progress * Math.PI * 2;
    return { x: centre + radius * Math.cos(angle), y: centre + radius * Math.sin(angle) };
  };

  const markerTouchesCard = (progress, card) => {
    const point = pointOnOrbit(progress);
    const left = card.x - card.width / 2;
    const right = card.x + card.width / 2;
    const top = card.y - card.height / 2;
    const bottom = card.y + card.height / 2;
    const closestX = Math.max(left, Math.min(point.x, right));
    const closestY = Math.max(top, Math.min(point.y, bottom));
    return Math.hypot(point.x - closestX, point.y - closestY) <= markerRadius;
  };

  const calculateContactWindow = (card, centreProgress) => {
    const samples = 4096;
    const contacts = [];
    for (let index = 0; index < samples; index += 1) {
      const progress = index / samples;
      if (markerTouchesCard(progress, card)) contacts.push(progress);
    }
    if (!contacts.length) return { entry: centreProgress, exit: (centreProgress + .075) % 1 };

    const seed = contacts.reduce((best, value) => {
      const distance = Math.min(Math.abs(value - centreProgress), 1 - Math.abs(value - centreProgress));
      return distance < best.distance ? { value, distance } : best;
    }, { value: contacts[0], distance: Infinity }).value;
    const seedIndex = Math.round(seed * samples) % samples;

    let entryIndex = seedIndex;
    while (markerTouchesCard(((entryIndex - 1 + samples) % samples) / samples, card)) {
      entryIndex = (entryIndex - 1 + samples) % samples;
      if (entryIndex === seedIndex) break;
    }
    let exitIndex = seedIndex;
    while (markerTouchesCard(((exitIndex + 1) % samples) / samples, card)) {
      exitIndex = (exitIndex + 1) % samples;
      if (exitIndex === seedIndex) break;
    }

    return { entry: entryIndex / samples, exit: ((exitIndex + 1) % samples) / samples };
  };

  const calculateStageTriggers = () => {
    const orbitRect = cycleOrbit.getBoundingClientRect();
    const scaleX = 440 / orbitRect.width;
    const scaleY = 440 / orbitRect.height;

    stagePoints = cycleStages.map((stage) => {
      const rect = stage.getBoundingClientRect();
      const x = (rect.left + rect.width / 2 - orbitRect.left) * scaleX;
      const y = (rect.top + rect.height / 2 - orbitRect.top) * scaleY;
      return { x, y, width: rect.width * scaleX, height: rect.height * scaleY };
    });

    stageTriggers = stagePoints.map(({ x, y }) => {
      const angle = Math.atan2(y - centre, x - centre);
      return ((angle + Math.PI / 2) / (Math.PI * 2) + 1) % 1;
    });
    stageWindows = stagePoints.map((card, index) => calculateContactWindow(card, stageTriggers[index]));
    stageTriggers = stageWindows.map((window) => window.entry);

    const cardEdgeToward = (card, targetX, targetY) => {
      const dx = targetX - card.x;
      const dy = targetY - card.y;
      const length = Math.hypot(dx, dy) || 1;
      const unitX = dx / length;
      const unitY = dy / length;
      const edgeDistance = Math.min(
        Math.abs(unitX) > .001 ? card.width / 2 / Math.abs(unitX) : Infinity,
        Math.abs(unitY) > .001 ? card.height / 2 / Math.abs(unitY) : Infinity
      );
      return { x: card.x + unitX * edgeDistance, y: card.y + unitY * edgeDistance };
    };

    const learnIndex = cycleStages.findIndex((stage) => stage.classList.contains('node-learn'));
    const learnCard = stagePoints[learnIndex];
    const source = cardEdgeToward(learnCard, centre, centre);
    transferCurve = {
      source,
      controlOne: { x: source.x + (centre - source.x) / 3, y: source.y + (centre - source.y) / 3 },
      controlTwo: { x: source.x + (centre - source.x) * 2 / 3, y: source.y + (centre - source.y) * 2 / 3 },
      trigger: stageTriggers[learnIndex]
    };
    beliefTransferPath?.setAttribute('d', `M ${source.x} ${source.y} C ${transferCurve.controlOne.x} ${transferCurve.controlOne.y}, ${transferCurve.controlTwo.x} ${transferCurve.controlTwo.y}, ${centre} ${centre}`);

    const selectIndex = cycleStages.findIndex((stage) => stage.classList.contains('node-select'));
    const selectCard = stagePoints[selectIndex];
    const target = cardEdgeToward(selectCard, centre, centre);
    const selectX = target.x - centre;
    const selectY = target.y - centre;
    const selectLength = Math.hypot(selectX, selectY) || 1;
    const selectUnitX = selectX / selectLength;
    const selectUnitY = selectY / selectLength;
    const librarySource = { x: centre + selectUnitX * 103, y: centre + selectUnitY * 103 };
    selectionCurve = {
      source: librarySource,
      controlOne: { x: librarySource.x + (target.x - librarySource.x) / 3, y: librarySource.y + (target.y - librarySource.y) / 3 },
      controlTwo: { x: librarySource.x + (target.x - librarySource.x) * 2 / 3, y: librarySource.y + (target.y - librarySource.y) * 2 / 3 },
      target,
      trigger: stageTriggers[selectIndex]
    };
    selectionTransferPath?.setAttribute('d', `M ${librarySource.x} ${librarySource.y} C ${selectionCurve.controlOne.x} ${selectionCurve.controlOne.y}, ${selectionCurve.controlTwo.x} ${selectionCurve.controlTwo.y}, ${target.x} ${target.y}`);

    const applyIndex = cycleStages.findIndex((stage) => stage.classList.contains('node-apply'));
    const applyCard = stagePoints[applyIndex];
    const handoffSource = cardEdgeToward(selectCard, applyCard.x, applyCard.y);
    const handoffTarget = cardEdgeToward(applyCard, selectCard.x, selectCard.y);
    applicationCurve = {
      source: handoffSource,
      controlOne: { x: handoffSource.x + (handoffTarget.x - handoffSource.x) / 3, y: handoffSource.y + (handoffTarget.y - handoffSource.y) / 3 },
      controlTwo: { x: handoffSource.x + (handoffTarget.x - handoffSource.x) * 2 / 3, y: handoffSource.y + (handoffTarget.y - handoffSource.y) * 2 / 3 },
      target: handoffTarget,
      trigger: stageTriggers[applyIndex]
    };
    applicationTransferPath?.setAttribute('d', `M ${handoffSource.x} ${handoffSource.y} C ${applicationCurve.controlOne.x} ${applicationCurve.controlOne.y}, ${applicationCurve.controlTwo.x} ${applicationCurve.controlTwo.y}, ${handoffTarget.x} ${handoffTarget.y}`);
  };

  const cubicPoint = (start, controlOne, controlTwo, end, progress) => {
    const inverse = 1 - progress;
    return {
      x: inverse ** 3 * start.x + 3 * inverse ** 2 * progress * controlOne.x + 3 * inverse * progress ** 2 * controlTwo.x + progress ** 3 * end.x,
      y: inverse ** 3 * start.y + 3 * inverse ** 2 * progress * controlOne.y + 3 * inverse * progress ** 2 * controlTwo.y + progress ** 3 * end.y
    };
  };

  const cubicTangent = (start, controlOne, controlTwo, end, progress) => {
    const inverse = 1 - progress;
    return {
      x: 3 * inverse ** 2 * (controlOne.x - start.x) + 6 * inverse * progress * (controlTwo.x - controlOne.x) + 3 * progress ** 2 * (end.x - controlTwo.x),
      y: 3 * inverse ** 2 * (controlOne.y - start.y) + 6 * inverse * progress * (controlTwo.y - controlOne.y) + 3 * progress ** 2 * (end.y - controlTwo.y)
    };
  };

  const renderTokenStream = (tokens, curve, overallProgress, easing) => {
    const offsets = [0, .14, .28];
    const travelSpan = .72;
    tokens.forEach((token, index) => {
      const localProgress = (overallProgress - offsets[index]) / travelSpan;
      if (localProgress < 0 || localProgress > 1) {
        token.style.opacity = 0;
        return;
      }
      const easedProgress = easing(localProgress);
      const point = cubicPoint(curve.source, curve.controlOne, curve.controlTwo, curve.target, easedProgress);
      const tangent = cubicTangent(curve.source, curve.controlOne, curve.controlTwo, curve.target, easedProgress);
      const direction = Math.atan2(tangent.y, tangent.x) * 180 / Math.PI;
      const fade = Math.min(1, localProgress / .12, (1 - localProgress) / .13);
      token.setAttribute('transform', `translate(${point.x} ${point.y}) rotate(${direction})`);
      token.style.opacity = Math.max(0, fade);
    });
  };

  const animateCycle = (time) => {
    if (cycleStart === undefined) cycleStart = time;
    const elapsed = (time - cycleStart) % duration;
    const progress = elapsed / duration;
    const { x, y } = pointOnOrbit(progress);
    orbitMarker.setAttribute('transform', `translate(${x} ${y})`);
    guidanceForeground?.setAttribute('transform', `translate(${x} ${y})`);

    cycleStages.forEach((stage, index) => {
      const { entry, exit } = stageWindows[index];
      const touching = entry <= exit
        ? progress >= entry && progress < exit
        : progress >= entry || progress < exit;
      stage.classList.toggle('is-cycle-active', touching);
    });

    if (stageWindows.length) {
      const guidanceStart = (stageWindows[applyStageIndex].exit - .012 + 1) % 1;
      const guidanceEnd = stageWindows[generateStageIndex].entry;
      const carryingGuidance = guidanceStart <= guidanceEnd
        ? progress >= guidanceStart && progress < guidanceEnd
        : progress >= guidanceStart || progress < guidanceEnd;
      orbitMarker.classList.toggle('carrying-guidance', carryingGuidance);
      guidanceForeground?.classList.toggle('carrying-guidance', carryingGuidance);
    }

    if (transferCurve && beliefTransferMarker && beliefTransferPath) {
      const transferStart = transferCurve.trigger * duration + transferDelay;
      const timeSinceTransfer = (elapsed - transferStart + duration) % duration;
      const transferring = timeSinceTransfer < transferDuration;

      if (transferring) {
        const transferProgress = timeSinceTransfer / transferDuration;
        beliefTransferMarker.style.opacity = 1;
        renderTokenStream(beliefTokens, { ...transferCurve, target: { x: centre, y: centre } }, transferProgress, (value) => value ** 2.65);
        beliefTransferPath.style.opacity = .58;
      } else {
        beliefTransferMarker.style.opacity = 0;
        beliefTransferPath.style.opacity = 0;
      }
      const receiving = timeSinceTransfer >= transferDuration * .76 && timeSinceTransfer < transferDuration + 950;
      beliefLibrary?.classList.toggle('belief-received', receiving);
    }

    if (selectionCurve && selectionTransferMarker && selectionTransferPath) {
      const selectionStart = selectionCurve.trigger * duration;
      const timeSinceSelection = (elapsed - selectionStart + duration) % duration;
      const selecting = timeSinceSelection < selectionDuration;

      if (selecting) {
        const selectionProgress = timeSinceSelection / selectionDuration;
        selectStage?.classList.add('selection-running');
        if (selectionLiveText) selectionLiveText.textContent = 'Ranking and filtering';
        selectionTransferMarker.style.opacity = 1;
        renderTokenStream(selectionTokens, selectionCurve, selectionProgress, (value) => value * value * (3 - 2 * value));
        selectionTransferPath.style.opacity = .62;
      } else {
        selectStage?.classList.remove('selection-running');
        selectionTransferMarker.style.opacity = 0;
        selectionTransferPath.style.opacity = 0;
      }
    }

    if (applicationCurve && applicationTransferMarker && applicationTransferPath) {
      const selectTrigger = selectionCurve.trigger;
      const injectTrigger = applicationCurve.trigger;
      const selectToInjectArc = (injectTrigger - selectTrigger + 1) % 1;
      const handoffStartProgress = (selectTrigger + selectToInjectArc / 2) % 1;
      const handoffDuration = selectToInjectArc / 2 * duration;
      const handoffStart = handoffStartProgress * duration;
      const timeSinceHandoff = (elapsed - handoffStart + duration) % duration;
      const handingOff = timeSinceHandoff < handoffDuration;

      if (handingOff) {
        const handoffProgress = timeSinceHandoff / handoffDuration;
        applicationTransferMarker.style.opacity = 1;
        renderTokenStream(applicationTokens, applicationCurve, handoffProgress, (value) => value * value * (3 - 2 * value));
        applicationTransferPath.style.opacity = .52;
      } else {
        applicationTransferMarker.style.opacity = 0;
        applicationTransferPath.style.opacity = 0;
      }
    }

    window.requestAnimationFrame(animateCycle);
  };

  calculateStageTriggers();
  if (document.fonts?.ready) document.fonts.ready.then(calculateStageTriggers);
  window.addEventListener('resize', calculateStageTriggers);
  window.requestAnimationFrame(animateCycle);
}
