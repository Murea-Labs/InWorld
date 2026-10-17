/* Small, full-frame previews. Sources are loaded only while visible. */
(function () {
  "use strict";
  var cards = Array.from(document.querySelectorAll(".motion-card"));
  if (!cards.length) return;
  var preference = window.matchMedia("(prefers-reduced-motion: reduce)");
  var pausedAll = preference.matches;
  var controls = Array.from(document.querySelectorAll(".motion-control"));
  var states = cards.map(function (card) {
    return { card: card, video: card.querySelector("video"), button: card.querySelector(".motion-card__toggle"), visible: false, pausedByUser: false, pending: false, blocked: false, startTimer: null, releaseTimer: null, settled: false };
  });

  function sync(state) {
    var paused = state.video.paused;
    state.button.setAttribute("aria-label", (paused ? "Play " : "Pause ") + state.card.dataset.label);
    state.button.querySelector("span").textContent = paused ? "▶" : "Ⅱ";
    state.card.classList.toggle("is-playing", !paused);
  }
  function eligible(state) {
    return state.visible && !document.hidden && !pausedAll && !state.pausedByUser && !state.blocked;
  }
  function update(state) {
    var video = state.video;
    if (!eligible(state)) {
      clearTimeout(state.startTimer);
      state.startTimer = null;
      state.settled = false;
      video.pause();
      if ((!state.visible || document.hidden) && video.getAttribute("src") && video.readyState < 2) {
        // Cancel a download that was still starting when the user moved on.
        video.removeAttribute("src");
        video.load();
        state.card.classList.remove("is-ready");
      }
      if ((!state.visible || document.hidden) && video.getAttribute("src") && !state.releaseTimer) {
        state.releaseTimer = setTimeout(function () {
          state.releaseTimer = null;
          if (state.visible && !document.hidden) return;
          video.removeAttribute("src");
          video.load();
          state.card.classList.remove("is-ready");
        }, 20000);
      }
      sync(state);
      return;
    }
    clearTimeout(state.releaseTimer);
    state.releaseTimer = null;
    if (!video.getAttribute("src")) {
      // Passing a row during a quick scroll should not download its clips.
      if (!state.settled) {
        if (!state.startTimer) state.startTimer = setTimeout(function () {
          state.startTimer = null;
          state.settled = true;
          update(state);
        }, 250);
        return;
      }
      video.src = video.dataset.src;
      video.load();
    }
    if (!video.paused || state.pending) return;
    state.pending = true;
    var result = video.play();
    if (result && result.then) result.then(function () {
      state.pending = false;
      if (!eligible(state)) video.pause();
      sync(state);
    }, function (error) {
      state.pending = false;
      if (error.name !== "AbortError") state.blocked = true;
      sync(state);
      // A rapid scroll can cancel a pending start before the card returns.
      if (error.name === "AbortError" && eligible(state)) update(state);
    });
    else state.pending = false;
  }
  function refresh() { states.forEach(update); }
  function syncControls() {
    controls.forEach(function (button) {
      button.textContent = pausedAll ? "Play previews" : "Pause previews";
      button.setAttribute("aria-pressed", String(pausedAll));
    });
  }
  controls.forEach(function (button) {
    button.hidden = false;
    button.addEventListener("click", function () {
      pausedAll = !pausedAll;
      states.forEach(function (state) {
        state.blocked = false;
        if (!pausedAll) state.pausedByUser = false;
      });
      syncControls();
      refresh();
    });
  });
  states.forEach(function (state) {
    state.video.muted = true;
    state.button.hidden = false;
    state.video.addEventListener("play", function () { sync(state); });
    state.video.addEventListener("pause", function () { sync(state); });
    state.video.addEventListener("loadeddata", function () { state.card.classList.add("is-ready"); });
    state.video.addEventListener("error", function () {
      state.blocked = true;
      state.card.classList.add("has-error");
      state.card.classList.remove("is-ready");
      state.button.setAttribute("aria-label", "Retry " + state.card.dataset.label);
    });
    state.button.addEventListener("click", function () {
      if (!state.video.paused) {
        state.pausedByUser = true;
        update(state);
        return;
      }
      state.pausedByUser = false;
      state.blocked = false;
      if (state.card.classList.contains("has-error")) {
        state.video.load();
        state.card.classList.remove("has-error");
      }
      if (pausedAll) {
        // An explicit play request starts this clip, retaining other pauses.
        states.forEach(function (other) { if (other !== state) other.pausedByUser = true; });
        pausedAll = false;
        syncControls();
      }
      update(state);
    });
    sync(state);
  });
  if ("IntersectionObserver" in window) {
    var byCard = new Map(states.map(function (state) { return [state.card, state]; }));
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        var state = byCard.get(entry.target);
        state.visible = entry.isIntersecting && entry.intersectionRatio >= 0.25;
        update(state);
      });
    }, { rootMargin: "-72px 0px 0px", threshold: [0, 0.25] });
    states.forEach(function (state) { observer.observe(state.card); });
  } else {
    // Older browsers retain manual playback without loading the whole library.
    pausedAll = true;
    controls.forEach(function (button) { button.hidden = true; });
    states.forEach(function (state) { state.visible = true; state.pausedByUser = true; });
  }
  document.addEventListener("visibilitychange", refresh);
  function changedPreference(event) {
    pausedAll = event.matches;
    syncControls();
    refresh();
  }
  if (preference.addEventListener) preference.addEventListener("change", changedPreference);
  else if (preference.addListener) preference.addListener(changedPreference);
  syncControls();

  document.querySelectorAll(".story-world").forEach(function (world) {
    var track = world.querySelector(".story-track");
    var navigation = world.querySelector(".story-nav");
    var previous = world.querySelector(".story-prev");
    var next = world.querySelector(".story-next");
    function buttons() {
      navigation.hidden = track.scrollWidth <= track.clientWidth + 4;
      previous.disabled = track.scrollLeft < 4;
      next.disabled = track.scrollLeft + track.clientWidth >= track.scrollWidth - 4;
    }
    [previous, next].forEach(function (button, index) {
      button.hidden = false;
      button.addEventListener("click", function () {
        track.scrollBy({ left: (index ? 1 : -1) * Math.max(240, track.clientWidth * 0.8), behavior: preference.matches ? "instant" : "smooth" });
      });
    });
    track.addEventListener("scroll", buttons, { passive: true });
    window.addEventListener("resize", buttons);
    buttons();
  });
})();
