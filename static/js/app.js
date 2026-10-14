/**
 * TrafficFlow AI Frontend Logic
 * Dataset: traffic.csv (Kaggle Traffic Prediction Dataset)
 */

function formatHour(h) {
  h = parseInt(h);
  const period = h >= 12 ? 'PM' : 'AM';
  let displayHour = h % 12;
  if (displayHour === 0) displayHour = 12;
  const timeStr = `${displayHour.toString().padStart(2, '0')}:00 ${period}`;

  let tag = '';
  if (h >= 8 && h <= 10) tag = ' (Morning Peak)';
  else if (h >= 17 && h <= 19) tag = ' (Evening Peak)';
  else if (h >= 0 && h <= 5) tag = ' (Midnight/Off-Peak)';
  else if (h >= 12 && h <= 15) tag = ' (Midday Steady)';

  return `${timeStr}${tag}`;
}

const JUNCTION_NAMES = {
  1: 'Junction 1 (Primary Highway Arterial)',
  2: 'Junction 2 (Secondary Radial Road)',
  3: 'Junction 3 (Commercial Collector Way)',
  4: 'Junction 4 (Local Branch Lane)'
};

// Viva & Examiner Quick Presets
const PRESETS = {
  j1_morning: {
    junction: 1,
    hour: 9,
    day_of_week: 2, // Wednesday
    day: 14,
    month: 10,
    year: 2026
  },
  j1_evening: {
    junction: 1,
    hour: 18,
    day_of_week: 4, // Friday
    day: 16,
    month: 10,
    year: 2026
  },
  j2_afternoon: {
    junction: 2,
    hour: 14,
    day_of_week: 2, // Wednesday
    day: 15,
    month: 7,
    year: 2026
  },
  j3_weekend: {
    junction: 3,
    hour: 16,
    day_of_week: 6, // Sunday
    day: 23,
    month: 8,
    year: 2026
  },
  j4_night: {
    junction: 4,
    hour: 2,
    day_of_week: 1, // Tuesday
    day: 10,
    month: 11,
    year: 2026
  }
};

document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('trafficForm');
  const hourSlider = document.getElementById('hour');
  const hourDisplay = document.getElementById('hourDisplay');
  const junctionLabel = document.getElementById('junctionLabel');
  const datetimePicker = document.getElementById('datetime_picker');
  const dayInput = document.getElementById('day');
  const monthSelect = document.getElementById('month');
  const yearSelect = document.getElementById('year');

  // Sync Hour Display
  function updateHourDisplay() {
    hourDisplay.textContent = formatHour(hourSlider.value);
  }
  hourSlider.addEventListener('input', updateHourDisplay);
  updateHourDisplay();

  // Sync Junction Header Label
  function updateJunctionLabel() {
    const selectedJunc = document.querySelector('input[name="junction"]:checked')?.value || 1;
    junctionLabel.textContent = JUNCTION_NAMES[selectedJunc] || `Junction ${selectedJunc}`;
  }
  document.querySelectorAll('input[name="junction"]').forEach(radio => {
    radio.addEventListener('change', updateJunctionLabel);
  });
  updateJunctionLabel();

  // Sync Datetime Picker with Sliders & Inputs
  datetimePicker.addEventListener('change', () => {
    if (!datetimePicker.value) return;
    const dt = new Date(datetimePicker.value);
    if (isNaN(dt.getTime())) return;

    hourSlider.value = dt.getHours();
    updateHourDisplay();

    dayInput.value = dt.getDate();
    monthSelect.value = dt.getMonth() + 1;
    yearSelect.value = dt.getFullYear();

    // Map JS getDay() (0=Sun, 1=Mon... 6=Sat) to Pandas dayofweek (0=Mon... 6=Sun)
    const jsDay = dt.getDay();
    const pandasDay = (jsDay + 6) % 7;
    const dayRadio = document.querySelector(`input[name="day_of_week"][value="${pandasDay}"]`);
    if (dayRadio) dayRadio.checked = true;
  });

  // 1-Click Viva Presets
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const key = btn.getAttribute('data-preset');
      const p = PRESETS[key];
      if (!p) return;

      // Set Junction
      const jRadio = document.querySelector(`input[name="junction"][value="${p.junction}"]`);
      if (jRadio) jRadio.checked = true;
      updateJunctionLabel();

      // Set Hour
      hourSlider.value = p.hour;
      updateHourDisplay();

      // Set Date fields
      dayInput.value = p.day;
      monthSelect.value = p.month;
      yearSelect.value = p.year;

      // Set Day of week
      const dRadio = document.querySelector(`input[name="day_of_week"][value="${p.day_of_week}"]`);
      if (dRadio) dRadio.checked = true;

      // Update datetime picker display
      const mStr = String(p.month).padStart(2, '0');
      const dStr = String(p.day).padStart(2, '0');
      const hStr = String(p.hour).padStart(2, '0');
      datetimePicker.value = `${p.year}-${mStr}-${dStr}T${hStr}:00`;

      // Trigger instant prediction
      submitPrediction();
    });
  });

  // Form submission
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    submitPrediction();
  });

  function submitPrediction() {
    const submitBtn = document.getElementById('predictBtn');
    submitBtn.disabled = true;
    submitBtn.innerHTML = `<span>⏳ Evaluating Ensemble...</span>`;

    const selectedModel = document.querySelector('input[name="model_choice"]:checked')?.value || 'both';
    const selectedJunction = parseInt(document.querySelector('input[name="junction"]:checked')?.value || 1);
    const selectedDayOfWeek = parseInt(document.querySelector('input[name="day_of_week"]:checked')?.value || 0);

    const payload = {
      model_choice: selectedModel,
      junction: selectedJunction,
      hour: parseInt(hourSlider.value),
      day_of_week: selectedDayOfWeek,
      day: parseInt(dayInput.value) || 1,
      month: parseInt(monthSelect.value) || 10,
      year: parseInt(yearSelect.value) || 2026,
      is_weekend: selectedDayOfWeek >= 5 ? 1 : 0
    };

    fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
      .then(res => res.json())
      .then(data => {
        submitBtn.disabled = false;
        submitBtn.innerHTML = `
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
          </svg>
          <span>Run Prediction</span>
        `;
        if (data.success) {
          renderResults(data);
        } else {
          alert('Prediction error: ' + data.error);
        }
      })
      .catch(err => {
        submitBtn.disabled = false;
        submitBtn.innerHTML = `<span>Run Prediction</span>`;
        console.error(err);
        alert('Could not connect to prediction server.');
      });
  }

  function renderResults(data) {
    document.getElementById('resultsPlaceholder').style.display = 'none';
    const container = document.getElementById('resultsContent');
    container.style.display = 'block';

    const preds = data.predictions;
    const overall = preds.overall_level;

    // Determine primary display volume
    let primaryVolume = 0;
    if (preds.consensus_average) {
      primaryVolume = preds.consensus_average;
    } else if (preds.random_forest) {
      primaryVolume = preds.random_forest.vehicles;
    } else if (preds.xgboost) {
      primaryVolume = preds.xgboost.vehicles;
    }

    document.getElementById('volumeValue').textContent = Math.round(primaryVolume).toLocaleString();

    // Congestion Badge
    const badge = document.getElementById('congestionBadge');
    badge.textContent = overall.level;
    badge.style.backgroundColor = `${overall.badge_color}22`;
    badge.style.color = overall.badge_color;
    badge.style.border = `1px solid ${overall.badge_color}66`;

    document.getElementById('congestionDesc').textContent = overall.description;

    // Traffic Capacity Meter (Max capacity scale ~150 vehicles/h)
    const pct = Math.min(100, Math.max(6, (primaryVolume / 150) * 100));
    const meterFill = document.getElementById('meterFill');
    meterFill.style.width = `${pct}%`;
    meterFill.style.backgroundColor = overall.badge_color;

    // Subcard Model Metrics
    const rfCard = document.getElementById('rfStatCard');
    const xgbCard = document.getElementById('xgbStatCard');
    const consensusBox = document.getElementById('consensusBox');

    if (preds.random_forest) {
      rfCard.style.display = 'block';
      document.getElementById('rfVolume').textContent = `${Math.round(preds.random_forest.vehicles)} v/h`;
      document.getElementById('rfMetrics').textContent = `R²: ${preds.random_forest.r2} | MAE: ±${preds.random_forest.mae}`;
    } else {
      rfCard.style.display = 'none';
    }

    if (preds.xgboost) {
      xgbCard.style.display = 'block';
      document.getElementById('xgbVolume').textContent = `${Math.round(preds.xgboost.vehicles)} v/h`;
      document.getElementById('xgbMetrics').textContent = `R²: ${preds.xgboost.r2} | MAE: ±${preds.xgboost.mae}`;
    } else {
      xgbCard.style.display = 'none';
    }

    if (preds.random_forest && preds.xgboost) {
      consensusBox.style.display = 'block';
      const diff = Math.abs(preds.random_forest.vehicles - preds.xgboost.vehicles);
      const agreement = diff <= 3.0 ? 'High Agreement (±' + diff.toFixed(1) + ' v/h)' : 'Moderate Spread (±' + diff.toFixed(1) + ' v/h)';
      document.getElementById('agreementText').textContent = agreement;
    } else {
      consensusBox.style.display = 'none';
    }

    // Scroll into view on mobile screens
    if (window.innerWidth < 960) {
      container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  // Run initial prediction on load
  submitPrediction();
});
