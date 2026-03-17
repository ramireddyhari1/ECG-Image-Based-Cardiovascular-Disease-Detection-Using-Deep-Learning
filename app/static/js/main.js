const form = document.getElementById("ecgForm");
const statusText = document.getElementById("statusText");
const uploadedImage = document.getElementById("uploadedImage");
const predictedConditionEl = document.getElementById("predictedCondition");
const confidenceScoreEl = document.getElementById("confidenceScore");
const medicalExplanationEl = document.getElementById("medicalExplanation");
const reportContentEl = document.getElementById("reportContent");
const downloadReportBtn = document.getElementById("downloadReportBtn");

let probabilityChart;
let waveformChart;
let latestReportText = "";

function setStatus(message, type = "") {
  statusText.textContent = message;
  statusText.className = "status";
  if (type) {
    statusText.classList.add(type);
  }
}

function initCharts() {
  const probCtx = document.getElementById("probabilityChart").getContext("2d");
  probabilityChart = new Chart(probCtx, {
    type: "bar",
    data: {
      labels: ["Normal", "Atrial Fibrillation", "Myocardial Infarction", "Tachycardia", "Bradycardia"],
      datasets: [
        {
          label: "Probability",
          data: [0, 0, 0, 0, 0],
          backgroundColor: ["#0e7ca8", "#2b9ed2", "#cc4b4b", "#ef8b3f", "#6e8cb8"],
          borderRadius: 8,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 0,
          max: 1,
          ticks: {
            callback: (value) => `${Math.round(value * 100)}%`,
          },
        },
      },
      plugins: {
        legend: { display: false },
      },
    },
  });

  const waveCtx = document.getElementById("waveformChart").getContext("2d");
  waveformChart = new Chart(waveCtx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Waveform Intensity",
          data: [],
          borderColor: "#0e7ca8",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.25,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          reverse: true,
          title: {
            display: true,
            text: "Relative Vertical Position",
          },
        },
      },
      plugins: {
        legend: { display: false },
      },
    },
  });
}

function updateProbabilityChart(probabilities) {
  const labels = Object.keys(probabilities);
  const values = labels.map((label) => probabilities[label]);
  probabilityChart.data.labels = labels;
  probabilityChart.data.datasets[0].data = values;
  probabilityChart.update();
}

function extractWaveformPoints(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (event) => {
      const img = new Image();
      img.onload = () => {
        const sampleWidth = 300;
        const sampleHeight = 140;
        const canvas = document.createElement("canvas");
        canvas.width = sampleWidth;
        canvas.height = sampleHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0, sampleWidth, sampleHeight);

        const pixels = ctx.getImageData(0, 0, sampleWidth, sampleHeight).data;
        const points = [];

        for (let x = 0; x < sampleWidth; x += 2) {
          let bestY = sampleHeight / 2;
          let bestScore = -1;

          for (let y = 0; y < sampleHeight; y++) {
            const idx = (y * sampleWidth + x) * 4;
            const r = pixels[idx];
            const g = pixels[idx + 1];
            const b = pixels[idx + 2];
            const darkness = 255 - (r + g + b) / 3;

            if (darkness > bestScore) {
              bestScore = darkness;
              bestY = y;
            }
          }

          points.push(bestY);
        }

        resolve(points);
      };
      img.onerror = () => reject(new Error("Unable to load ECG image for waveform extraction."));
      img.src = event.target.result;
    };
    reader.onerror = () => reject(new Error("Unable to read uploaded image."));
    reader.readAsDataURL(file);
  });
}

function updateWaveformChart(points) {
  waveformChart.data.labels = points.map((_, i) => i + 1);
  waveformChart.data.datasets[0].data = points;
  waveformChart.update();
}

function buildReport(data) {
  const confidencePct = (data.confidence * 100).toFixed(2);
  const timestamp = new Date().toLocaleString();
  const lines = [
    "ECG AI Report",
    "--------------------------------------",
    `Generated: ${timestamp}`,
    `Patient Name: ${data.patient.name || "Unknown"}`,
    `Age: ${data.patient.age || "Unknown"}`,
    `Gender: ${data.patient.gender || "Unknown"}`,
    "",
    `Predicted Condition: ${data.predicted_condition}`,
    `Confidence: ${confidencePct}%`,
    `Medical Explanation: ${data.explanation}`,
    "",
    "Class Probabilities:",
  ];

  Object.entries(data.probabilities).forEach(([condition, prob]) => {
    lines.push(`- ${condition}: ${(prob * 100).toFixed(2)}%`);
  });

  lines.push("");
  lines.push("Important: This AI output is for screening support only and must be validated by a cardiologist.");

  return lines.join("\n");
}

function downloadTextFile(filename, content) {
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const fileInput = document.getElementById("ecgImageInput");
  const file = fileInput.files[0];

  if (!file) {
    setStatus("Please select an ECG image.", "error");
    return;
  }

  setStatus("Processing ECG image and running CNN inference...", "");

  try {
    uploadedImage.src = URL.createObjectURL(file);
    const waveformPoints = await extractWaveformPoints(file);
    updateWaveformChart(waveformPoints);

    const formData = new FormData(form);
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const result = await response.json();

    if (!response.ok || !result.success) {
      throw new Error(result.error || "Prediction failed.");
    }

    uploadedImage.src = result.image_url;
    predictedConditionEl.textContent = result.predicted_condition;
    confidenceScoreEl.textContent = `${(result.confidence * 100).toFixed(2)}%`;
    medicalExplanationEl.textContent = result.explanation;

    updateProbabilityChart(result.probabilities);

    latestReportText = buildReport(result);
    reportContentEl.textContent = latestReportText;
    downloadReportBtn.disabled = false;

    setStatus("Analysis complete.", "success");
  } catch (error) {
    setStatus(error.message, "error");
  }
});

downloadReportBtn.addEventListener("click", () => {
  if (!latestReportText) return;
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
  downloadTextFile(`ecg_ai_report_${stamp}.txt`, latestReportText);
});

initCharts();
