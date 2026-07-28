document.addEventListener('DOMContentLoaded', () => {
    // -------------------------------------------------------------
    // 1. LIVE ANIMATED CANVAS BACKGROUND
    // -------------------------------------------------------------
    const canvas = document.getElementById('bg-canvas');
    const ctx = canvas.getContext('2d');

    function resizeCanvas() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    const particles = [];
    const particleCount = 45;

    for (let i = 0; i < particleCount; i++) {
        particles.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 0.6,
            vy: (Math.random() - 0.5) * 0.6,
            radius: Math.random() * 2 + 1,
            alpha: Math.random() * 0.5 + 0.2
        });
    }

    function animateParticles() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        for (let i = 0; i < particleCount; i++) {
            const p = particles[i];
            p.x += p.vx;
            p.y += p.vy;

            if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
            if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(59, 130, 246, ${p.alpha})`;
            ctx.fill();

            for (let j = i + 1; j < particleCount; j++) {
                const p2 = particles[j];
                const dx = p.x - p2.x;
                const dy = p.y - p2.y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < 130) {
                    ctx.beginPath();
                    ctx.moveTo(p.x, p.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.strokeStyle = `rgba(139, 92, 246, ${0.15 * (1 - dist / 130)})`;
                    ctx.lineWidth = 0.8;
                    ctx.stroke();
                }
            }
        }
        requestAnimationFrame(animateParticles);
    }
    animateParticles();

    // -------------------------------------------------------------
    // 2. PAGE NAVIGATION & STEPPER CONTROL
    // -------------------------------------------------------------
    const stepNav1 = document.getElementById('step-nav-1');
    const stepNav2 = document.getElementById('step-nav-2');
    const stepNav3 = document.getElementById('step-nav-3');

    const page1 = document.getElementById('page-1');
    const page2 = document.getElementById('page-2');
    const page3 = document.getElementById('page-3');

    const personalForm = document.getElementById('personal-info-form');
    const editPatientBtn = document.getElementById('edit-patient-btn');
    const goToCarePlanBtn = document.getElementById('go-to-care-plan-btn');
    const backToP2Btn = document.getElementById('back-to-p2-btn');
    const newAssessmentBtn = document.getElementById('new-assessment-btn');

    let patientData = {
        name: 'Eleanor Vance',
        age: 33,
        gender: 'Female',
        id: 'PAT-88241',
        phone: '',
        email: '',
        notes: ''
    };

    let latestPredictionData = null;

    function navigateToPage(pageNum) {
        [page1, page2, page3].forEach(p => p.classList.remove('active'));
        [stepNav1, stepNav2, stepNav3].forEach(s => s.classList.remove('active', 'completed'));

        if (pageNum === 1) {
            page1.classList.add('active');
            stepNav1.classList.add('active');
        } else if (pageNum === 2) {
            page2.classList.add('active');
            stepNav1.classList.add('completed');
            stepNav2.classList.add('active');
        } else if (pageNum === 3) {
            page3.classList.add('active');
            stepNav1.classList.add('completed');
            stepNav2.classList.add('completed');
            stepNav3.classList.add('active');
        }
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    // Page 1 Form Submit -> Go to Page 2
    personalForm.addEventListener('submit', (e) => {
        e.preventDefault();
        patientData.name = document.getElementById('fullName').value || 'Eleanor Vance';
        patientData.age = parseInt(document.getElementById('pAge').value) || 33;
        patientData.gender = document.getElementById('gender').value;
        patientData.id = document.getElementById('patientId').value || 'PAT-88241';
        patientData.phone = document.getElementById('contactPhone').value;
        patientData.email = document.getElementById('emailAddress').value;
        patientData.notes = document.getElementById('medicalHistory').value;

        // Update Page 2 Banners & Age sync
        document.getElementById('banner-name').textContent = patientData.name;
        document.getElementById('banner-age').textContent = `${patientData.age} yrs (${patientData.gender})`;
        document.getElementById('banner-id').textContent = patientData.id;

        document.getElementById('p3-patient-name').textContent = `${patientData.name} (${patientData.age} yrs, ID: ${patientData.id})`;

        navigateToPage(2);
    });

    editPatientBtn.addEventListener('click', () => navigateToPage(1));
    goToCarePlanBtn.addEventListener('click', () => navigateToPage(3));
    backToP2Btn.addEventListener('click', () => navigateToPage(2));

    newAssessmentBtn.addEventListener('click', () => {
        document.getElementById('results-container').classList.add('hidden');
        document.getElementById('care-plan-trigger-box').classList.add('hidden');
        personalForm.reset();
        navigateToPage(1);
    });

    // -------------------------------------------------------------
    // 3. PRESETS BUTTONS (Page 2)
    // -------------------------------------------------------------
    document.getElementById('fill-healthy-btn').addEventListener('click', () => {
        document.getElementById('pregnancies').value = 1;
        document.getElementById('glucose').value = 85;
        document.getElementById('bloodPressure').value = 66;
        document.getElementById('skinThickness').value = 20;
        document.getElementById('insulin').value = 80;
        document.getElementById('bmi').value = 22.4;
        document.getElementById('dpf').value = 0.25;
    });

    document.getElementById('fill-diabetic-btn').addEventListener('click', () => {
        document.getElementById('pregnancies').value = 6;
        document.getElementById('glucose').value = 168;
        document.getElementById('bloodPressure').value = 88;
        document.getElementById('skinThickness').value = 36;
        document.getElementById('insulin').value = 240;
        document.getElementById('bmi').value = 38.5;
        document.getElementById('dpf').value = 0.85;
    });

    // -------------------------------------------------------------
    // 4. ML MODEL PREDICTION & RESULT ANIMATION (Page 2)
    // -------------------------------------------------------------
    const clinicalForm = document.getElementById('clinical-form');
    const processBtn = document.getElementById('process-btn');
    const btnText = document.getElementById('btn-text');
    const btnSpinner = document.getElementById('btn-spinner');

    clinicalForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        btnText.textContent = 'ANALYZING...';
        btnSpinner.classList.remove('hidden');
        processBtn.disabled = true;

        const payload = {
            pregnancies: parseFloat(document.getElementById('pregnancies').value),
            glucose: parseFloat(document.getElementById('glucose').value),
            bloodPressure: parseFloat(document.getElementById('bloodPressure').value),
            skinThickness: parseFloat(document.getElementById('skinThickness').value),
            insulin: parseFloat(document.getElementById('insulin').value),
            bmi: parseFloat(document.getElementById('bmi').value),
            dpf: parseFloat(document.getElementById('dpf').value),
            age: parseFloat(patientData.age)
        };

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await response.json();
            btnText.textContent = 'PROCESS & ANALYZE DATA';
            btnSpinner.classList.add('hidden');
            processBtn.disabled = false;

            if (data.success) {
                latestPredictionData = data;
                renderResults(data);
            } else {
                alert('Prediction Error: ' + data.error);
            }
        } catch (err) {
            btnText.textContent = 'PROCESS & ANALYZE DATA';
            btnSpinner.classList.add('hidden');
            processBtn.disabled = false;
            alert('Failed to connect to backend server. Ensure python server.py is running.');
        }
    });

    function renderResults(data) {
        const resultsContainer = document.getElementById('results-container');
        resultsContainer.classList.remove('hidden');

        // Animate Ring Gauge
        const gaugeFill = document.getElementById('gauge-fill');
        const gaugePercent = document.getElementById('gauge-percent');
        const riskTier = document.getElementById('risk-tier');
        const riskClassification = document.getElementById('risk-classification');

        const prob = data.diabetic_probability;
        const maxOffset = 314;
        const targetOffset = maxOffset - (maxOffset * (prob / 100));

        gaugeFill.style.strokeDashoffset = maxOffset;
        gaugeFill.style.stroke = data.risk_color;

        setTimeout(() => {
            gaugeFill.style.strokeDashoffset = targetOffset;
        }, 50);

        // Counter Animation
        let count = 0;
        const duration = 1000;
        const steps = 40;
        const increment = prob / steps;
        const stepTime = duration / steps;

        const timer = setInterval(() => {
            count += increment;
            if (count >= prob) {
                count = prob;
                clearInterval(timer);
            }
            gaugePercent.textContent = `${count.toFixed(1)}%`;
        }, stepTime);

        riskTier.textContent = data.risk_tier;
        riskTier.style.color = data.risk_color;
        riskTier.style.borderColor = data.risk_color;
        riskClassification.textContent = data.label;

        // Render Risk Factors
        const riskList = document.getElementById('risk-factors-list');
        riskList.innerHTML = '';
        data.risk_factors.forEach(factor => {
            const li = document.createElement('li');
            li.textContent = factor;
            riskList.appendChild(li);
        });

        // FEATURE 2: Render Feature Contributions
        const contribGrid = document.getElementById('contrib-grid');
        contribGrid.innerHTML = '';
        if (data.feature_contributions && data.feature_contributions.length > 0) {
            data.feature_contributions.forEach(item => {
                const div = document.createElement('div');
                div.className = 'contrib-item';
                div.innerHTML = `
                    <div class="contrib-header">
                        <span>${item.feature}</span>
                        <strong>${item.percentage}%</strong>
                    </div>
                    <div class="contrib-bar-bg">
                        <div class="contrib-bar-fill" style="width: ${item.percentage}%"></div>
                    </div>
                `;
                contribGrid.appendChild(div);
            });
        }

        // FEATURE 3: Initialize What-If Sliders Sync
        const glucoseInput = parseFloat(document.getElementById('glucose').value);
        const bmiInput = parseFloat(document.getElementById('bmi').value);
        const bpInput = parseFloat(document.getElementById('bloodPressure').value);

        document.getElementById('sim-glucose').value = glucoseInput;
        document.getElementById('sim-bmi').value = bmiInput;
        document.getElementById('sim-bp').value = bpInput;

        updateSimLabels();

        // Care Plan Unlock Banner
        const carePlanTriggerBox = document.getElementById('care-plan-trigger-box');
        document.getElementById('p3-risk-score').textContent = `${data.label} (${data.diabetic_probability}% Risk Probability)`;

        if (data.prediction === 1 || data.diabetic_probability >= 35.0) {
            carePlanTriggerBox.classList.remove('hidden');
        } else {
            carePlanTriggerBox.classList.add('hidden');
        }

        resultsContainer.scrollIntoView({ behavior: 'smooth' });
    }

    // -------------------------------------------------------------
    // FEATURE 3: WHAT-IF LIFESTYLE SIMULATOR (Real-Time Recalculation)
    // -------------------------------------------------------------
    const simGlucose = document.getElementById('sim-glucose');
    const simBmi = document.getElementById('sim-bmi');
    const simBp = document.getElementById('sim-bp');

    function updateSimLabels() {
        document.getElementById('sim-glucose-val').textContent = `${simGlucose.value} mg/dL`;
        document.getElementById('sim-bmi-val').textContent = `${simBmi.value} kg/m²`;
        document.getElementById('sim-bp-val').textContent = `${simBp.value} mmHg`;
    }

    let simDebounceTimer = null;
    [simGlucose, simBmi, simBp].forEach(slider => {
        slider.addEventListener('input', () => {
            updateSimLabels();
            clearTimeout(simDebounceTimer);
            simDebounceTimer = setTimeout(runSimPrediction, 250);
        });
    });

    async function runSimPrediction() {
        if (!latestPredictionData) return;

        const payload = {
            pregnancies: parseFloat(document.getElementById('pregnancies').value),
            glucose: parseFloat(simGlucose.value),
            bloodPressure: parseFloat(simBp.value),
            skinThickness: parseFloat(document.getElementById('skinThickness').value),
            insulin: parseFloat(document.getElementById('insulin').value),
            bmi: parseFloat(simBmi.value),
            dpf: parseFloat(document.getElementById('dpf').value),
            age: parseFloat(patientData.age)
        };

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await response.json();

            if (data.success) {
                const simVal = document.getElementById('sim-risk-val');
                const diffBadge = document.getElementById('sim-diff-badge');

                simVal.textContent = `${data.diabetic_probability.toFixed(1)}%`;
                const diff = data.diabetic_probability - latestPredictionData.diabetic_probability;

                if (diff < 0) {
                    diffBadge.textContent = `${diff.toFixed(1)}% Risk Reduced`;
                    diffBadge.style.background = 'rgba(16, 185, 129, 0.2)';
                    diffBadge.style.color = '#10b981';
                } else if (diff > 0) {
                    diffBadge.textContent = `+${diff.toFixed(1)}% Risk Increased`;
                    diffBadge.style.background = 'rgba(239, 68, 68, 0.2)';
                    diffBadge.style.color = '#ef4444';
                } else {
                    diffBadge.textContent = `No Change`;
                    diffBadge.style.background = 'rgba(255, 255, 255, 0.1)';
                    diffBadge.style.color = '#94a3b8';
                }
            }
        } catch (err) {
            console.error('Sim error:', err);
        }
    }

    // -------------------------------------------------------------
    // FEATURE 5: BULK CSV UPLOAD & DOWNLOAD MODAL
    // -------------------------------------------------------------
    const csvModal = document.getElementById('csv-modal');
    const openCsvModalBtn = document.getElementById('open-csv-modal-btn');
    const closeCsvModal = document.getElementById('close-csv-modal');
    const csvDropzone = document.getElementById('csv-dropzone');
    const csvFileInput = document.getElementById('csv-file-input');
    const downloadCsvBtn = document.getElementById('download-csv-btn');
    let processedCsvResultText = null;

    openCsvModalBtn.addEventListener('click', () => csvModal.classList.remove('hidden'));
    closeCsvModal.addEventListener('click', () => csvModal.classList.add('hidden'));

    csvDropzone.addEventListener('click', () => csvFileInput.click());
    csvDropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        csvDropzone.style.borderColor = '#3b82f6';
    });
    csvDropzone.addEventListener('dragleave', () => csvDropzone.style.borderColor = 'rgba(59, 130, 246, 0.4)');

    csvDropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        csvDropzone.style.borderColor = 'rgba(59, 130, 246, 0.4)';
        if (e.dataTransfer.files.length > 0) {
            handleCsvFile(e.dataTransfer.files[0]);
        }
    });

    csvFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleCsvFile(e.target.files[0]);
        }
    });

    async function handleCsvFile(file) {
        if (!file.name.endsWith('.csv')) {
            alert('Please select a valid .csv file.');
            return;
        }

        const reader = new FileReader();
        reader.onload = async (e) => {
            const csvText = e.target.result;
            try {
                const response = await fetch('/api/predict_csv', {
                    method: 'POST',
                    headers: { 'Content-Type': 'text/csv' },
                    body: csvText
                });

                if (response.ok) {
                    processedCsvResultText = await response.text();
                    downloadCsvBtn.disabled = false;
                    renderCsvPreview(processedCsvResultText);
                } else {
                    alert('Error processing CSV on server.');
                }
            } catch (err) {
                alert('Failed to connect to backend CSV endpoint.');
            }
        };
        reader.readAsText(file);
    }

    function renderCsvPreview(csvText) {
        const lines = csvText.trim().split('\n');
        if (lines.length < 2) return;

        const tableBody = document.querySelector('#csv-results-table tbody');
        tableBody.innerHTML = '';

        for (let i = 1; i < Math.min(lines.length, 6); i++) {
            const cols = lines[i].split(',');
            if (cols.length >= 7) {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${cols[0]}</td>
                    <td>${cols[1]}</td>
                    <td>${cols[2]}</td>
                    <td>${cols[3]}</td>
                    <td><strong>${cols[4]}</strong></td>
                    <td>${cols[5]}%</td>
                    <td>${cols[6]}</td>
                `;
                tableBody.appendChild(tr);
            }
        }
        document.getElementById('csv-preview-box').classList.remove('hidden');
    }

    downloadCsvBtn.addEventListener('click', () => {
        if (!processedCsvResultText) return;
        const blob = new Blob([processedCsvResultText], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'diabetes_predictions_result.csv';
        a.click();
        URL.revokeObjectURL(url);
    });

    // -------------------------------------------------------------
    // FEATURE 6: PRINTABLE PDF REPORT GENERATION
    // -------------------------------------------------------------
    document.getElementById('print-p2-pdf-btn').addEventListener('click', () => window.print());
    document.getElementById('print-p3-pdf-btn').addEventListener('click', () => window.print());
});
