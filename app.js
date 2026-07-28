document.addEventListener('DOMContentLoaded', () => {
    initCanvasAnimation();
    initMultiPageApp();
});

/* -------------------------------------------------------------
 * 1. LIVE ANIMATED CANVAS BACKGROUND
 * ------------------------------------------------------------- */
function initCanvasAnimation() {
    const canvas = document.getElementById('bg-canvas');
    const ctx = canvas.getContext('2d');

    let width = canvas.width = window.innerWidth;
    let height = canvas.height = window.innerHeight;

    window.addEventListener('resize', () => {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    });

    const particles = [];
    const particleCount = Math.min(Math.floor(width / 20), 60);

    for (let i = 0; i < particleCount; i++) {
        particles.push({
            x: Math.random() * width,
            y: Math.random() * height,
            vx: (Math.random() - 0.5) * 0.6,
            vy: (Math.random() - 0.5) * 0.6,
            radius: Math.random() * 2 + 1,
            color: Math.random() > 0.5 ? 'rgba(59, 130, 246, ' : 'rgba(139, 92, 246, '
        });
    }

    function animate() {
        ctx.clearRect(0, 0, width, height);

        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];

            p.x += p.vx;
            p.y += p.vy;

            if (p.x < 0) p.x = width;
            if (p.x > width) p.x = 0;
            if (p.y < 0) p.y = height;
            if (p.y > height) p.y = 0;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
            ctx.fillStyle = p.color + '0.4)';
            ctx.fill();

            for (let j = i + 1; j < particles.length; j++) {
                const p2 = particles[j];
                const dx = p.x - p2.x;
                const dy = p.y - p2.y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < 120) {
                    ctx.beginPath();
                    ctx.moveTo(p.x, p.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.strokeStyle = p.color + (1 - dist / 120) * 0.15 + ')';
                    ctx.lineWidth = 0.8;
                    ctx.stroke();
                }
            }
        }

        requestAnimationFrame(animate);
    }

    animate();
}

/* -------------------------------------------------------------
 * 2. MULTI-PAGE WEBSITE NAVIGATION & PREDICTION HANDLERS
 * ------------------------------------------------------------- */
let currentPatient = {
    name: 'Eleanor Vance',
    age: 33,
    gender: 'Female',
    id: 'PAT-88241',
    phone: '',
    email: '',
    history: ''
};

let lastPredictionResult = null;

function initMultiPageApp() {
    const personalForm = document.getElementById('personal-info-form');
    const clinicalForm = document.getElementById('clinical-form');
    const editPatientBtn = document.getElementById('edit-patient-btn');
    const fillHealthyBtn = document.getElementById('fill-healthy-btn');
    const fillDiabeticBtn = document.getElementById('fill-diabetic-btn');
    const goToCarePlanBtn = document.getElementById('go-to-care-plan-btn');
    const backToP2Btn = document.getElementById('back-to-p2-btn');
    const newAssessmentBtn = document.getElementById('new-assessment-btn');

    // Strict 10-digit numeric phone validation
    const phoneInput = document.getElementById('contactPhone');
    if (phoneInput) {
        phoneInput.addEventListener('input', (e) => {
            e.target.value = e.target.value.replace(/[^0-9]/g, '').slice(0, 10);
        });
    }

    // Page 1: Personal Details Submission
    personalForm.addEventListener('submit', (e) => {
        e.preventDefault();

        const phoneVal = document.getElementById('contactPhone').value;
        if (phoneVal.length !== 10) {
            alert("Please enter a valid 10-digit phone number.");
            return;
        }

        currentPatient.name = document.getElementById('fullName').value || 'Eleanor Vance';
        currentPatient.age = document.getElementById('pAge').value || 33;
        currentPatient.gender = document.getElementById('gender').value || 'Female';
        currentPatient.id = document.getElementById('patientId').value || 'PAT-88241';
        currentPatient.phone = '+91 ' + phoneVal;
        currentPatient.email = document.getElementById('emailAddress').value || '';
        currentPatient.history = document.getElementById('medicalHistory').value || '';

        updatePatientBanners();
        switchPage(2);
    });

    // Edit Patient Details Button
    editPatientBtn.addEventListener('click', () => {
        switchPage(1);
    });

    // Preset buttons
    fillHealthyBtn.addEventListener('click', () => {
        setClinicalValues({
            pregnancies: 1,
            glucose: 85,
            bloodPressure: 66,
            skinThickness: 20,
            insulin: 80,
            bmi: 22.4,
            dpf: 0.25
        });
    });

    fillDiabeticBtn.addEventListener('click', () => {
        setClinicalValues({
            pregnancies: 6,
            glucose: 168,
            bloodPressure: 88,
            skinThickness: 36,
            insulin: 240,
            bmi: 38.5,
            dpf: 0.85
        });
    });

    // Page 2: Clinical Form Submission
    clinicalForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const processBtn = document.getElementById('process-btn');
        const btnText = document.getElementById('btn-text');
        const btnSpinner = document.getElementById('btn-spinner');

        processBtn.disabled = true;
        btnText.textContent = "ANALYZING...";
        btnSpinner.classList.remove('hidden');

        const formData = {
            pregnancies: parseFloat(document.getElementById('pregnancies').value),
            glucose: parseFloat(document.getElementById('glucose').value),
            bloodPressure: parseFloat(document.getElementById('bloodPressure').value),
            skinThickness: parseFloat(document.getElementById('skinThickness').value),
            insulin: parseFloat(document.getElementById('insulin').value),
            bmi: parseFloat(document.getElementById('bmi').value),
            dpf: parseFloat(document.getElementById('dpf').value),
            age: parseFloat(currentPatient.age)
        };

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });

            const result = await response.json();

            if (result.success) {
                lastPredictionResult = result;
                displayPage2Results(result);
            } else {
                alert("Error during prediction: " + result.error);
            }
        } catch (err) {
            console.error("API error:", err);
            alert("Failed to connect to prediction server.");
        } finally {
            processBtn.disabled = false;
            btnText.textContent = "PROCESS & ANALYZE DATA";
            btnSpinner.classList.add('hidden');
        }
    });

    // Button to Page 3 (Care Plan & Precautions)
    goToCarePlanBtn.addEventListener('click', () => {
        updatePage3CarePlan();
        switchPage(3);
    });

    // Navigation back/reset
    backToP2Btn.addEventListener('click', () => {
        switchPage(2);
    });

    newAssessmentBtn.addEventListener('click', () => {
        document.getElementById('personal-info-form').reset();
        document.getElementById('clinical-form').reset();
        document.getElementById('results-container').classList.add('hidden');
        document.getElementById('care-plan-trigger-box').classList.add('hidden');
        switchPage(1);
    });
}

function switchPage(pageNum) {
    document.querySelectorAll('.page-section').forEach(sec => sec.classList.remove('active'));
    document.querySelectorAll('.step-item').forEach(item => item.classList.remove('active', 'completed'));

    document.getElementById(`page-${pageNum}`).classList.add('active');

    for (let i = 1; i <= 3; i++) {
        const item = document.getElementById(`step-nav-${i}`);
        if (i < pageNum) {
            item.classList.add('completed');
        } else if (i === pageNum) {
            item.classList.add('active');
        }
    }

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function updatePatientBanners() {
    document.getElementById('banner-name').textContent = currentPatient.name;
    document.getElementById('banner-age').textContent = `${currentPatient.age} yrs (${currentPatient.gender})`;
    document.getElementById('banner-id').textContent = currentPatient.id;
}

function setClinicalValues(vals) {
    document.getElementById('pregnancies').value = vals.pregnancies;
    document.getElementById('glucose').value = vals.glucose;
    document.getElementById('bloodPressure').value = vals.bloodPressure;
    document.getElementById('skinThickness').value = vals.skinThickness;
    document.getElementById('insulin').value = vals.insulin;
    document.getElementById('bmi').value = vals.bmi;
    document.getElementById('dpf').value = vals.dpf;
}

/* -------------------------------------------------------------
 * 3. DISPLAY RESULTS & CARE PLAN UPDATES
 * ------------------------------------------------------------- */
function displayPage2Results(result) {
    const container = document.getElementById('results-container');
    const prob = result.diabetic_probability;
    const riskTier = document.getElementById('risk-tier');
    const riskClassification = document.getElementById('risk-classification');
    const gaugePercent = document.getElementById('gauge-percent');
    const gaugeFill = document.getElementById('gauge-fill');
    const riskFactorsList = document.getElementById('risk-factors-list');
    const carePlanBox = document.getElementById('care-plan-trigger-box');

    container.classList.remove('hidden');

    // Animate percentage
    let current = 0;
    const duration = 1000;
    const stepTime = 20;
    const steps = duration / stepTime;
    const increment = prob / steps;

    const timer = setInterval(() => {
        current += increment;
        if (current >= prob) {
            current = prob;
            clearInterval(timer);
        }
        gaugePercent.textContent = current.toFixed(1) + "%";
    }, stepTime);

    // SVG Gauge offset
    const circumference = 314.159;
    const offset = circumference - (prob / 100) * circumference;
    gaugeFill.style.strokeDashoffset = offset;
    gaugeFill.style.stroke = result.risk_color;

    // Badges & Classification Text
    riskTier.textContent = result.risk_tier.toUpperCase();
    riskTier.style.backgroundColor = `${result.risk_color}22`;
    riskTier.style.color = result.risk_color;
    riskTier.style.border = `1px solid ${result.risk_color}44`;

    riskClassification.textContent = result.label;

    // Populate Risk Factors
    riskFactorsList.innerHTML = '';
    result.risk_factors.forEach(factor => {
        const li = document.createElement('li');
        li.textContent = factor;
        riskFactorsList.appendChild(li);
    });

    // Conditionally show Page 3 Care Plan button if diabetic risk detected (prediction == 1 or prob >= 50%)
    if (result.prediction === 1 || prob >= 50.0) {
        carePlanBox.classList.remove('hidden');
    } else {
        carePlanBox.classList.add('hidden');
    }

    container.scrollIntoView({ behavior: 'smooth' });
}

function updatePage3CarePlan() {
    document.getElementById('p3-patient-name').textContent = `${currentPatient.name} (${currentPatient.age} yrs, ID: ${currentPatient.id})`;
    if (lastPredictionResult) {
        document.getElementById('p3-risk-score').textContent = `High Diabetic Risk Detected (${lastPredictionResult.diabetic_probability}% Risk Probability)`;
    }
}
