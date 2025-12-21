// Flask Molecular Analysis Interface - JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const generationForm = document.getElementById('generationForm');
    const generateBtn = document.getElementById('generateBtn');
    const progressSection = document.getElementById('progressSection');
    const resultsSection = document.getElementById('resultsSection');
    const errorSection = document.getElementById('errorSection');
    const progressFill = document.getElementById('progressFill');
    const progressMessage = document.getElementById('progressMessage');
    const conditionSelect = document.getElementById('condition');
    const conditionDescription = document.getElementById('conditionDescription');
    const numMoleculesInput = document.getElementById('numMolecules');
    const numMoleculesValue = document.getElementById('numMoleculesValue');
    const temperatureInput = document.getElementById('temperature');
    const temperatureValue = document.getElementById('temperatureValue');
    const downloadJsonBtn = document.getElementById('downloadJsonBtn');
    const downloadSmilesBtn = document.getElementById('downloadSmilesBtn');

    // Conditions data (will be populated from Flask template)
    const conditions = {
        0: {
            name: "Condition 1: LogP ≤ 3",
            description: "Single objective: logP ≤ 3"
        },
        1: {
            name: "Condition 2: Structural",
            description: "2 aromatic rings, 1 non-aromatic, functional groups, R-value [0.05-0.50]"
        },
        2: {
            name: "Condition 3: Lipinski Ro3",
            description: "LogP≤3, MW≤480, HBA≤3, HBD≤3, RotB≤3"
        },
        3: {
            name: "Condition 4: Structural + Lipinski",
            description: "Combination of conditions 2 and 3"
        }
    };

    // Global variables
    let currentTaskId = null;
    let statusCheckInterval = null;

    // Initialize
    initializeForm();

    // Event Listeners
    generationForm.addEventListener('submit', handleFormSubmit);
    conditionSelect.addEventListener('change', updateConditionDescription);
    numMoleculesInput.addEventListener('input', updateNumMoleculesValue);
    temperatureInput.addEventListener('input', updateTemperatureValue);
    downloadJsonBtn.addEventListener('click', () => downloadFile('json'));
    downloadSmilesBtn.addEventListener('click', () => downloadFile('smiles'));

    // Functions
    function initializeForm() {
        updateConditionDescription();
        updateNumMoleculesValue();
        updateTemperatureValue();
    }

    function updateConditionDescription() {
        const selectedCondition = conditionSelect.value;
        if (conditions[selectedCondition]) {
            conditionDescription.innerHTML = `<strong>${conditions[selectedCondition].name}:</strong> ${conditions[selectedCondition].description}`;
        }
    }

    function updateNumMoleculesValue() {
        numMoleculesValue.textContent = numMoleculesInput.value;
    }

    function updateTemperatureValue() {
        temperatureValue.textContent = temperatureInput.value;
    }

    function handleFormSubmit(e) {
        e.preventDefault();

        // Get form data
        const formData = new FormData(generationForm);
        const data = {
            condition: parseInt(formData.get('condition')),
            num_molecules: parseInt(formData.get('numMolecules')),
            temperature: parseFloat(formData.get('temperature')),
            calculate_intdiv: formData.has('calculateIntdiv')
        };

        // Disable form and show progress
        setFormDisabled(true);
        showSection(progressSection);
        hideSection(resultsSection);
        hideSection(errorSection);

        // Start generation
        fetch('/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(result => {
            if (result.error) {
                throw new Error(result.error);
            }
            currentTaskId = result.task_id;
            startStatusChecking();
        })
        .catch(error => {
            showError(error.message);
        });
    }

    function startStatusChecking() {
        statusCheckInterval = setInterval(() => {
            fetch(`/status/${currentTaskId}`)
            .then(response => response.json())
            .then(status => {
                updateProgress(status);

                if (status.status === 'completed') {
                    clearInterval(statusCheckInterval);
                    loadResults();
                } else if (status.status === 'error') {
                    clearInterval(statusCheckInterval);
                    showError(status.message);
                }
            })
            .catch(error => {
                clearInterval(statusCheckInterval);
                showError('Erreur de communication avec le serveur');
            });
        }, 1000);
    }

    function updateProgress(status) {
        progressMessage.textContent = status.message;

        // Estimate progress based on message
        let progress = 0;
        if (status.message.includes('Chargement du modèle')) progress = 10;
        else if (status.message.includes('Génération')) progress = 30;
        else if (status.message.includes('Validation')) progress = 50;
        else if (status.message.includes('Nouveauté')) progress = 70;
        else if (status.message.includes('Unicité')) progress = 85;
        else if (status.message.includes('Conditions')) progress = 95;
        else if (status.message.includes('terminée')) progress = 100;

        progressFill.style.width = `${progress}%`;
    }

    function loadResults() {
        fetch(`/results/${currentTaskId}`)
        .then(response => response.json())
        .then(results => {
            if (results.error) {
                throw new Error(results.error);
            }
            displayResults(results);
            hideSection(progressSection);
            showSection(resultsSection);
            setFormDisabled(false);
        })
        .catch(error => {
            showError(error.message);
        });
    }

    function displayResults(results) {
        // Update metrics
        document.getElementById('totalGenerated').textContent = results.total_generated.toLocaleString();
        document.getElementById('totalValid').textContent = results.total_valid.toLocaleString();
        document.getElementById('validityPercentage').textContent = `${results.validity_percentage.toFixed(1)}% des générées`;
        document.getElementById('totalNovel').textContent = results.total_novel.toLocaleString();
        document.getElementById('noveltyPercentage').textContent = `${results.novelty_percentage.toFixed(1)}% des valides`;
        document.getElementById('totalUniqueNovel').textContent = results.total_unique_novel.toLocaleString();
        document.getElementById('uniquenessPercentage').textContent = `${results.uniqueness_percentage.toFixed(1)}% des nouvelles`;

        // IntDiv section
        if (results.intdiv !== undefined && results.intdiv !== null) {
            document.getElementById('intdivValue').textContent = results.intdiv.toFixed(4);
            document.getElementById('intdivDescription').textContent = `Calculée sur ${Math.min(2000, results.total_unique_novel)} molécules uniques nouvelles`;
            showSection(document.getElementById('intdivSection'));
        } else {
            hideSection(document.getElementById('intdivSection'));
        }

        // Visualizations
        if (results.visualizations) {
            createHierarchyChart(results.visualizations.hierarchy);
            createConditionsChart(results.visualizations.conditions);
        }

        // Conditions details
        displayConditionsDetails(results);
    }

    function createHierarchyChart(data) {
        const trace1 = {
            x: data.categories,
            y: data.counts,
            name: 'Nombre',
            type: 'bar',
            marker: { color: '#2C3E50' }
        };

        const trace2 = {
            x: data.categories.slice(1),
            y: data.percentages.slice(1),
            name: 'Pourcentage',
            type: 'bar',
            marker: { color: '#3498DB' },
            yaxis: 'y2'
        };

        const layout = {
            title: 'Hiérarchie de qualité des molécules',
            yaxis: { title: 'Nombre de molécules' },
            yaxis2: {
                title: 'Pourcentage (%)',
                overlaying: 'y',
                side: 'right'
            },
            barmode: 'group',
            showlegend: false,
            font: { family: 'Arial, sans-serif', size: 12 },
            plot_bgcolor: 'white',
            paper_bgcolor: 'white'
        };

        Plotly.newPlot('hierarchyChart', [trace1, trace2], layout, { responsive: true });
    }

    function createConditionsChart(data) {
        const trace = {
            x: data.names,
            y: data.counts,
            type: 'bar',
            name: 'Nombre',
            marker: { color: '#2C3E50' }
        };

        const trace2 = {
            x: data.names,
            y: data.percentages,
            type: 'bar',
            name: 'Pourcentage',
            marker: { color: '#3498DB' },
            yaxis: 'y2'
        };

        const layout = {
            title: 'Satisfaction des conditions',
            yaxis: { title: 'Nombre de molécules' },
            yaxis2: {
                title: 'Pourcentage (%)',
                overlaying: 'y',
                side: 'right'
            },
            barmode: 'group',
            showlegend: false,
            font: { family: 'Arial, sans-serif', size: 12 },
            plot_bgcolor: 'white',
            paper_bgcolor: 'white'
        };

        Plotly.newPlot('conditionsChart', [trace, trace2], layout, { responsive: true });
    }

    function displayConditionsDetails(results) {
        const container = document.getElementById('conditionsContainer');
        container.innerHTML = '';

        for (let i = 0; i < 4; i++) {
            const conditionResults = results.condition_results;
            const count = conditionResults.condition_counts[i];
            const percentage = conditionResults.condition_percentages[i];
            const examples = conditionResults.examples_per_condition[i];

            const conditionDiv = document.createElement('div');
            conditionDiv.className = 'condition-item';

            conditionDiv.innerHTML = `
                <div class="condition-header">
                    <div class="condition-title">${conditions[i].name}</div>
                    <div class="condition-stats">${count}/${results.total_unique_novel} (${percentage.toFixed(1)}%)</div>
                </div>
                <div class="condition-description">${conditions[i].description}</div>
                ${examples.length > 0 ? `
                    <div class="examples-section">
                        <strong>Exemples de SMILES :</strong>
                        <div class="examples-grid">
                            ${examples.slice(0, 3).map(smiles => `<div class="example-item">${smiles}</div>`).join('')}
                        </div>
                    </div>
                ` : ''}
            `;

            container.appendChild(conditionDiv);
        }
    }

    function downloadFile(fileType) {
        if (!currentTaskId) return;

        const link = document.createElement('a');
        link.href = `/download/${currentTaskId}/${fileType}`;
        link.download = '';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    function showSection(section) {
        section.style.display = 'block';
    }

    function hideSection(section) {
        section.style.display = 'none';
    }

    function setFormDisabled(disabled) {
        const inputs = generationForm.querySelectorAll('input, select, button');
        inputs.forEach(input => {
            input.disabled = disabled;
        });

        if (disabled) {
            generateBtn.textContent = 'Génération en cours...';
            generateBtn.classList.add('loading');
        } else {
            generateBtn.textContent = 'Lancer la génération et l\'analyse';
            generateBtn.classList.remove('loading');
        }
    }

    function showError(message) {
        hideSection(progressSection);
        hideSection(resultsSection);
        showSection(errorSection);
        document.getElementById('errorMessage').textContent = message;
        setFormDisabled(false);
    }

    // Utility functions
    function formatNumber(num) {
        return num.toLocaleString();
    }

    function formatPercentage(num) {
        return num.toFixed(1) + '%';
    }
});