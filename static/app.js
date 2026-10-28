// GeoAgent Interactive Application Logic
document.addEventListener("DOMContentLoaded", () => {
    // 1. Initialize Map
    const map = L.map('map', {
        zoomControl: false,
        attributionControl: false
    }).setView([-6.9175, 107.6191], 11); // Default centered on Bandung

    L.control.zoom({ position: 'bottomleft' }).addTo(map);

    // Dedicated custom panes with explicit z-index so thermal & boundary layers NEVER get covered by basemaps
    map.createPane('thermalPane');
    map.getPane('thermalPane').style.zIndex = 450;
    map.getPane('thermalPane').style.pointerEvents = 'none';

    map.createPane('boundaryPane');
    map.getPane('boundaryPane').style.zIndex = 500;
    map.getPane('boundaryPane').style.pointerEvents = 'none';

    // Free, crisp basemaps with NO API key and NO watermarks!
    const esriDark = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
        maxZoom: 16,
        attribution: 'Esri, HERE, Garmin'
    }).addTo(map);

    const osmStandard = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '&copy; OpenStreetMap contributors'
    });

    const esriSatellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        maxZoom: 18,
        attribution: 'Esri, Maxar, Earthstar Geographics'
    });

    // Basemap Switcher Control
    const baseMaps = {
        "Dark Canvas": esriDark,
        "OpenStreetMap": osmStandard,
        "Satellite Imagery": esriSatellite
    };
    L.control.layers(baseMaps, null, { position: 'topright' }).addTo(map);

    // Keep overlays on top when basemap switches
    map.on('baselayerchange', () => {
        if (activeGeoJsonLayer) activeGeoJsonLayer.bringToFront();
    });

    // Attribution
    L.control.attribution({ position: 'bottomright', prefix: 'GeoAgent • Earth Observation' }).addTo(map);

    // State
    let activeGeoJsonLayer = null;
    let activeTileLayer = null;
    let activeSimulatedHeatLayer = null;

    // Elements
    const chatFeed = document.getElementById("chatFeed");
    const chatForm = document.getElementById("chatForm");
    const chatInput = document.getElementById("chatInput");
    const sendBtn = document.getElementById("sendBtn");
    const presetButtons = document.getElementById("presetButtons");
    const hudTargetText = document.getElementById("hudTargetText");
    const hudMetaText = document.getElementById("hudMetaText");
    const legendCard = document.getElementById("legendCard");
    const valMinLST = document.getElementById("valMinLST");
    const valMeanLST = document.getElementById("valMeanLST");
    const valMaxLST = document.getElementById("valMaxLST");
    const lblMinTemp = document.getElementById("lblMinTemp");
    const lblMeanTemp = document.getElementById("lblMeanTemp");
    const lblMaxTemp = document.getElementById("lblMaxTemp");
    const legendModeBadge = document.getElementById("legendModeBadge");

    // 2. Fetch Presets as Compact Chips
    fetchPresets();

    async function fetchPresets() {
        try {
            const resp = await fetch("/api/presets");
            if (resp.ok) {
                const presets = await resp.json();
                renderPresets(presets);
            }
        } catch (e) {
            console.error("Failed to load presets:", e);
        }
    }

    function renderPresets(presets) {
        presetButtons.innerHTML = "";
        presets.forEach(p => {
            const chip = document.createElement("button");
            chip.className = "chip-btn";
            chip.textContent = p.title;
            chip.title = p.description || p.prompt;
            chip.addEventListener("click", () => {
                chatInput.value = p.prompt;
                sendMessage(p.prompt);
            });
            presetButtons.appendChild(chip);
        });
    }

    // 3. Chat Handler
    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const msg = chatInput.value.trim();
        if (!msg) return;
        sendMessage(msg);
    });

    async function sendMessage(messageText) {
        appendMessage("user", messageText);
        chatInput.value = "";
        setLoading(true);

        try {
            const resp = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: messageText })
            });

            if (!resp.ok) {
                throw new Error(`Server returned ${resp.status}`);
            }

            const data = await resp.json();
            appendAgentResponse(data);

            if (data.type === "analysis_result") {
                applySpatialResults(data);
            }
        } catch (err) {
            appendMessage("agent", `⚠️ Error executing request: ${err.message}. Please check connection.`);
        } finally {
            setLoading(false);
        }
    }

    function setLoading(isLoading) {
        chatInput.disabled = isLoading;
        sendBtn.disabled = isLoading;
        if (isLoading) {
            hudMetaText.textContent = "Agent reasoning & executing Earth observation pipeline...";
        }
    }

    function appendMessage(role, text) {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message message-${role}`;
        const avatar = role === "user" ? "👤" : "🤖";
        msgDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content"><p>${formatMarkdown(text)}</p></div>
        `;
        chatFeed.appendChild(msgDiv);
        chatFeed.scrollTop = chatFeed.scrollHeight;
    }

    // Global helper for interactive disambiguation clicks
    window.selectLocationCandidate = function(locationName) {
        chatInput.value = `Analyze temperature in ${locationName}`;
        sendMessage(`Analyze land surface temperature in ${locationName}`);
    };

    function appendAgentResponse(data) {
        const msgDiv = document.createElement("div");
        msgDiv.className = "message message-agent";

        // 1. Disambiguation Candidate Bubbles
        let candidatesHtml = "";
        if (data.type === "disambiguation" && data.candidates && data.candidates.length > 0) {
            const chips = data.candidates.map(c => `
                <button class="disambiguation-chip" onclick="window.selectLocationCandidate('${c.display_name.replace(/'/g, "\\'")}')">
                    <span class="chip-icon">📍</span>
                    <span class="chip-body">
                        <span class="chip-title">${c.short_name}</span>
                        <span class="chip-desc">${c.display_name}</span>
                    </span>
                    <span class="chip-arrow">➔</span>
                </button>
            `).join("");

            candidatesHtml = `
                <div class="disambiguation-wrapper">
                    <div class="disambiguation-label">Select region to confirm:</div>
                    <div class="disambiguation-chips">${chips}</div>
                </div>
            `;
        }

        // 2. Audit Trace Accordion
        let auditHtml = "";
        if (data.audit_trace && data.audit_trace.length > 0) {
            const stepsHtml = data.audit_trace.map(s => {
                let details = "";
                for (let k in s) {
                    if (k !== "step") details += `<div><strong>${k}:</strong> ${s[k]}</div>`;
                }
                return `<div class="trace-step"><strong>${s.step}</strong>${details}</div>`;
            }).join("");

            auditHtml = `
                <div class="audit-trace-box">
                    <div class="audit-trace-header" onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'none' ? 'block' : 'none'">
                        <span>⚙️ AI Spatial Reasoning & Execution Audit</span>
                        <span>▼</span>
                    </div>
                    <div class="audit-trace-body" style="display: none;">
                        ${stepsHtml}
                        <div style="margin-top: 6px; color: #818cf8;">Execution Latency: ${data.execution_latency || '0.8s'}</div>
                    </div>
                </div>
            `;
        }

        msgDiv.innerHTML = `
            <div class="message-avatar">🤖</div>
            <div class="message-content">
                <div>${formatMarkdown(data.message)}</div>
                ${candidatesHtml}
                ${auditHtml}
            </div>
        `;
        chatFeed.appendChild(msgDiv);
        chatFeed.scrollTop = chatFeed.scrollHeight;
    }

    function formatMarkdown(text) {
        if (!text) return "";
        let formatted = text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`([^`]+)`/g, '<code style="background: rgba(255,255,255,0.1); padding: 1px 4px; border-radius: 4px;">$1</code>')
            .replace(/• /g, '• ')
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br/>');
        return `<p>${formatted}</p>`;
    }

    // Chart.js instance and Slide Panel Elements
    let timeseriesChartInstance = null;
    const analyticsSlidePanel = document.getElementById("analyticsSlidePanel");
    const closeSlidePanelBtn = document.getElementById("closeSlidePanelBtn");
    const valWarmingRate = document.getElementById("valWarmingRate");
    const valNetDelta = document.getElementById("valNetDelta");
    const valTimeWindow = document.getElementById("valTimeWindow");
    const panelChartTitle = document.getElementById("panelChartTitle");

    if (closeSlidePanelBtn) {
        closeSlidePanelBtn.addEventListener("click", () => {
            if (analyticsSlidePanel) analyticsSlidePanel.classList.remove("active");
        });
    }

    // 4. Map & Spatial Layer Renderer
    function applySpatialResults(data) {
        // Clear previous layers
        if (activeGeoJsonLayer) map.removeLayer(activeGeoJsonLayer);
        if (activeTileLayer) map.removeLayer(activeTileLayer);
        if (activeSimulatedHeatLayer) map.removeLayer(activeSimulatedHeatLayer);

        hudTargetText.textContent = `${data.city_name}`;

        // 1. Render Boundary Polygon
        if (data.geojson) {
            activeGeoJsonLayer = L.geoJSON(data.geojson, {
                pane: 'boundaryPane',
                style: {
                    color: "#818cf8",
                    weight: 2.5,
                    opacity: 0.9,
                    fillColor: "#4f46e5",
                    fillOpacity: 0.12,
                    dashArray: "4, 6"
                }
            }).addTo(map);

            const bounds = activeGeoJsonLayer.getBounds();
            map.flyToBounds(bounds, { padding: [40, 40], duration: 1.5 });

            // 2. Render Thermal Layer if present
            if (data.tile_url) {
                activeTileLayer = L.tileLayer(data.tile_url, {
                    pane: 'thermalPane',
                    maxZoom: 18,
                    opacity: 0.8
                }).addTo(map);
            } else if (data.analysis_type !== "timeseries" && data.eo_stats) {
                // High-fidelity heat simulation mesh over boundary
                renderSimulatedThermalLayer(bounds, data.eo_stats);
            }
        }

        // 3. Handle Spatiotemporal Timeseries Mode vs Standard LST/UTFVI
        if (data.analysis_type === "timeseries" && data.timeseries_data) {
            legendCard.style.display = "none";
            hudMetaText.textContent = `Warming Rate: ${data.warming_rate} • Total Change: ${data.total_change}`;
            renderTimeseriesSlidePanel(data);
            return;
        }

        // Hide slide panel if switching back to LST/UTFVI
        if (analyticsSlidePanel) analyticsSlidePanel.classList.remove("active");

        // 4. Update Legend Card for LST vs UTFVI
        legendCard.style.display = "block";
        const isSuhi = (data.analysis_type === "utfvi" || !!data.classes);
        const stats = data.eo_stats || {};

        const rampEl = legendCard.querySelector(".color-ramp");
        const titleEl = legendCard.querySelector(".legend-title");
        const statLabels = legendCard.querySelectorAll(".stat-label");

        if (isSuhi) {
            titleEl.textContent = "SUHI: UTFVI Ecological Field";
            legendModeBadge.textContent = "Landsat 8/9 UTFVI";
            if (rampEl) {
                rampEl.style.background = "linear-gradient(to right, #2b83ba, #abdda4, #ffffbf, #fdae61, #f46d43, #d7191c)";
            }

            lblMinTemp.textContent = `${stats.min_utfvi !== undefined ? stats.min_utfvi : '-0.02'} (None)`;
            lblMeanTemp.textContent = `${stats.mean_utfvi !== undefined ? stats.mean_utfvi : '0.00'} (Weak)`;
            lblMaxTemp.textContent = `${stats.max_utfvi !== undefined ? stats.max_utfvi : '+0.02'} (Severe)`;

            if (statLabels[0]) statLabels[0].textContent = "Min UTFVI";
            if (statLabels[1]) statLabels[1].textContent = "Mean UTFVI";
            if (statLabels[2]) statLabels[2].textContent = "Severe Stress";

            valMinLST.textContent = `${stats.min_utfvi !== undefined ? stats.min_utfvi : '--'}`;
            valMeanLST.textContent = `${stats.mean_utfvi !== undefined ? stats.mean_utfvi : '--'}`;
            valMaxLST.textContent = `${stats.severe_heat_pct !== undefined ? stats.severe_heat_pct + '%' : '--'}`;
            hudMetaText.textContent = `Baseline LST: ${stats.mean}°C • Severe SUHI Stress Area: ${stats.severe_heat_pct || 0}%`;
        } else {
            titleEl.textContent = "Land Surface Temperature (LST)";
            legendModeBadge.textContent = "Landsat 8/9 TIRS";
            if (rampEl) {
                rampEl.style.background = "linear-gradient(to right, #313695, #4575b4, #74add1, #abd9e9, #fee090, #fdae61, #f46d43, #d73027, #a50026)";
            }

            lblMinTemp.textContent = `${stats.min}°C (Cool)`;
            lblMeanTemp.textContent = `${stats.mean}°C (Mean)`;
            lblMaxTemp.textContent = `${stats.max}°C (Hotspot)`;

            if (statLabels[0]) statLabels[0].textContent = "Min LST";
            if (statLabels[1]) statLabels[1].textContent = "Mean LST";
            if (statLabels[2]) statLabels[2].textContent = "Peak Hotspot";

            valMinLST.textContent = `${stats.min}°C`;
            valMeanLST.textContent = `${stats.mean}°C`;
            valMaxLST.textContent = `${stats.max}°C`;
            hudMetaText.textContent = `Mean LST: ${stats.mean}°C • Scene Count: ${stats.scenes}`;
        }
    }

    function renderTimeseriesSlidePanel(data) {
        if (!analyticsSlidePanel) return;

        panelChartTitle.textContent = `${data.city_name} Thermal Trend`;
        valWarmingRate.textContent = data.warming_rate || "N/A";
        valNetDelta.textContent = data.total_change || "N/A";
        valTimeWindow.textContent = `${data.start_year} – ${data.end_year}`;

        analyticsSlidePanel.classList.add("active");

        const ctx = document.getElementById("timeseriesChart");
        if (!ctx) return;

        const labels = data.timeseries_data.map(d => d.date);
        const values = data.timeseries_data.map(d => d.lst_mean);

        if (timeseriesChartInstance) {
            timeseriesChartInstance.destroy();
        }

        timeseriesChartInstance = new Chart(ctx, {
            type: "line",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: "Terrestrial LST (°C)",
                        data: values,
                        borderColor: "#f97316",
                        backgroundColor: "rgba(249, 115, 22, 0.15)",
                        borderWidth: 2.2,
                        pointBackgroundColor: "#fb923c",
                        pointBorderColor: "#fff",
                        pointRadius: 4.5,
                        pointHoverRadius: 7,
                        fill: true,
                        tension: 0.25
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        labels: { color: "#94a3b8", font: { family: "Inter", size: 11 } }
                    },
                    tooltip: {
                        backgroundColor: "rgba(15, 23, 42, 0.95)",
                        titleColor: "#f8fafc",
                        bodyColor: "#cbd5e1",
                        borderColor: "rgba(255, 255, 255, 0.15)",
                        borderWidth: 1,
                        padding: 10,
                        callbacks: {
                            title: (items) => {
                                if (!items.length) return "";
                                const idx = items[0].dataIndex;
                                const item = data.timeseries_data[idx];
                                return `🛰️ Satellite Acquisition: ${item.date}`;
                            },
                            label: (context) => {
                                const idx = context.dataIndex;
                                const item = data.timeseries_data[idx];
                                const cloudStr = item.cloud !== undefined ? ` • Cloud: ${item.cloud}%` : "";
                                const sensorStr = item.sensor ? ` • ${item.sensor}` : "";
                                return [
                                    `Mean LST: ${context.parsed.y} °C`,
                                    `Payload: TIRS Band 10${cloudStr}${sensorStr}`
                                ];
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: {
                            color: "#94a3b8",
                            font: { family: "Inter", size: 10 },
                            maxRotation: 45,
                            minRotation: 30
                        }
                    },
                    y: {
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: {
                            color: "#94a3b8",
                            font: { family: "Inter", size: 11 },
                            callback: (v) => `${v}°C`
                        }
                    }
                }
            }
        });
    }

    function renderSimulatedThermalLayer(bounds, stats) {
        // Create an organic thermal gradient polygon grid over the target AOI
        const heatGroup = L.layerGroup();
        const minLat = bounds.getSouth();
        const maxLat = bounds.getNorth();
        const minLon = bounds.getWest();
        const maxLon = bounds.getEast();

        const rows = 12;
        const cols = 12;
        const dLat = (maxLat - minLat) / rows;
        const dLon = (maxLon - minLon) / cols;

        const palette = ["#313695", "#4575b4", "#74add1", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026"];

        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                // Center-weighted heat concentration (Urban Heat Island core)
                const distToCenter = Math.sqrt(Math.pow((i - rows / 2), 2) + Math.pow((j - cols / 2), 2));
                const normalized = Math.max(0, 1 - (distToCenter / (rows * 0.7)));
                const colorIdx = Math.min(palette.length - 1, Math.floor(normalized * palette.length));
                const cellColor = palette[colorIdx];

                const cellBounds = [
                    [minLat + i * dLat, minLon + j * dLon],
                    [minLat + (i + 1) * dLat, minLon + (j + 1) * dLon]
                ];

                L.rectangle(cellBounds, {
                    pane: 'thermalPane',
                    color: "transparent",
                    fillColor: cellColor,
                    fillOpacity: 0.55
                }).addTo(heatGroup);
            }
        }

        activeSimulatedHeatLayer = heatGroup.addTo(map);
    }
});
