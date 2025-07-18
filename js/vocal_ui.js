app.registerExtension({
    name: "ComfyUI.VocalNodes",
    init() {
        // Add custom styling
        const style = document.createElement('style');
        style.textContent = `
            .vocal-node-title {
                color: #ff7b7b;
                font-weight: bold;
            }
            .vocal-control-group {
                background-color: #2d1a1a;
                padding: 10px;
                border-radius: 8px;
                margin-bottom: 10px;
            }
        `;
        document.head.appendChild(style);
    },
    
    async setup() {
        // Register custom node UI behavior
        this.addVocalNodeUI();
    },
    
    addVocalNodeUI() {
        // Add genre-specific presets to VocalProcessorNode
        LiteGraph.registerNodeConstructed("VocalProcessorNode", (node) => {
            // Create preset selector
            const presetSelect = document.createElement('select');
            presetSelect.innerHTML = `
                <option value="custom">Custom Settings</option>
                <option value="rap">Rap/Trap Preset</option>
                <option value="rnb">R&B/Soul Preset</option>
                <option value="gospel">Gospel Preset</option>
                <option value="pop">Pop Preset</option>
            `;
            
            presetSelect.addEventListener('change', (e) => {
                const preset = e.target.value;
                if (preset !== 'custom') {
                    this.applyPreset(node, preset);
                }
            });
            
            // Add to node
            const container = document.createElement('div');
            container.className = 'vocal-control-group';
            container.appendChild(document.createTextNode('Presets: '));
            container.appendChild(presetSelect);
            node.addCustomWidget(container);
        });
    },
    
    applyPreset(node, preset) {
        // Apply preset values based on genre
        switch(preset) {
            case 'rap':
                node.setProperty('aggressiveness', 0.8);
                node.widgets.find(w => w.name === 'aggressiveness').value = 0.8;
                break;
            case 'rnb':
                node.setProperty('aggressiveness', 0.4);
                node.widgets.find(w => w.name === 'aggressiveness').value = 0.4;
                break;
            case 'gospel':
                node.setProperty('aggressiveness', 0.6);
                node.widgets.find(w => w.name === 'aggressiveness').value = 0.6;
                break;
            case 'pop':
                node.setProperty('aggressiveness', 0.5);
                node.widgets.find(w => w.name === 'aggressiveness').value = 0.5;
                break;
        }
        
        // Trigger value change
        node.onPropertyChanged('aggressiveness');
    }
});