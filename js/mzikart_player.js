import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "Mzikart.PlayerNode",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "MzikartPlayerNode") {
            const orig_nodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function () {
                orig_nodeCreated?.apply(this, arguments);
                
                // Create player UI
                this.createPlayerUI();
                
                // Setup WebSocket listener for real-time updates
                this.setupWebSocketListener();
            };
            
            nodeType.prototype.setupWebSocketListener = function() {
                const onMessageReceived = (event) => {
                    try {
                        const message = JSON.parse(event.data);
                        if (message.type === "mzikart_player_update" && 
                            message.node_id === this.id && 
                            this.playerControl) {
                            this.playerControl.updateFromWebSocket(message.data);
                        }
                    } catch (error) {
                        console.error("Error processing WebSocket message:", error);
                    }
                };
                
                // Use ComfyUI's existing WebSocket connection
                if (app.api && app.api.socket) {
                    app.api.socket.addEventListener("message", onMessageReceived);
                    
                    // Store reference to remove listener on cleanup
                    this._wsListener = onMessageReceived;
                }
            };
            
            // Clean up WebSocket listener when node is removed
            const orig_onRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function() {
                if (this._wsListener && app.api && app.api.socket) {
                    app.api.socket.removeEventListener("message", this._wsListener);
                }
                orig_onRemoved?.apply(this, arguments);
            };
            
            const orig_onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                orig_onExecuted?.apply(this, [message]);
                
                // Update player with new audio data
                if (message.ui && message.ui.audio_info) {
                    if (this.playerControl) {
                        this.playerControl.updatePlayerUI(message.ui);
                    }
                }
            };
        }
    },
});

class MzikartPlayerControl {
    constructor(node) {
        this.node = node;
        this.audioPlayer = null;
        this.playPauseBtn = null;
        this.stopBtn = null;
        this.progressSlider = null;
        this.volumeSlider = null;
        this.fileDropdown = null;
        this.durationDisplay = null;
        this.currentTimeDisplay = null;
        
        this.createUI();
        this.registerEventHandlers();
    }
    
    createUI() {
        // Create player container
        this.container = document.createElement('div');
        this.container.className = 'mzikart-player';
        this.container.style.padding = '10px';
        this.container.style.border = '1px solid #444';
        this.container.style.borderRadius = '5px';
        this.container.style.marginTop = '10px';
        
        // File browser
        const fileGroup = document.createElement('div');
        fileGroup.style.marginBottom = '10px';
        
        this.fileDropdown = document.createElement('select');
        this.fileDropdown.style.width = '100%';
        this.fileDropdown.style.padding = '5px';
        this.fileDropdown.style.backgroundColor = '#333';
        this.fileDropdown.style.color = '#fff';
        this.fileDropdown.style.border = '1px solid #555';
        
        fileGroup.appendChild(this.fileDropdown);
        this.container.appendChild(fileGroup);
        
        // Player controls
        const controlsGroup = document.createElement('div');
        controlsGroup.style.display = 'flex';
        controlsGroup.style.gap = '10px';
        controlsGroup.style.marginBottom = '10px';
        controlsGroup.style.alignItems = 'center';
        
        this.playPauseBtn = document.createElement('button');
        this.playPauseBtn.innerHTML = '▶';
        this.playPauseBtn.style.padding = '5px 10px';
        
        this.stopBtn = document.createElement('button');
        this.stopBtn.innerHTML = '⏹';
        this.stopBtn.style.padding = '5px 10px';
        
        controlsGroup.appendChild(this.playPauseBtn);
        controlsGroup.appendChild(this.stopBtn);
        
        // Volume control
        this.volumeSlider = document.createElement('input');
        this.volumeSlider.type = 'range';
        this.volumeSlider.min = '0';
        this.volumeSlider.max = '1';
        this.volumeSlider.step = '0.01';
        this.volumeSlider.value = '1';
        this.volumeSlider.style.width = '80px';
        
        controlsGroup.appendChild(document.createTextNode('🔊'));
        controlsGroup.appendChild(this.volumeSlider);
        this.container.appendChild(controlsGroup);
        
        // Progress bar
        const progressGroup = document.createElement('div');
        progressGroup.style.display = 'flex';
        progressGroup.style.gap = '5px';
        progressGroup.style.alignItems = 'center';
        
        this.currentTimeDisplay = document.createElement('div');
        this.currentTimeDisplay.textContent = '00:00';
        this.currentTimeDisplay.style.minWidth = '40px';
        this.currentTimeDisplay.style.textAlign = 'center';
        
        this.progressSlider = document.createElement('input');
        this.progressSlider.type = 'range';
        this.progressSlider.min = '0';
        this.progressSlider.max = '100';
        this.progressSlider.value = '0';
        this.progressSlider.style.flex = '1';
        
        this.durationDisplay = document.createElement('div');
        this.durationDisplay.textContent = '00:00';
        this.durationDisplay.style.minWidth = '40px';
        this.durationDisplay.style.textAlign = 'center';
        
        progressGroup.appendChild(this.currentTimeDisplay);
        progressGroup.appendChild(this.progressSlider);
        progressGroup.appendChild(this.durationDisplay);
        this.container.appendChild(progressGroup);
        
        // Add to node
        this.node.addDOMWidget('mzikart_player', this.container);
    }
    
    registerEventHandlers() {
        this.playPauseBtn.addEventListener('click', () => {
            this.togglePlayback();
        });
        
        this.stopBtn.addEventListener('click', () => {
            this.stopPlayback();
        });
        
        this.volumeSlider.addEventListener('input', () => {
            this.setVolume(parseFloat(this.volumeSlider.value));
        });
        
        this.progressSlider.addEventListener('input', () => {
            this.seekAudio(parseFloat(this.progressSlider.value));
        });
        
        this.fileDropdown.addEventListener('change', () => {
            this.loadSelectedFile();
        });
    }
    
    togglePlayback() {
        const action = this.playPauseBtn.innerHTML === '▶' ? 'play' : 'pause';
        this.sendControlCommand(action);
        this.playPauseBtn.innerHTML = action === 'play' ? '⏸' : '▶';
    }
    
    stopPlayback() {
        this.sendControlCommand('stop');
        this.playPauseBtn.innerHTML = '▶';
    }
    
    setVolume(volume) {
        this.sendControlCommand('set_volume', volume);
    }
    
    seekAudio(position) {
        this.sendControlCommand('set_position', position);
    }
    
    loadSelectedFile() {
        const selectedFile = this.fileDropdown.value;
        if (selectedFile) {
            this.sendControlCommand('select_file', selectedFile);
        }
    }
    
    sendControlCommand(action, value = null) {
        fetch('/mzikart/player/control', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                node_id: this.node.id,
                action: action,
                value: value
            })
        })
        .catch(error => {
            console.error('Error sending player command:', error);
        });
    }
    
    updateFromWebSocket(data) {
        // Update play/pause button
        if (data.playing !== undefined) {
            this.playPauseBtn.innerHTML = data.playing ? '⏸' : '▶';
        }
        
        // Update current position
        if (data.position !== undefined) {
            this.progressSlider.value = data.position;
            this.currentTimeDisplay.textContent = this.formatTime(data.position);
        }
        
        // Update volume
        if (data.volume !== undefined) {
            this.volumeSlider.value = data.volume;
        }
    }
    
    updatePlayerUI(data) {
        // Update file dropdown
        this.fileDropdown.innerHTML = '';
        const emptyOption = document.createElement('option');
        emptyOption.value = '';
        emptyOption.textContent = 'Select audio file...';
        this.fileDropdown.appendChild(emptyOption);
        
        if (data.file_list && Array.isArray(data.file_list)) {
            data.file_list.forEach(file => {
                const option = document.createElement('option');
                option.value = file;
                option.textContent = file;
                if (file === this.node.audio_file) {
                    option.selected = true;
                }
                this.fileDropdown.appendChild(option);
            });
        }
        
        // Update duration display
        if (data.audio_info && data.audio_info.duration_str) {
            this.durationDisplay.textContent = data.audio_info.duration_str;
            this.progressSlider.max = data.audio_info.duration || 100;
        }
        
        // Update current position
        if (data.position !== undefined) {
            this.progressSlider.value = data.position;
            this.currentTimeDisplay.textContent = this.formatTime(data.position);
        }
        
        // Update volume
        if (data.volume !== undefined) {
            this.volumeSlider.value = data.volume;
        }
        
        // Update play/pause button
        if (data.playing !== undefined) {
            this.playPauseBtn.innerHTML = data.playing ? '⏸' : '▶';
        }
    }
    
    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
}

// Add player control to node
app.registerExtension({
    name: "Mzikart.PlayerNode.UI",
    nodeCreated(node) {
        if (node.getTitle() === "🎵 Mzikart Player") {
            node.playerControl = new MzikartPlayerControl(node);
        }
    },
});