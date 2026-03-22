/**
 * HearmemanAI Prompter - Premium UI with Dynamic Image Loader
 * Uses JS-side uploads with lightweight file references to avoid workflow bloat.
 */

import { app } from "../../scripts/app.js";

console.log("[HearmemanAI] Loading extension...");

// Brand color palette
const COLORS = {
    background: "#1E0A2E",
    backgroundLight: "#2D1B3D",
    backgroundDark: "#150720",
    primary: "#E63946",
    secondary: "#E91E8C",
    tertiary: "#9B2D7B",
    highlight: "#FFB627",
    text: "#FFFFFF",
    textMuted: "#B8A0C8",
    border: "#3D2B4D"
};

// Inject global styles
function injectStyles() {
    if (document.getElementById("hearmeman-styles")) return;

    const css = `
        :root {
            /* ... keep your color variables ... */
            --hearmeman-background: ${COLORS.background};
            --hearmeman-background-light: ${COLORS.backgroundLight};
            --hearmeman-background-dark: ${COLORS.backgroundDark};
            --hearmeman-primary: ${COLORS.primary};
            --hearmeman-secondary: ${COLORS.secondary};
            --hearmeman-tertiary: ${COLORS.tertiary};
            --hearmeman-highlight: ${COLORS.highlight};
            --hearmeman-text: ${COLORS.text};
            --hearmeman-text-muted: ${COLORS.textMuted};
        }

        .hearmeman-image-loader {
            background: var(--hearmeman-background-dark);
            border: 2px dashed var(--hearmeman-secondary);
            border-radius: 8px;
            padding: 10px;
            /* Fixed height for stability */
            height: 340px; 
            
            /* FIX 1: Width calculation */
            width: calc(100% - 20px) !important;
            
            /* FIX 2: Margins - Top | Right | Bottom (30px) | Left */
            margin: 10px 10px 30px 10px !important;
            
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            position: relative;
        }

        .hearmeman-image-loader:hover {
            border-color: var(--hearmeman-highlight);
            box-shadow: 0 0 15px rgba(233, 30, 140, 0.19);
        }

        /* --- Empty State --- */
        .hearmeman-drop-zone {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: var(--hearmeman-text-muted);
            font-size: 13px;
            text-align: center;
        }

        .hearmeman-drop-zone-icon { font-size: 40px; margin-bottom: 15px; opacity: 0.8; }
        .hearmeman-add-btn {
            background: linear-gradient(135deg, var(--hearmeman-primary), var(--hearmeman-secondary));
            border: none; border-radius: 6px; color: #fff; padding: 10px 20px;
            font-weight: bold; cursor: pointer; margin-top: 15px;
        }

        /* --- Grid Layout --- */
        .hearmeman-thumbnails {
            flex: 1;
            display: grid;
            gap: 8px;
            width: 100%;
            height: 100%;
            overflow: hidden;
            /* Defaults will be set by JS classes */
        }

        /* Grid Variations */
        .grid-1 { grid-template-columns: 1fr; grid-template-rows: 1fr; }
        .grid-2 { grid-template-columns: 1fr 1fr; grid-template-rows: 1fr; }
        .grid-4 { grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr; }
        .grid-6 { grid-template-columns: 1fr 1fr 1fr; grid-template-rows: 1fr 1fr; }

        .hearmeman-thumb {
            position: relative;
            width: 100%;
            height: 100%;
            border-radius: 6px;
            overflow: hidden;
            border: 1px solid var(--hearmeman-border);
            background: #000;
        }
        
        .hearmeman-thumb img {
            width: 100%; height: 100%; object-fit: cover;
            transition: transform 0.3s ease;
        }
        
        .hearmeman-thumb:hover img { transform: scale(1.05); }

        /* Remove Button (Top Right) */
        .hearmeman-thumb-remove {
            position: absolute; top: 5px; right: 5px;
            width: 24px; height: 24px;
            background: rgba(230, 57, 70, 0.9);
            border: none; border-radius: 50%; color: white;
            font-weight: bold; cursor: pointer; display: none;
            align-items: center; justify-content: center;
        }
        .hearmeman-thumb:hover .hearmeman-thumb-remove { display: flex; }

        /* --- Footer Controls (Pagination & Clear) --- */
        .hearmeman-footer {
            height: 30px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-top: 8px;
            padding: 0 4px;
        }

        .hearmeman-paginator {
            display: flex; gap: 10px; align-items: center;
            color: var(--hearmeman-text-muted); font-size: 12px;
        }
        
        .hearmeman-page-btn {
            background: var(--hearmeman-border); border: none;
            color: white; border-radius: 4px; width: 24px; height: 24px;
            cursor: pointer; display: flex; align-items: center; justify-content: center;
        }
        .hearmeman-page-btn:hover { background: var(--hearmeman-highlight); color: black; }
        .hearmeman-page-btn:disabled { opacity: 0.3; cursor: default; pointer-events: none; }

        .hearmeman-clear-btn {
            background: transparent; border: 1px solid var(--hearmeman-primary);
            color: var(--hearmeman-primary); font-size: 11px; padding: 4px 10px;
            border-radius: 4px; cursor: pointer;
        }
        .hearmeman-clear-btn:hover { background: var(--hearmeman-primary); color: white; }
    `;

    const style = document.createElement("style");
    style.id = "hearmeman-styles";
    style.textContent = css;
    document.head.appendChild(style);
    console.log("[HearmemanAI] Styles injected");
}

// Build a server image URL
function buildImageUrl(meta) {
    const name = meta.name;
    const type = meta.type || "input";
    const subfolder = meta.subfolder || "";
    const params = new URLSearchParams({
        filename: name,
        type,
        subfolder
    });
    return `view?${params.toString()}`;
}

// Image Loader Widget Factory
function createImageLoader(node, metaWidget) {
    const widget = {
        images: [],
        element: null,
        metaWidget: metaWidget || null,
        currentPage: 0,
        pageSize: 6 // Maximum images per page
    };

    // 1. Structure Creation
    const container = document.createElement("div");
    container.className = "hearmeman-image-loader";

    // Drop Zone (Empty State)
    const dropZone = document.createElement("div");
    dropZone.className = "hearmeman-drop-zone";
    dropZone.innerHTML = `
        <div class="hearmeman-drop-zone-icon">�</div>
        <div class="hearmeman-drop-zone-text">Drag & drop images here</div>
        <button class="hearmeman-add-btn">+ Add Images</button>
    `;

    // File Input (Hidden)
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.multiple = true;
    fileInput.accept = "image/*";
    fileInput.style.display = "none";

    // Thumbnail Grid
    const thumbsContainer = document.createElement("div");
    thumbsContainer.className = "hearmeman-thumbnails";
    thumbsContainer.style.display = "none";

    // Footer (Pagination + Clear)
    const footer = document.createElement("div");
    footer.className = "hearmeman-footer";
    footer.style.display = "none";

    // Paginator HTML
    footer.innerHTML = `
        <div class="hearmeman-paginator">
            <button class="hearmeman-page-btn prev-btn">◀</button>
            <span class="page-info">1 / 1</span>
            <button class="hearmeman-page-btn next-btn">▶</button>
        </div>
        <button class="hearmeman-clear-btn">Clear All</button>
    `;

    // Error Label
    const errorLabel = document.createElement("div");
    errorLabel.style.display = "none";
    errorLabel.style.color = "#ff8080";
    errorLabel.style.fontSize = "11px";
    errorLabel.style.textAlign = "center";
    errorLabel.style.position = "absolute";
    errorLabel.style.bottom = "40px";
    errorLabel.style.width = "100%";

    container.appendChild(dropZone);
    container.appendChild(fileInput);
    container.appendChild(thumbsContainer);
    container.appendChild(footer);
    container.appendChild(errorLabel);

    // 2. Logic: Sync Data
    function syncToMetaWidget() {
        if (!widget.metaWidget) return;
        const meta = widget.images.map(img => ({
            name: img.name,
            subfolder: img.subfolder || "",
            type: img.type || "input"
        }));
        try {
            widget.metaWidget.value = JSON.stringify(meta);
        } catch (err) {
            console.warn("Serialization failed", err);
        }
    }

    // 3. Logic: Update UI (The Brains)
    function updateUI() {
        const total = widget.images.length;
        const hasImages = total > 0;

        // Toggle States
        dropZone.style.display = hasImages ? "none" : "flex";
        thumbsContainer.style.display = hasImages ? "grid" : "none";
        footer.style.display = hasImages ? "flex" : "none";

        if (!hasImages) {
            // Reset Node Size for Empty State (Increased to 600)
            node.size[1] = 600;
            node.setDirtyCanvas(true, true);
            return;
        }

        // Pagination Math
        const totalPages = Math.ceil(total / widget.pageSize);
        // Ensure current page is valid
        if (widget.currentPage >= totalPages) widget.currentPage = totalPages - 1;
        if (widget.currentPage < 0) widget.currentPage = 0;

        const startIndex = widget.currentPage * widget.pageSize;
        const endIndex = startIndex + widget.pageSize;
        const pageImages = widget.images.slice(startIndex, endIndex);

        // Smart Grid Class Selection
        // Removes old classes
        thumbsContainer.classList.remove('grid-1', 'grid-2', 'grid-4', 'grid-6');

        const visibleCount = pageImages.length;
        if (visibleCount === 1) thumbsContainer.classList.add('grid-1');
        else if (visibleCount === 2) thumbsContainer.classList.add('grid-2');
        else if (visibleCount <= 4) thumbsContainer.classList.add('grid-4');
        else thumbsContainer.classList.add('grid-6');

        // Render Thumbnails
        thumbsContainer.innerHTML = "";
        pageImages.forEach((img, pageIndex) => {
            const realIndex = startIndex + pageIndex; // Index in the main array

            const thumb = document.createElement("div");
            thumb.className = "hearmeman-thumb";
            thumb.innerHTML = `
                <img src="${img.url}" alt="${img.name}">
                <button class="hearmeman-thumb-remove" title="Remove">×</button>
            `;

            // Remove Event
            thumb.querySelector(".hearmeman-thumb-remove").onclick = (e) => {
                e.stopPropagation();
                widget.images.splice(realIndex, 1);
                // If page becomes empty, go back one page
                if (widget.images.length > 0 && widget.images.length <= startIndex) {
                    widget.currentPage--;
                }
                updateUI();
                syncToMetaWidget();
            };
            thumbsContainer.appendChild(thumb);
        });

        // Update Paginator Controls
        const prevBtn = footer.querySelector(".prev-btn");
        const nextBtn = footer.querySelector(".next-btn");
        const pageInfo = footer.querySelector(".page-info");

        pageInfo.textContent = `${widget.currentPage + 1} / ${totalPages}`;

        // Only show prev/next buttons if we actually have multiple pages
        prevBtn.style.visibility = totalPages > 1 ? "visible" : "hidden";
        nextBtn.style.visibility = totalPages > 1 ? "visible" : "hidden";
        prevBtn.disabled = widget.currentPage === 0;
        nextBtn.disabled = widget.currentPage === totalPages - 1;

        // Keep Node Size Fixed & Stable (Increased to 600)
        node.size[1] = 600;
        node.setDirtyCanvas(true, true);
    }

    // 4. Event Handlers
    // File Handling
    async function handleFiles(files) {
        if (!files || !files.length) return;
        const imageFiles = Array.from(files).filter(f => f.type.startsWith("image/"));

        for (const file of imageFiles) {
            try {
                const formData = new FormData();
                formData.append("image", file);
                formData.append("subfolder", `HearmemanAI/${node.id}`);
                formData.append("type", "input");
                formData.append("overwrite", "true");

                const res = await fetch("/upload/image", { method: "POST", body: formData });
                if (!res.ok) throw new Error("Upload Failed");

                const info = await res.json();
                const meta = { name: info.name, subfolder: info.subfolder, type: info.type };

                widget.images.push({ ...meta, url: buildImageUrl(meta) });
            } catch (e) {
                console.error(e);
            }
        }
        // Jump to the last page to see new upload
        widget.currentPage = Math.ceil(widget.images.length / widget.pageSize) - 1;
        updateUI();
        syncToMetaWidget();
    }

    // DOM Events
    dropZone.querySelector(".hearmeman-add-btn").onclick = (e) => { e.stopPropagation(); fileInput.click(); };
    dropZone.onclick = () => fileInput.click();
    fileInput.onchange = (e) => { handleFiles(e.target.files); fileInput.value = ""; };

    footer.querySelector(".hearmeman-clear-btn").onclick = () => {
        widget.images = [];
        widget.currentPage = 0;
        updateUI();
        syncToMetaWidget();
    };

    footer.querySelector(".prev-btn").onclick = () => {
        if (widget.currentPage > 0) {
            widget.currentPage--;
            updateUI();
        }
    };

    footer.querySelector(".next-btn").onclick = () => {
        const totalPages = Math.ceil(widget.images.length / widget.pageSize);
        if (widget.currentPage < totalPages - 1) {
            widget.currentPage++;
            updateUI();
        }
    };

    // Drag & Drop
    container.ondragenter = (e) => { e.preventDefault(); container.classList.add("dragover"); };
    container.ondragover = (e) => { e.preventDefault(); };
    container.ondragleave = (e) => { e.preventDefault(); container.classList.remove("dragover"); };
    container.ondrop = (e) => {
        e.preventDefault();
        container.classList.remove("dragover");
        handleFiles(e.dataTransfer.files);
    };

    // Restore State
    if (metaWidget && metaWidget.value) {
        try {
            const loaded = JSON.parse(metaWidget.value);
            if (Array.isArray(loaded)) {
                widget.images = loaded.map(item => ({
                    ...item,
                    url: buildImageUrl(item)
                }));
                updateUI();
            }
        } catch (e) { }
    }

    widget.element = container;
    widget.updateUI = updateUI;
    return widget;
}

// Register extension
app.registerExtension({
    name: "HearmemanAI.Prompter",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "HearmemanAI_Prompter") return;

        injectStyles();

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origOnNodeCreated?.apply(this, arguments);

            this.color = COLORS.backgroundLight;
            this.bgcolor = COLORS.background;
            this._isPremium = true;

            let metaWidget = null;
            if (this.widgets) {
                for (let i = 0; i < this.widgets.length; i++) {
                    const w = this.widgets[i];
                    if (w.name === "loader_images_meta") {
                        metaWidget = w;
                        w.computeSize = () => [0, -4];
                        w.type = "hidden";
                        if (w.element) w.element.style.display = "none";
                        break;
                    }
                }
            }

            const loader = createImageLoader(this, metaWidget);
            this.addDOMWidget("image_loader", "custom", loader.element, {
                serialize: false,
                hideOnZoom: false
            });
            this._imageLoader = loader;

            // FIX: Default height increased to accommodate text area + carousel
            this.size[0] = 380;
            this.size[1] = 600;
        };

        // Custom drawing with FIXED Context Management
        const origDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            origDrawForeground?.apply(this, arguments);
            if (!this._isPremium) return;

            // 1. SAVE THE CONTEXT (Isolate changes)
            ctx.save();

            const h = LiteGraph.NODE_TITLE_HEIGHT;

            // Gradient Header
            const grad = ctx.createLinearGradient(0, -h, this.size[0], 0);
            grad.addColorStop(0, COLORS.primary);
            grad.addColorStop(0.5, COLORS.secondary);
            grad.addColorStop(1, COLORS.tertiary);
            ctx.fillStyle = grad;
            ctx.beginPath();
            ctx.roundRect ? ctx.roundRect(0, -h, this.size[0], h, [8, 8, 0, 0]) : ctx.rect(0, -h, this.size[0], h);
            ctx.fill();

            // Title text (This was previously leaking font styles)
            ctx.fillStyle = COLORS.text;
            ctx.font = "bold 13px 'Segoe UI', Arial";
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            ctx.fillText("✨ " + this.title, 12, -h / 2);

            // Border
            ctx.strokeStyle = COLORS.secondary + "60";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.roundRect ? ctx.roundRect(0, -h, this.size[0], this.size[1] + h, 8) : ctx.rect(0, -h, this.size[0], this.size[1] + h);
            ctx.stroke();

            // Badge (Diamond)
            ctx.save();
            ctx.translate(this.size[0] - 25, -h / 2);
            ctx.rotate(Math.PI / 4);
            ctx.shadowColor = COLORS.highlight;
            ctx.shadowBlur = 6;
            ctx.fillStyle = COLORS.highlight;
            ctx.fillRect(-4, -4, 8, 8);
            ctx.restore();

            // 2. RESTORE THE CONTEXT (Cleanup!)
            // This was missing before, causing the text bleed
            ctx.restore();
        };

        // Slot colors
        const origDrawBg = nodeType.prototype.onDrawBackground;
        nodeType.prototype.onDrawBackground = function (ctx) {
            origDrawBg?.apply(this, arguments);
            if (!this._isPremium) return;

            this.outputs?.forEach(s => {
                if (s.name === "images") {
                    s.color_on = COLORS.primary;
                    s.color_off = COLORS.primary + "80";
                } else {
                    s.color_on = COLORS.highlight;
                    s.color_off = COLORS.highlight + "80";
                }
            });

            this.inputs?.forEach(s => {
                if (s.name === "character_image") {
                    s.color_on = COLORS.secondary;
                    s.color_off = COLORS.secondary + "80";
                }
            });
        };
    }
});

console.log("[HearmemanAI] Extension registered ✨");