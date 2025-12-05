#!/bin/bash

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Configuration
PROJECT_DIR="/teamspace/studios/this_studio/.lightning_studio/wildlife-analysis"
FASTAPI_PORT=8080
STREAMLIT_PORT=8502
LOG_DIR="$PROJECT_DIR/logs"

# Create logs directory
mkdir -p "$LOG_DIR"

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}🚀 Wild Arabia - Complete Startup Script${NC}"
echo -e "${MAGENTA}🤖 Powered by Qwen 2.5 + DINOv2-Large${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
    else
        echo -e "${RED}✗${NC} $2"
    fi
}

# Function to kill previous processes
cleanup() {
    echo -e "${YELLOW}Cleaning up old processes...${NC}"
    
    # Kill old FastAPI
    OLD_FASTAPI=$(lsof -ti:$FASTAPI_PORT)
    if [ ! -z "$OLD_FASTAPI" ]; then
        kill -9 $OLD_FASTAPI 2>/dev/null
        print_status 0 "Killed old FastAPI (PID: $OLD_FASTAPI)"
    fi
    
    # Kill old Streamlit
    OLD_STREAMLIT=$(lsof -ti:$STREAMLIT_PORT)
    if [ ! -z "$OLD_STREAMLIT" ]; then
        kill -9 $OLD_STREAMLIT 2>/dev/null
        print_status 0 "Killed old Streamlit (PID: $OLD_STREAMLIT)"
    fi
    
    # Kill old Cloudflare
    pkill -f cloudflared 2>/dev/null && print_status 0 "Killed old Cloudflare tunnel"
    
    # Remove old PID file
    [ -f "$PROJECT_DIR/.pids" ] && rm "$PROJECT_DIR/.pids"
    
    echo ""
}

# Navigate to project
echo -e "${BLUE}Step 1: Verify Project Directory${NC}"
if [ ! -d "$PROJECT_DIR" ]; then
    echo -e "${RED}❌ Project directory not found: $PROJECT_DIR${NC}"
    echo -e "${CYAN}Trying current directory...${NC}"
    PROJECT_DIR="."
fi
cd "$PROJECT_DIR" || { echo -e "${RED}❌ Cannot change to project directory${NC}"; exit 1; }
print_status 0 "In project directory: $(pwd)"
echo ""

# Check required files
echo -e "${BLUE}Step 2: Check Required Files${NC}"
required_files=("main.py" "streamlit_app.py" "model_loader.py" "inference.py" "rag_utils.py")
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        print_status 0 "$file found"
    else
        print_status 1 "$file NOT found"
    fi
done
echo ""

# Check Python and packages
echo -e "${BLUE}Step 3: Check Dependencies${NC}"
python3 --version
echo -e "${CYAN}Checking Python packages...${NC}"
python3 -c "import fastapi; print('  ✓ FastAPI: OK')" 2>/dev/null || echo -e "  ${RED}✗ FastAPI: MISSING${NC}"
python3 -c "import streamlit; print('  ✓ Streamlit: OK')" 2>/dev/null || echo -e "  ${RED}✗ Streamlit: MISSING${NC}"
python3 -c "import torch; print('  ✓ PyTorch: OK')" 2>/dev/null || echo -e "  ${RED}✗ PyTorch: MISSING${NC}"
python3 -c "import transformers; print('  ✓ Transformers: OK')" 2>/dev/null || echo -e "  ${RED}✗ Transformers: MISSING${NC}"
python3 -c "import PIL; print('  ✓ Pillow: OK')" 2>/dev/null || echo -e "  ${RED}✗ Pillow: MISSING${NC}"
echo ""

# Check GPU availability
echo -e "${BLUE}Step 4: Check GPU/Hardware${NC}"
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    print_status 0 "GPU available: $GPU_NAME"
    GPU_MEM=$(python3 -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')" 2>/dev/null)
    echo -e "  ${CYAN}GPU Memory: $GPU_MEM${NC}"
else
    print_status 1 "GPU not available (will use CPU - slower)"
    echo -e "  ${YELLOW}⚠${NC}  Models will run on CPU. Expect slower inference."
fi
echo ""

# Check RAM
echo -e "${BLUE}Step 5: Check System Resources${NC}"
TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
echo -e "  ${CYAN}Total RAM: ${TOTAL_RAM}GB${NC}"
if [ "$TOTAL_RAM" -lt 8 ]; then
    echo -e "  ${YELLOW}⚠${NC}  Low RAM detected. Models require ~14GB. Consider using smaller versions."
elif [ "$TOTAL_RAM" -lt 14 ]; then
    echo -e "  ${YELLOW}⚠${NC}  RAM may be tight. Monitor memory usage."
else
    print_status 0 "Sufficient RAM for all models"
fi
echo ""

# Note about models
echo -e "${BLUE}Step 6: Model Configuration${NC}"
print_status 0 "All models run locally - no API keys required!"
echo -e "  ${GREEN}✓${NC}  Vision Primary: WildArabia (ConvNeXt) - 89 classes"
echo -e "  ${GREEN}✓${NC}  Vision Fallback: DINOv2-Large - Self-supervised"
echo -e "  ${GREEN}✓${NC}  Text Generation: Qwen 2.5-3B - Local inference"
echo -e "  ${GREEN}✓${NC}  Privacy-focused (data stays local)"
echo -e "  ${GREEN}✓${NC}  Works offline after model download"
echo -e "  ${CYAN}ℹ${NC}  First run will download models (~2-3GB each)"
echo ""

# Cleanup old processes
cleanup

# Start FastAPI
echo -e "${BLUE}Step 7: Start FastAPI Backend${NC}"
echo -e "${YELLOW}Note: First startup may take 5-10 minutes to download models...${NC}"
nohup python3 main.py > "$LOG_DIR/fastapi.log" 2>&1 &
FASTAPI_PID=$!
print_status 0 "FastAPI started (PID: $FASTAPI_PID)"

# Wait for FastAPI to be ready (longer timeout for model download)
echo -e "${YELLOW}Waiting for FastAPI to start (loading models, this may take a while)...${NC}"
TIMEOUT=300  # 5 minutes for model loading
for i in $(seq 1 $TIMEOUT); do
    if curl -s http://localhost:$FASTAPI_PORT/health > /dev/null 2>&1; then
        print_status 0 "FastAPI is responding"
        break
    fi
    if [ $i -eq $TIMEOUT ]; then
        print_status 1 "FastAPI failed to start (timeout after ${TIMEOUT}s)"
        echo -e "${CYAN}Check logs: tail -50 $LOG_DIR/fastapi.log${NC}"
        echo -e "${YELLOW}If models are still downloading, wait longer and check logs${NC}"
    fi
    # Show progress indicator
    if [ $((i % 10)) -eq 0 ]; then
        echo -ne "${CYAN}[$i/${TIMEOUT}s]${NC} "
    fi
    echo -n "."
    sleep 1
done
echo ""
echo ""

# Check if backend actually started successfully
if curl -s http://localhost:$FASTAPI_PORT/health > /dev/null 2>&1; then
    # Get backend status
    echo -e "${BLUE}Backend Status:${NC}"
    HEALTH_JSON=$(curl -s http://localhost:$FASTAPI_PORT/health)
    echo "$HEALTH_JSON" | python3 -m json.tool 2>/dev/null | grep -E "(qwen_available|vision_primary|vision_fallback|cuda_available|vision_fallback_type)" | sed 's/^/  /'
    echo ""
    
    # Check specific models
    PRIMARY_STATUS=$(echo "$HEALTH_JSON" | python3 -c "import sys, json; data=json.load(sys.stdin); print('✅' if data.get('models',{}).get('vision_primary') else '❌')" 2>/dev/null)
    FALLBACK_STATUS=$(echo "$HEALTH_JSON" | python3 -c "import sys, json; data=json.load(sys.stdin); print('✅' if data.get('models',{}).get('vision_fallback') else '❌')" 2>/dev/null)
    QWEN_STATUS=$(echo "$HEALTH_JSON" | python3 -c "import sys, json; data=json.load(sys.stdin); print('✅' if data.get('models',{}).get('qwen_available') else '❌')" 2>/dev/null)
    
    echo -e "${BLUE}Model Status Summary:${NC}"
    echo -e "  ${PRIMARY_STATUS} WildArabia (Primary)"
    echo -e "  ${FALLBACK_STATUS} DINOv2-Large (Fallback)"
    echo -e "  ${QWEN_STATUS} Qwen 2.5-3B (Text Gen)"
    echo ""
else
    echo -e "${RED}❌ Backend health check failed${NC}"
    echo -e "${CYAN}Last 30 lines of log:${NC}"
    tail -30 "$LOG_DIR/fastapi.log"
    echo ""
fi

# Create Streamlit config
echo -e "${BLUE}Step 8: Configure Streamlit${NC}"
mkdir -p .streamlit
cat > .streamlit/config.toml << 'EOF'
[server]
port = 8502
address = "0.0.0.0"
headless = true
enableCORS = true
enableXsrfProtection = false
maxUploadSize = 200

[client]
toolbarMode = "auto"

[browser]
gatherUsageStats = false

[deprecation]
showfileUploaderEncoding = false
showPyplotGlobalUse = false

[logger]
level = "info"
EOF
print_status 0 "Streamlit config created"
echo ""

# Start Streamlit
echo -e "${BLUE}Step 9: Start Streamlit Frontend${NC}"
nohup streamlit run streamlit_app.py \
    --server.port $STREAMLIT_PORT \
    --server.address 0.0.0.0 \
    --server.headless true \
    > "$LOG_DIR/streamlit.log" 2>&1 &
STREAMLIT_PID=$!
print_status 0 "Streamlit started (PID: $STREAMLIT_PID)"

# Wait for Streamlit to be ready
echo -e "${YELLOW}Waiting for Streamlit to start...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:$STREAMLIT_PORT > /dev/null 2>&1; then
        print_status 0 "Streamlit is responding"
        break
    fi
    if [ $i -eq 30 ]; then
        print_status 1 "Streamlit failed to start"
        echo -e "${CYAN}Check logs: tail -20 $LOG_DIR/streamlit.log${NC}"
    fi
    echo -n "."
    sleep 1
done
echo ""
echo ""

# Start Cloudflare Tunnel (if available)
echo -e "${BLUE}Step 10: Start Cloudflare Tunnel${NC}"
TUNNEL_READY=false
TUNNEL_PID=0
if command -v cloudflared &> /dev/null; then
    nohup cloudflared tunnel --url http://localhost:$STREAMLIT_PORT --no-autoupdate \
        > "$LOG_DIR/cloudflare.log" 2>&1 &
    TUNNEL_PID=$!
    print_status 0 "Cloudflare tunnel started (PID: $TUNNEL_PID)"
    
    # Wait for URL
    echo -e "${YELLOW}Generating public URL...${NC}"
    sleep 5
    URL=$(grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG_DIR/cloudflare.log" | tail -1)
    if [ ! -z "$URL" ]; then
        TUNNEL_READY=true
        TUNNEL_URL="$URL"
    fi
else
    echo -e "${YELLOW}⚠${NC} Cloudflare tunnel not installed (optional)"
    TUNNEL_READY=false
fi
echo ""

# Save PIDs
echo "$FASTAPI_PID $STREAMLIT_PID $TUNNEL_PID" > "$PROJECT_DIR/.pids"

# Final status
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ ALL SERVICES STARTED!${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

echo -e "${BLUE}🔗 Access Points:${NC}"
echo -e "  ${CYAN}Local Streamlit:${NC}    http://localhost:$STREAMLIT_PORT"
echo -e "  ${CYAN}Local API:${NC}          http://localhost:$FASTAPI_PORT"
echo -e "  ${CYAN}API Docs:${NC}           http://localhost:$FASTAPI_PORT/docs"
echo -e "  ${CYAN}Health Check:${NC}       http://localhost:$FASTAPI_PORT/health"
if [ "$TUNNEL_READY" = true ]; then
    echo -e "  ${CYAN}Public URL:${NC}        ${YELLOW}$TUNNEL_URL${NC}"
fi
echo ""

echo -e "${BLUE}📊 Processes:${NC}"
echo -e "  ${CYAN}FastAPI (PID: $FASTAPI_PID)${NC}"
echo -e "  ${CYAN}Streamlit (PID: $STREAMLIT_PID)${NC}"
if [ ! -z "$TUNNEL_PID" ] && [ "$TUNNEL_PID" != "0" ]; then
    echo -e "  ${CYAN}Cloudflare (PID: $TUNNEL_PID)${NC}"
fi
echo ""

echo -e "${BLUE}🤖 AI Models:${NC}"
echo -e "  ${MAGENTA}Vision Primary:${NC}   WildArabia (ConvNeXt) - 89 classes"
echo -e "  ${MAGENTA}Vision Fallback:${NC} DINOv2-Large (Self-supervised)"
echo -e "  ${MAGENTA}Language:${NC}        Qwen 2.5-3B (local, no API key)"
echo -e "  ${MAGENTA}RAG:${NC}             ChromaDB + Wikipedia"
echo ""

echo -e "${BLUE}📝 Logs:${NC}"
echo -e "  ${CYAN}FastAPI:${NC}    tail -f $LOG_DIR/fastapi.log"
echo -e "  ${CYAN}Streamlit:${NC}   tail -f $LOG_DIR/streamlit.log"
if [ -f "$LOG_DIR/cloudflare.log" ]; then
    echo -e "  ${CYAN}Cloudflare:${NC}  tail -f $LOG_DIR/cloudflare.log"
fi
echo ""

echo -e "${BLUE}🛑 Stop Services:${NC}"
echo -e "  ${CYAN}bash stop.sh${NC}"
echo -e "  ${CYAN}or: kill $FASTAPI_PID $STREAMLIT_PID${NC}\n"

echo -e "${BLUE}💡 Useful Commands:${NC}"
echo -e "  ${CYAN}Check backend:${NC}  curl http://localhost:$FASTAPI_PORT/health | python3 -m json.tool"
echo -e "  ${CYAN}Test Qwen:${NC}      curl http://localhost:$FASTAPI_PORT/validate_qwen"
echo -e "  ${CYAN}Monitor GPU:${NC}    watch -n 1 nvidia-smi"
echo -e "  ${CYAN}Monitor RAM:${NC}    watch -n 1 free -h"
echo ""

echo -e "${YELLOW}⚠${NC}  ${YELLOW}First Run Notes:${NC}"
echo -e "  • Models will download automatically:"
echo -e "    - WildArabia: ~500MB"
echo -e "    - DINOv2-Large: ~1.1GB"
echo -e "    - Qwen 2.5-3B: ~6GB"
echo -e "  • Initial startup may take 5-10 minutes"
echo -e "  • Subsequent starts will be much faster"
echo -e "  • Monitor logs if services don't start immediately"
echo -e "  • DINOv2 provides excellent fallback for wildlife classification"
echo ""

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

# Keep script running
wait