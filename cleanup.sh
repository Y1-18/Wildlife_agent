#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}🧹 Wild Arabia Cleanup Script${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

# Kill specific PIDs if they exist
if [ -f .pids ]; then
    echo -e "${YELLOW}📋 Stopping saved processes...${NC}"
    PIDS=$(cat .pids)
    for pid in $PIDS; do
        if ps -p $pid > /dev/null 2>&1; then
            echo -e "  ${RED}✗${NC} Killing PID: $pid"
            kill -9 $pid 2>/dev/null
        fi
    done
    rm .pids
    echo ""
fi

# Kill all Python processes
echo -e "${YELLOW}🐍 Stopping Python processes...${NC}"
pkill -9 python 2>/dev/null && echo -e "  ${GREEN}✓${NC} Python stopped" || echo -e "  ${CYAN}○${NC} No Python processes"
pkill -9 python3 2>/dev/null

# Kill specific services
echo -e "\n${YELLOW}🔧 Stopping specific services...${NC}"
pkill -9 uvicorn 2>/dev/null && echo -e "  ${GREEN}✓${NC} Uvicorn stopped" || echo -e "  ${CYAN}○${NC} Uvicorn not running"
pkill -9 streamlit 2>/dev/null && echo -e "  ${GREEN}✓${NC} Streamlit stopped" || echo -e "  ${CYAN}○${NC} Streamlit not running"
pkill -9 cloudflared 2>/dev/null && echo -e "  ${GREEN}✓${NC} Cloudflare stopped" || echo -e "  ${CYAN}○${NC} Cloudflare not running"

# Kill processes on specific ports
echo -e "\n${YELLOW}🔌 Freeing ports...${NC}"
if lsof -ti:8080 > /dev/null 2>&1; then
    kill -9 $(lsof -t -i:8080) 2>/dev/null
    echo -e "  ${GREEN}✓${NC} Port 8080 freed"
else
    echo -e "  ${CYAN}○${NC} Port 8080 already free"
fi

if lsof -ti:8502 > /dev/null 2>&1; then
    kill -9 $(lsof -t -i:8502) 2>/dev/null
    echo -e "  ${GREEN}✓${NC} Port 8502 freed"
else
    echo -e "  ${CYAN}○${NC} Port 8502 already free"
fi

# Clean temporary files
echo -e "\n${YELLOW}🗑️  Cleaning temporary files...${NC}"
rm -rf /tmp/classify_* 2>/dev/null
rm -rf /tmp/analyze_* 2>/dev/null
rm -rf __pycache__ 2>/dev/null
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
echo -e "  ${GREEN}✓${NC} Temp files cleaned"

# Clean log files (optional)
if [ "$1" == "--clean-logs" ]; then
    echo -e "\n${YELLOW}📝 Cleaning log files...${NC}"
    rm -f fastapi.log streamlit.log cloudflare.log 2>/dev/null
    echo -e "  ${GREEN}✓${NC} Log files removed"
fi

# Clear GPU memory if available
if command -v nvidia-smi &> /dev/null; then
    echo -e "\n${YELLOW}🎮 Clearing GPU memory...${NC}"
    nvidia-smi --gpu-reset 2>/dev/null || true
    echo -e "  ${GREEN}✓${NC} GPU reset attempted"
fi

# Show status
echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}📊 System Status${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

# Check for remaining Python processes
echo -e "${CYAN}Python Processes:${NC}"
if ps aux | grep -E "python|uvicorn|streamlit" | grep -v grep > /dev/null; then
    ps aux | grep -E "python|uvicorn|streamlit" | grep -v grep
    echo -e "${YELLOW}⚠️  Some processes still running${NC}"
else
    echo -e "  ${GREEN}✓${NC} None found"
fi

# Check port status
echo -e "\n${CYAN}Port Status:${NC}"
if lsof -i :8080 2>/dev/null | grep -v COMMAND; then
    echo -e "  ${YELLOW}⚠️  Port 8080 in use${NC}"
else
    echo -e "  ${GREEN}✓${NC} Port 8080 free"
fi

if lsof -i :8502 2>/dev/null | grep -v COMMAND; then
    echo -e "  ${YELLOW}⚠️  Port 8502 in use${NC}"
else
    echo -e "  ${GREEN}✓${NC} Port 8502 free"
fi

echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Cleanup Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

echo -e "${CYAN}💡 Usage:${NC}"
echo -e "  ${YELLOW}./cleanup.sh${NC}              - Clean processes and temp files"
echo -e "  ${YELLOW}./cleanup.sh --clean-logs${NC} - Also remove log files"
echo -e "  ${YELLOW}./start_all.sh${NC}            - Start all services again\n"