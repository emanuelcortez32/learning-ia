#!/bin/bash

check_requirements() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED} Error: Docker no esta instalado${NC}"
        exit 1
    fi

    if ! docker compose version &> /dev/null; then
        echo -e "${RED} Error: Docker Compose no esta disponible${NC}"
        exit 1
    fi
}

cmd_up() {
    echo -e "${BLUE} Iniciando servicios en modo desarrollo ...${NC}"
    echo "Pendiente de implementacion"
}

cmd_up_all() {
    echo -e "${BLUE} Iniciando servicios en modo desarrollo ...${NC}"
    docker compose build
    docker compose up -d

    echo -e "${GREEN} Servicios Iniciados !${NC}"
    echo ""
    docker compose ps
    echo ""
}

cmd_down() {
    echo -e "${YELLOW} Deteniendo servicios ...${NC}"
    docker compose down
    echo -e "${GREEN} Servicios detenidos${NC}"
}

cmd_clean() {
    echo -e "${YELLOW} Limpiando Contenedores y volumenes...${NC}"
    docker compose kill || echo "no containers to kill"
    docker compose down -v --remove-orphans || echo "no volumes to remove"
    docker rm -s -f v || echo "no containers to remove"
    docker system prune -f
    echo -e "${GREEN} Limpieza completada${NC}"
}

main() {
    check_requirements

    case "${1:-}" in
        "up")
            cmd_up
            ;;
        "up-all")
            cmd_up_all
            ;;
        "down")
            cmd_down
            ;;
        "clean")
            cmd_clean
            ;;
        *)
        
            echo -e "${RED} Comando desconocido: $1${NC}"
            echo ""
            exit 1
            ;;
    esac
}

main "$@"