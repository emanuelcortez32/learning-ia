from functools import wraps
from lib.logger import logger

def safe_tool(func):
    
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            logger.debug(f"[MCP-DEV] Ejecutando herramienta '{func.__name__}' con argumentos: {kwargs if kwargs else args}")

            result = await func(*args, **kwargs)

            logger.debug(f"[MCP-DEV] Herramiena '{func.__name__}' resultado: '{result}'")
            logger.info(f"[MCP-DEV] Herramienta '{func.__name__}' ejecutada exitosamente")

            if isinstance(result, dict) and "success" in result:
                return result
            
            if isinstance(result, dict):
                return {
                    "success": True,
                    "data": result
                }
            
            return {
                "success": True,
                "data": result
            }
        
        except Exception as e:
            logger.error(f"[MCP-DEV] Error en la herramienta '{func.__name__}': {str(e)}")
            logger.error(f"[MCP-DEV] Revisar la implementacion de '{func.__name__}' - Detalles del error: {type(e).__name__}")

            return {
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": f"Error inesperado en {func.__name__}",
                    "details": str(e)
                }
            }
        
    return wrapper