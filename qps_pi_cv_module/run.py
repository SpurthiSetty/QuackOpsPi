#!/usr/bin/env python3
"""
QuackOps Pi CV Module - Main Entry Point

This script initializes and runs the Raspberry Pi companion computer
module for the QuackOps autonomous drone delivery system.

Usage:
    python -m qps_pi_cv_module
    python run.py
    python run.py --config /path/to/config.json
    python run.py --simulation
    python run.py --debug
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from qps_pi_cv_module import PiCvModule, Config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="QuackOps Raspberry Pi CV Module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default configuration
    python run.py
    
    # Run with custom config file
    python run.py --config /path/to/config.json
    
    # Run in simulation mode (no real hardware)
    python run.py --simulation
    
    # Run with debug logging
    python run.py --debug
    
    # Generate a default config file
    python run.py --generate-config
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to configuration file (JSON format)"
    )
    
    parser.add_argument(
        "--simulation", "-s",
        action="store_true",
        help="Run in simulation mode (no real hardware)"
    )
    
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--generate-config", "-g",
        type=str,
        metavar="PATH",
        help="Generate a default config file and exit"
    )
    
    parser.add_argument(
        "--backend-url", "-b",
        type=str,
        default=None,
        help="Override backend URL"
    )
    
    parser.add_argument(
        "--mavsdk-address", "-m",
        type=str,
        default=None,
        help="Override MAVSDK system address"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Generate config file if requested
    if args.generate_config:
        config = Config()
        config.save(args.generate_config)
        print(f"Generated default config at: {args.generate_config}")
        return 0
    
    # Load configuration
    if args.config:
        print(f"Loading config from: {args.config}")
        config = Config.from_file(args.config)
    else:
        print("Using default configuration with environment overrides")
        config = Config.from_environment()
    
    # Apply command line overrides
    if args.simulation:
        config.simulation_mode = True
        print("Running in SIMULATION mode")
    
    if args.debug:
        config.debug_mode = True
        config.log_level = "DEBUG"
    
    if args.backend_url:
        config.communication.backend_base_url = args.backend_url
        # Derive WebSocket URL from HTTP URL
        ws_url = args.backend_url.replace("http://", "ws://").replace("https://", "wss://")
        config.communication.websocket_url = f"{ws_url}/ws/drone"
    
    if args.mavsdk_address:
        config.flight.mavsdk_system_address = args.mavsdk_address
    
    # Setup logging
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger(__name__)
    
    # Print configuration summary
    logger.info("=" * 60)
    logger.info("QuackOps Pi CV Module")
    logger.info("=" * 60)
    logger.info(f"Simulation Mode: {config.simulation_mode}")
    logger.info(f"Backend URL: {config.communication.backend_base_url}")
    logger.info(f"MAVSDK Address: {config.flight.mavsdk_system_address}")
    logger.info(f"ArUco Dictionary: {config.vision.aruco_dictionary}")
    logger.info("=" * 60)
    
    # Create and run module
    try:
        module = PiCvModule(config)
        asyncio.run(module.run())
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
