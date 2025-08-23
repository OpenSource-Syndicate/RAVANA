# RAVANA AGI Installation Guide

This guide provides detailed instructions for installing and setting up the RAVANA AGI system.

## System Requirements

### Minimum Requirements

- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 20.04+, CentOS 8+)
- **Processor**: 2 GHz dual-core or better
- **Memory**: 8 GB RAM (16 GB recommended)
- **Storage**: 10 GB available space
- **Python**: 3.8 or higher

### Recommended Requirements

- **Operating System**: Latest stable versions of Windows, macOS, or Linux
- **Processor**: 3 GHz quad-core or better
- **Memory**: 16 GB RAM (32 GB recommended for heavy use)
- **Storage**: 50 GB available space (SSD recommended)
- **Python**: 3.10 or higher

## Prerequisites

Before installing RAVANA AGI, ensure you have the following installed:

1. **Python 3.8+**: Download from [python.org](https://www.python.org/downloads/)
2. **Git**: Download from [git-scm.com](https://git-scm.com/downloads)
3. **uv**: Python package manager (installation instructions below)

## Installation Steps

### 1. Install uv (Python Package Manager)

RAVANA uses `uv` for fast dependency management. Install it using:

**Windows (PowerShell):**
```powershell
pip install uv
```

**macOS/Linux:**
```bash
pip install uv
# Or using curl:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the Repository

```bash
git clone https://github.com/OpenSource-Syndicate/RAVANA.git
cd RAVANA
```

### 3. Create a Virtual Environment

```bash
uv venv
```

This creates a virtual environment in the `.venv` directory.

### 4. Activate the Virtual Environment

**Windows (Command Prompt):**
```cmd
.venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

### 5. Install Dependencies

```bash
uv pip install -e .
```

This installs all required dependencies and sets up the package in development mode.

## Configuration

### Initial Configuration

After installation, you'll need to configure the system. The default configuration is in `core/config.json`.

For basic operation, you may need to set up API keys for language models:

```bash
# Set OpenAI API key
export OPENAI_API_KEY="your-openai-api-key"

# Set Anthropic API key (if using Claude)
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### Database Setup

RAVANA uses SQLite by default, which requires no additional setup. For PostgreSQL or other databases:

1. Install the database server
2. Create a database for RAVANA
3. Update the connection string in `core/config.json`

## First Run

### Running the System

To start RAVANA AGI:

```bash
uv run main.py
```

The system will:
1. Initialize all modules
2. Load configuration
3. Start the autonomous loop
4. Begin processing and learning

### Running in Development Mode

For development and debugging:

```bash
uv run main.py --debug
```

This enables:
- Detailed logging
- Debug-level output
- Additional validation checks

## Verification

### System Check

After starting, verify the system is running correctly:

1. Check the console output for initialization messages
2. Look for "System initialized successfully" message
3. Verify no error messages appear during startup

### Test Components

Run the built-in self-test:

```bash
uv run tests/test_system_simple.py
```

This verifies:
- Core system components
- Basic functionality
- Module initialization
- Service connectivity

## Common Installation Issues

### Python Version Issues

**Problem**: "Python 3.8+ required"
**Solution**: Install a compatible Python version and ensure it's in your PATH

### Dependency Installation Failures

**Problem**: Errors during `uv pip install -e .`
**Solutions**:
1. Update pip: `pip install --upgrade pip`
2. Clear cache: `uv pip cache purge`
3. Check network connectivity
4. Ensure build tools are installed (Windows: Visual Studio Build Tools)

### Virtual Environment Issues

**Problem**: "Command not found" or wrong Python version
**Solution**: 
1. Ensure virtual environment is activated
2. Verify activation with `which python` (macOS/Linux) or `where python` (Windows)
3. Reactivate if necessary

### Permission Errors

**Problem**: Permission denied during installation
**Solutions**:
1. Run terminal as administrator (Windows)
2. Use `--user` flag: `pip install --user uv`
3. Check directory permissions

## Platform-Specific Instructions

### Windows

1. **PowerShell Execution Policy**: If you encounter issues running scripts:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

2. **Path Issues**: Ensure Python and Git are in your PATH environment variable.

### macOS

1. **Xcode Command Line Tools**: Install if not already present:
   ```bash
   xcode-select --install
   ```

2. **Homebrew**: Consider using Homebrew for Python management:
   ```bash
   brew install python
   ```

### Linux (Ubuntu/Debian)

1. **System Dependencies**:
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip git
   ```

2. **User Installation**:
   ```bash
   pip3 install --user uv
   ```

## Advanced Installation

### Docker Installation

For containerized deployment, use the provided Dockerfile:

```bash
docker build -t ravana-agi .
docker run -it ravana-agi
```

### Development Installation

For development work:

```bash
uv pip install -e ".[dev]"
```

This installs additional development dependencies including:
- Testing frameworks
- Linting tools
- Debugging utilities

### Production Installation

For production deployment:

```bash
uv pip install .
```

This installs the package without development dependencies.

## Post-Installation Setup

### Environment Variables

Set up required environment variables:

```bash
# Add to your shell profile (.bashrc, .zshrc, etc.)
export RAVANA_DATA_DIR="/path/to/data"
export RAVANA_LOG_DIR="/path/to/logs"
```

### Service Configuration

For running as a system service:

1. Create a systemd unit file (Linux) or service configuration (Windows)
2. Set up appropriate user permissions
3. Configure auto-restart policies

### Monitoring Setup

Set up monitoring for production use:

1. Configure log rotation
2. Set up health check endpoints
3. Implement alerting for critical failures
4. Monitor resource usage

## Updating RAVANA

### Pulling Updates

To update to the latest version:

```bash
git pull origin main
uv pip install -e .
```

### Version Management

For specific versions:

```bash
git checkout v1.2.0
uv pip install -e .
```

### Migration Notes

Check release notes for:
- Breaking changes
- Configuration updates
- Data migration requirements
- Dependency changes

## Uninstallation

### Removing the Virtual Environment

To completely remove the installation:

```bash
# Deactivate virtual environment
deactivate

# Remove the directory
rm -rf RAVANA/
```

### Cleaning Up Data

To remove stored data:

```bash
# Remove data directory
rm -rf data/

# Remove logs
rm -rf logs/

# Remove database (if using SQLite)
rm ravana_agi.db
```

## Support

### Getting Help

If you encounter issues:

1. Check the [FAQ](faq.md)
2. Review existing GitHub issues
3. Create a new issue with:
   - System information
   - Error messages
   - Steps to reproduce
   - Installation method used

### Community Support

- GitHub Discussions: [RAVANA Discussions](https://github.com/OpenSource-Syndicate/RAVANA/discussions)
- Email: semalalikithsai@gmail.com

## Next Steps

After successful installation:

1. [Configure the system](configuration.md)
2. [Review the architecture](architecture.md)
3. [Explore the modules](../modules/index.md)
4. [Run the tutorials](tutorials.md)
5. [Join the community](community.md)