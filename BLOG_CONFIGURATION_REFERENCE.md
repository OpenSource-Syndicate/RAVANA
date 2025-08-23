# RAVANA Blog Integration Configuration Reference

## Environment Variables

This document provides a comprehensive reference for all blog-related configuration options in the RAVANA AGI system.

## Core Configuration

### RAVANA_BLOG_ENABLED
- **Type**: Boolean
- **Default**: `true`
- **Description**: Master switch to enable or disable blog integration functionality
- **Example**: `RAVANA_BLOG_ENABLED=true`
- **Notes**: When disabled, all blog publishing attempts will be skipped

### RAVANA_BLOG_API_URL
- **Type**: String (URL)
- **Default**: `https://ravana-blog.netlify.app/api/publish`
- **Description**: The API endpoint for publishing blog posts
- **Example**: `RAVANA_BLOG_API_URL=https://ravana-blog.netlify.app/api/publish`
- **Notes**: Must be a valid HTTPS URL. Do not include trailing slash

### RAVANA_BLOG_AUTH_TOKEN
- **Type**: String
- **Default**: `ravana_secret_token_2024`
- **Description**: Bearer token for API authentication
- **Example**: `RAVANA_BLOG_AUTH_TOKEN=your_secret_token_here`
- **Security**: Keep this token secure and do not commit to version control

## Content Generation Settings

### BLOG_DEFAULT_STYLE
- **Type**: String (Enum)
- **Default**: `technical`
- **Description**: Default writing style when none is specified
- **Valid Values**: `technical`, `casual`, `academic`, `creative`, `philosophical`
- **Example**: `BLOG_DEFAULT_STYLE=technical`

### BLOG_MAX_CONTENT_LENGTH
- **Type**: Integer
- **Default**: `5000`
- **Description**: Maximum allowed content length in characters
- **Example**: `BLOG_MAX_CONTENT_LENGTH=5000`
- **Range**: 1000-50000 (recommended)

### BLOG_MIN_CONTENT_LENGTH
- **Type**: Integer
- **Default**: `500`
- **Description**: Minimum required content length in characters
- **Example**: `BLOG_MIN_CONTENT_LENGTH=500`
- **Range**: 100-2000 (recommended)

### BLOG_AUTO_TAGGING_ENABLED
- **Type**: Boolean
- **Default**: `true`
- **Description**: Enable automatic tag extraction from content
- **Example**: `BLOG_AUTO_TAGGING_ENABLED=true`
- **Notes**: When disabled, only custom tags will be used

### BLOG_MAX_TAGS
- **Type**: Integer
- **Default**: `10`
- **Description**: Maximum number of tags per blog post
- **Example**: `BLOG_MAX_TAGS=10`
- **Range**: 1-20 (recommended)

## Publishing Behavior Settings

### BLOG_AUTO_PUBLISH_ENABLED
- **Type**: Boolean
- **Default**: `false`
- **Description**: Allow autonomous publishing without explicit approval
- **Example**: `BLOG_AUTO_PUBLISH_ENABLED=false`
- **Security**: Enable with caution in production environments

### BLOG_REQUIRE_APPROVAL
- **Type**: Boolean
- **Default**: `true`
- **Description**: Require approval before publishing (future feature)
- **Example**: `BLOG_REQUIRE_APPROVAL=true`
- **Notes**: Currently used for logging; approval workflow not yet implemented

### BLOG_PUBLISH_FREQUENCY_HOURS
- **Type**: Integer
- **Default**: `24`
- **Description**: Minimum hours between autonomous blog posts
- **Example**: `BLOG_PUBLISH_FREQUENCY_HOURS=24`
- **Notes**: Prevents spam posting in autonomous mode

## API Communication Settings

### BLOG_TIMEOUT_SECONDS
- **Type**: Integer
- **Default**: `30`
- **Description**: HTTP request timeout in seconds
- **Example**: `BLOG_TIMEOUT_SECONDS=30`
- **Range**: 10-300 (recommended)

### BLOG_RETRY_ATTEMPTS
- **Type**: Integer
- **Default**: `3`
- **Description**: Number of retry attempts for failed API calls
- **Example**: `BLOG_RETRY_ATTEMPTS=3`
- **Range**: 1-10 (recommended)

### BLOG_RETRY_BACKOFF_FACTOR
- **Type**: Float
- **Default**: `2.0`
- **Description**: Exponential backoff factor for retries
- **Example**: `BLOG_RETRY_BACKOFF_FACTOR=2.0`
- **Notes**: Retry delay = backoff_factor ^ attempt_number

### BLOG_MAX_RETRY_DELAY
- **Type**: Integer
- **Default**: `60`
- **Description**: Maximum delay between retries in seconds
- **Example**: `BLOG_MAX_RETRY_DELAY=60`
- **Notes**: Caps the exponential backoff delay

## Content Quality Settings

### BLOG_MEMORY_CONTEXT_DAYS
- **Type**: Integer
- **Default**: `7`
- **Description**: Number of days of memory to include in content context
- **Example**: `BLOG_MEMORY_CONTEXT_DAYS=7`
- **Range**: 1-30 (recommended)

### BLOG_INCLUDE_MOOD_CONTEXT
- **Type**: Boolean
- **Default**: `true`
- **Description**: Include current mood state in content generation
- **Example**: `BLOG_INCLUDE_MOOD_CONTEXT=true`
- **Notes**: Requires emotional intelligence module to be active

## Configuration Examples

### Development Environment
```bash
# Development setup - safe defaults
RAVANA_BLOG_ENABLED=true
RAVANA_BLOG_API_URL=https://ravana-blog.netlify.app/api/publish
RAVANA_BLOG_AUTH_TOKEN=dev_token_123
BLOG_DEFAULT_STYLE=casual
BLOG_MAX_CONTENT_LENGTH=3000
BLOG_MIN_CONTENT_LENGTH=300
BLOG_AUTO_PUBLISH_ENABLED=false
BLOG_REQUIRE_APPROVAL=true
BLOG_TIMEOUT_SECONDS=15
BLOG_RETRY_ATTEMPTS=2
```

### Production Environment
```bash
# Production setup - optimized for reliability
RAVANA_BLOG_ENABLED=true
RAVANA_BLOG_API_URL=https://ravana-blog.netlify.app/api/publish
RAVANA_BLOG_AUTH_TOKEN=prod_secure_token_xyz789
BLOG_DEFAULT_STYLE=technical
BLOG_MAX_CONTENT_LENGTH=5000
BLOG_MIN_CONTENT_LENGTH=500
BLOG_AUTO_PUBLISH_ENABLED=false
BLOG_REQUIRE_APPROVAL=true
BLOG_PUBLISH_FREQUENCY_HOURS=48
BLOG_TIMEOUT_SECONDS=30
BLOG_RETRY_ATTEMPTS=3
BLOG_RETRY_BACKOFF_FACTOR=2.0
BLOG_MAX_RETRY_DELAY=60
```

### Autonomous Mode
```bash
# Autonomous publishing setup (use with caution)
RAVANA_BLOG_ENABLED=true
RAVANA_BLOG_API_URL=https://ravana-blog.netlify.app/api/publish
RAVANA_BLOG_AUTH_TOKEN=autonomous_token_abc456
BLOG_DEFAULT_STYLE=technical
BLOG_AUTO_PUBLISH_ENABLED=true
BLOG_REQUIRE_APPROVAL=false
BLOG_PUBLISH_FREQUENCY_HOURS=72
BLOG_MAX_CONTENT_LENGTH=4000
BLOG_MEMORY_CONTEXT_DAYS=14
BLOG_INCLUDE_MOOD_CONTEXT=true
```

### High-Performance Setup
```bash
# Optimized for speed and reliability
BLOG_TIMEOUT_SECONDS=45
BLOG_RETRY_ATTEMPTS=5
BLOG_RETRY_BACKOFF_FACTOR=1.5
BLOG_MAX_RETRY_DELAY=30
BLOG_MAX_CONTENT_LENGTH=6000
BLOG_AUTO_TAGGING_ENABLED=true
BLOG_MAX_TAGS=15
```

## Validation Rules

### Required Variables
- `RAVANA_BLOG_API_URL` must be a valid HTTPS URL
- `RAVANA_BLOG_AUTH_TOKEN` must not be empty when blog is enabled

### Logical Constraints
- `BLOG_MIN_CONTENT_LENGTH` must be less than `BLOG_MAX_CONTENT_LENGTH`
- `BLOG_TIMEOUT_SECONDS` must be greater than 0
- `BLOG_RETRY_ATTEMPTS` must be at least 1
- `BLOG_MAX_TAGS` should be reasonable (1-20)

### Security Considerations
- Never commit auth tokens to version control
- Use environment-specific tokens
- Regularly rotate authentication tokens
- Monitor API usage and rate limits

## Testing Configuration

To test the configuration, use the validation script:

```bash
python validate_blog_integration.py
```

This will check:
- All required variables are set
- Values are within valid ranges
- API connectivity works
- Authentication is successful

## Configuration Loading

The configuration is loaded in `core/config.py`:

```python
# Example configuration access
from core.config import Config

if Config.BLOG_ENABLED:
    api_url = Config.BLOG_API_URL
    timeout = Config.BLOG_TIMEOUT_SECONDS
    # ... use configuration
```

## Dynamic Configuration

Some settings can be overridden at runtime:

```python
# Override settings for specific requests
action = BlogPublishAction(system, data_service)
result = await action.execute(
    topic="Special Topic",
    style="creative",  # Override default style
    dry_run=True       # Override publishing behavior
)
```

## Troubleshooting Configuration Issues

### Common Problems

1. **Blog Disabled Unexpectedly**
   - Check `RAVANA_BLOG_ENABLED` is set to `true`
   - Verify boolean parsing (case-insensitive)

2. **Authentication Failures**
   - Verify `RAVANA_BLOG_AUTH_TOKEN` is correct
   - Check for extra whitespace or special characters
   - Ensure token has required permissions

3. **Timeouts and Retries**
   - Increase `BLOG_TIMEOUT_SECONDS` for slow networks
   - Adjust `BLOG_RETRY_ATTEMPTS` based on reliability needs
   - Monitor `BLOG_MAX_RETRY_DELAY` for reasonable limits

4. **Content Length Issues**
   - Ensure `BLOG_MIN_CONTENT_LENGTH` is achievable
   - Set `BLOG_MAX_CONTENT_LENGTH` based on platform limits
   - Monitor generated content lengths

### Debug Commands

```bash
# Check current configuration
python -c "from core.config import Config; print(f'Enabled: {Config.BLOG_ENABLED}')"

# Test API connectivity
python -c "
from core.actions.blog_api import BlogAPIInterface
import asyncio
api = BlogAPIInterface()
print('Config valid:', api.validate_config())
"

# Validate all settings
python validate_blog_integration.py
```

## Performance Tuning

### For High-Volume Publishing
- Increase `BLOG_TIMEOUT_SECONDS` to 45-60
- Set `BLOG_RETRY_ATTEMPTS` to 2-3
- Use `BLOG_RETRY_BACKOFF_FACTOR` of 1.5
- Monitor API rate limits

### For Reliable Publishing
- Set `BLOG_RETRY_ATTEMPTS` to 5
- Use `BLOG_RETRY_BACKOFF_FACTOR` of 2.0
- Set reasonable `BLOG_MAX_RETRY_DELAY`
- Enable comprehensive logging

### For Content Quality
- Set `BLOG_MEMORY_CONTEXT_DAYS` to 14-30
- Enable `BLOG_INCLUDE_MOOD_CONTEXT`
- Use `BLOG_AUTO_TAGGING_ENABLED=true`
- Adjust content length limits appropriately

## Migration Guide

When upgrading or changing configuration:

1. **Backup Current Settings**
   ```bash
   env | grep BLOG_ > blog_config_backup.env
   ```

2. **Test New Configuration**
   ```bash
   # Use dry run to test new settings
   python demo_blog_usage.py
   ```

3. **Gradual Rollout**
   - Start with `BLOG_AUTO_PUBLISH_ENABLED=false`
   - Test with `dry_run=true`
   - Monitor logs and performance
   - Enable autonomous features gradually

4. **Rollback Plan**
   - Keep previous configuration available
   - Document all changes
   - Have rollback procedures ready