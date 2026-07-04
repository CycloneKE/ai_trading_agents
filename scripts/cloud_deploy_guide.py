"""
Cloud Deployment Guide for 24/7 AI Trading
"""

def create_cloud_deployment():
    """Create cloud deployment instructions."""
    
    guide = """
# 24/7 AI TRADING AGENT - CLOUD DEPLOYMENT

## OPTION 1: AWS EC2 (Recommended)
1. Create AWS account
2. Launch t3.micro instance (Free tier)
3. Upload your files
4. Run: python always_online_agent.py
5. Cost: $0-10/month

## OPTION 2: Google Cloud
1. Create GCP account ($300 free credit)
2. Create VM instance
3. Deploy with Docker
4. Cost: $5-15/month

## OPTION 3: DigitalOcean
1. Create droplet ($5/month)
2. Upload trading agent
3. Run 24/7
4. Cost: $5/month

## OPTION 4: Heroku (Easiest)
1. Create Heroku account
2. Deploy with git push
3. Enable worker dyno
4. Cost: $7/month

## QUICK START COMMANDS:
# Keep PC awake (Windows)
powercfg /change standby-timeout-ac 0

# Run agent in background
start /min python always_online_agent.py

# Check if running
tasklist | findstr python
"""
    
    with open('CLOUD_DEPLOYMENT.md', 'w') as f:
        f.write(guide)
    
    print("Cloud deployment guide created!")
    print("Options:")
    print("1. AWS EC2: $0-10/month")
    print("2. DigitalOcean: $5/month") 
    print("3. Keep PC running: $10/month electricity")
    print("4. Raspberry Pi: $2/month electricity")

if __name__ == '__main__':
    create_cloud_deployment()