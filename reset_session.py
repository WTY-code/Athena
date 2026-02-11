import sys
sys.path.append('/root/mcts')
from config_sdk import ConfigSDK, ConfigSDKError

def reset_session():
    sdk = ConfigSDK(base_url="http://192.168.0.96:8321/")
    print("Checking for active sessions...")
    try:
        # We can't directly ask "give me active session" easily without STATUS
        # But we can try to start a dummy session or use internal API if available.
        # However, checking status via raw request is easier as I did with curl.
        import requests
        resp = requests.post("http://192.168.0.96:8321/", json={"method": "SESSION_STATUS"})
        data = resp.json()
        
        if data.get("active"):
            session_id = data.get("session_id")
            print(f"Found active session: {session_id}. Terminating...")
            sdk.session_end(session_id)
            print("Session terminated.")
        else:
            print("No active session found.")
            
    except Exception as e:
        print(f"Error resetting session: {e}")

if __name__ == "__main__":
    reset_session()
