#!/usr/bin/env python3
"""
Network Connectivity Diagnostics for API Connection Issues
Diagnoses network-level problems affecting multiple APIs simultaneously
"""

import socket
import requests
import time
import subprocess
import sys
import os
from typing import Dict, List, Tuple, Optional

def test_dns_resolution(domains: List[str]) -> Dict[str, bool]:
    """Test DNS resolution using socket (no external deps)"""
    results = {}
    
    for domain in domains:
        try:
            # Use socket.gethostbyname for basic DNS resolution
            ip = socket.gethostbyname(domain)
            results[domain] = True
            print(f"✅ DNS OK: {domain} -> {ip}")
        except socket.gaierror as e:
            results[domain] = False
            print(f"❌ DNS FAIL: {domain} -> {str(e)}")
        except Exception as e:
            results[domain] = False
            print(f"❌ DNS ERROR: {domain} -> {str(e)}")
    
    return results

def test_tcp_connectivity(endpoints: List[Tuple[str, int]]) -> Dict[str, bool]:
    """Test raw TCP connectivity"""
    results = {}
    
    for host, port in endpoints:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            result = sock.connect_ex((host, port))
            sock.close()
            
            success = result == 0
            results[f"{host}:{port}"] = success
            status = "✅ OPEN" if success else "❌ CLOSED"
            print(f"{status}: {host}:{port}")
            
        except Exception as e:
            results[f"{host}:{port}"] = False
            print(f"❌ TCP ERROR: {host}:{port} -> {str(e)}")
    
    return results

def test_http_connectivity(urls: List[str]) -> Dict[str, Dict]:
    """Test HTTP connectivity with detailed timing"""
    results = {}
    
    for url in urls:
        start_time = time.time()
        try:
            response = requests.get(
                url, 
                timeout=30,
                headers={'User-Agent': 'NetworkDiagnostic/1.0'}
            )
            
            elapsed = time.time() - start_time
            results[url] = {
                'success': response.status_code < 400,
                'status_code': response.status_code,
                'response_time': elapsed,
                'size': len(response.content)
            }
            
            print(f"✅ HTTP OK: {url} -> {response.status_code} ({elapsed:.2f}s)")
            
        except requests.exceptions.ConnectTimeout:
            results[url] = {'success': False, 'error': 'Connection timeout'}
            print(f"❌ TIMEOUT: {url}")
            
        except requests.exceptions.ConnectionError as e:
            results[url] = {'success': False, 'error': f'Connection error: {str(e)}'}
            print(f"❌ CONN ERROR: {url} -> {str(e)}")
            
        except Exception as e:
            results[url] = {'success': False, 'error': str(e)}
            print(f"❌ HTTP ERROR: {url} -> {str(e)}")
    
    return results

def test_api_endpoints() -> Dict[str, Dict]:
    """Test specific API endpoints that are failing"""
    api_tests = {
        'coingecko_ping': 'https://api.coingecko.com/api/v3/ping',
        'coingecko_markets': 'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&ids=bitcoin&per_page=1',
        'anthropic_status': 'https://status.anthropic.com',
        'google_dns': 'https://8.8.8.8',  # Basic connectivity test
    }
    
    print("\n🔍 Testing API Endpoints...")
    return test_http_connectivity(list(api_tests.values()))

def check_system_network_config():
    """Check system-level network configuration"""
    print("\n🔧 System Network Configuration:")
    
    try:
        # Check default gateway
        result = subprocess.run(['route', '-n'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Route table accessible")
        else:
            print("❌ Cannot access route table")
    except:
        print("❌ Route command failed")
    
    try:
        # Check if behind proxy
        import os
        http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
        https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
        
        if http_proxy or https_proxy:
            print(f"⚠️  Proxy detected: HTTP={http_proxy}, HTTPS={https_proxy}")
        else:
            print("✅ No proxy environment variables")
    except:
        print("❌ Cannot check proxy settings")

def diagnose_network_issues():
    """Main diagnostic function"""
    print("🚀 Network Connectivity Diagnostics Starting...\n")
    
    # Critical domains to test
    domains = [
        'api.coingecko.com',
        'api.anthropic.com',
        'google.com',
        '8.8.8.8'
    ]
    
    # Critical TCP endpoints
    tcp_endpoints = [
        ('api.coingecko.com', 443),
        ('api.anthropic.com', 443),
        ('8.8.8.8', 53),  # DNS
        ('1.1.1.1', 53),  # Cloudflare DNS
    ]
    
    print("1️⃣ Testing DNS Resolution...")
    dns_results = test_dns_resolution(domains)
    
    print("\n2️⃣ Testing TCP Connectivity...")
    tcp_results = test_tcp_connectivity(tcp_endpoints)
    
    print("\n3️⃣ Testing HTTP Connectivity...")
    api_results = test_api_endpoints()
    
    print("\n4️⃣ System Configuration...")
    check_system_network_config()
    
    # Summary analysis
    print("\n📊 DIAGNOSTIC SUMMARY:")
    print("=" * 50)
    
    dns_failures = sum(1 for success in dns_results.values() if not success)
    tcp_failures = sum(1 for success in tcp_results.values() if not success)
    api_failures = sum(1 for result in api_results.values() if not result.get('success', False))
    
    if dns_failures > 0:
        print(f"❌ DNS Issues: {dns_failures}/{len(dns_results)} domains failed")
        print("   -> Check DNS servers (8.8.8.8, 1.1.1.1)")
        print("   -> Verify /etc/resolv.conf")
    
    if tcp_failures > 0:
        print(f"❌ TCP Issues: {tcp_failures}/{len(tcp_results)} connections failed")
        print("   -> Firewall blocking outbound connections")
        print("   -> ISP/network infrastructure issues")
    
    if api_failures > 0:
        print(f"❌ API Issues: {api_failures}/{len(api_results)} APIs failed")
        print("   -> SSL/TLS certificate issues")
        print("   -> Proxy/middleware blocking requests")
    
    if dns_failures == 0 and tcp_failures == 0 and api_failures == 0:
        print("✅ All tests passed - network appears healthy")
        print("   -> Issue may be transient or application-specific")
    
    # Specific recommendations
    print("\n🔧 RECOMMENDED ACTIONS:")
    if dns_failures > 0:
        print("1. sudo systemctl restart systemd-resolved")
        print("2. echo 'nameserver 8.8.8.8' | sudo tee /etc/resolv.conf")
    
    if tcp_failures > 0:
        print("3. Check firewall: sudo ufw status")
        print("4. Test with different network (mobile hotspot)")
    
    if api_failures > 0:
        print("5. Update CA certificates: sudo apt update && sudo apt install ca-certificates")
        print("6. Check system time: timedatectl status")

if __name__ == "__main__":
    try:
        diagnose_network_issues()
    except KeyboardInterrupt:
        print("\n⏹️  Diagnostic interrupted by user")
    except Exception as e:
        print(f"\n❌ Diagnostic failed: {str(e)}")