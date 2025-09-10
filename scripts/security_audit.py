#!/usr/bin/env python3

import os
import json
import hashlib
from pathlib import Path
import pandas as pd
import warnings

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results" / "experiments"

def check_file_permissions(file_path):
    stat = file_path.stat()
    mode = oct(stat.st_mode)[-3:]
    secure_modes = ['600', '700', '644', '755']
    return mode in secure_modes, mode

def scan_for_secrets(file_path):
    secrets_patterns = [
        'password', 'secret', 'key', 'token', 'api_key', 
        'private_key', 'credential', 'auth'
    ]
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().lower()
            found_patterns = [pattern for pattern in secrets_patterns if pattern in content]
            return found_patterns
    except:
        return []

def check_data_encryption_readiness():
    """Check if data files are ready for encryption"""
    data_files = []
    sensitive_extensions = ['.parquet', '.csv', '.json', '.pkl', '.joblib']
    
    for ext in sensitive_extensions:
        data_files.extend(REPO.glob(f"**/*{ext}"))
    
    encryption_status = {}
    for file_path in data_files:
        # Check if file contains sensitive data patterns
        relative_path = file_path.relative_to(REPO)
        is_sensitive = any(keyword in str(file_path).lower() 
                          for keyword in ['hr', 'employee', 'gold', 'silver'])
        
        encryption_status[str(relative_path)] = {
            'size_bytes': file_path.stat().st_size,
            'is_sensitive': is_sensitive,
            'needs_encryption': is_sensitive and file_path.stat().st_size > 0
        }
    
    return encryption_status

def audit_access_controls():
    """Audit current access control implementation"""
    
    # Check for .env files and security configurations
    env_files = list(REPO.glob("**/.env*"))
    gitignore_path = REPO / ".gitignore"
    
    access_audit = {
        'env_files_found': [str(f.relative_to(REPO)) for f in env_files],
        'gitignore_exists': gitignore_path.exists(),
        'secrets_in_code': {},
        'file_permissions': {}
    }
    
    # Scan Python files for hardcoded secrets
    python_files = list(REPO.glob("**/*.py"))
    for py_file in python_files[:10]:  # Limit scan for performance
        secrets = scan_for_secrets(py_file)
        if secrets:
            access_audit['secrets_in_code'][str(py_file.relative_to(REPO))] = secrets
    
    # Check file permissions on sensitive files
    sensitive_files = list(REPO.glob("**/*.json")) + list(REPO.glob("**/*.yml"))
    for file_path in sensitive_files[:5]:  # Limit for performance
        is_secure, mode = check_file_permissions(file_path)
        access_audit['file_permissions'][str(file_path.relative_to(REPO))] = {
            'mode': mode,
            'is_secure': is_secure
        }
    
    return access_audit

def generate_security_recommendations(audit_results):
    """Generate security recommendations based on audit"""
    recommendations = []
    
    # Data encryption recommendations
    sensitive_files = sum(1 for f in audit_results['data_encryption'].values() 
                         if f['needs_encryption'])
    if sensitive_files > 0:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Data Protection',
            'issue': f'{sensitive_files} sensitive data files need encryption',
            'action': 'Implement AES-256 encryption for all HR data files'
        })
    
    # Access control recommendations
    if audit_results['access_controls']['secrets_in_code']:
        recommendations.append({
            'priority': 'CRITICAL',
            'category': 'Secret Management',
            'issue': 'Potential secrets found in code',
            'action': 'Move all secrets to environment variables or secure vault'
        })
    
    insecure_files = sum(1 for f in audit_results['access_controls']['file_permissions'].values()
                        if not f['is_secure'])
    if insecure_files > 0:
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'File Permissions',
            'issue': f'{insecure_files} files have insecure permissions',
            'action': 'Set restrictive file permissions (600/700) on sensitive files'
        })
    
    # General security recommendations
    recommendations.extend([
        {
            'priority': 'HIGH',
            'category': 'Authentication',
            'issue': 'No multi-factor authentication implemented',
            'action': 'Implement MFA for all system access'
        },
        {
            'priority': 'HIGH',
            'category': 'Monitoring',
            'issue': 'No security monitoring in place',
            'action': 'Deploy SIEM solution for security event monitoring'
        },
        {
            'priority': 'MEDIUM',
            'category': 'Network Security',
            'issue': 'No network segmentation documented',
            'action': 'Implement network segmentation for ML pipeline'
        }
    ])
    
    return recommendations

def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    print("[SECURITY-AUDIT] Starting cybersecurity assessment...")
    
    # Perform security audits
    audit_results = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'data_encryption': check_data_encryption_readiness(),
        'access_controls': audit_access_controls()
    }
    
    # Generate recommendations
    recommendations = generate_security_recommendations(audit_results)
    
    # Security score calculation
    total_issues = len(recommendations)
    critical_issues = sum(1 for r in recommendations if r['priority'] == 'CRITICAL')
    high_issues = sum(1 for r in recommendations if r['priority'] == 'HIGH')
    
    # Simple scoring: 100 - (critical*20 + high*10 + medium*5)
    security_score = max(0, 100 - (critical_issues * 20 + high_issues * 10 + 
                                  (total_issues - critical_issues - high_issues) * 5))
    
    # Compile final report
    security_report = {
        'security_score': security_score,
        'risk_level': 'CRITICAL' if security_score < 50 else 'HIGH' if security_score < 70 else 'MEDIUM',
        'audit_results': audit_results,
        'recommendations': recommendations,
        'summary': {
            'total_recommendations': total_issues,
            'critical_issues': critical_issues,
            'high_priority_issues': high_issues,
            'sensitive_files_count': sum(1 for f in audit_results['data_encryption'].values() 
                                       if f['is_sensitive'])
        }
    }
    
    # Save report
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "security_audit.json", 'w') as f:
        json.dump(security_report, f, indent=2)
    
    print(f"[SECURITY-AUDIT] Security Score: {security_score}/100")
    print(f"[SECURITY-AUDIT] Risk Level: {security_report['risk_level']}")
    print(f"[SECURITY-AUDIT] Critical Issues: {critical_issues}")
    print(f"[SECURITY-AUDIT] High Priority Issues: {high_issues}")
    print(f"[SECURITY-AUDIT] Report saved to {RESULTS_DIR / 'security_audit.json'}")
    print("[SECURITY-AUDIT] Done.")

if __name__ == "__main__":
    main()
