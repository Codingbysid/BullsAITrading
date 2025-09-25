"""
Authentication and Security Module for QuantAI Portfolio Manager.

This module handles user authentication, authorization, and security
features including password hashing, session management, and rate limiting.
"""

import hashlib
import hmac
import secrets
import time
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging
from ..database.db_manager import QuantAIDatabase

class SecurityManager:
    """Handles all security-related operations"""
    
    def __init__(self, db: QuantAIDatabase, settings):
        self.db = db
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.failed_attempts = {}  # Track failed login attempts
        self.active_sessions = {}  # Track active sessions
    
    def hash_password(self, password: str, salt: str = None) -> Tuple[str, str]:
        """Hash password with salt using PBKDF2"""
        if salt is None:
            salt = secrets.token_hex(32)
        
        # Use PBKDF2 with SHA-256
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # 100,000 iterations
        )
        
        return password_hash.hex(), salt
    
    def verify_password(self, password: str, stored_hash: str, salt: str) -> bool:
        """Verify password against stored hash"""
        try:
            password_hash, _ = self.hash_password(password, salt)
            return hmac.compare_digest(password_hash, stored_hash)
        except Exception as e:
            self.logger.error(f"Password verification error: {e}")
            return False
    
    def validate_password_strength(self, password: str) -> Dict[str, bool]:
        """Validate password strength"""
        validation = {
            'min_length': len(password) >= self.settings.security.password_min_length,
            'has_uppercase': any(c.isupper() for c in password),
            'has_lowercase': any(c.islower() for c in password),
            'has_digit': any(c.isdigit() for c in password),
            'has_special': any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password) if self.settings.security.password_require_special else True
        }
        
        validation['is_valid'] = all(validation.values())
        return validation
    
    def check_rate_limit(self, identifier: str, endpoint: str) -> bool:
        """Check if request is within rate limits"""
        current_time = time.time()
        key = f"{identifier}:{endpoint}"
        
        if key not in self.failed_attempts:
            self.failed_attempts[key] = []
        
        # Clean old attempts
        self.failed_attempts[key] = [
            attempt_time for attempt_time in self.failed_attempts[key]
            if current_time - attempt_time < 60  # 1 minute window
        ]
        
        # Check rate limit
        rate_limits = self.settings.get_rate_limits()
        limit = rate_limits.get(endpoint, 10)
        
        if len(self.failed_attempts[key]) >= limit:
            return False
        
        # Record this attempt
        self.failed_attempts[key].append(current_time)
        return True
    
    def check_login_attempts(self, username: str) -> bool:
        """Check if user has exceeded login attempts"""
        current_time = time.time()
        key = f"login:{username}"
        
        if key not in self.failed_attempts:
            return True
        
        # Clean old attempts
        self.failed_attempts[key] = [
            attempt_time for attempt_time in self.failed_attempts[key]
            if current_time - attempt_time < self.settings.security.lockout_duration
        ]
        
        return len(self.failed_attempts[key]) < self.settings.security.max_login_attempts
    
    def record_failed_login(self, username: str):
        """Record failed login attempt"""
        current_time = time.time()
        key = f"login:{username}"
        
        if key not in self.failed_attempts:
            self.failed_attempts[key] = []
        
        self.failed_attempts[key].append(current_time)
        self.logger.warning(f"Failed login attempt for user: {username}")
    
    def create_session(self, user_id: int, username: str) -> str:
        """Create user session"""
        session_id = secrets.token_urlsafe(32)
        session_data = {
            'user_id': user_id,
            'username': username,
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'expires_at': datetime.now() + timedelta(seconds=self.settings.security.session_timeout)
        }
        
        self.active_sessions[session_id] = session_data
        self.logger.info(f"Session created for user {username}")
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Dict]:
        """Validate session and return user data"""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        
        # Check if session is expired
        if datetime.now() > session['expires_at']:
            del self.active_sessions[session_id]
            return None
        
        # Update last activity
        session['last_activity'] = datetime.now()
        return session
    
    def invalidate_session(self, session_id: str):
        """Invalidate user session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            self.logger.info(f"Session invalidated: {session_id}")
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        current_time = datetime.now()
        expired_sessions = [
            session_id for session_id, session in self.active_sessions.items()
            if current_time > session['expires_at']
        ]
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
        
        if expired_sessions:
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user with enhanced security"""
        try:
            # Check rate limiting
            if not self.check_rate_limit(username, 'auth'):
                self.logger.warning(f"Rate limit exceeded for user: {username}")
                return None
            
            # Check login attempts
            if not self.check_login_attempts(username):
                self.logger.warning(f"Too many failed attempts for user: {username}")
                return None
            
            # Get user from database
            cursor = self.db.connection.cursor(dictionary=True)
            query = """
                SELECT user_id, username, password_hash, salt, is_active, last_login
                FROM users WHERE username = %s
            """
            cursor.execute(query, (username,))
            user = cursor.fetchone()
            cursor.close()
            
            if not user:
                self.record_failed_login(username)
                return None
            
            if not user['is_active']:
                self.logger.warning(f"Attempted login to inactive account: {username}")
                return None
            
            # Verify password
            if not self.verify_password(password, user['password_hash'], user['salt']):
                self.record_failed_login(username)
                return None
            
            # Update last login
            cursor = self.db.connection.cursor()
            cursor.execute(
                "UPDATE users SET last_login = %s WHERE user_id = %s",
                (datetime.now(), user['user_id'])
            )
            self.db.connection.commit()
            cursor.close()
            
            # Create session
            session_id = self.create_session(user['user_id'], username)
            
            return {
                'user_id': user['user_id'],
                'username': username,
                'session_id': session_id,
                'authenticated': True
            }
            
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return None
    
    def register_user(self, username: str, password: str, email: str = None,
                     risk_tolerance: str = 'Moderate', experience: str = 'Beginner') -> Dict:
        """Register new user with enhanced security"""
        try:
            # Validate password strength
            password_validation = self.validate_password_strength(password)
            if not password_validation['is_valid']:
                return {
                    'success': False,
                    'error': 'Password does not meet strength requirements',
                    'validation': password_validation
                }
            
            # Check if username already exists
            cursor = self.db.connection.cursor()
            cursor.execute("SELECT user_id FROM users WHERE username = %s", (username,))
            if cursor.fetchone():
                cursor.close()
                return {
                    'success': False,
                    'error': 'Username already exists'
                }
            
            # Hash password
            password_hash, salt = self.hash_password(password)
            
            # Create user
            query = """
                INSERT INTO users (username, password_hash, salt, email, risk_tolerance, investment_experience)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (username, password_hash, salt, email, risk_tolerance, experience))
            user_id = cursor.lastrowid
            self.db.connection.commit()
            cursor.close()
            
            self.logger.info(f"User registered successfully: {username}")
            
            return {
                'success': True,
                'user_id': user_id,
                'message': 'User registered successfully'
            }
            
        except Exception as e:
            self.logger.error(f"Registration error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def change_password(self, user_id: int, old_password: str, new_password: str) -> Dict:
        """Change user password"""
        try:
            # Get current password hash
            cursor = self.db.connection.cursor(dictionary=True)
            cursor.execute(
                "SELECT password_hash, salt FROM users WHERE user_id = %s",
                (user_id,)
            )
            user = cursor.fetchone()
            cursor.close()
            
            if not user:
                return {'success': False, 'error': 'User not found'}
            
            # Verify old password
            if not self.verify_password(old_password, user['password_hash'], user['salt']):
                return {'success': False, 'error': 'Current password is incorrect'}
            
            # Validate new password
            password_validation = self.validate_password_strength(new_password)
            if not password_validation['is_valid']:
                return {
                    'success': False,
                    'error': 'New password does not meet strength requirements',
                    'validation': password_validation
                }
            
            # Hash new password
            new_password_hash, new_salt = self.hash_password(new_password)
            
            # Update password
            cursor = self.db.connection.cursor()
            cursor.execute(
                "UPDATE users SET password_hash = %s, salt = %s WHERE user_id = %s",
                (new_password_hash, new_salt, user_id)
            )
            self.db.connection.commit()
            cursor.close()
            
            self.logger.info(f"Password changed for user: {user_id}")
            
            return {
                'success': True,
                'message': 'Password changed successfully'
            }
            
        except Exception as e:
            self.logger.error(f"Password change error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_security_summary(self) -> Dict:
        """Get security system summary"""
        return {
            'active_sessions': len(self.active_sessions),
            'failed_attempts_tracked': len(self.failed_attempts),
            'rate_limits': self.settings.get_rate_limits(),
            'security_features': {
                'password_hashing': 'PBKDF2-SHA256',
                'session_management': True,
                'rate_limiting': True,
                'login_attempt_tracking': True,
                'password_strength_validation': True
            }
        }
