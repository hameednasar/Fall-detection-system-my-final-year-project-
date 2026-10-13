import 'package:flutter/material.dart';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

const String kBaseUrl = 'http://192.168.137.149:8000';

class AuthService {
  static const _tokenKey = 'auth_token';
  static const _userKey = 'user_email';
  static const _nameKey = 'user_name';

  static Future<String?> login(String email, String password) async {
    try {
      final res = await http
          .post(
            Uri.parse('$kBaseUrl/auth/login'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({'email': email, 'password': password}),
          )
          .timeout(const Duration(seconds: 10));

      if (res.statusCode == 200) {
        final data = jsonDecode(res.body) as Map<String, dynamic>;
        final prefs = await SharedPreferences.getInstance();
        await prefs.setString(_tokenKey, data['access_token'] as String);
        await prefs.setString(_userKey, email.trim().toLowerCase());
        final name = data['name'] as String? ?? '';
        if (name.isNotEmpty) await prefs.setString(_nameKey, name);
        return null;
      }
      final err = jsonDecode(res.body) as Map<String, dynamic>;
      return err['detail'] as String? ?? 'Login failed';
    } catch (e) {
      return 'Cannot reach server.\n\n'
          '• Check IP address in auth_screen.dart\n'
          '• Current IP: $kBaseUrl\n'
          '• Make sure backend.py is running\n'
          '• Phone and PC must be on same WiFi';
    }
  }

  static Future<String?> signup(
      String name, String email, String password) async {
    try {
      final res = await http
          .post(
            Uri.parse('$kBaseUrl/auth/signup'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(
                {'name': name, 'email': email, 'password': password}),
          )
          .timeout(const Duration(seconds: 10));

      if (res.statusCode == 201) {
        final prefs = await SharedPreferences.getInstance();
        await prefs.setString(_nameKey, name.trim());
        return null;
      }
      final err = jsonDecode(res.body) as Map<String, dynamic>;
      return err['detail'] as String? ?? 'Sign up failed';
    } catch (e) {
      return 'Cannot reach server.\n\n'
          '• Check IP address in auth_screen.dart\n'
          '• Make sure backend.py is running';
    }
  }

  static Future<String?> getToken() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_tokenKey);
  }

  static Future<String> getUserEmail() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_userKey) ?? '';
  }

  static Future<String> getUserName() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_nameKey) ?? '';
  }

  static Future<void> logout() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_tokenKey);
    await prefs.remove(_userKey);
    await prefs.remove(_nameKey);
  }

  static Future<bool> checkBackend() async {
    try {
      final res = await http
          .get(Uri.parse('$kBaseUrl/health'))
          .timeout(const Duration(seconds: 4));
      return res.statusCode == 200;
    } catch (_) {
      return false;
    }
  }
}

class AuthScreen extends StatefulWidget {
  const AuthScreen({super.key});

  @override
  State<AuthScreen> createState() => _AuthScreenState();
}

class _AuthScreenState extends State<AuthScreen>
    with SingleTickerProviderStateMixin {
  bool _isLogin = true;
  bool _loading = false;
  bool _obscure = true;
  bool _backendOk = false;

  final _formKey = GlobalKey<FormState>();
  final _nameCtrl = TextEditingController();
  final _emailCtrl = TextEditingController();
  final _passCtrl = TextEditingController();

  late AnimationController _animCtrl;
  late Animation<double> _fadeAnim;
  late Animation<Offset> _slideAnim;

  @override
  void initState() {
    super.initState();
    _animCtrl = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 500));
    _fadeAnim = CurvedAnimation(parent: _animCtrl, curve: Curves.easeOut);
    _slideAnim = Tween<Offset>(begin: const Offset(0, 0.06), end: Offset.zero)
        .animate(CurvedAnimation(parent: _animCtrl, curve: Curves.easeOut));
    _animCtrl.forward();
    _checkBackend();
  }

  Future<void> _checkBackend() async {
    final ok = await AuthService.checkBackend();
    if (mounted) setState(() => _backendOk = ok);
  }

  @override
  void dispose() {
    _animCtrl.dispose();
    _nameCtrl.dispose();
    _emailCtrl.dispose();
    _passCtrl.dispose();
    super.dispose();
  }

  void _toggleMode() {
    _animCtrl.reset();
    setState(() => _isLogin = !_isLogin);
    _animCtrl.forward();
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) return;
    setState(() => _loading = true);

    final String? error = _isLogin
        ? await AuthService.login(_emailCtrl.text.trim(), _passCtrl.text)
        : await AuthService.signup(
            _nameCtrl.text.trim(),
            _emailCtrl.text.trim(),
            _passCtrl.text,
          );

    setState(() => _loading = false);
    if (!mounted) return;

    if (error != null) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        content: Text(error),
        backgroundColor: const Color(0xFFE53935),
        behavior: SnackBarBehavior.floating,
        duration: const Duration(seconds: 5),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ));
    } else {
      if (!_isLogin) {
        setState(() => _isLogin = true);
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
          content: Text('Account created! Please log in.'),
          behavior: SnackBarBehavior.floating,
        ));
      } else {
        Navigator.of(context).pushReplacementNamed('/home');
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0A0F1E),
      body: Stack(children: [
        Positioned(
            top: -80,
            right: -60,
            child: _Blob(
                color: const Color(0xFF1565C0).withOpacity(0.4), size: 260)),
        Positioned(
            bottom: -60,
            left: -40,
            child: _Blob(
                color: const Color(0xFF0D47A1).withOpacity(0.3), size: 220)),
        SafeArea(
          child: SingleChildScrollView(
            padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 24),
            child: FadeTransition(
              opacity: _fadeAnim,
              child: SlideTransition(
                position: _slideAnim,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const SizedBox(height: 32),

                    Row(children: [
                      Container(
                        width: 54,
                        height: 54,
                        decoration: BoxDecoration(
                          color: const Color(0xFF1976D2),
                          borderRadius: BorderRadius.circular(16),
                        ),
                        child: const Icon(Icons.monitor_heart,
                            color: Colors.white, size: 30),
                      ),
                      const SizedBox(width: 14),
                      const Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text('FallPredict',
                                style: TextStyle(
                                    color: Colors.white,
                                    fontSize: 22,
                                    fontWeight: FontWeight.w800)),
                            Text('UWB + IMU Fall Detection',
                                style: TextStyle(
                                    color: Colors.white38, fontSize: 12)),
                          ]),
                    ]),

                    const SizedBox(height: 16),

                    GestureDetector(
                      onTap: _checkBackend,
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 12, vertical: 6),
                        decoration: BoxDecoration(
                          color: _backendOk
                              ? const Color(0xFF0D2B0D)
                              : const Color(0xFF2B0D0D),
                          borderRadius: BorderRadius.circular(20),
                          border: Border.all(
                              color: (_backendOk ? Colors.green : Colors.red)
                                  .withOpacity(0.5)),
                        ),
                        child: Row(mainAxisSize: MainAxisSize.min, children: [
                          Container(
                            width: 7,
                            height: 7,
                            decoration: BoxDecoration(
                                shape: BoxShape.circle,
                                color: _backendOk
                                    ? Colors.greenAccent
                                    : Colors.redAccent),
                          ),
                          const SizedBox(width: 6),
                          Text(
                            _backendOk
                                ? 'Backend connected'
                                : 'Backend offline – tap to retry',
                            style: TextStyle(
                                color: _backendOk
                                    ? Colors.greenAccent
                                    : Colors.redAccent,
                                fontSize: 12,
                                fontWeight: FontWeight.w600),
                          ),
                        ]),
                      ),
                    ),

                    const SizedBox(height: 28),

                    Text(
                      _isLogin ? 'Welcome back' : 'Create account',
                      style: const TextStyle(
                          color: Colors.white,
                          fontSize: 28,
                          fontWeight: FontWeight.w800),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      _isLogin
                          ? 'Sign in to continue monitoring'
                          : 'Join to protect yourself and loved ones',
                      style: TextStyle(
                          color: Colors.white.withOpacity(0.55), fontSize: 15),
                    ),
                    const SizedBox(height: 40),

                    Container(
                      padding: const EdgeInsets.all(24),
                      decoration: BoxDecoration(
                        color: const Color(0xFF111827),
                        borderRadius: BorderRadius.circular(24),
                        border:
                            Border.all(color: Colors.white.withOpacity(0.07)),
                        boxShadow: [
                          BoxShadow(
                              color: Colors.black.withOpacity(0.4),
                              blurRadius: 32,
                              offset: const Offset(0, 12))
                        ],
                      ),
                      child: Form(
                        key: _formKey,
                        child: Column(children: [
                          if (!_isLogin) ...[
                            _InputField(
                              controller: _nameCtrl,
                              label: 'Full Name',
                              icon: Icons.person_outline,
                              validator: (v) => (v == null || v.trim().isEmpty)
                                  ? 'Enter your name'
                                  : null,
                            ),
                            const SizedBox(height: 16),
                          ],
                          _InputField(
                            controller: _emailCtrl,
                            label: 'Email Address',
                            icon: Icons.email_outlined,
                            keyboardType: TextInputType.emailAddress,
                            validator: (v) {
                              if (v == null || v.isEmpty) return 'Enter email';
                              if (!v.contains('@')) return 'Invalid email';
                              return null;
                            },
                          ),
                          const SizedBox(height: 16),
                          _InputField(
                            controller: _passCtrl,
                            label: 'Password',
                            icon: Icons.lock_outline,
                            obscure: _obscure,
                            suffixIcon: IconButton(
                              icon: Icon(
                                  _obscure
                                      ? Icons.visibility_outlined
                                      : Icons.visibility_off_outlined,
                                  color: Colors.white38,
                                  size: 20),
                              onPressed: () =>
                                  setState(() => _obscure = !_obscure),
                            ),
                            validator: (v) {
                              if (v == null || v.isEmpty)
                                return 'Enter password';
                              if (!_isLogin && v.length < 6)
                                return 'Min 6 characters';
                              return null;
                            },
                          ),
                          const SizedBox(height: 28),
                          SizedBox(
                            width: double.infinity,
                            height: 56,
                            child: ElevatedButton(
                              onPressed: _loading ? null : _submit,
                              style: ElevatedButton.styleFrom(
                                backgroundColor: const Color(0xFF1976D2),
                                foregroundColor: Colors.white,
                                disabledBackgroundColor:
                                    const Color(0xFF1976D2).withOpacity(0.5),
                                elevation: 0,
                                shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(14)),
                              ),
                              child: _loading
                                  ? const SizedBox(
                                      width: 22,
                                      height: 22,
                                      child: CircularProgressIndicator(
                                          color: Colors.white,
                                          strokeWidth: 2.5))
                                  : Text(
                                      _isLogin ? 'Sign In' : 'Create Account',
                                      style: const TextStyle(
                                          fontSize: 16,
                                          fontWeight: FontWeight.w700)),
                            ),
                          ),
                        ]),
                      ),
                    ),

                    const SizedBox(height: 28),
                    Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                      Text(
                          _isLogin
                              ? "Don't have an account? "
                              : 'Already have an account? ',
                          style: TextStyle(
                              color: Colors.white.withOpacity(0.45),
                              fontSize: 14)),
                      GestureDetector(
                        onTap: _toggleMode,
                        child: Text(
                          _isLogin ? 'Sign Up' : 'Sign In',
                          style: const TextStyle(
                              color: Color(0xFF64B5F6),
                              fontSize: 14,
                              fontWeight: FontWeight.w700),
                        ),
                      ),
                    ]),
                  ],
                ),
              ),
            ),
          ),
        ),
      ]),
    );
  }
}

class _InputField extends StatelessWidget {
  final TextEditingController controller;
  final String label;
  final IconData icon;
  final bool obscure;
  final Widget? suffixIcon;
  final TextInputType? keyboardType;
  final String? Function(String?)? validator;

  const _InputField({
    required this.controller,
    required this.label,
    required this.icon,
    this.obscure = false,
    this.suffixIcon,
    this.keyboardType,
    this.validator,
  });

  @override
  Widget build(BuildContext context) {
    return TextFormField(
      controller: controller,
      obscureText: obscure,
      keyboardType: keyboardType,
      validator: validator,
      style: const TextStyle(color: Colors.white, fontSize: 15),
      decoration: InputDecoration(
        labelText: label,
        labelStyle:
            TextStyle(color: Colors.white.withOpacity(0.4), fontSize: 14),
        prefixIcon: Icon(icon, color: const Color(0xFF42A5F5), size: 20),
        suffixIcon: suffixIcon,
        filled: true,
        fillColor: const Color(0xFF1A2235),
        border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: BorderSide.none),
        focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: const BorderSide(color: Color(0xFF1976D2), width: 1.5)),
        errorBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: const BorderSide(color: Color(0xFFE53935), width: 1)),
        focusedErrorBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: const BorderSide(color: Color(0xFFE53935), width: 1.5)),
        errorStyle: const TextStyle(color: Color(0xFFEF9A9A)),
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 16, vertical: 18),
      ),
    );
  }
}

class _Blob extends StatelessWidget {
  final Color color;
  final double size;
  const _Blob({required this.color, required this.size});

  @override
  Widget build(BuildContext context) => Container(
      width: size,
      height: size,
      decoration: BoxDecoration(shape: BoxShape.circle, color: color));
}
