import 'dart:async';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:intl/intl.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:url_launcher/url_launcher.dart';
import 'auth_screen.dart';
import 'live_sensor_screen.dart';
import 'fall_api_service.dart';

class AppColors {
  static const bg = Color(0xFF0B1120);
  static const surface = Color(0xFF141E33);
  static const card = Color(0xFF1A2540);
  static const border = Color(0xFF243050);
  static const accent = Color(0xFF3D7BFF);
  static const accentSoft = Color(0xFF5B95FF);
  static const white = Color(0xFFFFFFFF);
  static const white70 = Color(0xB3FFFFFF);
  static const white40 = Color(0x66FFFFFF);
  static const white15 = Color(0x26FFFFFF);
  static const danger = Color(0xFFFF4757);
  static const warning = Color(0xFFFFB730);
  static const success = Color(0xFF2ED573);
}

class NotificationService {
  static final _plugin = FlutterLocalNotificationsPlugin();
  static bool _initialized = false;

  static Future<void> init() async {
    if (_initialized) return;
    const android = AndroidInitializationSettings('@mipmap/ic_launcher');
    const ios = DarwinInitializationSettings(
      requestAlertPermission: true,
      requestBadgePermission: true,
      requestSoundPermission: true,
    );
    await _plugin
        .initialize(const InitializationSettings(android: android, iOS: ios));
    _initialized = true;
  }

  static Future<void> showFallAlert(
      {required String title, required String body}) async {
    await init();
    const androidDetails = AndroidNotificationDetails(
      'fall_alerts',
      'Fall Alerts',
      channelDescription: 'Critical fall detection alerts',
      importance: Importance.max,
      priority: Priority.high,
      playSound: true,
      enableVibration: true,
      color: Color(0xFFFF4757),
    );
    const iosDetails = DarwinNotificationDetails(
        presentAlert: true, presentBadge: true, presentSound: true);
    await _plugin.show(
      DateTime.now().millisecondsSinceEpoch ~/ 1000,
      title,
      body,
      const NotificationDetails(android: androidDetails, iOS: iosDetails),
    );
  }
}

class FallEvent {
  final DateTime timestamp;
  final String type;
  final double confidence;
  final String source;
  FallEvent(this.timestamp, this.type, this.confidence,
      {this.source = 'UWB Sensor'});
}

class FallService {
  static final FallService instance = FallService._internal();
  factory FallService() => instance;
  FallService._internal();

  final List<FallEvent> history = [];
  bool isMonitoring = false;
  StreamSubscription<AccelerometerEvent>? _accelSub;

  void addEvent(FallEvent event) {
    history.insert(0, event);
    NotificationService.showFallAlert(
      title: ' Fall Detected!',
      body:
          '${event.source} detected a fall at ${DateFormat('hh:mm a').format(event.timestamp)}',
    );
  }

  void startMonitoring(Function(double) onSMV, Function(String) onFall) {
    if (isMonitoring) return;
    isMonitoring = true;
    _accelSub = accelerometerEventStream().listen((e) {
      double smv = sqrt(e.x * e.x + e.y * e.y + e.z * e.z) / 9.81;
      onSMV(smv);
      if (smv > 3.8) _trigger(onFall);
    });
  }

  void _trigger(Function(String) onFall) {
    if (history.isNotEmpty &&
        DateTime.now().difference(history.first.timestamp).inSeconds < 8)
      return;
    final event =
        FallEvent(DateTime.now(), 'Fall Detected', 94.0, source: 'Phone IMU');
    addEvent(event);
    onFall('Sudden movement detected!');
  }

  void stopMonitoring() {
    _accelSub?.cancel();
    isMonitoring = false;
  }
}


void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await NotificationService.init();
  final token = await AuthService.getToken();
  runApp(MyApp(isLoggedIn: token != null));
}

class MyApp extends StatelessWidget {
  final bool isLoggedIn;
  const MyApp({super.key, required this.isLoggedIn});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'FallPredict',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        scaffoldBackgroundColor: AppColors.bg,
        colorScheme: const ColorScheme.dark(
            primary: AppColors.accent, surface: AppColors.surface),
        appBarTheme: const AppBarTheme(
          backgroundColor: AppColors.bg,
          foregroundColor: AppColors.white,
          elevation: 0,
          centerTitle: true,
          titleTextStyle: TextStyle(
              color: AppColors.white,
              fontSize: 18,
              fontWeight: FontWeight.w700,
              letterSpacing: -0.3),
          systemOverlayStyle: SystemUiOverlayStyle(
              statusBarColor: Colors.transparent,
              statusBarIconBrightness: Brightness.light),
        ),
        useMaterial3: true,
      ),
      initialRoute: isLoggedIn ? '/home' : '/auth',
      routes: {
        '/auth': (_) => const AuthScreen(),
        '/home': (_) => const MainScreen(),
      },
    );
  }
}

class MainScreen extends StatefulWidget {
  const MainScreen({super.key});
  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  int _idx = 0;
  final _pages = const [
    DashboardScreen(),
    SOSMonitorScreen(),
    LiveSensorScreen(),
    HistoryScreen(),
    ProfileScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(index: _idx, children: _pages),
      bottomNavigationBar: Container(
        decoration: const BoxDecoration(
          color: AppColors.surface,
          border: Border(top: BorderSide(color: AppColors.border, width: 1)),
        ),
        child: BottomNavigationBar(
          currentIndex: _idx,
          onTap: (i) => setState(() => _idx = i),
          backgroundColor: Colors.transparent,
          elevation: 0,
          selectedItemColor: AppColors.accent,
          unselectedItemColor: AppColors.white40,
          selectedLabelStyle:
              const TextStyle(fontSize: 11, fontWeight: FontWeight.w600),
          unselectedLabelStyle: const TextStyle(fontSize: 11),
          type: BottomNavigationBarType.fixed,
          items: const [
            BottomNavigationBarItem(
                icon: Icon(Icons.dashboard_outlined),
                activeIcon: Icon(Icons.dashboard),
                label: 'Dashboard'),
            BottomNavigationBarItem(
                icon: Icon(Icons.sos_outlined),
                activeIcon: Icon(Icons.sos),
                label: 'SOS'),
            BottomNavigationBarItem(
                icon: Icon(Icons.sensors_outlined),
                activeIcon: Icon(Icons.sensors),
                label: 'Live UWB'),
            BottomNavigationBarItem(
                icon: Icon(Icons.history_outlined),
                activeIcon: Icon(Icons.history),
                label: 'History'),
            BottomNavigationBarItem(
                icon: Icon(Icons.person_outline),
                activeIcon: Icon(Icons.person),
                label: 'Profile'),
          ],
        ),
      ),
    );
  }
}

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});
  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  Timer? _timer;
  bool _connected = false;
  bool _fall = false;
  bool _prevFall = false;
  int _riskScore = 0;
  String _riskLabel = 'LOW';
  double _confidence = 0.0;
  String _lastUpdate = '--';
  int _rows = 0;
  final List<FlSpot> _chart = [];
  int _tick = 0;

  @override
  void initState() {
    super.initState();
    _fetch();
    _timer = Timer.periodic(const Duration(seconds: 1), (_) => _fetch());
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  Future<void> _fetch() async {
    final d = await FallApiService.getLatestStatus();
    if (!mounted || d == null) return;

    final newFall = d['fall_detected'] ?? false;
    // Add to history + notify on new fall edge
    if (newFall && !_prevFall) {
      FallService.instance.addEvent(FallEvent(
        DateTime.now(),
        'Fall Detected',
        ((d['confidence'] ?? 0.0) as num).toDouble() * 100,
        source: 'UWB Sensor',
      ));
    }
    _prevFall = newFall;

    setState(() {
      _connected = d['sensor_connected'] ?? false;
      _fall = newFall;
      _riskScore = ((d['risk_score'] ?? 0) as num).toInt();
      _riskLabel = d['risk_label'] ?? 'LOW';
      _confidence = ((d['confidence'] ?? 0.0) as num).toDouble();
      _lastUpdate = d['timestamp'] ?? '--';
      _rows = ((d['rows_collected'] ?? 0) as num).toInt();
      _chart.add(FlSpot(_tick.toDouble(), _riskScore.toDouble()));
      if (_chart.length > 40) _chart.removeAt(0);
      _tick++;
    });
  }

  Color get _riskColor => _riskLabel == 'HIGH'
      ? AppColors.danger
      : _riskLabel == 'MEDIUM'
          ? AppColors.warning
          : AppColors.success;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      appBar: AppBar(
        title: const Text('Dashboard'),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 16),
            child: _LiveBadge(connected: _connected),
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          if (_fall) ...[
            _FallAlertBanner(),
            const SizedBox(height: 16),
          ],

          // Risk card
          _AppCard(
            child: Column(children: [
              Row(children: [
                Expanded(
                  child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text('RISK SCORE',
                            style: TextStyle(
                                color: AppColors.white40,
                                fontSize: 11,
                                letterSpacing: 1.5,
                                fontWeight: FontWeight.w600)),
                        const SizedBox(height: 8),
                        Text('$_riskScore',
                            style: TextStyle(
                                color: _riskColor,
                                fontSize: 56,
                                fontWeight: FontWeight.w900,
                                height: 1)),
                        const SizedBox(height: 6),
                        _RiskChip(label: _riskLabel, color: _riskColor),
                      ]),
                ),
                SizedBox(
                  width: 90,
                  height: 90,
                  child: Stack(alignment: Alignment.center, children: [
                    CircularProgressIndicator(
                      value: _riskScore / 100,
                      strokeWidth: 8,
                      backgroundColor: AppColors.white15,
                      valueColor: AlwaysStoppedAnimation(_riskColor),
                      strokeCap: StrokeCap.round,
                    ),
                    Text('${_riskScore}%',
                        style: TextStyle(
                            color: _riskColor,
                            fontWeight: FontWeight.bold,
                            fontSize: 13)),
                  ]),
                ),
              ]),
              const SizedBox(height: 20),
              const Divider(color: AppColors.border, height: 1),
              const SizedBox(height: 16),
              Row(children: [
                _MiniStat(
                    label: 'Confidence',
                    value: '${(_confidence * 100).toStringAsFixed(1)}%',
                    color: _riskColor),
                _VertDivider(),
                _MiniStat(
                    label: 'Rows',
                    value: '$_rows',
                    color: AppColors.accentSoft),
                _VertDivider(),
                _MiniStat(
                    label: 'Status',
                    value: _connected ? 'Live' : 'Offline',
                    color: _connected ? AppColors.success : AppColors.warning),
              ]),
            ]),
          ),

          const SizedBox(height: 16),

          if (_chart.length > 1)
            _AppCard(
              child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('Risk Score History',
                        style: TextStyle(
                            color: AppColors.white,
                            fontWeight: FontWeight.w700,
                            fontSize: 14)),
                    const SizedBox(height: 16),
                    SizedBox(
                      height: 140,
                      child: LineChart(LineChartData(
                        gridData: FlGridData(
                          show: true,
                          drawVerticalLine: false,
                          getDrawingHorizontalLine: (_) => const FlLine(
                              color: AppColors.border, strokeWidth: 1),
                        ),
                        titlesData: const FlTitlesData(show: false),
                        borderData: FlBorderData(show: false),
                        minY: 0,
                        maxY: 100,
                        lineBarsData: [
                          LineChartBarData(
                            spots: _chart,
                            isCurved: true,
                            color: _riskColor,
                            barWidth: 2.5,
                            dotData: const FlDotData(show: false),
                            belowBarData: BarAreaData(
                                show: true,
                                color: _riskColor.withOpacity(0.08)),
                          ),
                        ],
                      )),
                    ),
                  ]),
            ),

          const SizedBox(height: 12),
          Text('Last update: $_lastUpdate',
              style: const TextStyle(color: AppColors.white40, fontSize: 11)),
        ]),
      ),
    );
  }
}

class SOSMonitorScreen extends StatefulWidget {
  const SOSMonitorScreen({super.key});
  @override
  State<SOSMonitorScreen> createState() => _SOSMonitorScreenState();
}

class _SOSMonitorScreenState extends State<SOSMonitorScreen>
    with SingleTickerProviderStateMixin {
  final _svc = FallService.instance;
  double _smv = 1.0;
  bool _monitoring = false;
  bool _sosSent = false;
  late AnimationController _pulseCtrl;

  String _c1Name = '', _c1Phone = '';
  String _c2Name = '', _c2Phone = '';
  final _c1NameCtrl = TextEditingController();
  final _c1PhoneCtrl = TextEditingController();
  final _c2NameCtrl = TextEditingController();
  final _c2PhoneCtrl = TextEditingController();

  @override
  void initState() {
    super.initState();
    _pulseCtrl = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 800))
      ..repeat(reverse: true);
    _loadContacts();
  }

  @override
  void dispose() {
    _svc.stopMonitoring();
    _pulseCtrl.dispose();
    _c1NameCtrl.dispose();
    _c1PhoneCtrl.dispose();
    _c2NameCtrl.dispose();
    _c2PhoneCtrl.dispose();
    super.dispose();
  }

  Future<void> _loadContacts() async {
    final p = await SharedPreferences.getInstance();
    setState(() {
      _c1Name = p.getString('c1_name') ?? '';
      _c1Phone = p.getString('c1_phone') ?? '';
      _c2Name = p.getString('c2_name') ?? '';
      _c2Phone = p.getString('c2_phone') ?? '';
      _c1NameCtrl.text = _c1Name;
      _c1PhoneCtrl.text = _c1Phone;
      _c2NameCtrl.text = _c2Name;
      _c2PhoneCtrl.text = _c2Phone;
    });
  }

  Future<void> _saveContacts() async {
    final p = await SharedPreferences.getInstance();
    await p.setString('c1_name', _c1NameCtrl.text.trim());
    await p.setString('c1_phone', _c1PhoneCtrl.text.trim());
    await p.setString('c2_name', _c2NameCtrl.text.trim());
    await p.setString('c2_phone', _c2PhoneCtrl.text.trim());
    setState(() {
      _c1Name = _c1NameCtrl.text.trim();
      _c1Phone = _c1PhoneCtrl.text.trim();
      _c2Name = _c2NameCtrl.text.trim();
      _c2Phone = _c2PhoneCtrl.text.trim();
    });
    if (mounted) {
      Navigator.of(context).pop();
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
        content: Text('Emergency contacts saved'),
        backgroundColor: AppColors.success,
        behavior: SnackBarBehavior.floating,
      ));
    }
  }

  void _toggleMonitor() {
    if (_monitoring) {
      _svc.stopMonitoring();
      setState(() => _monitoring = false);
    } else {
      _svc.startMonitoring(
        (smv) => setState(() => _smv = smv),
        (_) => setState(() {}),
      );
      setState(() {
        _monitoring = true;
        _sosSent = false;
      });
    }
  }

  Future<void> _sendSOS() async {
    HapticFeedback.heavyImpact();
    setState(() => _sosSent = true);

    await NotificationService.showFallAlert(
      title: '🚨 SOS Activated!',
      body: 'Emergency contacts have been alerted.',
    );

    FallService.instance.addEvent(
        FallEvent(DateTime.now(), 'Manual SOS', 100.0, source: 'Manual SOS'));

    // Call primary contact
    final callPhone = _c1Phone.isNotEmpty ? _c1Phone : _c2Phone;
    if (callPhone.isNotEmpty) {
      final uri = Uri.parse('tel:$callPhone');
      if (await canLaunchUrl(uri)) await launchUrl(uri);
    }

    if (mounted) {
      showDialog(
        context: context,
        builder: (_) => AlertDialog(
          backgroundColor: AppColors.card,
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
          title: const Row(children: [
            Icon(Icons.check_circle, color: AppColors.success),
            SizedBox(width: 8),
            Text('SOS Sent', style: TextStyle(color: AppColors.white)),
          ]),
          content: Text(
            'Emergency contacts notified.\n\n'
            '${_c1Name.isNotEmpty ? "• $_c1Name\n" : ""}'
            '${_c2Name.isNotEmpty ? "• $_c2Name" : ""}',
            style: const TextStyle(color: AppColors.white70),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child:
                  const Text('OK', style: TextStyle(color: AppColors.accent)),
            )
          ],
        ),
      );
    }
  }

  void _editContacts() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: AppColors.card,
      shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.vertical(top: Radius.circular(24))),
      builder: (_) => Padding(
        padding: EdgeInsets.fromLTRB(
            24, 24, 24, MediaQuery.of(context).viewInsets.bottom + 24),
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          Container(
              width: 36,
              height: 4,
              decoration: BoxDecoration(
                  color: AppColors.border,
                  borderRadius: BorderRadius.circular(2))),
          const SizedBox(height: 20),
          const Text('Emergency Contacts',
              style: TextStyle(
                  color: AppColors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.w700)),
          const SizedBox(height: 24),
          _ContactField(
              label: 'Contact 1 Name',
              ctrl: _c1NameCtrl,
              icon: Icons.person_outline),
          const SizedBox(height: 10),
          _ContactField(
              label: 'Contact 1 Phone',
              ctrl: _c1PhoneCtrl,
              icon: Icons.phone_outlined,
              type: TextInputType.phone),
          const SizedBox(height: 16),
          _ContactField(
              label: 'Contact 2 Name',
              ctrl: _c2NameCtrl,
              icon: Icons.person_outline),
          const SizedBox(height: 10),
          _ContactField(
              label: 'Contact 2 Phone',
              ctrl: _c2PhoneCtrl,
              icon: Icons.phone_outlined,
              type: TextInputType.phone),
          const SizedBox(height: 24),
          SizedBox(
            width: double.infinity,
            height: 52,
            child: ElevatedButton(
              onPressed: _saveContacts,
              style: ElevatedButton.styleFrom(
                  backgroundColor: AppColors.accent,
                  foregroundColor: AppColors.white,
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(14)),
                  elevation: 0),
              child: const Text('Save Contacts',
                  style: TextStyle(fontSize: 15, fontWeight: FontWeight.w700)),
            ),
          ),
        ]),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final hasContacts = _c1Phone.isNotEmpty || _c2Phone.isNotEmpty;
    return Scaffold(
      backgroundColor: AppColors.bg,
      appBar: AppBar(
        title: const Text('Emergency SOS'),
        actions: [
          IconButton(
            icon: const Icon(Icons.contacts_outlined, color: AppColors.white70),
            onPressed: _editContacts,
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(children: [
          // Contacts card
          _AppCard(
            child:
                Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
                const Text('Emergency Contacts',
                    style: TextStyle(
                        color: AppColors.white,
                        fontWeight: FontWeight.w700,
                        fontSize: 14)),
                GestureDetector(
                  onTap: _editContacts,
                  child: const Text('Edit',
                      style: TextStyle(
                          color: AppColors.accent,
                          fontSize: 13,
                          fontWeight: FontWeight.w600)),
                ),
              ]),
              const SizedBox(height: 14),
              if (!hasContacts)
                const Text(
                    'No contacts added. Tap Edit to add emergency contacts.',
                    style: TextStyle(color: AppColors.white40, fontSize: 13))
              else ...[
                if (_c1Name.isNotEmpty)
                  _ContactTile(
                      name: _c1Name, phone: _c1Phone, label: 'Primary'),
                if (_c2Name.isNotEmpty) ...[
                  const SizedBox(height: 8),
                  _ContactTile(
                      name: _c2Name, phone: _c2Phone, label: 'Secondary'),
                ],
              ],
            ]),
          ),

          const SizedBox(height: 32),

          // SOS button
          AnimatedBuilder(
            animation: _pulseCtrl,
            builder: (_, child) {
              final glow = _sosSent ? _pulseCtrl.value * 40.0 : 8.0;
              return Container(
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  boxShadow: [
                    BoxShadow(
                      color: AppColors.danger.withOpacity(_sosSent ? 0.5 : 0.2),
                      blurRadius: 30 + glow,
                      spreadRadius: glow / 4,
                    ),
                  ],
                ),
                child: child,
              );
            },
            child: GestureDetector(
              onTap: _sendSOS,
              child: Container(
                width: 180,
                height: 180,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: _sosSent
                      ? AppColors.danger
                      : AppColors.danger.withOpacity(0.12),
                  border: Border.all(
                      color: AppColors.danger, width: _sosSent ? 3 : 2),
                ),
                child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(Icons.sos,
                          size: 64,
                          color: _sosSent ? AppColors.white : AppColors.danger),
                      const SizedBox(height: 8),
                      Text(
                        _sosSent ? 'SENT!' : 'TAP TO SEND',
                        style: TextStyle(
                            color:
                                _sosSent ? AppColors.white : AppColors.danger,
                            fontWeight: FontWeight.w900,
                            fontSize: 12,
                            letterSpacing: 1.5),
                      ),
                    ]),
              ),
            ),
          ),

          const SizedBox(height: 8),
          const Text('Calls your primary emergency contact',
              style: TextStyle(color: AppColors.white40, fontSize: 12)),

          const SizedBox(height: 32),

          // IMU Monitor toggle
          _AppCard(
            child: Column(children: [
              Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
                const Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Phone IMU Monitor',
                          style: TextStyle(
                              color: AppColors.white,
                              fontWeight: FontWeight.w700,
                              fontSize: 14)),
                      SizedBox(height: 4),
                      Text('Auto-detects falls via accelerometer',
                          style: TextStyle(
                              color: AppColors.white40, fontSize: 12)),
                    ]),
                Switch(
                  value: _monitoring,
                  onChanged: (_) => _toggleMonitor(),
                  activeColor: AppColors.accent,
                ),
              ]),
              if (_monitoring) ...[
                const SizedBox(height: 16),
                const Divider(color: AppColors.border, height: 1),
                const SizedBox(height: 16),
                Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text('Acceleration',
                          style: TextStyle(
                              color: AppColors.white40, fontSize: 13)),
                      Text('${_smv.toStringAsFixed(2)}g',
                          style: TextStyle(
                              color: _smv > 3.8
                                  ? AppColors.danger
                                  : AppColors.accentSoft,
                              fontWeight: FontWeight.bold,
                              fontSize: 16)),
                    ]),
                const SizedBox(height: 10),
                ClipRRect(
                  borderRadius: BorderRadius.circular(6),
                  child: LinearProgressIndicator(
                    value: (_smv / 5).clamp(0.0, 1.0),
                    minHeight: 8,
                    backgroundColor: AppColors.white15,
                    valueColor: AlwaysStoppedAnimation(
                        _smv > 3.8 ? AppColors.danger : AppColors.accent),
                  ),
                ),
                if (_smv > 3.8) ...[
                  const SizedBox(height: 12),
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                    decoration: BoxDecoration(
                      color: AppColors.danger.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(8),
                      border:
                          Border.all(color: AppColors.danger.withOpacity(0.4)),
                    ),
                    child: const Row(children: [
                      Icon(Icons.warning_rounded,
                          color: AppColors.danger, size: 16),
                      SizedBox(width: 8),
                      Text('Fall detected by phone!',
                          style: TextStyle(
                              color: AppColors.danger,
                              fontWeight: FontWeight.w700,
                              fontSize: 13)),
                    ]),
                  ),
                ],
              ],
            ]),
          ),
        ]),
      ),
    );
  }
}

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});
  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    final h = FallService.instance.history;
    return Scaffold(
      backgroundColor: AppColors.bg,
      appBar: AppBar(
        title: const Text('Fall History'),
        actions: [
          if (h.isNotEmpty)
            TextButton(
              onPressed: () {
                FallService.instance.history.clear();
                setState(() {});
              },
              child: const Text('Clear',
                  style: TextStyle(color: AppColors.danger, fontSize: 13)),
            ),
        ],
      ),
      body: h.isEmpty
          ? Center(
              child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                  Container(
                    width: 80,
                    height: 80,
                    decoration: BoxDecoration(
                        color: AppColors.white15, shape: BoxShape.circle),
                    child: const Icon(Icons.history,
                        size: 40, color: AppColors.white40),
                  ),
                  const SizedBox(height: 16),
                  const Text('No fall events yet',
                      style: TextStyle(
                          color: AppColors.white,
                          fontSize: 18,
                          fontWeight: FontWeight.w700)),
                  const SizedBox(height: 8),
                  const Text('Events appear here when a fall is detected',
                      style: TextStyle(color: AppColors.white40, fontSize: 13)),
                ]))
          : ListView.builder(
              padding: const EdgeInsets.all(20),
              itemCount: h.length,
              itemBuilder: (_, i) {
                final e = h[i];
                final isSOS = e.source == 'Manual SOS';
                final color = isSOS
                    ? AppColors.danger
                    : e.confidence > 80
                        ? AppColors.danger
                        : AppColors.warning;
                final icon = isSOS ? Icons.sos : Icons.warning_amber_rounded;

                return Container(
                  margin: const EdgeInsets.only(bottom: 12),
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: AppColors.card,
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(color: color.withOpacity(0.3), width: 1),
                  ),
                  child: Row(children: [
                    Container(
                      width: 44,
                      height: 44,
                      decoration: BoxDecoration(
                          color: color.withOpacity(0.12),
                          shape: BoxShape.circle),
                      child: Icon(icon, color: color, size: 22),
                    ),
                    const SizedBox(width: 14),
                    Expanded(
                        child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                          Text(e.type,
                              style: const TextStyle(
                                  color: AppColors.white,
                                  fontWeight: FontWeight.w700,
                                  fontSize: 14)),
                          const SizedBox(height: 4),
                          Text(
                            DateFormat('MMM dd, yyyy  hh:mm a')
                                .format(e.timestamp),
                            style: const TextStyle(
                                color: AppColors.white40, fontSize: 12),
                          ),
                          const SizedBox(height: 6),
                          _SourceChip(source: e.source),
                        ])),
                    Column(
                        crossAxisAlignment: CrossAxisAlignment.end,
                        children: [
                          Text('${e.confidence.toInt()}%',
                              style: TextStyle(
                                  color: color,
                                  fontWeight: FontWeight.w800,
                                  fontSize: 20)),
                          const Text('confidence',
                              style: TextStyle(
                                  color: AppColors.white40, fontSize: 11)),
                        ]),
                  ]),
                );
              }),
    );
  }
}

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});
  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  String _email = '', _name = '';

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final e = await AuthService.getUserEmail();
    final n = await AuthService.getUserName();
    if (mounted) {
      setState(() {
        _email = e;
        _name = n;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      appBar: AppBar(
        title: const Text('Profile'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(children: [
          const SizedBox(height: 8),
          Container(
            width: 88,
            height: 88,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              gradient: const LinearGradient(
                colors: [AppColors.accent, Color(0xFF1A56DB)],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
              boxShadow: [
                BoxShadow(
                    color: AppColors.accent.withOpacity(0.4),
                    blurRadius: 20,
                    spreadRadius: 2)
              ],
            ),
            child: Center(
              child: Text(
                _name.isNotEmpty ? _name[0].toUpperCase() : 'U',
                style: const TextStyle(
                    fontSize: 36,
                    color: AppColors.white,
                    fontWeight: FontWeight.w800),
              ),
            ),
          ),
          const SizedBox(height: 16),
          Text(_name,
              style: const TextStyle(
                  color: AppColors.white,
                  fontSize: 22,
                  fontWeight: FontWeight.w800)),
          const SizedBox(height: 4),
          Text(_email,
              style: const TextStyle(color: AppColors.white40, fontSize: 14)),
          const SizedBox(height: 32),
          _AppCard(
            child: Column(children: [
              _ProfileTile(
                  icon: Icons.sensors,
                  label: 'UWB Sensor',
                  value: 'See Live UWB tab',
                  color: AppColors.accentSoft),
              const Divider(color: AppColors.border, height: 1),
              _ProfileTile(
                  icon: Icons.warning_amber_rounded,
                  label: 'Fall Events',
                  value: '${FallService.instance.history.length}',
                  color: AppColors.warning),
              const Divider(color: AppColors.border, height: 1),
              _ProfileTile(
                  icon: Icons.monitor_heart,
                  label: 'Phone Monitor',
                  value:
                      FallService.instance.isMonitoring ? 'Active' : 'Inactive',
                  color: FallService.instance.isMonitoring
                      ? AppColors.success
                      : AppColors.white40),
            ]),
          ),
          const SizedBox(height: 24),
          SizedBox(
            width: double.infinity,
            height: 52,
            child: ElevatedButton.icon(
              onPressed: () async {
                await AuthService.logout();
                if (context.mounted) {
                  Navigator.of(context).pushReplacementNamed('/auth');
                }
              },
              icon: const Icon(Icons.logout),
              label: const Text('Logout',
                  style: TextStyle(fontSize: 15, fontWeight: FontWeight.w700)),
              style: ElevatedButton.styleFrom(
                backgroundColor: AppColors.danger.withOpacity(0.1),
                foregroundColor: AppColors.danger,
                elevation: 0,
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(14),
                    side: const BorderSide(color: AppColors.danger, width: 1)),
              ),
            ),
          ),
        ]),
      ),
    );
  }
}


class _AppCard extends StatelessWidget {
  final Widget child;
  const _AppCard({required this.child});
  @override
  Widget build(BuildContext context) => Container(
        width: double.infinity,
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: AppColors.card,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: AppColors.border),
        ),
        child: child,
      );
}

class _FallAlertBanner extends StatelessWidget {
  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
        decoration: BoxDecoration(
          color: AppColors.danger.withOpacity(0.1),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: AppColors.danger, width: 1.5),
        ),
        child: const Row(children: [
          Icon(Icons.warning_rounded, color: AppColors.danger, size: 28),
          SizedBox(width: 12),
          Text('⚠️  FALL DETECTED!',
              style: TextStyle(
                  color: AppColors.danger,
                  fontSize: 18,
                  fontWeight: FontWeight.w900)),
        ]),
      );
}

class _LiveBadge extends StatelessWidget {
  final bool connected;
  const _LiveBadge({required this.connected});
  @override
  Widget build(BuildContext context) =>
      Row(mainAxisSize: MainAxisSize.min, children: [
        Container(
          width: 8,
          height: 8,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: connected ? AppColors.success : AppColors.warning,
            boxShadow: [
              BoxShadow(
                  color: (connected ? AppColors.success : AppColors.warning)
                      .withOpacity(0.7),
                  blurRadius: 6)
            ],
          ),
        ),
        const SizedBox(width: 6),
        Text(connected ? 'LIVE' : 'OFFLINE',
            style: TextStyle(
                fontSize: 11,
                fontWeight: FontWeight.w700,
                color: connected ? AppColors.success : AppColors.warning,
                letterSpacing: 1)),
      ]);
}

class _RiskChip extends StatelessWidget {
  final String label;
  final Color color;
  const _RiskChip({required this.label, required this.color});
  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
        decoration: BoxDecoration(
          color: color.withOpacity(0.12),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: color.withOpacity(0.4)),
        ),
        child: Text(label,
            style: TextStyle(
                color: color,
                fontWeight: FontWeight.w700,
                fontSize: 11,
                letterSpacing: 1)),
      );
}

class _MiniStat extends StatelessWidget {
  final String label, value;
  final Color color;
  const _MiniStat(
      {required this.label, required this.value, required this.color});
  @override
  Widget build(BuildContext context) => Expanded(
        child: Column(children: [
          Text(value,
              style: TextStyle(
                  color: color, fontWeight: FontWeight.w800, fontSize: 16)),
          const SizedBox(height: 2),
          Text(label,
              style: const TextStyle(color: AppColors.white40, fontSize: 11)),
        ]),
      );
}

class _VertDivider extends StatelessWidget {
  @override
  Widget build(BuildContext context) =>
      Container(width: 1, height: 32, color: AppColors.border);
}

class _ContactField extends StatelessWidget {
  final String label;
  final TextEditingController ctrl;
  final IconData icon;
  final TextInputType? type;
  const _ContactField(
      {required this.label, required this.ctrl, required this.icon, this.type});
  @override
  Widget build(BuildContext context) => TextFormField(
        controller: ctrl,
        keyboardType: type,
        style: const TextStyle(color: AppColors.white),
        decoration: InputDecoration(
          labelText: label,
          labelStyle: const TextStyle(color: AppColors.white40, fontSize: 13),
          prefixIcon: Icon(icon, color: AppColors.accentSoft, size: 20),
          filled: true,
          fillColor: AppColors.surface,
          border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(12),
              borderSide: BorderSide.none),
          focusedBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(12),
              borderSide: const BorderSide(color: AppColors.accent)),
        ),
      );
}

class _ContactTile extends StatelessWidget {
  final String name, phone, label;
  const _ContactTile(
      {required this.name, required this.phone, required this.label});
  @override
  Widget build(BuildContext context) => Row(children: [
        Container(
          width: 38,
          height: 38,
          decoration: BoxDecoration(
              color: AppColors.accent.withOpacity(0.12),
              shape: BoxShape.circle),
          child: const Icon(Icons.person_outline,
              color: AppColors.accentSoft, size: 18),
        ),
        const SizedBox(width: 12),
        Expanded(
            child:
                Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text(name,
              style: const TextStyle(
                  color: AppColors.white,
                  fontWeight: FontWeight.w600,
                  fontSize: 14)),
          Text(phone,
              style: const TextStyle(color: AppColors.white40, fontSize: 12)),
        ])),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
          decoration: BoxDecoration(
              color: AppColors.accent.withOpacity(0.12),
              borderRadius: BorderRadius.circular(20)),
          child: Text(label,
              style: const TextStyle(
                  color: AppColors.accentSoft,
                  fontSize: 10,
                  fontWeight: FontWeight.w600)),
        ),
      ]);
}

class _SourceChip extends StatelessWidget {
  final String source;
  const _SourceChip({required this.source});
  @override
  Widget build(BuildContext context) {
    final isSOS = source == 'Manual SOS';
    final color = isSOS ? AppColors.danger : AppColors.accentSoft;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
          color: color.withOpacity(0.1),
          borderRadius: BorderRadius.circular(20)),
      child: Text(source,
          style: TextStyle(
              color: color, fontSize: 10, fontWeight: FontWeight.w600)),
    );
  }
}

class _ProfileTile extends StatelessWidget {
  final IconData icon;
  final String label, value;
  final Color color;
  const _ProfileTile(
      {required this.icon,
      required this.label,
      required this.value,
      required this.color});
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 14),
        child: Row(children: [
          Icon(icon, color: color, size: 20),
          const SizedBox(width: 14),
          Expanded(
              child: Text(label,
                  style:
                      const TextStyle(color: AppColors.white, fontSize: 14))),
          Text(value,
              style: TextStyle(
                  color: color, fontWeight: FontWeight.w700, fontSize: 14)),
        ]),
      );
}
