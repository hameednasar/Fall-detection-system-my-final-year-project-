import 'dart:async';
import 'package:flutter/material.dart';
import 'fall_api_service.dart';
import 'main.dart' show AppColors;

class LiveSensorScreen extends StatefulWidget {
  const LiveSensorScreen({super.key});
  @override
  State<LiveSensorScreen> createState() => _LiveSensorScreenState();
}

class _LiveSensorScreenState extends State<LiveSensorScreen>
    with SingleTickerProviderStateMixin {
  Timer? _timer;
  bool _backendOnline = false;
  bool _sensorConnected = false;
  bool _fallDetected = false;
  double _confidence = 0.0;
  String _riskLabel = 'LOW';
  String _timestamp = '--';
  int _rowsCollected = 0;
  double _xAcc = 0, _yAcc = 0, _zAcc = 0;
  double _xGyro = 0, _yGyro = 0, _zGyro = 0;
  bool _loading = true;
  late AnimationController _pulseCtrl;

  @override
  void initState() {
    super.initState();
    _pulseCtrl = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 800))
      ..repeat(reverse: true);
    _startPolling();
  }

  @override
  void dispose() {
    _timer?.cancel();
    _pulseCtrl.dispose();
    super.dispose();
  }

  void _startPolling() {
    _fetchStatus();
    _timer = Timer.periodic(const Duration(seconds: 1), (_) => _fetchStatus());
  }

  Future<void> _fetchStatus() async {
    final data = await FallApiService.getLatestStatus();
    if (!mounted) return;
    if (data == null) {
      setState(() {
        _loading = false;
        _backendOnline = false;
        _sensorConnected = false;
      });
      return;
    }
    setState(() {
      _loading = false;
      _backendOnline = true;
      _sensorConnected = data['sensor_connected'] ?? false;
      _fallDetected = data['fall_detected'] ?? false;
      _confidence = ((data['confidence'] ?? 0.0) as num).toDouble() * 100;
      _riskLabel = data['risk_label'] ?? 'LOW';
      _timestamp = data['timestamp'] ?? '--';
      _rowsCollected = ((data['rows_collected'] ?? 0) as num).toInt();
      _xAcc = ((data['xAcc'] ?? 0.0) as num).toDouble();
      _yAcc = ((data['yAcc'] ?? 0.0) as num).toDouble();
      _zAcc = ((data['zAcc'] ?? 0.0) as num).toDouble();
      _xGyro = ((data['xGyro'] ?? 0.0) as num).toDouble();
      _yGyro = ((data['yGyro'] ?? 0.0) as num).toDouble();
      _zGyro = ((data['zGyro'] ?? 0.0) as num).toDouble();
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
        title: const Text('Live UWB Sensor'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh, color: AppColors.white40, size: 20),
            onPressed: () {
              setState(() => _loading = true);
              _fetchStatus();
            },
          ),
        ],
      ),
      body: _loading
          ? const Center(
              child: CircularProgressIndicator(color: AppColors.accent))
          : SingleChildScrollView(
              padding: const EdgeInsets.all(20),
              child: Column(children: [
                Row(children: [
                  Expanded(
                      child: _StatusBadge(
                    label:
                        _backendOnline ? 'Backend Online' : 'Backend Offline',
                    color:
                        _backendOnline ? AppColors.success : AppColors.danger,
                  )),
                  const SizedBox(width: 10),
                  Expanded(
                      child: _StatusBadge(
                    label: _sensorConnected
                        ? 'Sensor Connected'
                        : 'Sensor Offline',
                    color: _sensorConnected
                        ? AppColors.success
                        : AppColors.warning,
                  )),
                ]),

                const SizedBox(height: 20),

                AnimatedBuilder(
                  animation: _pulseCtrl,
                  builder: (_, child) {
                    final glow = _fallDetected ? _pulseCtrl.value * 24.0 : 0.0;
                    return Container(
                      width: double.infinity,
                      padding: const EdgeInsets.symmetric(
                          vertical: 32, horizontal: 24),
                      decoration: BoxDecoration(
                        color: AppColors.card,
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(
                            color: _riskColor.withOpacity(0.5), width: 1.5),
                        boxShadow: [
                          BoxShadow(
                            color: _riskColor.withOpacity(0.2),
                            blurRadius: 20 + glow,
                            spreadRadius: glow / 5,
                          ),
                        ],
                      ),
                      child: child,
                    );
                  },
                  child: Column(children: [
                    Icon(
                      _fallDetected
                          ? Icons.warning_rounded
                          : Icons.verified_user_outlined,
                      size: 64,
                      color: _riskColor,
                    ),
                    const SizedBox(height: 12),
                    Text(
                      _fallDetected ? '⚠️  FALL DETECTED!' : 'No Fall Detected',
                      style: TextStyle(
                          color: _riskColor,
                          fontSize: 22,
                          fontWeight: FontWeight.w900),
                    ),
                    const SizedBox(height: 6),
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 12, vertical: 4),
                      decoration: BoxDecoration(
                        color: _riskColor.withOpacity(0.12),
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(color: _riskColor.withOpacity(0.4)),
                      ),
                      child: Text('Risk: $_riskLabel',
                          style: TextStyle(
                              color: _riskColor,
                              fontWeight: FontWeight.w700,
                              fontSize: 12,
                              letterSpacing: 1)),
                    ),
                  ]),
                ),

                const SizedBox(height: 16),

                Row(children: [
                  Expanded(
                      child: _StatCard(
                    label: 'Confidence',
                    value: '${_confidence.toStringAsFixed(1)}%',
                    icon: Icons.analytics_outlined,
                    color: _riskColor,
                  )),
                  const SizedBox(width: 12),
                  Expanded(
                      child: _StatCard(
                    label: 'Rows Sent',
                    value: '$_rowsCollected',
                    icon: Icons.swap_vert,
                    color: AppColors.accentSoft,
                  )),
                ]),

                const SizedBox(height: 12),

                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: AppColors.card,
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(color: AppColors.border),
                  ),
                  child: Column(children: [
                    Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Text('Confidence Level',
                              style: TextStyle(
                                  color: AppColors.white40, fontSize: 13)),
                          Text('${_confidence.toStringAsFixed(1)}%',
                              style: TextStyle(
                                  color: _riskColor,
                                  fontWeight: FontWeight.bold,
                                  fontSize: 13)),
                        ]),
                    const SizedBox(height: 10),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(6),
                      child: LinearProgressIndicator(
                        value: _confidence / 100,
                        minHeight: 10,
                        backgroundColor: AppColors.white15,
                        valueColor: AlwaysStoppedAnimation(_riskColor),
                      ),
                    ),
                  ]),
                ),

                const SizedBox(height: 12),

                if (_sensorConnected)
                  Container(
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: AppColors.card,
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(color: AppColors.border),
                    ),
                    child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(children: [
                            Container(
                              width: 6,
                              height: 6,
                              decoration: const BoxDecoration(
                                  shape: BoxShape.circle,
                                  color: AppColors.success),
                            ),
                            const SizedBox(width: 8),
                            const Text('Live Sensor Values',
                                style: TextStyle(
                                    color: AppColors.white,
                                    fontWeight: FontWeight.w700,
                                    fontSize: 13)),
                          ]),
                          const SizedBox(height: 16),
                          Row(children: [
                            Expanded(
                                child: _ValColumn(
                              title: 'Accel (m/s²)',
                              values: {'X': _xAcc, 'Y': _yAcc, 'Z': _zAcc},
                              color: AppColors.accentSoft,
                            )),
                            Container(
                                width: 1, height: 80, color: AppColors.border),
                            Expanded(
                                child: _ValColumn(
                              title: 'Gyro (°/s)',
                              values: {'X': _xGyro, 'Y': _yGyro, 'Z': _zGyro},
                              color: AppColors.warning,
                            )),
                          ]),
                        ]),
                  ),

                const SizedBox(height: 12),

                Container(
                  padding: const EdgeInsets.all(14),
                  decoration: BoxDecoration(
                    color: AppColors.card,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: AppColors.border),
                  ),
                  child: Row(children: [
                    const Icon(Icons.access_time_outlined,
                        color: AppColors.white40, size: 14),
                    const SizedBox(width: 8),
                    Expanded(
                        child: Text('Last update: $_timestamp',
                            style: const TextStyle(
                                color: AppColors.white40, fontSize: 11))),
                  ]),
                ),

                const SizedBox(height: 16),

                if (!_backendOnline)
                  _InfoBox(
                    color: AppColors.danger,
                    title: 'Backend server is offline',
                    steps: const [
                      '1. Open terminal on your Mac',
                      '2. cd to your fall_backend folder',
                      '3. Run: uvicorn backend:app --host 0.0.0.0 --port 8000',
                      '4. Check IP in auth_screen.dart matches your Mac IP',
                    ],
                  )
                else if (!_sensorConnected)
                  _InfoBox(
                    color: AppColors.warning,
                    title: 'Sensor not connected',
                    steps: const [
                      '1. Connect ESP32 to your Mac via USB cable',
                      '2. Open a new terminal in fall_backend folder',
                      '3. Run: /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 sensor_reader.py',
                      '4. Make sure Mac and phone are on same WiFi',
                    ],
                  ),
              ]),
            ),
    );
  }
}


class _StatusBadge extends StatelessWidget {
  final String label;
  final Color color;
  const _StatusBadge({required this.label, required this.color});

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        decoration: BoxDecoration(
          color: AppColors.card,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: color.withOpacity(0.5)),
        ),
        child: Row(mainAxisSize: MainAxisSize.min, children: [
          Container(
            width: 7,
            height: 7,
            decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: color,
                boxShadow: [
                  BoxShadow(color: color.withOpacity(0.6), blurRadius: 6)
                ]),
          ),
          const SizedBox(width: 8),
          Flexible(
            child: Text(label,
                style: TextStyle(
                    color: color, fontWeight: FontWeight.w600, fontSize: 12),
                overflow: TextOverflow.ellipsis),
          ),
        ]),
      );
}

class _StatCard extends StatelessWidget {
  final String label, value;
  final IconData icon;
  final Color color;
  const _StatCard(
      {required this.label,
      required this.value,
      required this.icon,
      required this.color});

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: AppColors.card,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: AppColors.border),
        ),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Icon(icon, color: color, size: 20),
          const SizedBox(height: 10),
          Text(value,
              style: TextStyle(
                  color: color, fontSize: 22, fontWeight: FontWeight.w800)),
          const SizedBox(height: 4),
          Text(label,
              style: const TextStyle(color: AppColors.white40, fontSize: 12)),
        ]),
      );
}

class _ValColumn extends StatelessWidget {
  final String title;
  final Map<String, double> values;
  final Color color;
  const _ValColumn(
      {required this.title, required this.values, required this.color});

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text(title,
              style: TextStyle(
                  color: color, fontSize: 11, fontWeight: FontWeight.w600)),
          const SizedBox(height: 10),
          ...values.entries.map((e) => Padding(
                padding: const EdgeInsets.only(bottom: 6),
                child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text('${e.key}:',
                          style: const TextStyle(
                              color: AppColors.white40, fontSize: 13)),
                      Text(e.value.toStringAsFixed(3),
                          style: const TextStyle(
                              color: AppColors.white,
                              fontSize: 13,
                              fontFamily: 'monospace',
                              fontWeight: FontWeight.w600)),
                    ]),
              )),
        ]),
      );
}

class _InfoBox extends StatelessWidget {
  final Color color;
  final String title;
  final List<String> steps;
  const _InfoBox(
      {required this.color, required this.title, required this.steps});

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: color.withOpacity(0.06),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: color.withOpacity(0.35)),
        ),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [
            Icon(Icons.info_outline, color: color, size: 16),
            const SizedBox(width: 8),
            Text(title,
                style: TextStyle(
                    color: color, fontWeight: FontWeight.bold, fontSize: 13)),
          ]),
          const SizedBox(height: 10),
          ...steps.map((s) => Padding(
                padding: const EdgeInsets.only(bottom: 5),
                child: Text(s,
                    style: const TextStyle(
                        color: AppColors.white70, fontSize: 12, height: 1.5)),
              )),
        ]),
      );
}
