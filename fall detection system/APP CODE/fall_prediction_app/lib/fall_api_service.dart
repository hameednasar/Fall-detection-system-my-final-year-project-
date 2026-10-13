import 'dart:convert';
import 'package:http/http.dart' as http;
import 'auth_screen.dart' show AuthService, kBaseUrl;

class PredictionResult {
  final bool fallDetected;
  final double confidence;
  final String riskLabel;
  final int riskScore;
  final String timestamp;

  const PredictionResult({
    required this.fallDetected,
    required this.confidence,
    required this.riskLabel,
    required this.riskScore,
    required this.timestamp,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> j) => PredictionResult(
        fallDetected: j['fall_detected'] as bool,
        confidence: (j['confidence'] as num).toDouble(),
        riskLabel: j['risk_label'] as String,
        riskScore: (j['risk_score'] as num).toInt(),
        timestamp: j['timestamp'] as String,
      );
}

class FallApiService {
  
  static Future<bool> pingBackend() async {
    try {
      final res = await http
          .get(Uri.parse('$kBaseUrl/health/sensor'))
          .timeout(const Duration(seconds: 4));
      return res.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  static Future<Map<String, dynamic>?> getLatestStatus() async {
    try {
      final res = await http
          .get(Uri.parse('$kBaseUrl/status/latest'))
          .timeout(const Duration(seconds: 4));
      if (res.statusCode == 200) {
        return jsonDecode(res.body) as Map<String, dynamic>;
      }
    } catch (_) {}
    return null;
  }

  static Future<PredictionResult?> predict(List<List<double>> window) async {
    final token = await AuthService.getToken();
    if (token == null) return null;
    try {
      final res = await http
          .post(
            Uri.parse('$kBaseUrl/predict/fall'),
            headers: {
              'Content-Type': 'application/json',
              'Authorization': 'Bearer $token',
            },
            body: jsonEncode({'window': window}),
          )
          .timeout(const Duration(seconds: 6));
      if (res.statusCode == 200) {
        return PredictionResult.fromJson(
            jsonDecode(res.body) as Map<String, dynamic>);
      }
    } catch (_) {}
    return null;
  }
}
