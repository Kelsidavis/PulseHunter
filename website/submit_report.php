<?php
// Set directory and ensure it exists
$saveDir = "reports/";
if (!is_dir($saveDir)) {
    mkdir($saveDir, 0755, true);
}

// Read raw POST data
$data = file_get_contents("php://input");

// Validate JSON
$decoded = json_decode($data, true);
if ($decoded === null) {
    http_response_code(400);
    echo json_encode(["status" => "error", "message" => "Invalid JSON"]);
    exit;
}

// Generate unique filename for the individual report
$timestamp = date("Ymd_His");
$filename = $saveDir . "report_" . $timestamp . ".json";
file_put_contents($filename, $data);

// Rebuild the combined `reports.json`
$allReports = [];
foreach (glob($saveDir . "report_*.json") as $file) {
    $contents = file_get_contents($file);
    $report = json_decode($contents, true);
    if ($report && isset($report["detections"])) {
        $allReports = array_merge($allReports, $report["detections"]);
    }
}

// Save combined detections to reports.json
file_put_contents($saveDir . "reports.json", json_encode($allReports, JSON_PRETTY_PRINT));

echo json_encode(["status" => "success", "message" => "Report received", "saved_as" => $filename]);
?>
