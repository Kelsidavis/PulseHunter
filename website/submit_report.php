<?php
$reportDir = __DIR__ . "/reports/";
if (!is_dir($reportDir)) {
    mkdir($reportDir, 0775, true);
}

$data = file_get_contents("php://input");
if (!$data) {
    http_response_code(400);
    echo "No data received.";
    exit;
}

$timestamp = date("Ymd_His");
$filename = $reportDir . "report_" . $timestamp . ".json";

if (file_put_contents($filename, $data)) {
    http_response_code(200);
    echo "Report saved: " . basename($filename);
} else {
    http_response_code(500);
    echo "Failed to save report.";
}
?>
