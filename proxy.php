<?php

header("Access-Control-Allow-Origin: *");
header("Access-Control-Allow-Methods: GET, POST, OPTIONS");
header("Access-Control-Allow-Headers: Content-Type, Authorization");

// handle preflight
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200);
    exit;
}
// this is a simple proxy script that forwards requests to the target server and returns the response
// put in /var/www/html/proxy.php and access via http://your-server-ip/proxy.php/your-endpoint
// target server URL is http://professionalsnail.ddns.net:5050, so for example to access http://professionalsnail.ddns.net:5050/api/data, you would access http://your-server-ip/proxy.php/api/data
$targetBase = "http://professionalsnail.ddns.net:5050";

// Get path AFTER proxy.php
$path = parse_url($_SERVER["REQUEST_URI"], PHP_URL_PATH);
$script = $_SERVER["SCRIPT_NAME"];

// remove "/proxy.php"
$forwardPath = str_replace($script, "", $path);

// final URL
$url = $targetBase . $forwardPath;

// append query string if exists
if (!empty($_SERVER["QUERY_STRING"])) {
    $url .= "?" . $_SERVER["QUERY_STRING"];
}

$ch = curl_init($url);

curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_CUSTOMREQUEST, $_SERVER["REQUEST_METHOD"]);
curl_setopt($ch, CURLOPT_POSTFIELDS, file_get_contents("php://input"));

$headers = [];
foreach (getallheaders() as $key => $value) {
    if (strtolower($key) !== "host") {
        $headers[] = "$key: $value";
    }
}

curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);

$response = curl_exec($ch);
$httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);

curl_close($ch);

http_response_code($httpCode);
echo $response;