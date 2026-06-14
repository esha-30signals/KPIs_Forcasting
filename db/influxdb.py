"""
InfluxDB Helper for time-series database operations
"""
import os
import ssl
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

import logging
logger = logging.getLogger(__name__)
from helpers.json_utils import safe_json_dumps

# Global client instance
_client = None
_write_api = None
_query_api = None

# Global variables (set during connect)
INFLUX_BUCKET = None
INFLUX_ORG = None


def connect():
    """Initialize InfluxDB connection (call once at startup)"""
    global _client, _write_api, _query_api, INFLUX_BUCKET, INFLUX_ORG

    try:
        if _client is None:

            # Get environment variables
            url = os.getenv("INFLUX_URL")
            token = os.getenv("INFLUX_TOKEN")
            INFLUX_ORG = os.getenv("INFLUX_ORG")
            INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "vibelets")

            # Check if required environment variables are set
            if not all([url, token, INFLUX_ORG]):
                raise ValueError("Missing InfluxDB config. Required: INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG")

            # Disable SSL verification for testing
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            # Create client
            _client = InfluxDBClient(
                url=url,
                token=token,
                org=INFLUX_ORG,
                timeout=10000,
                ssl_ca_cert=None,
                verify_ssl=False
            )

            # Create APIs
            _write_api = _client.write_api(write_options=SYNCHRONOUS)
            _query_api = _client.query_api()

            # Test the connection by checking health
            health = _client.health()
            if health.status != "pass":
                raise RuntimeError(f"InfluxDB health check failed: {health.status}")

            print("✅ InfluxDB connection established successfully")

        return _client

    except Exception as e:
        if _client:
            _client.close()
            _client = None
            _write_api = None
            _query_api = None
        raise RuntimeError(f"Database connection error: {str(e)}")


def get_client():
    """Get the InfluxDB client instance"""
    if _client is None:
        raise RuntimeError("InfluxDB not connected. Call connect() first.")
    return _client


def disconnect():
    """Close InfluxDB connection (call at shutdown)"""
    global _client, _write_api, _query_api, INFLUX_BUCKET, INFLUX_ORG
    if _client:
        _client.close()
        _client = None
        _write_api = None
        _query_api = None
        INFLUX_BUCKET = None
        INFLUX_ORG = None
        print("InfluxDB connection closed")


def write_point(measurement: str, tags: Dict[str, str], fields: Dict[str, Any], timestamp=None):
    """
    Write a single data point to InfluxDB

    Args:
        measurement: Measurement name
        tags: Dictionary of tag key-value pairs
        fields: Dictionary of field key-value pairs
        timestamp: Optional timestamp (datetime or int)
    """
    if _write_api is None:
        connect()

    try:
        point = Point(measurement)

        # Add tags
        for tag_key, tag_value in tags.items():
            point.tag(tag_key, tag_value)

        # Add fields
        for field_key, field_value in fields.items():
            point.field(field_key, field_value)

        # Add timestamp if provided
        if timestamp:
            point.time(timestamp)

        # Write the point
        _write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)

    except Exception as e:
        raise RuntimeError(f"Write point error: {str(e)}")


def write_points(points: List[Dict[str, Any]]):
    """
    Write multiple data points to InfluxDB

    Args:
        points: List of point dictionaries with keys: measurement, tags, fields, timestamp (optional)
    """
    if _write_api is None:
        connect()

    try:
        records = []
        for point_data in points:
            point = Point(point_data["measurement"])

            # Add tags
            if "tags" in point_data:
                for tag_key, tag_value in point_data["tags"].items():
                    point.tag(tag_key, tag_value)

            # Add fields
            for field_key, field_value in point_data["fields"].items():
                point.field(field_key, field_value)

            # Add timestamp if provided
            if "timestamp" in point_data:
                point.time(point_data["timestamp"])

            records.append(point)

        # Write all points
        _write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=records)

    except Exception as e:
        raise RuntimeError(f"Write points error: {str(e)}")


def query_flux(flux_query: str) -> List[Dict[str, Any]]:
    """
    Execute a Flux query and return results

    Args:
        flux_query: Flux query string

    Returns:
        List of dictionaries containing query results
    """
    if _query_api is None:
        connect()

    try:
        # print(f"query_flux flux_query: {flux_query}")
        # Execute query
        result = _query_api.query(flux_query)

        # logger.info(f"query_flux result: {safe_json_dumps(result)}")
        
        records = []
        for table in result:
            for record in table.records:
                records.append(record.values)

        return records

    except Exception as e:
        raise RuntimeError(f"Query error: {str(e)}")


def query_range(measurement: str, start: str, stop: str = None, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Query data points from a measurement within a time range

    Args:
        measurement: Measurement name
        start: Start time (e.g., "-1h", "2023-01-01T00:00:00Z")
        stop: Optional stop time (defaults to now)
        filters: Optional filters for tags/fields

    Returns:
        List of data points
    """
    if _query_api is None:
        connect()

    try:
        # Build Flux query
        flux_query = f'''
        from(bucket: "{INFLUX_BUCKET}")
        |> range(start: {start}'''

        if stop:
            flux_query += f', stop: {stop}'

        flux_query += f''')
        |> filter(fn: (r) => r["_measurement"] == "{measurement}")'''

        # Add filters
        if filters:
            for key, value in filters.items():
                if isinstance(value, str):
                    flux_query += f'''
        |> filter(fn: (r) => r["{key}"] == "{value}")'''
                else:
                    flux_query += f'''
        |> filter(fn: (r) => r["{key}"] == {value})'''

        flux_query += '''
        |> yield(name: "results")'''

        return query_flux(flux_query)

    except Exception as e:
        raise RuntimeError(f"Range query error: {str(e)}")


def get_measurements() -> List[str]:
    """
    Get list of all measurements in the bucket

    Returns:
        List of measurement names
    """
    if _query_api is None:
        connect()

    try:
        flux_query = f'''
        import "influxdata/influxdb/schema"
        schema.measurements(bucket: "{INFLUX_BUCKET}")
        '''

        result = _query_api.query(flux_query)
        measurements = []

        for table in result:
            for record in table.records:
                if record.get_value():
                    measurements.append(record.get_value())

        return measurements

    except Exception as e:
        raise RuntimeError(f"Get measurements error: {str(e)}")
