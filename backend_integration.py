# backend_integration.py
"""
Backend integration for Atlas GeoAI Vehicle Detection Frontend
This file provides the backend API endpoints to integrate with your Python vehicle detection script.
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import tempfile
import json
import zipfile
from werkzeug.utils import secure_filename
import subprocess
import sys
import time
import geoai
import geopandas as gpd
from geoai.dinov3 import DINOv3GeoProcessor, visualize_similarity_results
import numpy as np
from PIL import Image
import base64
import io
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}
MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB

# Track active processing jobs to prevent duplicates
active_jobs = set()

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def run_ship_detection(image_path, output_dir):
    """Run ship detection using geoai-py with user's specific script"""
    try:
        print(f"Starting ship detection for: {image_path}")
        start_time = time.time()
        
        # Initialize the ship detector (using user's script)
        print("Initializing ShipDetector...")
        detector = geoai.ShipDetector()
        print("ShipDetector initialized successfully")
        
        # Generate ship masks with UNIVERSAL detection parameters (works for all maritime imagery)
        print("Generating ship masks...")
        masks_path = os.path.join(output_dir, "ships_masks.tif")
        detector.generate_masks(
            image_path,
            output_path=masks_path,
            confidence_threshold=0.4,    # Balanced for universal detection
            mask_threshold=0.5,           # Balanced threshold for various imagery
            overlap=0.6,                  # Good overlap for different boat densities
            chip_size=(256, 256),         # Balanced chip size for various resolutions
            batch_size=4,                 # Keep your batch size
        )
        print(f"Masks generated: {masks_path}")
        
        # Vectorize masks with UNIVERSAL parameters (works for all maritime imagery)
        print("Vectorizing masks...")
        geojson_path = os.path.join(output_dir, "ships_masks.geojson")
        gdf = detector.vectorize_masks(
            masks_path,
            output_path=geojson_path,
            confidence_threshold=0.4,     # Balanced threshold for vectorization
            min_object_area=25,            # Universal minimum for various boat sizes
            max_object_size=100000,        # Universal maximum for various ship sizes
        )
        print(f"Vectorization complete: {len(gdf)} objects found")
        
        # Add geometric properties (from user's script)
        print("Adding geometric properties...")
        gdf = geoai.add_geometric_properties(gdf)
        
        # Filter ships with UNIVERSAL parameters (works for all maritime imagery)
        print("Filtering ships...")
        gdf_filtered = gdf[
            (gdf["area_m2"] > 15) &        # Universal minimum for various boat sizes
            (gdf["area_m2"] < 100000) &   # Universal maximum for various ship sizes
            (gdf["minor_length_m"] > 2)    # Universal minimum length for various boats
        ]
        print(f"Filtered ships: {len(gdf_filtered)}")
        
        # Calculate statistics
        total_ships = len(gdf_filtered)
        avg_confidence = gdf_filtered["confidence"].mean() if total_ships > 0 else 0
        processing_time = time.time() - start_time
        
        # Classify ships by size (UNIVERSAL categories for all maritime imagery)
        small_boats = len(gdf_filtered[(gdf_filtered["area_m2"] >= 15) & (gdf_filtered["area_m2"] < 100)])
        medium_boats = len(gdf_filtered[(gdf_filtered["area_m2"] >= 100) & (gdf_filtered["area_m2"] < 1000)])
        large_boats = len(gdf_filtered[(gdf_filtered["area_m2"] >= 1000) & (gdf_filtered["area_m2"] < 10000)])
        mega_ships = len(gdf_filtered[(gdf_filtered["area_m2"] >= 10000) & (gdf_filtered["area_m2"] < 100000)])
        
        print(f"Ship detection complete: {total_ships} ships found in {processing_time:.2f}s")
        
        # Prepare results
        results = {
            'total_ships': total_ships,
            'small_boats': small_boats,
            'medium_boats': medium_boats,
            'large_boats': large_boats,
            'mega_ships': mega_ships,
            'confidence': round(avg_confidence, 3),
            'processing_time': round(processing_time, 2),
            'detection_type': 'ship'
        }
        
        # Save filtered results to different formats
        output_files = {}
        
        # Save as GeoJSON with error handling
        try:
            geojson_output = os.path.join(output_dir, 'ships.geojson')
            gdf_filtered.to_file(geojson_output, driver='GeoJSON')
            output_files['polygons_geojson'] = geojson_output
            print(f"GeoJSON saved successfully: {geojson_output}")
        except Exception as e:
            print(f"Error saving GeoJSON: {e}")
            # Try alternative GeoJSON creation
            try:
                geojson_output = os.path.join(output_dir, 'ships.geojson')
                gdf_filtered.to_file(geojson_output, driver='GeoJSON', engine='pyogrio')
                output_files['polygons_geojson'] = geojson_output
                print(f"GeoJSON saved with pyogrio: {geojson_output}")
            except Exception as e2:
                print(f"Error saving GeoJSON with pyogrio: {e2}")
                # Create GeoJSON manually
                try:
                    geojson_output = os.path.join(output_dir, 'ships.geojson')
                    gdf_filtered.to_file(geojson_output, driver='GeoJSON', engine='fiona')
                    output_files['polygons_geojson'] = geojson_output
                    print(f"GeoJSON saved with fiona: {geojson_output}")
                except Exception as e3:
                    print(f"Error saving GeoJSON with fiona: {e3}")
                    # Skip GeoJSON if all methods fail
                    print("Skipping GeoJSON creation due to errors")
        
        # Save as Shapefile
        try:
            shp_output = os.path.join(output_dir, 'ships.shp')
            gdf_filtered.to_file(shp_output, driver='ESRI Shapefile')
            output_files['polygons_shp'] = shp_output
            print(f"Shapefile saved successfully: {shp_output}")
        except Exception as e:
            print(f"Error saving Shapefile: {e}")
        
        # Save as KML
        try:
            kml_output = os.path.join(output_dir, 'ships.kml')
            gdf_filtered.to_file(kml_output, driver='KML')
            output_files['polygons_kml'] = kml_output
            print(f"KML saved successfully: {kml_output}")
        except Exception as e:
            print(f"Error saving KML: {e}")
        
        # Save statistics
        stats_output = os.path.join(output_dir, 'statistics.json')
        with open(stats_output, 'w') as f:
            json.dump(results, f, indent=2)
        output_files['statistics'] = stats_output
        
        # Clean up intermediate files
        if os.path.exists(masks_path):
            os.remove(masks_path)
        if os.path.exists(geojson_path):
            os.remove(geojson_path)
        
        print("Ship detection completed successfully")
        return results, output_files
        
    except Exception as e:
        print(f"Error in ship detection: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def run_building_detection(image_path, output_dir):
    """
    Run building footprint extraction using geoai-py package (USA model)
    """
    try:
        start_time = time.time()
        
        # Initialize the building footprint extractor
        extractor = geoai.BuildingFootprintExtractor()
        
        # Try multiple approaches for building detection
        geojson_path = os.path.join(output_dir, "buildings.geojson")
        
        try:
            # Method 1: Direct vector extraction (most comprehensive)
            gdf = extractor.process_raster(
                image_path,
                output_path=geojson_path,
                batch_size=4,
                confidence_threshold=0.3,  # Lowered from 0.5 for more sensitive detection
                overlap=0.5,              # Increased from 0.25 for better coverage
                nms_iou_threshold=0.3,    # Lowered from 0.5 for less aggressive filtering
                min_object_area=50,       # Lowered from 100 for smaller buildings
                max_object_area=None,     # No upper limit for buildings
                mask_threshold=0.3,      # Lowered from 0.5 for better mask generation
                simplify_tolerance=1.0,
            )
        except Exception as e:
            print(f"Method 1 failed: {e}")
            # Method 2: Raster to vector approach
            try:
                masks_path = os.path.join(output_dir, "building_masks.tif")
                extractor.save_masks_as_geotiff(
                    raster_path=image_path,
                    output_path=masks_path,
                    confidence_threshold=0.3,
                    mask_threshold=0.3,
                )
                
                gdf = extractor.masks_to_vector(
                    mask_path=masks_path,
                    output_path=geojson_path,
                    simplify_tolerance=1.0,
                )
            except Exception as e2:
                print(f"Method 2 failed: {e2}")
                # Method 3: Generate masks and vectorize
                masks_path = os.path.join(output_dir, "building_masks.tif")
                extractor.generate_masks(
                    raster_path=image_path,
                    output_dir=masks_path,
                    min_object_area=50,
                    confidence_threshold=0.3,
                    threshold=0.3,
                )
                
                gdf = geoai.orthogonalize(
                    input_path=masks_path, 
                    output_path=geojson_path, 
                    epsilon=1.0
                )
        
        # Regularize building footprints for better geometry
        gdf_regularized = extractor.regularize_buildings(
            gdf=gdf,
            min_area=50,                # Lowered from 100
            angle_threshold=20,         # Increased from 15 for more tolerance
            orthogonality_threshold=0.2, # Lowered from 0.3 for more tolerance
            rectangularity_threshold=0.5, # Lowered from 0.7 for more tolerance
        )
        
        # Add geometric properties
        gdf_final = geoai.add_geometric_properties(gdf_regularized)
        
        # Filter buildings by area with building-specific parameters
        gdf_filtered = gdf_final[
            (gdf_final["area_m2"] > 20) &     # Lowered from 50 for smaller buildings
            (gdf_final["area_m2"] < 20000) &  # Increased from 10000 for larger buildings
            (gdf_final["minor_length_m"] > 2)  # Lowered from 3 for smaller buildings
        ]
        
        # Calculate statistics
        total_buildings = len(gdf_filtered)
        avg_confidence = gdf_filtered["confidence"].mean() if total_buildings > 0 else 0
        processing_time = time.time() - start_time
        
        # Classify buildings by size (rough classification)
        small_buildings = len(gdf_filtered[(gdf_filtered["area_m2"] >= 20) & (gdf_filtered["area_m2"] < 200)])
        medium_buildings = len(gdf_filtered[(gdf_filtered["area_m2"] >= 200) & (gdf_filtered["area_m2"] < 1000)])
        large_buildings = len(gdf_filtered[(gdf_filtered["area_m2"] >= 1000) & (gdf_filtered["area_m2"] < 5000)])
        commercial_buildings = len(gdf_filtered[(gdf_filtered["area_m2"] >= 500) & (gdf_filtered["area_m2"] < 20000)])
        
        # Prepare results
        results = {
            'total_buildings': total_buildings,
            'small_buildings': small_buildings,
            'medium_buildings': medium_buildings,
            'large_buildings': large_buildings,
            'commercial_buildings': commercial_buildings,
            'confidence': round(avg_confidence, 3),
            'processing_time': round(processing_time, 2),
            'detection_type': 'building'
        }
        
        # Save filtered results to different formats
        output_files = {}
        
        # Save as GeoJSON with error handling
        try:
            geojson_output = os.path.join(output_dir, 'buildings.geojson')
            gdf_filtered.to_file(geojson_output, driver='GeoJSON')
            output_files['polygons_geojson'] = geojson_output
            print(f"GeoJSON saved successfully: {geojson_output}")
        except Exception as e:
            print(f"Error saving GeoJSON: {e}")
            # Try alternative GeoJSON creation
            try:
                geojson_output = os.path.join(output_dir, 'buildings.geojson')
                gdf_filtered.to_file(geojson_output, driver='GeoJSON', engine='pyogrio')
                output_files['polygons_geojson'] = geojson_output
                print(f"GeoJSON saved with pyogrio: {geojson_output}")
            except Exception as e2:
                print(f"Error saving GeoJSON with pyogrio: {e2}")
                # Create GeoJSON manually
                try:
                    geojson_output = os.path.join(output_dir, 'buildings.geojson')
                    gdf_filtered.to_file(geojson_output, driver='GeoJSON', engine='fiona')
                    output_files['polygons_geojson'] = geojson_output
                    print(f"GeoJSON saved with fiona: {geojson_output}")
                except Exception as e3:
                    print(f"Error saving GeoJSON with fiona: {e3}")
                    # Skip GeoJSON if all methods fail
                    print("Skipping GeoJSON creation due to errors")
        
        # Save as Shapefile
        try:
            shp_output = os.path.join(output_dir, 'buildings.shp')
            gdf_filtered.to_file(shp_output, driver='ESRI Shapefile')
            output_files['polygons_shp'] = shp_output
            print(f"Shapefile saved successfully: {shp_output}")
        except Exception as e:
            print(f"Error saving Shapefile: {e}")
        
        # Save as KML
        try:
            kml_output = os.path.join(output_dir, 'buildings.kml')
            gdf_filtered.to_file(kml_output, driver='KML')
            output_files['polygons_kml'] = kml_output
            print(f"KML saved successfully: {kml_output}")
        except Exception as e:
            print(f"Error saving KML: {e}")
        
        # Save statistics
        stats_output = os.path.join(output_dir, 'statistics.json')
        with open(stats_output, 'w') as f:
            json.dump(results, f, indent=2)
        output_files['statistics'] = stats_output
        
        # Clean up intermediate files
        if os.path.exists(geojson_path):
            os.remove(geojson_path)
        
        return results, output_files
        
    except Exception as e:
        raise Exception(f"Building detection failed: {str(e)}")

def run_vehicle_detection(image_path, output_dir):
    """
    Run vehicle detection using geoai-py package
    """
    try:
        start_time = time.time()
        
        # Initialize the car detector
        detector = geoai.CarDetector()
        
        # Generate masks for vehicle detection with optimized parameters
        masks_path = os.path.join(output_dir, "cars_masks.tif")
        detector.generate_masks(
            raster_path=image_path,
            output_path=masks_path,
            confidence_threshold=0.2,  # Lowered from 0.3 for more sensitive detection
            mask_threshold=0.3,       # Lowered from 0.5 for better mask generation
            overlap=0.5,              # Increased from 0.25 for better coverage
            chip_size=(512, 512),     # Increased from 400x400 for better context
        )
        
        # Convert masks to vector polygons with optimized parameters
        geojson_path = os.path.join(output_dir, "cars.geojson")
        gdf = detector.vectorize_masks(
            masks_path=masks_path,
            output_path=geojson_path,
            min_object_area=50,    # Lowered from 100 to catch smaller vehicles
            max_object_area=5000, # Increased from 2000 to catch larger vehicles/trucks
        )
        
        # Add geometric properties
        gdf = geoai.add_geometric_properties(gdf)
        
        # Filter vehicles by area with more inclusive parameters
        gdf_filtered = gdf[
            (gdf["area_m2"] > 3) &     # Lowered from 8 to catch smaller vehicles
            (gdf["area_m2"] < 120) &  # Increased from 60 to catch larger vehicles/trucks
            (gdf["minor_length_m"] > 0.5)  # Lowered from 1 to catch smaller vehicles
        ]
        
        # Calculate statistics
        total_vehicles = len(gdf_filtered)
        avg_confidence = gdf_filtered["confidence"].mean() if total_vehicles > 0 else 0
        processing_time = time.time() - start_time
        
        # Classify vehicles by size (rough classification)
        cars = len(gdf_filtered[(gdf_filtered["area_m2"] >= 3) & (gdf_filtered["area_m2"] < 25)])
        trucks = len(gdf_filtered[(gdf_filtered["area_m2"] >= 25) & (gdf_filtered["area_m2"] < 80)])
        buses = len(gdf_filtered[(gdf_filtered["area_m2"] >= 80) & (gdf_filtered["area_m2"] < 120)])
        motorcycles = len(gdf_filtered[(gdf_filtered["area_m2"] >= 3) & (gdf_filtered["area_m2"] < 8) & (gdf_filtered["minor_length_m"] < 2)])
        
        # Prepare results
        results = {
            'total_vehicles': total_vehicles,
            'cars': cars,
            'trucks': trucks,
            'buses': buses,
            'motorcycles': motorcycles,
            'confidence': round(avg_confidence, 3),
            'processing_time': round(processing_time, 2),
            'detection_type': 'vehicle'
        }
        
        # Save filtered results to different formats
        output_files = {}
        
        # Save as GeoJSON with error handling
        try:
            geojson_output = os.path.join(output_dir, 'vehicles.geojson')
            gdf_filtered.to_file(geojson_output, driver='GeoJSON')
            output_files['polygons_geojson'] = geojson_output
            print(f"GeoJSON saved successfully: {geojson_output}")
        except Exception as e:
            print(f"Error saving GeoJSON: {e}")
            # Try alternative GeoJSON creation
            try:
                geojson_output = os.path.join(output_dir, 'vehicles.geojson')
                gdf_filtered.to_file(geojson_output, driver='GeoJSON', engine='pyogrio')
                output_files['polygons_geojson'] = geojson_output
                print(f"GeoJSON saved with pyogrio: {geojson_output}")
            except Exception as e2:
                print(f"Error saving GeoJSON with pyogrio: {e2}")
                # Create GeoJSON manually
                try:
                    geojson_output = os.path.join(output_dir, 'vehicles.geojson')
                    gdf_filtered.to_file(geojson_output, driver='GeoJSON', engine='fiona')
                    output_files['polygons_geojson'] = geojson_output
                    print(f"GeoJSON saved with fiona: {geojson_output}")
                except Exception as e3:
                    print(f"Error saving GeoJSON with fiona: {e3}")
                    # Skip GeoJSON if all methods fail
                    print("Skipping GeoJSON creation due to errors")
        
        # Save as Shapefile
        try:
            shp_output = os.path.join(output_dir, 'vehicles.shp')
            gdf_filtered.to_file(shp_output, driver='ESRI Shapefile')
            output_files['polygons_shp'] = shp_output
            print(f"Shapefile saved successfully: {shp_output}")
        except Exception as e:
            print(f"Error saving Shapefile: {e}")
        
        # Save as KML
        try:
            kml_output = os.path.join(output_dir, 'vehicles.kml')
            gdf_filtered.to_file(kml_output, driver='KML')
            output_files['polygons_kml'] = kml_output
            print(f"KML saved successfully: {kml_output}")
        except Exception as e:
            print(f"Error saving KML: {e}")
        
        # Save statistics
        stats_output = os.path.join(output_dir, 'statistics.json')
        with open(stats_output, 'w') as f:
            json.dump(results, f, indent=2)
        output_files['statistics'] = stats_output
        
        # Clean up intermediate files
        if os.path.exists(masks_path):
            os.remove(masks_path)
        if os.path.exists(geojson_path):
            os.remove(geojson_path)
        
        return results, output_files
        
    except Exception as e:
        raise Exception(f"Vehicle detection failed: {str(e)}")

def run_solar_panel_detection(image_path, output_dir):
    """Run solar panel detection using geoai-py"""
    try:
        print(f"Starting solar panel detection for: {image_path}")
        start_time = time.time()
        
        # Initialize the solar panel detector
        print("Initializing SolarPanelDetector...")
        detector = geoai.SolarPanelDetector()
        print("SolarPanelDetector initialized successfully")
        
        # Generate solar panel masks with OPTIMIZED detection parameters for faster processing
        print("Generating solar panel masks...")
        masks_path = os.path.join(output_dir, "solar_panels_masks.tif")
        detector.generate_masks(
            image_path,
            output_path=masks_path,
            confidence_threshold=0.4,    # Balanced for universal detection
            mask_threshold=0.5,           # Balanced threshold for various imagery
            overlap=0.5,                  # Reduced overlap for faster processing (50% vs 60%)
            chip_size=(512, 512),         # Larger chip size for fewer batches and faster processing
            batch_size=8,                 # Increased batch size for faster processing
        )
        print(f"Masks generated: {masks_path}")
        
        # Vectorize masks with UNIVERSAL parameters
        print("Vectorizing masks...")
        geojson_path = os.path.join(output_dir, "solar_panels_masks.geojson")
        gdf = detector.vectorize_masks(
            masks_path,
            output_path=geojson_path,
            confidence_threshold=0.4,     # Balanced threshold for vectorization
            min_object_area=25,            # Universal minimum for various panel sizes
            max_object_size=100000,        # Universal maximum for large solar farms
        )
        print(f"Vectorization complete: {len(gdf)} objects found")
        
        # Add geometric properties
        print("Adding geometric properties...")
        gdf = geoai.add_geometric_properties(gdf)
        
        # Filter solar panels with UNIVERSAL parameters
        print("Filtering solar panels...")
        gdf_filtered = gdf[
            (gdf["area_m2"] > 15) &        # Universal minimum for various panel sizes
            (gdf["area_m2"] < 100000) &   # Universal maximum for large solar farms
            (gdf["minor_length_m"] > 2)   # Universal minimum length for various panels
        ]
        print(f"Filtered solar panels: {len(gdf_filtered)}")
        
        # Calculate statistics
        total_panels = len(gdf_filtered)
        avg_confidence = gdf_filtered["confidence"].mean() if total_panels > 0 else 0
        processing_time = time.time() - start_time
        
        # Classify solar panels by size (UNIVERSAL categories)
        small_panels = len(gdf_filtered[(gdf_filtered["area_m2"] >= 15) & (gdf_filtered["area_m2"] < 50)])
        medium_panels = len(gdf_filtered[(gdf_filtered["area_m2"] >= 50) & (gdf_filtered["area_m2"] < 200)])
        large_panels = len(gdf_filtered[(gdf_filtered["area_m2"] >= 200) & (gdf_filtered["area_m2"] < 1000)])
        solar_farms = len(gdf_filtered[(gdf_filtered["area_m2"] >= 1000) & (gdf_filtered["area_m2"] < 100000)])
        
        print(f"Solar panel detection complete: {total_panels} panels found in {processing_time:.2f}s")
        
        # Prepare results
        results = {
            'total_panels': total_panels,
            'small_panels': small_panels,
            'medium_panels': medium_panels,
            'large_panels': large_panels,
            'solar_farms': solar_farms,
            'confidence': round(avg_confidence, 3),
            'processing_time': round(processing_time, 2),
            'detection_type': 'solar_panel'
        }
        
        # Save filtered results to different formats
        output_files = {}
        
        # Save as GeoJSON with error handling
        try:
            geojson_output = os.path.join(output_dir, 'solar_panels.geojson')
            gdf_filtered.to_file(geojson_output, driver='GeoJSON')
            output_files['polygons_geojson'] = geojson_output
            print(f"GeoJSON saved successfully: {geojson_output}")
        except Exception as e:
            print(f"Error saving GeoJSON: {e}")
            # Try alternative GeoJSON creation
            try:
                geojson_output = os.path.join(output_dir, 'solar_panels.geojson')
                gdf_filtered.to_file(geojson_output, driver='GeoJSON', engine='pyogrio')
                output_files['polygons_geojson'] = geojson_output
                print(f"GeoJSON saved with pyogrio: {geojson_output}")
            except Exception as e2:
                print(f"Error saving GeoJSON with pyogrio: {e2}")
                # Create GeoJSON manually
                try:
                    geojson_output = os.path.join(output_dir, 'solar_panels.geojson')
                    gdf_filtered.to_file(geojson_output, driver='GeoJSON', engine='fiona')
                    output_files['polygons_geojson'] = geojson_output
                    print(f"GeoJSON saved with fiona: {geojson_output}")
                except Exception as e3:
                    print(f"Error saving GeoJSON with fiona: {e3}")
                    # Skip GeoJSON if all methods fail
                    print("Skipping GeoJSON creation due to errors")
        
        # Save as Shapefile
        try:
            shp_output = os.path.join(output_dir, 'solar_panels.shp')
            gdf_filtered.to_file(shp_output, driver='ESRI Shapefile')
            output_files['polygons_shp'] = shp_output
            print(f"Shapefile saved successfully: {shp_output}")
        except Exception as e:
            print(f"Error saving Shapefile: {e}")
        
        # Save as KML
        try:
            kml_output = os.path.join(output_dir, 'solar_panels.kml')
            gdf_filtered.to_file(kml_output, driver='KML')
            output_files['polygons_kml'] = kml_output
            print(f"KML saved successfully: {kml_output}")
        except Exception as e:
            print(f"Error saving KML: {e}")
        
        # Save statistics
        stats_output = os.path.join(output_dir, 'statistics.json')
        with open(stats_output, 'w') as f:
            json.dump(results, f, indent=2)
        output_files['statistics'] = stats_output
        
        # Clean up intermediate files
        if os.path.exists(masks_path):
            os.remove(masks_path)
        if os.path.exists(geojson_path):
            os.remove(geojson_path)
        
        print("Solar panel detection completed successfully")
        return results, output_files
        
    except Exception as e:
        print(f"Error in solar panel detection: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def run_pf_simplifind(image_path, query_coords, output_dir):
    """Run PF-SimpliFind similarity analysis using DINOv3"""
    try:
        print(f"Starting PF-SimpliFind analysis for: {image_path}")
        print(f"Query coordinates: {query_coords}")
        start_time = time.time()
        
        # Initialize the DINOv3 processor
        print("Initializing DINOv3GeoProcessor...")
        processor = DINOv3GeoProcessor()
        print("DINOv3GeoProcessor initialized successfully")
        
        # Load and preprocess image
        print("Loading and preprocessing image...")
        data, metadata = processor.load_image(image_path)
        image = processor.preprocess_image_for_dinov3(data)
        features, h_patches, w_patches = processor.extract_features(image)
        
        print(f"Image size: {image.size}")
        print(f"Features shape: {features.shape}")
        print(f"Patch grid: {h_patches} x {w_patches}")
        
        # Convert pixel coordinates to patch coordinates
        img_w, img_h = data.shape[2], data.shape[1]
        patch_x = int((query_coords['x'] / img_w) * w_patches)
        patch_y = int((query_coords['y'] / img_h) * h_patches)
        
        print(f"Query pixel coordinates: {query_coords}")
        print(f"Query patch coordinates: ({patch_x}, {patch_y})")
        
        # Compute similarity
        print("Computing similarity...")
        similarities = processor.compute_patch_similarity(features, patch_x, patch_y)
        similarity_array = similarities.cpu().numpy()
        
        print(f"Similarity range: {similarity_array.min():.3f} - {similarity_array.max():.3f}")
        
        # Create overlay visualization
        print("Creating similarity overlay...")
        overlay_img = processor.create_similarity_overlay(
            source=image_path, 
            similarity_data=similarity_array, 
            colormap="turbo", 
            alpha=0.6
        )
        
        # Save overlay as image file
        overlay_path = os.path.join(output_dir, 'similarity_overlay.png')
        overlay_pil = Image.fromarray((overlay_img * 255).astype(np.uint8))
        overlay_pil.save(overlay_path)
        print(f"Overlay saved to: {overlay_path}")
        
        # Save similarity data as numpy array
        similarity_data_path = os.path.join(output_dir, 'similarity_data.npy')
        np.save(similarity_data_path, similarity_array)
        
        # Convert overlay to base64 for frontend display
        buffer = io.BytesIO()
        overlay_pil.save(buffer, format='PNG')
        overlay_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Calculate statistics
        processing_time = time.time() - start_time
        max_similarity = float(similarity_array.max())
        min_similarity = float(similarity_array.min())
        mean_similarity = float(similarity_array.mean())
        
        # Count high similarity areas (above 0.7)
        high_similarity_count = int(np.sum(similarity_array > 0.7))
        medium_similarity_count = int(np.sum((similarity_array > 0.5) & (similarity_array <= 0.7)))
        low_similarity_count = int(np.sum(similarity_array <= 0.5))
        
        print(f"PF-SimpliFind analysis complete in {processing_time:.2f}s")
        
        # Prepare results
        results = {
            'query_coords': query_coords,
            'patch_coords': [patch_x, patch_y],
            'patch_grid_size': [h_patches, w_patches],
            'max_similarity': round(max_similarity, 3),
            'min_similarity': round(min_similarity, 3),
            'mean_similarity': round(mean_similarity, 3),
            'high_similarity_areas': high_similarity_count,
            'medium_similarity_areas': medium_similarity_count,
            'low_similarity_areas': low_similarity_count,
            'processing_time': round(processing_time, 2),
            'detection_type': 'pf_simplifind',
            'overlay_base64': overlay_base64
        }
        
        # Save results to different formats
        output_files = {}
        
        # Save overlay image
        output_files['overlay_image'] = overlay_path
        
        # Save similarity data
        output_files['similarity_data'] = similarity_data_path
        
        # Save statistics
        stats_output = os.path.join(output_dir, 'statistics.json')
        with open(stats_output, 'w') as f:
            # Create a copy without base64 data for JSON storage
            stats_data = results.copy()
            del stats_data['overlay_base64']  # Remove base64 data from JSON
            json.dump(stats_data, f, indent=2)
        output_files['statistics'] = stats_output
        
        print("PF-SimpliFind analysis completed successfully")
        return results, output_files
        
    except Exception as e:
        print(f"Error in PF-SimpliFind analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload from frontend"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            
            # Get file info
            file_size = os.path.getsize(file_path)
            file_info = {
                'filename': filename,
                'size': file_size,
                'path': file_path,
                'status': 'uploaded'
            }
            
            return jsonify(file_info)
        else:
            return jsonify({'error': 'Invalid file type'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/convert-image/<filename>', methods=['GET'])
def convert_image_for_display(filename):
    """Convert TIFF or other geospatial images to web-compatible format"""
    try:
        print(f"Converting image: {filename}")
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        print(f"File path: {filepath}")
        
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return jsonify({'error': 'File not found'}), 404
        
        # Check if it's a TIFF file
        if filename.lower().endswith(('.tif', '.tiff')):
            print("Processing TIFF file...")
            # Read TIFF with rasterio
            with rasterio.open(filepath) as src:
                print(f"TIFF info - Bands: {src.count}, Shape: {src.shape}, CRS: {src.crs}")
                # Read the first 3 bands (RGB) or just the first band if grayscale
                if src.count >= 3:
                    print("Reading as RGB image...")
                    # RGB image
                    data = src.read([1, 2, 3])
                    data = np.transpose(data, (1, 2, 0))
                else:
                    print("Reading as grayscale image...")
                    # Grayscale image
                    data = src.read(1)
                
                print(f"Data shape: {data.shape}, dtype: {data.dtype}, min: {data.min()}, max: {data.max()}")
                
                # Normalize data to 0-255 range
                if data.dtype != np.uint8:
                    # Handle different data types
                    if data.max() <= 1.0:
                        # Assume data is in 0-1 range
                        data = (data * 255).astype(np.uint8)
                    else:
                        # Scale to 0-255 range
                        data_min, data_max = data.min(), data.max()
                        if data_max > data_min:
                            data = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
                        else:
                            data = np.zeros_like(data, dtype=np.uint8)
                
                # Convert to PIL Image
                print("Converting to PIL Image...")
                if len(data.shape) == 3:
                    pil_image = Image.fromarray(data, 'RGB')
                else:
                    pil_image = Image.fromarray(data, 'L')
                
                print(f"PIL Image created: {pil_image.size}, mode: {pil_image.mode}")
                
                # Convert to base64
                buffer = io.BytesIO()
                pil_image.save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                
                print("TIFF conversion successful!")
                return jsonify({
                    'success': True,
                    'image_data': f'data:image/png;base64,{img_str}',
                    'width': pil_image.width,
                    'height': pil_image.height,
                    'format': 'PNG'
                })
        
        else:
            # For non-TIFF files, try to open with PIL directly
            try:
                with Image.open(filepath) as pil_image:
                    # Convert to RGB if necessary
                    if pil_image.mode not in ('RGB', 'L'):
                        pil_image = pil_image.convert('RGB')
                    
                    buffer = io.BytesIO()
                    pil_image.save(buffer, format='PNG')
                    img_str = base64.b64encode(buffer.getvalue()).decode()
                    
                    return jsonify({
                        'success': True,
                        'image_data': f'data:image/png;base64,{img_str}',
                        'width': pil_image.width,
                        'height': pil_image.height,
                        'format': 'PNG'
                    })
            except Exception as e:
                return jsonify({'error': f'Unable to process image: {str(e)}'}), 400
    
    except Exception as e:
        print(f"Error converting image: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Image conversion failed: {str(e)}'}), 500

@app.route('/api/process', methods=['POST'])
def process_image():
    """Process uploaded image for vehicle detection"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        detection_type = data.get('detection_type', 'vehicle')  # Default to vehicle detection
        
        print(f"Processing request - filename: {filename}, detection_type: {detection_type}")
        if data.get('query_coords'):
            print(f"Query coordinates: {data.get('query_coords')}")
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Check if this file is already being processed
        job_key = f"{filename}_{detection_type}"
        if job_key in active_jobs:
            return jsonify({'error': 'This file is already being processed. Please wait for the current job to complete.'}), 409
        
        # Mark job as active
        active_jobs.add(job_key)
        
        try:
            # Create output directory for this processing job
            job_id = f"job_{int(time.time())}"
            output_dir = os.path.join(RESULTS_FOLDER, job_id)
            os.makedirs(output_dir, exist_ok=True)
            
            # Run detection based on type
            if detection_type == 'building':
                results, output_files = run_building_detection(file_path, output_dir)
            elif detection_type == 'ship':
                results, output_files = run_ship_detection(file_path, output_dir)
            elif detection_type == 'solar_panel':
                results, output_files = run_solar_panel_detection(file_path, output_dir)
            elif detection_type == 'pf_simplifind':
                # For PF-SimpliFind, we need query coordinates
                query_coords = data.get('query_coords')
                if not query_coords:
                    return jsonify({'error': 'Query coordinates required for PF-SimpliFind'}), 400
                results, output_files = run_pf_simplifind(file_path, query_coords, output_dir)
            else:
                results, output_files = run_vehicle_detection(file_path, output_dir)
            
            # Add job info to results
            results['job_id'] = job_id
            results['output_files'] = output_files
            results['detection_type'] = detection_type
            
            return jsonify(results)
        
        finally:
            # Always remove job from active set when done (success or failure)
            active_jobs.discard(job_key)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<job_id>/<format>')
def download_results(job_id, format):
    """Download results in specified format"""
    try:
        output_dir = os.path.join(RESULTS_FOLDER, job_id)
        if not os.path.exists(output_dir):
            return jsonify({'error': 'Results not found'}), 404
        
        # Map format to file extension (check if it's building or vehicle detection)
        stats_file = os.path.join(output_dir, 'statistics.json')
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            detection_type = stats.get('detection_type', 'vehicle')
        else:
            detection_type = 'vehicle'
        
        # Map format to file extension based on detection type
        if detection_type == 'building':
            format_map = {
                'shp': 'buildings.shp',
                'kml': 'buildings.kml',
                'geojson': 'buildings.geojson'
            }
        elif detection_type == 'ship':
            format_map = {
                'shp': 'ships.shp',
                'kml': 'ships.kml',
                'geojson': 'ships.geojson'
            }
        elif detection_type == 'solar_panel':
            format_map = {
                'shp': 'solar_panels.shp',
                'kml': 'solar_panels.kml',
                'geojson': 'solar_panels.geojson'
            }
        elif detection_type == 'pf_simplifind':
            format_map = {
                'png': 'similarity_overlay.png',
                'npy': 'similarity_data.npy',
                'json': 'statistics.json'
            }
        else:
            format_map = {
                'shp': 'vehicles.shp',
                'kml': 'vehicles.kml',
                'geojson': 'vehicles.geojson'
            }
        
        if format not in format_map:
            return jsonify({'error': 'Invalid format'}), 400
        
        filename = format_map[format]
        file_path = os.path.join(output_dir, filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(file_path, as_attachment=True, download_name=filename)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status/<job_id>')
def get_job_status(job_id):
    """Get processing job status"""
    try:
        output_dir = os.path.join(RESULTS_FOLDER, job_id)
        if not os.path.exists(output_dir):
            return jsonify({'error': 'Job not found'}), 404
        
        # Check if processing is complete
        stats_file = os.path.join(output_dir, 'statistics.json')
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                results = json.load(f)
            return jsonify({'status': 'completed', 'results': results})
        else:
            return jsonify({'status': 'processing'})
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Atlas GeoAI Backend is running'})

@app.route('/')
def serve_frontend():
    """Serve the frontend HTML file"""
    try:
        return send_file('PF - Atlas.html')
    except Exception as e:
        return f"Error serving frontend: {str(e)}", 500

@app.route('/api/active-jobs')
def get_active_jobs():
    """Get list of currently active processing jobs"""
    return jsonify({'active_jobs': list(active_jobs), 'count': len(active_jobs)})

@app.route('/api/processing-status/<filename>/<detection_type>')
def get_processing_status(filename, detection_type):
    """Check if a specific file is currently being processed"""
    job_key = f"{filename}_{detection_type}"
    is_processing = job_key in active_jobs
    
    # Check if results exist - look for any job directory that contains results for this file/type
    results_exist = False
    results_dir = None
    
    # Search through all job directories to find one with results for this file/type
    if os.path.exists(RESULTS_FOLDER):
        for job_dir in os.listdir(RESULTS_FOLDER):
            if job_dir.startswith('job_'):
                potential_dir = os.path.join(RESULTS_FOLDER, job_dir)
                # Check if this directory has results files
                if os.path.isdir(potential_dir):
                    # For PF-SimpliFind, look for similarity_overlay.png
                    if detection_type == 'pf_simplifind':
                        if os.path.exists(os.path.join(potential_dir, 'similarity_overlay.png')):
                            results_exist = True
                            results_dir = potential_dir
                            break
                    # For other detection types, look for statistics.json
                    elif os.path.exists(os.path.join(potential_dir, 'statistics.json')):
                        results_exist = True
                        results_dir = potential_dir
                        break
    
    if is_processing:
        status = 'processing'
        message = 'Processing in progress...'
    elif results_exist:
        status = 'completed'
        message = 'Processing completed successfully'
    else:
        status = 'not_found'
        message = 'No processing job found'
    
    return jsonify({
        'status': status,
        'is_processing': is_processing,
        'job_key': job_key,
        'message': message,
        'results_available': results_exist,
        'results_dir': results_dir
    })

if __name__ == '__main__':
    print("Starting Atlas GeoAI Backend Server...")
    print("Frontend URL: http://localhost:3000")
    print("Backend API: http://localhost:5000")
    print("API Documentation: http://localhost:5000/api/health")
    
    # Use debug=False to prevent auto-restart which interferes with job tracking
    app.run(debug=False, host='0.0.0.0', port=5000)

# For Vercel deployment
app = app
