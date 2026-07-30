# -*- coding: utf-8 -*-

"""A GUI-based signature clustering interface."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import random
from datetime import datetime
import shutil
import traceback
import json
import re
import sys
import subprocess
import tempfile
import shlex
import zipfile

import numpy as np
from PIL import Image, ImageTk
from scipy.spatial.distance import pdist
from rapidfuzz.distance import DamerauLevenshtein
import hnswlib

from signature_clustering import SignatureFeatureExtractor, SignatureClustering, \
    test_configs, default_config


class SmoothScroller:
    """
    Handles smooth scrolling with acceleration and deceleration for Tkinter canvas widgets.
    Provides platform-specific optimizations for Windows, macOS, and Linux.
    """
    def __init__(self, widget, axis='y', platform=None):
        """
        Initialize the scroller for a widget.
        
        Args:
            widget: The canvas widget to scroll
            axis: 'x' for horizontal scrolling, 'y' for vertical scrolling
            platform: Platform identifier ('windows', 'macos', 'linux'), or None to auto-detect
        """
        self.widget = widget
        self.axis = axis

        # Detect platform if not specified
        if platform is None:
            platform = self._detect_platform()
        self.platform = platform

        # Scrolling state
        self.velocity = 0.0
        self.last_event_time = 0
        self.animation_id = None
        self.is_scrolling = False

        # For direct scrolling on macOS
        self.scroll_queue = []
        self.processing_scroll = False

        # Load platform-specific parameters
        self.params = self._get_platform_params()

    def _detect_platform(self):
        """Detect the current platform."""
        if sys.platform.startswith('win'):
            return 'windows'
        elif sys.platform.startswith('darwin'):
            return 'macos'
        else:
            return 'linux'

    def _get_platform_params(self):
        """Get scrolling parameters for the current platform."""
        # Default parameters for all platforms
        default_params = {
            'initial_velocity_factor': 0.5,     # Initial velocity multiplier
            'acceleration_factor': 1.2,         # How quickly acceleration builds up
            'deceleration_factor': 0.9,         # How quickly scrolling slows down (higher = slower)
            'max_velocity': 30.0,               # Maximum scrolling velocity
            'min_velocity': 0.5,                # Velocity threshold to stop scrolling
            'animation_interval': 10,           # Animation frame interval (ms)
            'scroll_multiplier': 1.0,           # Base scroll speed multiplier
            'use_pixels': False,                # Use pixels instead of units for scrolling
            'acceleration_threshold': 200,      # Time threshold (ms) for acceleration
        }

        # Platform-specific parameter adjustments
        platform_params = {
            'windows': {
                'initial_velocity_factor': 0.4, # Lower initial velocity
                'acceleration_factor': 1.2,     # Moderate acceleration
                'deceleration_factor': 0.88,    # Faster deceleration
                'max_velocity': 20.0,           # Lower max velocity
                'min_velocity': 0.3,            # Higher threshold to stop
                'animation_interval': 10,       # Standard animation frames
                'scroll_multiplier': 0.8,       # Slower scrolling for mouse wheels
                'use_pixels': False,            # Use unit-based scrolling
                'direct_scroll': False,         # Use the inertial model
            },
            'macos': {
                # macOS-specific parameters with acceleration curve
                'scroll_multiplier': 2.0,       # Base sensitivity (lower is more precise)
                'acceleration_factor': 6.0,     # Controls how aggressive the acceleration curve is
                'min_delta': 0.0,               # Min scroll delta to register (0 = full precision)
                'max_delta': 200.0,             # No upper limit by default
                'direct_scroll': True,          # Use direct scrolling model for macOS
                'line_height': 1,               # Scale factor for pixel scrolling
                'debounce_time': 150,           # Time (ms) to wait for scroll end
                'use_pixels': True,             # Use pixel-based scrolling
                'animation_interval': 1,        # Fast animation interval for smoothness
            },
            'linux': {
                'initial_velocity_factor': 0.3, # Lower initial velocity
                'acceleration_factor': 1.1,     # Gentle acceleration
                'deceleration_factor': 0.85,    # Faster deceleration
                'max_velocity': 18.0,           # Lower max velocity
                'min_velocity': 0.4,            # Higher threshold to stop
                'animation_interval': 12,       # Slower animation frames
                'scroll_multiplier': 0.9,       # Slower scrolling
                'use_pixels': False,            # Use unit-based scrolling
                'direct_scroll': False,         # Use the inertial model
            }
        }

        # Start with default parameters
        params = default_params.copy()

        # Apply platform-specific adjustments
        if self.platform in platform_params:
            for key, value in platform_params[self.platform].items():
                params[key] = value

        return params

    def handle_scroll_event(self, event):
        """
        Handle a scroll event and start smooth scrolling if needed.
        
        Args:
            event: The scroll event
            
        Returns:
            "break" to prevent further event propagation
        """
        # For macOS, use the direct scrolling model which better matches
        # the trackpad's natural behavior
        if self.platform == 'macos' and self.params.get('direct_scroll', False):
            return self._handle_macos_direct_scroll(event)
        else:
            # For other platforms, use the inertial model
            return self._handle_inertial_scroll(event)

    def _handle_macos_direct_scroll(self, event):
        """
        Special handling for macOS trackpad events using direct scrolling with acceleration curve.
        This mimics the native macOS scrolling behavior with non-linear acceleration.
        """
        # Get the raw delta value
        delta = event.delta if hasattr(event, 'delta') else 0

        # If no delta, we can't process this event
        if delta == 0:
            return "break"

        # Set scroll direction (negate for natural scrolling direction)
        direction = -1 if delta > 0 else 1

        # Get magnitude of delta (absolute value)
        delta_magnitude = abs(delta)

        # Apply acceleration curve: small deltas stay small, large deltas grow quadratically
        # Formula: adjusted_delta = base_delta * (1 + acceleration_factor * base_delta)
        acceleration_factor = self.params.get('acceleration_factor', 0.01)
        adjusted_delta = delta_magnitude * (1 + acceleration_factor * delta_magnitude)

        # Apply scroll multiplier to adjust overall sensitivity
        adjusted_delta *= self.params.get('scroll_multiplier', 1.0)

        # Apply minimum threshold if specified
        min_delta = self.params.get('min_delta', 0.0)
        if adjusted_delta < min_delta:
            adjusted_delta = 0

        # Apply maximum threshold if specified
        max_delta = self.params.get('max_delta', float('inf'))
        if adjusted_delta > max_delta:
            adjusted_delta = max_delta

        # Apply the scrolling immediately with the adjusted delta
        self._apply_scroll(direction * adjusted_delta)

        # Optionally cancel any pending stop actions and set a new one
        if hasattr(self, 'debounce_id') and self.debounce_id:
            self.widget.after_cancel(self.debounce_id)

        # Set a debounce to detect when scrolling stops
        self.debounce_id = self.widget.after(
            self.params.get('debounce_time', 150),
            self._on_scroll_end
        )

        return "break"

    def _handle_inertial_scroll(self, event):
        """Handle scroll with inertial behavior for Windows/Linux."""
        # Get the scroll direction and amount
        direction, amount = self._get_scroll_info(event)
        if direction is None:
            return "break"

        # Calculate time delta for acceleration
        current_time = event.time if hasattr(event, 'time') \
            else int(self.widget.winfo_toplevel().winfo_pointerx())
        time_delta = current_time - self.last_event_time
        self.last_event_time = current_time

        # If this is a new scroll or the direction changed
        if not self.is_scrolling or (self.velocity * direction < 0):
            # Reset velocity for new scroll direction
            self.velocity = direction * amount * self.params['initial_velocity_factor']
        else:
            # For ongoing scrolls in same direction, accelerate if events are close together
            if time_delta < self.params['acceleration_threshold']:
                self.velocity = self.velocity * self.params['acceleration_factor']

        # Cap at max velocity
        self.velocity = max(min(self.velocity, self.params['max_velocity']), \
                            -self.params['max_velocity'])

        # Start or continue animation
        self.is_scrolling = True
        if self.animation_id is None:
            self._animate_scroll()

        return "break"

    def _on_scroll_end(self):
        """Called when scrolling appears to have stopped."""
        # Reset debounce ID
        self.debounce_id = None

    def _get_scroll_info(self, event):
        """Extract scroll direction and amount from event."""

        direction = None
        amount = 1.0

        # Handle different event types for different platforms
        if hasattr(event, 'delta'):
            # Windows and macOS
            if event.delta != 0:
                direction = -1 if event.delta > 0 else 1
                # On macOS, use raw delta values; on Windows normalize to approx 1.0 per notch
                amount = abs(event.delta) if \
                    sys.platform.startswith('darwin') else abs(event.delta) / 120.0
        elif hasattr(event, 'num'):
            # Linux
            if event.num == 4:  # Scroll up
                direction = -1
            elif event.num == 5:  # Scroll down
                direction = 1

        # Apply platform multiplier
        amount *= self.params['scroll_multiplier']

        return direction, amount

    def _animate_scroll(self):
        """Animate the smooth scrolling effect."""
        if abs(self.velocity) < self.params['min_velocity'] or not self.is_scrolling:
            # Stop animation when velocity gets too small
            self.is_scrolling = False
            self.velocity = 0.0
            self.animation_id = None
            return

        # Apply the scroll
        self._apply_scroll(self.velocity)

        # Decelerate
        self.velocity *= self.params['deceleration_factor']

        # Schedule next animation frame
        self.animation_id = self.widget.after(
            self.params['animation_interval'],
            self._animate_scroll
        )

    def _apply_scroll(self, amount):
        """Apply the actual scroll to the widget."""
        if self.axis == 'y':
            if self.params.get('use_pixels', False):
                # Pixel-based scrolling (smoother on some platforms)
                current_view = self.widget.yview()
                canvas_height = self.widget.winfo_height()
                if canvas_height <= 1:  # Canvas not fully initialized
                    canvas_height = 500  # Fallback value

                # Convert pixels to canvas units
                scroll_region = self.widget.cget('scrollregion')
                if scroll_region:
                    try:
                        _, _, _, canvas_full_height = map(int, scroll_region.split())
                        # Scale amount by line height for more natural scrolling on macOS
                        if self.platform == 'macos':
                            line_height = self.params.get('line_height', 16)
                            amount = amount / line_height
                        scroll_amount = amount / canvas_full_height
                        self.widget.yview_moveto(current_view[0] + scroll_amount)
                    except (ValueError, IndexError):
                        # Fallback to normal units
                        self.widget.yview_scroll(int(amount), "units")
                else:
                    self.widget.yview_scroll(int(amount), "units")
            else:
                # Unit-based scrolling
                self.widget.yview_scroll(int(amount), "units")
        else:
            # Horizontal scrolling (similar logic)
            if self.params.get('use_pixels', False):
                current_view = self.widget.xview()
                canvas_width = self.widget.winfo_width()
                if canvas_width <= 1:  # Canvas not fully initialized
                    canvas_width = 500  # Fallback value

                # Convert pixels to canvas units
                scroll_region = self.widget.cget('scrollregion')
                if scroll_region:
                    try:
                        _, _, canvas_full_width, _ = map(int, scroll_region.split())
                        # Scale amount by line height for more natural scrolling on macOS
                        if self.platform == 'macos':
                            line_height = self.params.get('line_height', 16)
                            amount = amount / line_height
                        scroll_amount = amount / canvas_full_width
                        self.widget.xview_moveto(current_view[0] + scroll_amount)
                    except (ValueError, IndexError):
                        # Fallback to normal units
                        self.widget.xview_scroll(int(amount), "units")
                else:
                    self.widget.xview_scroll(int(amount), "units")
            else:
                # Unit-based scrolling
                self.widget.xview_scroll(int(amount), "units")

    def stop_scrolling(self):
        """Stop any ongoing scrolling animation."""
        self.is_scrolling = False
        if self.animation_id is not None:
            self.widget.after_cancel(self.animation_id)
            self.animation_id = None
        self.velocity = 0.0

        # Cancel any pending debounce
        if hasattr(self, 'debounce_id') and self.debounce_id:
            self.widget.after_cancel(self.debounce_id)
            self.debounce_id = None


class HNSWIndex:
    """
    Manages the HNSW (Hierarchical Navigable Small World) index for efficient 
    nearest neighbor search in high-dimensional spaces.
    """
    def __init__(self, distance_metric='cosine', dim=None):
        self.index = None
        self.distance_metric = distance_metric
        self.dim = dim
        self.id_to_signature = {}
        self.signature_to_id = {}
        self.next_id = 0
        self.vector_cache = {}  # Cache to store vectors for retrieval

    def initialize(self, dim, expected_elements=1000):
        """
        Initialize the HNSW index with the specified dimension.
        Automatically scales parameters based on expected dataset size.
        
        Args:
            dim: Dimension of feature vectors
            expected_elements: Expected number of elements to be indexed
        """

        self.dim = dim

        # Map any distance metric to hnswlib space types
        if self.distance_metric in ['cosine', 'correlation']:
            space = 'cosine'
        else:
            space = 'l2'  # Default to L2 for other metrics

        self.index = hnswlib.Index(space=space, dim=dim)

        # Scale parameters based on dataset size
        # For larger datasets, we need higher M and ef values
        if expected_elements < 1000:
            # Small dataset
            m_param = 16
            ef_construction = 200
            ef_query = 50
        elif expected_elements < 5000:
            # Medium dataset
            m_param = 32
            ef_construction = 400
            ef_query = 100
        else:
            # Large dataset
            m_param = 48
            ef_construction = 600
            ef_query = 150

        # Initialize with scaled parameters
        max_elements = max(expected_elements * 2, 100000)  # Ensure enough capacity

        try:
            self.index.init_index(max_elements=max_elements, \
                                  ef_construction=ef_construction, M=m_param)
            self.index.set_ef(ef_query)
            print(f"HNSW index initialized with parameters: M={m_param}, \
                  ef_construction={ef_construction}, ef_query={ef_query}")
        except Exception as e:
            # Handle initialization errors
            print(f"Error initializing HNSW index: {e}")
            # Try again with higher values as a fallback
            try:
                m_param = 64
                ef_construction = 800
                ef_query = 200
                print(f"Retrying with higher parameters: M={m_param}, \
                      ef_construction={ef_construction}, ef_query={ef_query}")
                self.index.init_index(max_elements=max_elements, \
                                      ef_construction=ef_construction, M=m_param)
                self.index.set_ef(ef_query)
            except Exception as e2:
                print(f"Critical error initializing HNSW index: {e2}")
                raise

    def add_vector(self, signature, vector):
        """Add a vector to the index with its associated signature."""
        try:
            # Ensure vector is flattened and has consistent dtype and memory layout
            if vector is None or not isinstance(vector, np.ndarray):
                print(f"Error: Invalid vector type for {signature}")
                return

            # Create a fresh copy with consistent properties to avoid memory issues
            vector = np.array(vector, dtype=np.float64, order='C').flatten()

            if self.index is None:
                # Initialize if not already done
                if self.dim is None:
                    self.dim = len(vector)

                # Estimate dataset size based on vector dimension and system state
                if hasattr(self, 'expected_elements'):
                    expected_elements = self.expected_elements
                else:
                    # Estimate from context if available
                    expected_elements = 5000  # Default to expect a large dataset

                self.initialize(self.dim, expected_elements)

            # Check if signature is already in the index
            if signature in self.signature_to_id:
                # Skip if already indexed - vectors should be deterministic for same signature
                return

            # Additional validation
            if len(vector) != self.dim:
                print(f"Error: Vector dimension mismatch for {signature}. " \
                      f"Expected {self.dim}, got {len(vector)}")
                return

            # Assign a new ID
            item_id = self.next_id
            self.next_id += 1

            # Reshape for hnswlib (expects 2D array)
            vector_reshaped = vector.reshape(1, -1)

            # Add to index with error handling
            try:
                self.index.add_items(vector_reshaped, np.array([item_id]))

                # Update mappings
                self.id_to_signature[item_id] = signature
                self.signature_to_id[signature] = item_id

                # Store vector in cache for retrieval (make a copy to avoid memory issues)
                self.vector_cache[signature] = vector.copy()
            except Exception as e:
                print(f"Error adding vector to HNSW index: {e}")
                # Try to handle specific errors
                if "ef or M is too small" in str(e):
                    print("Increasing search parameters and retrying...")
                    # Increase ef parameter and retry
                    current_ef = self.index.get_ef()
                    self.index.set_ef(current_ef * 2)
                    try:
                        self.index.add_items(vector_reshaped, np.array([item_id]))

                        # Update mappings if successful
                        self.id_to_signature[item_id] = signature
                        self.signature_to_id[signature] = item_id

                        # Store vector in cache for retrieval
                        self.vector_cache[signature] = vector.copy()
                    except Exception as e2:
                        print(f"Failed to add vector even with increased parameters: {e2}")
                        # Roll back ID increment if vector wasn't added
                        self.next_id -= 1
                else:
                    # Roll back ID increment for other errors
                    self.next_id -= 1
        except Exception as e:
            print(f"Critical error in add_vector: {e}")
            # Roll back ID increment if there was an error
            if hasattr(self, 'next_id'):
                self.next_id -= 1

    def get_nearest_neighbors(self, query_vector, k=10):
        """Get k nearest neighbors to the query vector."""
        try:
            # Validate the query vector
            if query_vector is None or \
                not isinstance(query_vector, np.ndarray) or query_vector.size == 0:

                print("Invalid query vector")
                return []

            # Ensure vector is flattened with consistent properties
            query_vector = np.array(query_vector, dtype=np.float64, order='C').flatten()

            # Ensure k doesn't exceed the number of items in the index
            k = min(k, self.next_id)
            if k == 0:
                return []

            # Reshape for hnswlib (expects 2D array)
            query_vector_reshaped = query_vector.reshape(1, -1)

            # Query the index with error handling
            try:
                # Get both labels (IDs) and distances
                labels, distances = self.index.knn_query(query_vector_reshaped, k=k)

                # Convert IDs to signatures
                neighbors = []
                for i in range(min(k, len(labels[0]))):
                    item_id = labels[0][i]
                    distance = distances[0][i]

                    if item_id in self.id_to_signature:
                        signature = self.id_to_signature[item_id]
                        neighbors.append((signature, distance))

                return neighbors
            except Exception as e:
                print(f"Error in HNSW nearest neighbor search: {e}")
                # Try to handle "ef or M is too small" errors
                if "ef or M is too small" in str(e):
                    print("Increasing search parameters and retrying...")
                    current_ef = self.index.get_ef()
                    new_ef = current_ef * 2
                    print(f"Increasing ef from {current_ef} to {new_ef}")
                    self.index.set_ef(new_ef)

                    # Try again with increased parameters
                    try:
                        labels, distances = self.index.knn_query(query_vector_reshaped, k=k)

                        # Convert IDs to signatures
                        neighbors = []
                        for i in range(min(k, len(labels[0]))):
                            item_id = labels[0][i]
                            distance = distances[0][i]

                            if item_id in self.id_to_signature:
                                signature = self.id_to_signature[item_id]
                                neighbors.append((signature, distance))

                        return neighbors
                    except Exception as e2:
                        print(f"Failed even with increased parameters: {e2}")
                        # Fall back to returning an empty list
                        return []
                else:
                    # For other errors, return empty list
                    return []
        except Exception as e:
            print(f"Critical error in get_nearest_neighbors: {e}")
            return []

    def get_distance(self, sig1, sig2):
        """Get the distance between two signatures in the index."""
        try:
            # Get vectors from cache
            if sig1 in self.vector_cache and sig2 in self.vector_cache:
                vector1 = self.vector_cache[sig1]
                vector2 = self.vector_cache[sig2]

                # Validate vectors
                if vector1 is None or vector2 is None or vector1.size == 0 or vector2.size == 0:
                    return None

                # Ensure consistent format
                vector1 = np.array(vector1, dtype=np.float64, order='C').flatten()
                vector2 = np.array(vector2, dtype=np.float64, order='C').flatten()

                # Calculate distance based on the metric
                if self.distance_metric in ['cosine', 'correlation']:
                    # Calculate cosine distance
                    dot_product = np.dot(vector1, vector2)
                    norm1 = np.linalg.norm(vector1)
                    norm2 = np.linalg.norm(vector2)

                    if norm1 == 0 or norm2 == 0:
                        return 1.0  # Maximum distance for orthogonal vectors

                    similarity = dot_product / (norm1 * norm2)
                    # Clip to [0,1] range to handle floating point errors
                    similarity = max(0, min(1, similarity))
                    distance = 1.0 - similarity
                else:
                    # Calculate Euclidean distance for other metrics
                    distance = np.linalg.norm(vector1 - vector2)

                return distance
        except Exception as e:
            print(f"Error calculating distance in HNSW: {e}")

        return None

    def save_index(self, filepath):
        """
        Save the HNSW index to a file.
        
        Args:
            filepath: Path to save the index
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.index is None:
                print("No index to save")
                return False

            self.index.save_index(filepath)

            # Save metadata alongside the index
            metadata = {
                'dim': self.dim,
                'distance_metric': self.distance_metric,
                'next_id': self.next_id,
                'id_to_signature': self.id_to_signature,
                'signature_to_id': {k: v for k, v in self.signature_to_id.items()}
            }

            # Save metadata to a separate file with the same base name
            metadata_path = f"{filepath}.metadata"
            with open(metadata_path, 'w', encoding="utf-8") as f:
                # Convert dictionary keys to strings for JSON serialization
                serializable_metadata = {
                    'dim': metadata['dim'],
                    'distance_metric': metadata['distance_metric'],
                    'next_id': metadata['next_id'],
                    'id_to_signature': {str(k): v for k, v in metadata['id_to_signature'].items()},
                    'signature_to_id': {k: v for k, v in metadata['signature_to_id'].items()}
                }
                json.dump(serializable_metadata, f)

            return True
        except Exception as e:
            print(f"Error saving HNSW index: {e}")
            return False

    def load_index(self, filepath):
        """
        Load the HNSW index from a file.
        
        Args:
            filepath: Path to the index file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(filepath):
                print(f"Index file not found: {filepath}")
                return False

            # Check if metadata file exists
            metadata_path = f"{filepath}.metadata"
            if not os.path.exists(metadata_path):
                print(f"Metadata file not found: {metadata_path}")
                return False

            # Load metadata
            with open(metadata_path, 'r', encoding="utf-8") as f:
                metadata = json.load(f)

            # Set class attributes from metadata
            self.dim = metadata['dim']
            self.distance_metric = metadata['distance_metric']
            self.next_id = metadata['next_id']

            # Convert id_to_signature keys back to integers
            self.id_to_signature = {int(k): v for k, v in metadata['id_to_signature'].items()}
            self.signature_to_id = metadata['signature_to_id']

            # Map distance metric to space type
            if self.distance_metric in ['cosine', 'correlation']:
                space = 'cosine'
            else:
                space = 'l2'

            # Create and load the index
            self.index = hnswlib.Index(space=space, dim=self.dim)

            # Use a reasonable max_elements value based on the current index size
            max_elements = max(len(self.signature_to_id) * 2, 100000)

            print(f"Loading HNSW index from {filepath} with {len(self.signature_to_id)} elements")
            self.index.load_index(filepath, max_elements=max_elements)

            # Set search parameters to reasonable defaults
            ef_query = min(max(100, len(self.signature_to_id) // 10), 400) # Scale with dataset size
            self.index.set_ef(ef_query)

            print(f"HNSW index loaded successfully with {len(self.signature_to_id)} signatures")
            return True
        except Exception as e:
            print(f"Error loading HNSW index: {e}")
            traceback.print_exc()
            # Reset the index to prevent partial loading issues
            self.index = None
            return False

    def is_indexed(self, signature):
        """Check if a signature is in the index."""
        return signature in self.signature_to_id

    def get_indexed_count(self):
        """Get the number of vectors in the index."""
        return len(self.signature_to_id)

    def set_expected_elements(self, count):
        """
        Set the expected number of elements to help scale parameters appropriately.
        Call this before initialization if you know the dataset size.
        """
        self.expected_elements = count


class SignatureClusteringApp:
    """Main signature clustering application class"""

    def __init__(self, root):
        self.root = root
        self.root.title("Signature Clustering Assistant")
        self.root.geometry("1300x710")
        self.root.protocol("WM_DELETE_WINDOW", self._show_save_before_quit_dialog)

        # These should be adapted to work with your existing code
        self.base_directory = ""  # Directory containing all signatures
        self.output_directory = ""  # Where to save cluster results
        self.save_file = ""

        # Current application state
        self.current_mode = "DISCOVERY"  # DISCOVERY, COMPLETION, VERIFICATION
        self.current_reference = None  # Reference signature for comparison
        self.current_reference_cluster = None  # Cluster ID of reference
        self.current_grid_signatures = []  # Signatures currently in grid
        self.selected_signatures = []  # Currently selected signatures in grid

        # Discovery mode variables
        self.discovery_grid_cols = 6  # Always use 6 columns in discovery mode
        self.discovery_grid_layout = []  # Persistent ordered list of signatures for discovery mode
        self.discovery_grid_needs_fresh_arrangement = True  # Flag to track if we need a refresh

        # Completion mode variables
        self.completion_furthest_page = 1  # Track furthest visited page in completion mode
        self.completion_grid_length = 0   # Track current grid length
        self.completion_grid_signatures_cache = []  # Cache of current grid signatures
        self.completion_reference_changed_manually = False  # Track manual reference changes

        # Tracking clusters and constraints
        self.clusters = {}  # Dict: cluster_id -> list of signature paths
        self.unclustered_signatures = []  # List of signatures not yet clustered
        self.cannot_link_constraints = []  # Keep for backward compatibility
        self.cannot_link_map = {}  # Dict mapping image -> set of images it cannot link with

        # Default to showing only non-rejected
        self.rejection_filter_var = tk.StringVar(value="Non-rejected")

        # Add tracking for complete clusters
        self.complete_clusters = set()  # Set of cluster_ids that have been completed

        # Add tracking for user-selected reference signatures
        self.user_selected_references = set()  # Set of cluster_ids with user-selected references

        # Configuration for display
        self.grid_cols = 5 if self.current_mode == "DISCOVERY" else 4
        self.grid_size = self.grid_cols * 3
        self.thumbnail_size = (180, 120)  # Size of signature thumbnails

        # Progress tracking
        self.total_signatures = 0
        self.clustered_signatures = 0

        self.shown_signatures = set()

        # Add filtering variables
        self.cluster_sizes = []
        self.min_existing_size = 1
        self.max_existing_size = 1

        # Initialize last applied search values with defaults
        self.last_applied_search_text = ""
        self.last_applied_filter_type = "Incomplete"
        self.last_applied_sort_option = "Visual Similarity"

        self.last_applied_grid_membership = "Both"
        self.last_applied_grid_filter = "Incomplete"
        self.last_applied_grid_sort = "Visual Similarity"
        self.last_applied_grid_use_name_query = False
        self.last_applied_grid_name_query = ""
        self.last_applied_rejection_filter = "Non-rejected"

        # Add HNSW index
        self.hnsw_index = None  # Will be initialized when needed

        # Track the last manually selected cell index
        self.last_selected_index = None

        # ======================================================
        # NEW: Pagination state variables
        # ======================================================
        self.signatures_per_page = 120  # Default to 120 signatures per page

        # Current page for each mode
        self.current_page = {
            "DISCOVERY": 1,
            "COMPLETION": 1,
            "VERIFICATION": 1
        }

        # Total pages for each mode
        self.total_pages = {
            "DISCOVERY": 1,
            "COMPLETION": 1,
            "VERIFICATION": 1
        }

        # Cache for full signature lists in each mode
        self.full_signature_lists = {
            "DISCOVERY": [],
            "COMPLETION": [],
            "VERIFICATION": []
        }

        # Flag to track if pagination has been initialized
        self.pagination_initialized = False

        # ======================================================
        # NEW: Lazy loading state variables
        # ======================================================
        self.lazy_discovery_arranged = []  # Incrementally built discovery arrangement
        self.lazy_discovery_last_page = 1  # Last viewed page when leaving discovery mode
        self.lazy_completion_arranged = []  # Incrementally built completion arrangement
        self.lazy_calculation_extended_for_rejected = False  # Extended for rejected signatures
        self.lazy_calculation_complete = False  # Whether discovery arrangement is calculated

        # Completion mode lazy loading state variables
        self.lazy_completion_arranged = []  # Incrementally built completion arrangement
        self.completion_remaining_candidates = []  # Remaining candidates to arrange
        self.completion_calculation_complete = False  # Whether completion arrangement is calculated

        # Configuration for clustering algorithm

        # Start with default configuration
        self.clustering_params = default_config.copy()

        # Update with the first test configuration (your best config)
        if test_configs and len(test_configs) > 0:
            for key, value in test_configs[0].items():
                # Skip the 'name' field
                if key != 'name':
                    self.clustering_params[key] = value

        # Feature caching to improve responsiveness
        self.feature_extractor = None  # Will be initialized after parameters are set
        self.clustering = None  # Will be initialized alongside feature_extractor
        self.features_cache = {}  # Path -> (hu_moments, lbp_hist, hog_feats, zernike_moments, etc.)
        self.combined_vectors_cache = {}  # Path -> combined feature vector

        # Create the UI
        self._create_menu()
        self._create_main_layout()

        # Update mode-specific UI elements
        self._update_mode_specific_ui()

        if self.current_mode == "DISCOVERY":
            self.left_frame.pack_forget()  # Hide reference frame initially in discovery mode

        # Disable UI until project is loaded
        self._set_ui_enabled(False)

        # Add tracking for displayed images (vs. reference images)
        self.current_displayed_signature = None  # Currently displayed signature in ref panel
        self.cluster_displayed_signatures = {}  # Track displayed signatures for each cluster

        # Track displayed signatures in grid
        self.grid_displayed_signatures = {}  # signature_path -> displayed_signature_path

        self.root.after(100, self.initialize_phase1_fixes)

    def initialize_phase1_fixes(self):
        """Initialize all Phase 1 fixes"""
        print("Initializing Phase 1 fixes...")

        # Create mode cache if it doesn't exist
        if not hasattr(self, 'mode_grid_cache'):
            self.mode_grid_cache = {}

        # Initialize smooth scrollers dictionary
        if not hasattr(self, 'smooth_scrollers'):
            self.smooth_scrollers = {}

        # Initialize keyboard handling
        self._initialize_keyboard_handling()

        # Re-bind main frame click event to handle focus
        if hasattr(self, 'main_frame'):
            self.main_frame.bind("<Button-1>", self._take_focus_from_entry, add="+")

        # Add focus management to all major frames
        frames_to_bind = [
            'main_frame', 'top_frame', 'middle_frame', 'left_frame', 
            'right_frame', 'reference_frame', 'scrollable_frame'
        ]

        for frame_name in frames_to_bind:
            if hasattr(self, frame_name):
                frame = getattr(self, frame_name)
                frame.bind("<Button-1>", self._take_focus_from_entry, add="+")

        # Set up initial scrolling with smooth scrolling
        self._setup_mousewheel_scrolling()

        # Force refresh grid to apply fixes
        self.root.after(500, self._force_refresh_current_mode)

        print("Phase 1 fixes initialized")

    def _get_current_scroll_position(self):
        """Get the current scroll position of the main canvas"""
        try:
            if hasattr(self, 'canvas') and self.canvas:
                return self.canvas.canvasy(0)
        except Exception:
            pass
        return 0

    def _restore_scroll_position(self, position):
        """Restore the scroll position of the main canvas"""
        try:
            if hasattr(self, 'canvas') and self.canvas and position is not None:
                # Update the canvas to ensure it has proper scroll region
                self.canvas.update_idletasks()

                # Get the total scrollable height
                scroll_region = self.canvas.cget("scrollregion")
                if scroll_region:
                    # Parse the scrollregion (x1, y1, x2, y2)
                    _, y1, __, y2 = map(float, scroll_region.split())
                    total_height = y2 - y1
                    canvas_height = self.canvas.winfo_height()

                    if total_height > canvas_height:
                        # Calculate the fraction to scroll to
                        fraction = position / total_height
                        # Ensure fraction is within bounds
                        fraction = max(0.0, min(1.0, fraction))
                        self.canvas.yview_moveto(fraction)
        except Exception as e:
            print(f"Error restoring scroll position: {e}")

    def _force_refresh_current_mode(self):
        """Force refresh the current mode to ensure it displays correctly"""
        try:
            current_mode = self.current_mode
            print(f"Forcing refresh of current mode: {current_mode}")

            # Always clear the mode cache for the current mode
            if hasattr(self, 'mode_grid_cache') and current_mode in self.mode_grid_cache:
                self.mode_grid_cache.pop(current_mode, None)

            # Refresh the grid
            self._refresh_grid()

            # Re-establish scrolling after refresh
            self.root.after(100, self._setup_mousewheel_scrolling)
        except Exception as e:
            print(f"Error in _force_refresh_current_mode: {e}")

    def update_clustering_parameters(self, new_params):
        """Update clustering parameters and recalculate as needed"""
        # Update parameters
        for key, value in new_params.items():
            self.clustering_params[key] = value

        # Recreate feature extractor with new parameters
        self.feature_extractor = SignatureFeatureExtractor(self.clustering_params)
        self.clustering = SignatureClustering(self.clustering_params)

        # Clear feature and distance caches to force recalculation
        self.features_cache = {}

        # Clear shown signatures to ensure fresh selection
        self.shown_signatures = set()

        # Refresh grid with new parameters
        self._refresh_grid()

        self.status_var.set("Parameters updated")

    def _create_menu(self):
        """Create the application menu bar"""
        menubar = tk.Menu(self.root)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Signature Directory", command=self._open_directory)
        file_menu.add_command(label="Export Clusters", command=self._export_clusters)
        file_menu.add_separator()
        file_menu.add_command(label="Save", command=lambda: self._save_progress(self.save_file))
        file_menu.add_command(label="Save As...", command=self._save_progress)
        file_menu.add_command(label="Load Progress", command=self._load_progress)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._show_save_before_quit_dialog)
        menubar.add_cascade(label="File", menu=file_menu)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Open Signatures in File Explorer", \
                              command=self._open_signatures_in_file_explorer)
        view_menu.add_separator()
        view_menu.add_command(label="Increase Thumbnail Size", \
                              command=lambda: self._change_thumbnail_size(20))
        view_menu.add_command(label="Decrease Thumbnail Size", \
                              command=lambda: self._change_thumbnail_size(-20))
        menubar.add_cascade(label="View", menu=view_menu)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="How to Use", command=self._show_help)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def _open_signatures_in_file_explorer(self, called_from_menu_option=True):
        """Opens selected signatures in the system's file explorer with multiple file selection"""
        # Collect paths of all selected signatures

        selected_paths = []
        if called_from_menu_option:
            for frame in self.signature_frames:
                if frame.selected and hasattr(frame, 'displayed_signature'):
                    # Use the currently displayed signature, not necessarily the reference
                    if frame.displayed_signature and os.path.exists(frame.displayed_signature):
                        selected_paths.append(frame.displayed_signature)
        else:
            if hasattr(self, 'current_displayed_signature') and self.current_displayed_signature:
                if os.path.exists(self.current_displayed_signature):
                    selected_paths.append(self.current_displayed_signature)

        if not selected_paths:
            messagebox.showinfo("No Signatures Selected", \
                                "Please select at least one signature to open in file explorer.")
            return

        # Group paths by directory to minimize the number of explorer windows
        directories = {}
        for path in selected_paths:
            directory = os.path.dirname(path)
            if directory not in directories:
                directories[directory] = []
            directories[directory].append(path)

        # Confirm if multiple directories will be opened
        num_dirs = len(directories)
        num_files = len(selected_paths)
        if num_dirs > 1:
            confirm = messagebox.askyesno(
                "Multiple Directories",
                f"You are about to open {num_files} signatures across {num_dirs} directories. "
                f"This will open {num_dirs} windows. Are you sure you want to continue?"
            )
            if not confirm:
                return

        # Open file explorer for each directory
        for directory, files in directories.items():
            try:
                if sys.platform.startswith("win32"):
                    # On Windows, use PowerShell to select multiple files

                    # Create a temporary PowerShell script to select multiple files
                    with tempfile.NamedTemporaryFile(suffix='.ps1', \
                                                     delete=False, mode='w') as ps_file:

                        ps_file.write('$shell = New-Object -ComObject shell.application\n')
                        dir_replaced = directory.replace("\\", "\\\\")
                        ps_file.write(f'$folder = $shell.namespace("{dir_replaced}")\n')

                        # First just open the folder
                        ps_file.write('$folder.Self.InvokeVerb("explore")\n')
                        ps_file.write('Start-Sleep -Milliseconds 500\n')  # Give it time to open

                        # Then select all the specified files
                        ps_file.write('$selectedItems = @()\n')
                        for file in files:
                            filename = os.path.basename(file)
                            ps_file.write(f'$item = $folder.parsename("{filename}")\n')
                            ps_file.write('if ($item -ne $null) { $selectedItems += $item }\n')

                        # Select all items at once
                        ps_file.write('if ($selectedItems.Count -gt 0) {\n')
                        ps_file.write('    $folder.GetFolder.SelectItem($selectedItems[0], 1)\n')
                        ps_file.write('    for ($i=1; $i -lt $selectedItems.Count; $i++) {\n')
                        ps_file.write(\
                            '        $folder.GetFolder.SelectItem($selectedItems[$i], 4)\n')
                        ps_file.write('    }\n')
                        ps_file.write('}\n')

                        ps_script = ps_file.name

                    # Execute the PowerShell script with bypassing execution policy
                    subprocess.Popen(['powershell', '-ExecutionPolicy', 'Bypass', '-File',
                                      ps_script], creationflags=subprocess.CREATE_NO_WINDOW)

                    # Schedule the temp file for deletion after a delay
                    self.root.after(5000, lambda file=ps_script: \
                                    os.unlink(file) if os.path.exists(file) else None)

                elif sys.platform.startswith("darwin"):  # macOS
                    # On macOS, use AppleScript to select multiple files without duplicate windows

                    # Create a temporary AppleScript file
                    with tempfile.NamedTemporaryFile(\
                        suffix='.scpt', delete=False, mode='w') as script_file:

                        script_file.write('tell application "Finder"\n')
                        script_file.write('  activate\n')

                        # Create a list of file references
                        script_file.write('  set myFiles to {')
                        file_aliases = []
                        for file in files:
                            file_aliases.append(f'POSIX file "{file}" as alias')
                        script_file.write(', '.join(file_aliases))
                        script_file.write('}\n')

                        # Just select the files - this opens a window with the files highlighted
                        # The key is to NOT use "reveal" which would open a second window
                        script_file.write('  select myFiles\n')
                        script_file.write('end tell\n')

                        applescript = script_file.name

                    # Execute the AppleScript
                    subprocess.Popen(['osascript', applescript])

                    # Schedule the temp file for deletion after a delay
                    self.root.after(5000, lambda file=applescript: \
                                    os.unlink(file) if os.path.exists(file) else None)

                else:  # Linux
                    # Try different file managers with multi-select capabilities

                    # Try nautilus first, which supports the --select option for multiple files
                    try:
                        file_args = []
                        for file in files:
                            file_args.append(shlex.quote(file))

                        # Nautilus: GNOME file manager
                        cmd = f"nautilus --select {' '.join(file_args)}"
                        result = \
                            subprocess.run(cmd, shell=True, stderr=subprocess.PIPE, check=False)

                        # If nautilus fails, try dolphin (KDE)
                        if result.returncode != 0:
                            cmd = f"dolphin --select {' '.join(file_args)}"
                            result = \
                                subprocess.run(cmd, shell=True, stderr=subprocess.PIPE, check=False)

                            # If dolphin fails, just open the directory
                            if result.returncode != 0:
                                subprocess.Popen(['xdg-open', directory])
                    except Exception as e:
                        # Fallback to just opening the directory
                        print(f"Error with file manager: {e}")
                        subprocess.Popen(['xdg-open', directory])

                self.status_var.set(f"Opened directory containing "\
                                    f"{len(files)} selected signature(s)")
            except Exception as e:
                print(f"Error opening directory {directory}: {e}")
                self.status_var.set(f"Error opening directory: {str(e)}")

    def _reset_verification_index(self):
        """Reset the verification index to force showing the first page of signatures."""
        self.verification_index = 0
        print("DEBUG: Verification index reset to 0")

    def _deselect_all(self):
        """Deselect all selected cells in the grid"""
        self.selected_signatures = []
        self.last_selected_index = None  # Reset last selected index

        # Update visual appearance of all frames
        for frame in self.signature_frames:
            if frame.selected:
                frame.selected = False
                frame.config(relief="solid", borderwidth=2)

        self.status_var.set("All cells deselected")

    def _create_main_layout(self):
        """Create the main application layout with cluster selection panel"""
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.main_frame.configure(takefocus=1)  # Make it focusable

        # Bind click handler to main frame to take focus from entry
        self.main_frame.bind("<Button-1>", self._take_focus_from_entry)

        # Top section: Mode selector and stats
        self.top_frame = ttk.Frame(self.main_frame)
        self.top_frame.pack(fill=tk.X, pady=(0, 10))

        # Mode selector
        mode_frame = ttk.LabelFrame(self.top_frame, text="Operating Mode")
        mode_frame.pack(side=tk.LEFT, padx=5)

        self.mode_var = tk.StringVar(value="DISCOVERY")
        ttk.Radiobutton(mode_frame, text="Discovery", variable=self.mode_var,
                        value="DISCOVERY", command=self._change_mode).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(mode_frame, text="Completion", variable=self.mode_var,
                        value="COMPLETION", command=self._change_mode).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(mode_frame, text="Verification", variable=self.mode_var,
                        value="VERIFICATION", command=self._change_mode).pack(side=tk.LEFT, padx=10)

        # Stats frame
        stats_frame = ttk.LabelFrame(self.top_frame, text="Progress")
        stats_frame.pack(side=tk.RIGHT, padx=5)

        ttk.Label(stats_frame, text="Clustered:").grid(row=0, column=0, sticky=tk.W, padx=(0, 1))
        self.clustered_label = ttk.Label(stats_frame, text="0/0 (0%)")
        self.clustered_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 5))

        ttk.Label(stats_frame, text="Complete:").grid(row=0, column=2, sticky=tk.W, padx=(5, 1))
        self.complete_label = ttk.Label(stats_frame, text="0/0 (0%)")
        self.complete_label.grid(row=0, column=3, sticky=tk.W, padx=(0, 0))

        # Middle section: Split into left and right panes
        self.middle_frame = ttk.Frame(self.main_frame)
        self.middle_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Left pane: Reference signature (if applicable)
        self.left_frame = ttk.LabelFrame(self.middle_frame, text="Reference Signature")
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, expand=False, padx=(0, 5))

        self.reference_frame = ttk.Frame(self.left_frame, width=200)
        self.reference_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Placeholder for reference image
        self.reference_canvas = tk.Canvas(self.reference_frame, width=200, height=150, bg="#f0f0f0")
        self.reference_canvas.pack(pady=5)

        self.reference_info = tk.Label(
            self.reference_frame,
            text="No reference selected",
            anchor=tk.CENTER,
            foreground="black",  # Use foreground instead of fg for cross-platform compatibility
            cursor="hand2"  # Hand cursor when hovering over it
        )
        self.reference_info.pack(pady=5)

        self.reference_info.config(font=("TkDefaultFont", 13, "underline"))
        self.reference_info.bind("<Button-1>", lambda _: self.\
                                 _open_signatures_in_file_explorer(called_from_menu_option=False))

        self.reference_cluster_info = ttk.Label(self.reference_frame, text="")
        self.reference_cluster_info.pack(pady=5)

        # Reference controls
        ref_control_frame = ttk.Frame(self.reference_frame)
        ref_control_frame.pack(fill=tk.X, pady=10)

        self.change_ref_btn = ttk.Button(ref_control_frame, text="Change Reference",
                                        command=self._change_reference)
        self.change_ref_btn.pack(side=tk.LEFT, padx=5)

        self.next_cluster_btn = ttk.Button(ref_control_frame, text="Next Cluster",
                                            command=self._next_cluster)
        self.next_cluster_btn.pack(side=tk.LEFT, padx=5)

        # Add a small cluster confidence indicator
        confidence_frame = ttk.LabelFrame(self.reference_frame, text="Cluster Completeness")
        confidence_frame.pack(fill=tk.X, pady=5)  # Reduced padding

        self.confidence_label = ttk.Label(confidence_frame, text="Unknown")
        self.confidence_label.pack(side=tk.TOP, pady=2)  # Reduced padding

        self.confidence_bar = \
            ttk.Progressbar(confidence_frame, orient="horizontal", length=180, mode="determinate")
        self.confidence_bar.pack(pady=2)  # Reduced padding

        # NEW: Add cluster selector frame with FIXED HEIGHT
        cluster_selector_frame = ttk.LabelFrame(self.reference_frame, text="Available Clusters")
        # IMPORTANT: height=150 constrains the cluster selector height
        cluster_selector_frame.pack(fill=tk.X, pady=5)

        # Create a canvas with scrollbar for the cluster list - WITH FIXED HEIGHT
        self.cluster_canvas = tk.Canvas(cluster_selector_frame, width=180, height=150)
        cluster_scrollbar = ttk.Scrollbar(cluster_selector_frame, orient="vertical",
                                          command=self.cluster_canvas.yview)
        self.cluster_scrollable_frame = ttk.Frame(self.cluster_canvas)

        # Configure the canvas
        self.cluster_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.cluster_canvas.configure(scrollregion=self.cluster_canvas.bbox("all"))
        )

        self.cluster_canvas.create_window((0, 0), window=self.cluster_scrollable_frame, anchor="nw")
        self.cluster_canvas.configure(yscrollcommand=cluster_scrollbar.set)

        # Pack the canvas and scrollbar
        self.cluster_canvas.pack(side="left", fill="both", expand=True)
        cluster_scrollbar.pack(side="right", fill="y")

        # Right pane: Scrollable grid of signatures
        self.right_frame = ttk.LabelFrame(self.middle_frame, text="Signature Grid")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Add filter controls for completion mode
        self.filter_frame = ttk.Frame(self.right_frame)
        self.filter_frame.pack(fill=tk.X, padx=10, pady=(5, 0))

        # Initially hide the filter controls (they'll be shown only in completion mode)
        self.filter_frame.pack_forget()

        # Create cluster size filter controls
        self._create_cluster_filter_controls()

        # Grid container with scrollbar
        grid_container = ttk.Frame(self.right_frame)
        grid_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Canvas for scrolling - THIS IS CRITICAL FOR SCROLLING
        self.canvas = tk.Canvas(grid_container)
        self.grid_scrollbar = \
            ttk.Scrollbar(grid_container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        # Configure the scrollable frame to resize with its contents
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        # Create a window in the canvas for the scrollable frame
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.grid_scrollbar.set)

        # Pack the canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.grid_scrollbar.pack(side="right", fill="y")

        # Initialize scrolling capabilities
        self._setup_mousewheel_scrolling()
        self.root.after(100, self._setup_widget_scrolling)  # Delay to ensure widgets are created

        # REMOVED: Static grid creation - we'll create the grid dynamically
        # self.signature_frames will be populated in _update_grid_display
        self.signature_frames = []

        # Bottom section: Action buttons
        self.bottom_frame = ttk.Frame(self.main_frame)
        self.bottom_frame.pack(fill=tk.X, pady=(10, 0), side=tk.BOTTOM)

        # Left side buttons
        self.left_buttons = ttk.Frame(self.bottom_frame)
        self.left_buttons.pack(side=tk.LEFT)

        # CHANGED: Create separate frames for mode-specific buttons
        # We'll show/hide these based on current mode

        # Discovery mode navigation buttons (will be shown/hidden based on mode)
        self.discovery_nav_buttons = ttk.Frame(self.left_buttons)

        # Base refresh button (always visible)
        self.refresh_btn = ttk.Button(self.discovery_nav_buttons, text="Reload Grid (R)",
                                        command=self._refresh_grid)
        self.refresh_btn.pack(side=tk.LEFT, padx=5)

        # Center buttons - ADJUST THE PADX HERE for space between button groups
        self.center_buttons = ttk.Frame(self.bottom_frame)
        self.center_buttons.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=False)

        # Default buttons

        self.new_cluster_btn = ttk.Button(self.center_buttons, text="New Cluster (N)",
                                            command=self._create_new_cluster)
        self.new_cluster_btn.pack(side=tk.LEFT, padx=5)

        self.reject_btn = ttk.Button(self.center_buttons, text="Reject (X)",
                                        command=self._reject_selected)
        self.reject_btn.pack(side=tk.LEFT, padx=5)

        # Right buttons - for deselect all functionality
        self.right_buttons = ttk.Frame(self.bottom_frame)
        self.right_buttons.pack(side=tk.LEFT, padx=0)

        # ======================================================
        # NEW: Pagination controls in the right side of bottom frame
        # ======================================================
        self.pagination_frame = ttk.Frame(self.bottom_frame)
        self.pagination_frame.pack(side=tk.RIGHT, padx=(0, 0))

        # Page size controls
        page_size_frame = ttk.Frame(self.pagination_frame)
        page_size_frame.pack(side=tk.LEFT, padx=(0, 2))

        ttk.Label(page_size_frame, text="Cells/Page:").pack(side=tk.LEFT, padx=(0, 1))

        self.page_size_var = tk.StringVar(value=str(self.signatures_per_page))
        self.page_size_entry = ttk.Entry(page_size_frame, width=3, textvariable=self.page_size_var)
        self.page_size_entry.pack(side=tk.LEFT, padx=(0, 0))

        self.apply_page_size_btn = ttk.Button(page_size_frame, text="Apply", width=4,
                                              command=self._apply_page_size)
        self.apply_page_size_btn.pack(side=tk.LEFT, padx=(0, 5))

        # Page navigation controls
        page_nav_frame = ttk.Frame(self.pagination_frame)
        page_nav_frame.pack(side=tk.LEFT)

        # Page indicator as read-only label
        page_indicator_frame = ttk.Frame(page_nav_frame)
        page_indicator_frame.pack(side=tk.LEFT, padx=(0, 0))

        ttk.Label(page_indicator_frame, text="Page:").pack(side=tk.LEFT, padx=(0, 2))

        self.current_page_var = tk.StringVar(value="1")
        self.current_page_entry = \
            ttk.Entry(page_indicator_frame, width=3, textvariable=self.current_page_var)
        self.current_page_entry.pack(side=tk.LEFT, padx=(0, 0))

        self.total_pages_label = ttk.Label(page_indicator_frame, text="of 1")
        self.total_pages_label.pack(side=tk.LEFT, padx=(0, 0))

        self.go_to_page_btn = ttk.Button(page_nav_frame, text="Go", width=2,
                                 command=self._go_to_custom_page)
        self.go_to_page_btn.pack(side=tk.LEFT, padx=(2, 2))

        self.prev_page_btn = ttk.Button(page_nav_frame, text="←", width=2,
                                        command=self._go_to_prev_page)
        self.prev_page_btn.pack(side=tk.LEFT, padx=(0, 0))

        self.next_page_btn = ttk.Button(page_nav_frame, text="→", width=2,
                                    command=self._go_to_next_page)
        self.next_page_btn.pack(side=tk.LEFT, padx=(0, 0))

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.statusbar = \
            ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Keyboard shortcuts with improved focus checking
        self.root.bind("g", lambda _: self._handle_keyboard_shortcut(_, self._group_selected))
        self.root.bind("n", lambda _: self._handle_keyboard_shortcut(_, self._create_new_cluster))
        self.root.bind("x", lambda _: self._handle_keyboard_shortcut(_, self._reject_selected))
        self.root.bind("r", lambda _: self._handle_keyboard_shortcut(_, self._handle_refresh_grid))
        self.root.bind("<space>", lambda _: self._handle_keyboard_shortcut(
            _, self._toggle_selection_of_focused))

    def _apply_page_size(self):
        """Apply a new page size from the entry field"""
        try:

            # Abandon the page change if the entered value cannot be interpreted as an integer.
            try:
                new_size = int(self.page_size_var.get())
            except ValueError:
                self.page_size_var.set(str(self.signatures_per_page))
                return

            # Abandon the page change if the entered value is less than one.
            if new_size < 1:
                self.page_size_var.set(str(self.signatures_per_page))
                return

            if new_size > 500:
                result = messagebox.askyesno(
                    "Large Page Size", 
                    f"Page size of {new_size} is quite large "
                    "and might affect performance. Continue?",
                    icon='warning'
                )
                if not result:
                    self.page_size_var.set(str(self.signatures_per_page))  # Reset to current
                    return

            # For discovery mode, simply maintain position in the grid
            if self.current_mode == "DISCOVERY":
                old_size = self.signatures_per_page

                # Calculate the index of the first visible signature
                first_visible_idx = (self.current_page["DISCOVERY"] - 1) * old_size

                # Update the page size
                self.signatures_per_page = new_size

                # Calculate the new page that would contain this same index
                new_page = (first_visible_idx // new_size) + 1
                self.current_page["DISCOVERY"] = new_page

                # Need to clear cached grids to force recalculation with new page size
                if hasattr(self, 'discovery_current_grid'):
                    self.discovery_current_grid = []

                # No need to modify discovery_grid_layout - it's the complete list
                # We just need to update pagination

                # Force full refresh with new page size
                self._refresh_grid()

                self.status_var.set(f"Page size changed from {old_size} to {new_size}. "
                                    "Now on page {new_page}.")
                return

            # For other modes, calculate which page will contain the first visible item
            old_size = self.signatures_per_page
            current_page = self.current_page[self.current_mode]

            # Calculate index of first visible item on current page
            first_item_index = (current_page - 1) * old_size

            # Calculate new page that will contain this item
            new_page = (first_item_index // new_size) + 1

            # Update page size and go to calculated page
            self.signatures_per_page = new_size
            self.current_page[self.current_mode] = new_page

            # Update total pages and refresh grid
            self._update_pagination_controls()

            self.selected_signatures = []
            self.last_selected_index = None

            self._refresh_grid()

            self.status_var.set(f"Page size changed from {old_size} to {new_size}. "
                                "Moved to page {new_page}.")

        except ValueError:
            self.status_var.set("Invalid page size. Please enter a number.")
            self.page_size_var.set(str(self.signatures_per_page))  # Reset to current

    def _go_to_custom_page(self):
        """Navigate to the provided page"""
        current_page = self.current_page[self.current_mode]
        total_pages = self.total_pages[self.current_mode]

        try:
            custom_page = int(self.current_page_var.get())
        except ValueError:
            self.current_page_var.set(str(current_page))
            return

        if not 1 <= custom_page <= total_pages:
            self.current_page_var.set(str(current_page))
            return

        self.current_page[self.current_mode] = custom_page

        # Track furthest visited page in completion mode
        if self.current_mode == "COMPLETION":
            self.completion_furthest_page = max(self.completion_furthest_page, custom_page)

        # Clear appropriate caches when changing pages
        if self.current_mode == "DISCOVERY" and hasattr(self, 'discovery_current_grid'):
            self.discovery_current_grid = []

        self._update_pagination_controls()
        self.selected_signatures = []
        self.last_selected_index = None
        self._refresh_grid()

    def _go_to_prev_page(self):
        """Navigate to the previous page"""
        current_page = self.current_page[self.current_mode]

        if current_page > 1:
            self.current_page[self.current_mode] = current_page - 1

            # Track furthest visited page in completion mode
            if self.current_mode == "COMPLETION":
                self.completion_furthest_page = max(self.completion_furthest_page, current_page - 1)

            # Clear appropriate caches when changing pages
            if self.current_mode == "DISCOVERY" and hasattr(self, 'discovery_current_grid'):
                self.discovery_current_grid = []

            self._update_pagination_controls()
            self.selected_signatures = []
            self.last_selected_index = None
            self._refresh_grid()

    def _go_to_next_page(self):
        """Navigate to the next page"""
        current_page = self.current_page[self.current_mode]
        total_pages = self.total_pages[self.current_mode]

        if current_page < total_pages:
            self.current_page[self.current_mode] = current_page + 1

            # Track furthest visited page in completion mode
            if self.current_mode == "COMPLETION":
                self.completion_furthest_page = max(self.completion_furthest_page, current_page + 1)

            # Clear appropriate caches when changing pages
            if self.current_mode == "DISCOVERY" and hasattr(self, 'discovery_current_grid'):
                self.discovery_current_grid = []

            self._update_pagination_controls()
            self.selected_signatures = []
            self.last_selected_index = None
            self._refresh_grid()

    def _update_pagination_controls(self):
        """Update pagination controls based on current state"""
        # Update current page display
        self.current_page_var.set(str(self.current_page[self.current_mode]))

        # Update total pages display
        total_pages = self.total_pages[self.current_mode]
        self.total_pages_label.config(text=f"of {total_pages}")

        # Enable/disable navigation buttons
        current_page = self.current_page[self.current_mode]

        if current_page <= 1:
            self.prev_page_btn.config(state=tk.DISABLED)
        else:
            self.prev_page_btn.config(state=tk.NORMAL)

        if current_page >= total_pages:
            self.next_page_btn.config(state=tk.DISABLED)
        else:
            self.next_page_btn.config(state=tk.NORMAL)

    def _take_focus_from_entry(self, event=None):
        """
        Take focus away from entry widgets and ensure keyboard shortcuts work.
        Also prevents further input to the entries when clicked away.
        """
        # If no event provided, just force focus to main frame
        if event is None:
            self.main_frame.focus_set()
            return

        # Get the widget that was clicked
        clicked_widget = event.widget

        # Get the current widget with focus
        focused_widget = self.root.focus_get()

        # Only change focus if the clicked widget isn't a text entry
        if not any([
            isinstance(clicked_widget, tk.Entry),
            isinstance(clicked_widget, ttk.Entry),
            isinstance(clicked_widget, tk.Text)
        ]):
            # If we have a text entry widget currently focused
            if focused_widget and any([
                isinstance(focused_widget, tk.Entry),
                isinstance(focused_widget, ttk.Entry),
                isinstance(focused_widget, tk.Text)
            ]):
                # Force focus to main_frame which doesn't capture text input
                # This is the key step that prevents further input to the entries
                self.main_frame.focus_set()

        # Don't stop the event from propagating
        return None

    def _on_entry_focus_out(self, event=None):
        """Handle the entry widget losing focus"""
        # Do nothing - we only want to apply changes when buttons are clicked
        pass

    def _handle_keyboard_shortcut(self, _, callback_function):
        """
        Handle keyboard shortcut events, checking if a text widget has focus first
        
        Args:
            event: The keyboard event
            callback_function: The function to call if shortcut should be processed
        
        Returns:
            "break" if event is handled to prevent further processing
        """
        # Get the widget that currently has focus
        focused_widget = self.root.focus_get()

        # Check if the focused widget is an Entry or Text widget or similar
        if focused_widget and any([
            isinstance(focused_widget, tk.Entry),
            isinstance(focused_widget, ttk.Entry),
            isinstance(focused_widget, tk.Text),
            hasattr(focused_widget, 'edit_modified')  # Text widgets have this method
        ]):
            # If a text widget has focus, don't process the shortcut
            return None

        # Otherwise, execute the callback function
        if callback_function:
            callback_function()

        # Return "break" to prevent the event from propagating
        return "break"

    def _fresh_arrangement_discovery(self):
        """
        Perform a complete fresh arrangement of the discovery mode grid.
        Now only calculates first page initially with lazy loading.
        """
        if self.current_mode != "DISCOVERY":
            return

        # Ask the user if they really want to do a fresh arrangement
        confirm_fresh_arrangement = messagebox.askyesno(
            "Confirm Fresh Arrangement", 
            "Are you sure you want to rearrange the discovery mode grid? " \
            "This will reset your current position to page 1.")

        if not confirm_fresh_arrangement:
            return

        self.status_var.set("Preparing fresh arrangement of discovery grid...")
        self.root.update()

        # Reset lazy loading state
        self._reset_lazy_loading_state("DISCOVERY")

        # Reset to page 1
        self.current_page["DISCOVERY"] = 1
        self.lazy_discovery_last_page = 1

        # Clear any cached grid data
        if hasattr(self, 'discovery_current_grid'):
            self.discovery_current_grid = []

        # Clear discovery mode from cache
        if hasattr(self, 'mode_grid_cache') and "DISCOVERY" in self.mode_grid_cache:
            del self.mode_grid_cache["DISCOVERY"]

        # Clear the full signature list to force regeneration
        if "DISCOVERY" in self.full_signature_lists:
            self.full_signature_lists["DISCOVERY"] = []

        # Force full refresh of the grid (will trigger lazy loading)
        self._refresh_grid()

        self.status_var.set("Fresh arrangement ready - navigate to see more pages")

    def _handle_fresh_arrangement(self):
        """Handle 'f' key shortcut for fresh arrangement based on current mode"""
        if self.current_mode == "DISCOVERY":
            self._fresh_arrangement_discovery()

    def _initialize_keyboard_handling(self):
        """Set up robust keyboard event handling"""
        # Ensure main_frame is focusable
        self.main_frame.configure(takefocus=1)

        # Remove any existing bindings to avoid duplicates
        try:
            self.root.unbind("g")
            self.root.unbind("n")
            self.root.unbind("x")
            self.root.unbind("r")
            self.root.unbind("<space>")
            self.root.unbind("f")
            self.root.unbind("d")
        except Exception:
            pass

        # Bind key events with improved handling
        self.root.bind("g", lambda _: self._handle_keyboard_shortcut(_, self._group_selected))
        self.root.bind("n", lambda _: self._handle_keyboard_shortcut(_, self._create_new_cluster))
        self.root.bind("x", lambda _: self._handle_keyboard_shortcut(_, self._reject_selected))
        self.root.bind("r", lambda _: self._handle_keyboard_shortcut(_, self._handle_refresh_grid))
        self.root.bind("<space>", lambda _: self._handle_keyboard_shortcut(
            _, self._toggle_selection_of_focused))
        self.root.bind(
            "f", lambda _: self._handle_keyboard_shortcut(_, self._handle_fresh_arrangement))
        self.root.bind("d", lambda _: self._handle_keyboard_shortcut(_, self._deselect_all))

        # Set initial focus to main_frame
        self.main_frame.focus_set()

    def _create_cluster_filter_controls(self):
        """FIXED VERSION: Create filter controls with proper event bindings"""
        # Create a frame for the filter controls
        filter_frame = ttk.Frame(self.filter_frame)
        filter_frame.pack(side=tk.LEFT, padx=(0, 0))

        # Add Rejection filter dropdown
        rejection_frame = ttk.Frame(self.filter_frame)
        rejection_frame.pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(rejection_frame, text="Rej.:").pack(side=tk.LEFT, padx=(0, 0))
        self.rejection_filter_dropdown = ttk.Combobox(
            rejection_frame,
            textvariable=self.rejection_filter_var,
            values=["Non-rejected", "Rejected", "Both"],
            state="readonly",
            width=7
        )
        self.rejection_filter_dropdown.pack(side=tk.LEFT)

        # Replace "Include Unclustered:" checkbox with "Membership:" dropdown
        ttk.Label(filter_frame, text="Mem.:").pack(side=tk.LEFT, padx=(0, 0))
        self.membership_var = tk.StringVar(value="Both")  # Default to "Both"
        self.membership_dropdown = ttk.Combobox(
            filter_frame,
            textvariable=self.membership_var,
            values=["Unclustered", "Clustered", "Both"],
            state="readonly",
            width=6
        )
        self.membership_dropdown.pack(side=tk.LEFT, padx=(0, 0))
        # Add event handler for membership changes
        self.membership_dropdown.bind("<<ComboboxSelected>>", self._handle_membership_change)

        # Add Completion dropdown (renamed from "filter")
        filter_frame = ttk.Frame(filter_frame)
        filter_frame.pack(side=tk.LEFT, padx=(4, 4))
        ttk.Label(filter_frame, text="Com.:").pack(side=tk.LEFT, padx=(0, 0))
        self.grid_filter_var = tk.StringVar(value="Incomplete")  # Default to "Incomplete"
        self.grid_filter_dropdown = ttk.Combobox(
            filter_frame,
            textvariable=self.grid_filter_var,
            values=["Incomplete", "Complete", "Both"],  # Changed "All" to "Both"
            state="readonly",
            width=6
        )
        self.grid_filter_dropdown.pack(side=tk.LEFT)
        # FIXED: Add event handler for completion filter changes
        self.grid_filter_dropdown.bind("<<ComboboxSelected>>",
                                       self._handle_completion_filter_change)

        # Now add Sort label and dropdown
        sort_frame = ttk.Frame(self.filter_frame)
        sort_frame.pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(sort_frame, text="Sort:").pack(side=tk.LEFT, padx=(0, 0))

        # Dropdown for sort options
        self.sort_completion_var = tk.StringVar(value="Visual Similarity")
        self.sort_completion_dropdown = ttk.Combobox(
            sort_frame,
            textvariable=self.sort_completion_var,
            values=["Visual Similarity", "Query Similarity", "A→Z", "Z→A",
                    "Size (↓)", "Size (↑)" "Path (↓)", "Path (↑)", "Path Similarity"],
            state="readonly",
            width=9
        )
        self.sort_completion_dropdown.pack(side=tk.LEFT)
        # Add event handler for changing sort option (to enable/disable name query checkbox)
        self.sort_completion_dropdown.bind("<<ComboboxSelected>>",
                                           self._update_name_query_checkbox_state)

        # CRITICAL: Ensure all event handlers are properly bound
        self.membership_dropdown.bind("<<ComboboxSelected>>", self._handle_membership_change)
        self.grid_filter_dropdown.bind("<<ComboboxSelected>>",
                                       self._handle_completion_filter_change)
        self.sort_completion_dropdown.bind("<<ComboboxSelected>>",
                                           self._update_name_query_checkbox_state)

        # Also bind to trace changes on the StringVar objects as backup
        self.membership_var.trace_add("write", lambda *_: self._handle_membership_change())
        self.grid_filter_var.trace_add("write", lambda *_: self._handle_completion_filter_change())
        self.sort_completion_var.trace_add(
            "write", lambda *_: self._update_name_query_checkbox_state())

        # Name query frame
        name_frame = ttk.Frame(self.filter_frame)
        name_frame.pack(side=tk.LEFT, padx=(0, 0))

        # Add checkbox for enabling/disabling name query
        self.use_name_query_var = tk.BooleanVar(value=False)  # Unchecked by default
        self.use_name_query_check = ttk.Checkbutton(
            name_frame,
            variable=self.use_name_query_var,
            command=self._update_name_query_entry_state
        )
        self.use_name_query_check.pack(side=tk.LEFT, padx=(0, 0))

        # Add the "Name Query:" label and entry field
        ttk.Label(name_frame, text="Query:").pack(side=tk.LEFT, padx=(0, 0))
        self.name_query_var = tk.StringVar()
        self.name_query_entry = ttk.Entry(name_frame, textvariable=self.name_query_var,
                                        width=15, state="disabled")
        self.name_query_entry.pack(side=tk.LEFT)

        # Add hotkey binding for grid search entry
        self.name_query_entry.bind("<Return>", lambda e: self._update_cluster_filters())

        # Create button frame on far right
        button_frame = ttk.Frame(self.filter_frame)
        button_frame.pack(side=tk.RIGHT)

        # Add magnifying glass button (replaces "Apply" text)
        self.search_grid_btn = ttk.Button(
            button_frame,
            text="🔍",
            width=3,
            command=self._update_cluster_filters
        )
        self.search_grid_btn.pack(side=tk.LEFT, padx=(0, 0))

        # Add "X" button
        self.clear_grid_search_btn = ttk.Button(
            button_frame,
            text="✕",
            width=3,
            command=self._reset_grid_filters
        )
        self.clear_grid_search_btn.pack(side=tk.LEFT)

        # Initialize cluster sizes tracking (keeping this from original code)
        self.cluster_sizes = []
        self.min_existing_size = 1
        self.max_existing_size = 1

    def _reset_grid_filters(self):
        """Reset grid filter controls to default values without applying them"""
        # Reset to default values
        self.membership_var.set("Both")
        self.grid_filter_var.set("Incomplete")
        self.rejection_filter_var.set("Non-rejected")
        self.sort_completion_var.set("Visual Similarity")
        self.use_name_query_var.set(False)

        # Set empty string in the name query field (not the cluster name)
        self.name_query_var.set("")

        # Ensure name query entry is disabled (since checkbox is unchecked)
        if hasattr(self, 'name_query_entry'):
            self.name_query_entry.config(state="disabled")

        # Update states based on new values
        self._handle_membership_change()
        self._update_name_query_checkbox_state()

        # Update status
        self.status_var.set("Grid filter values reset to defaults (not applied)")

    def _handle_membership_change(self):
        """FIXED VERSION: Handle changes to the membership dropdown with proper constraints"""
        membership_option = self.membership_var.get()

        if membership_option == "Unclustered":
            # Enforce settings for Unclustered option
            # 1. Set Completion to "Incomplete" and disable
            self.grid_filter_var.set("Incomplete")
            self.grid_filter_dropdown.config(state="disabled")

            # 2. For Sort dropdown: only allow "Visual Similarity",
            # "Path (↓)", "Path (↑)" and "Path Similarity"
            current_sort = self.sort_completion_var.get()
            # Update dropdown values to only show allowed options
            self.sort_completion_dropdown.config(values=["Visual Similarity", "Path (↓)",
                                                         "Path (↑)", "Path Similarity"])

            # Set sort option: keep "Path (↓)", "Path (↑)" or "Path Similarity"
            # if already selected, otherwise use "Visual Similarity"
            if current_sort == "Path (↓)":
                self.sort_completion_var.set("Path (↓)")
            elif current_sort == "Path (↑)":
                self.sort_completion_var.set("Path (↑)")
            elif current_sort == "Path Similarity":
                self.sort_completion_var.set("Path Similarity")
            else:
                self.sort_completion_var.set("Visual Similarity")

            self.sort_completion_dropdown.config(state="readonly")  # Keep enabled

            # 3. Uncheck Name Query and disable, but preserve the entry text
            self.use_name_query_var.set(False)
            self.use_name_query_check.config(state="disabled")
            self.name_query_entry.config(state="disabled")

        elif membership_option == "Clustered":
            # Re-enable all controls for Clustered option
            self.grid_filter_dropdown.config(state="readonly")

            # Restore all sort options
            self.sort_completion_dropdown.config(values=["Visual Similarity", "Query Similarity",
                                                         "A→Z", "Z→A", "Size (↓)", "Size (↑)",
                                                         "Path (↓)", "Path (↑)", "Path Similarity"])
            self.sort_completion_dropdown.config(state="readonly")

            self.use_name_query_check.config(state="normal")

            # Update name query state based on sort option
            self._update_name_query_checkbox_state()

        else:  # "Both"
            # Check if completion filter is set to "Complete"
            if self.grid_filter_var.get() == "Complete":
                # Force membership to "Clustered" when completion is "Complete"
                self.membership_var.set("Clustered")
                self.membership_dropdown.config(state="disabled")
            else:
                # Re-enable all controls for Both option
                self.membership_dropdown.config(state="readonly")
                self.grid_filter_dropdown.config(state="readonly")

                # Restore all sort options
                self.sort_completion_dropdown.config(
                    values=["Visual Similarity", "Query Similarity", "A→Z", "Z→A",
                            "Size (↓)", "Size (↑)", "Path (↓)", "Path (↑)", "Path Similarity"])
                self.sort_completion_dropdown.config(state="readonly")

                self.use_name_query_check.config(state="normal")

                # Update name query state based on sort option
                self._update_name_query_checkbox_state()

    def _handle_completion_filter_change(self, _=None):
        """NEW METHOD: Handle changes to the completion filter dropdown"""
        completion_option = self.grid_filter_var.get()

        if completion_option == "Complete":
            # Force membership to "Clustered" and disable
            self.membership_var.set("Clustered")
            self.membership_dropdown.config(state="disabled")
        else:
            # Re-enable membership dropdown
            self.membership_dropdown.config(state="readonly")

            # Re-check membership constraints
            self._handle_membership_change()

    def _is_strict_substring_match(self, query, text):
        """
        NEW METHOD: Check if query is a strict case-insensitive substring of text
        
        Args:
            query: The search query
            text: The text to search in
            
        Returns:
            bool: True if query is found as substring in text (case-insensitive)
        """
        if not query or not text:
            return not query  # Empty query matches everything

        return query.lower() in str(text).lower()

    def _update_name_query_entry_state(self):
        """Update the state of the name query entry based on checkbox state"""
        if self.use_name_query_var.get():
            self.name_query_entry.config(state="normal")
            self.name_query_entry.focus_set()
        else:
            self.name_query_entry.config(state="disabled")

    def _update_name_query_checkbox_state(self, _=None):
        """ENHANCED VERSION: Update name query checkbox state
        based on selected sort option and membership"""
        # Skip if we're in "Unclustered" mode - controls should remain disabled
        if hasattr(self, 'membership_var') and self.membership_var.get() == "Unclustered":
            # FORCE unchecked and disabled for Unclustered
            self.use_name_query_var.set(False)
            self.use_name_query_check.config(state="disabled")
            self.name_query_entry.config(state="disabled")
            return

        sort_option = self.sort_completion_var.get()

        if sort_option == "Query Similarity":
            # When Query Similarity is selected, force checkbox to be checked and disabled
            self.use_name_query_var.set(True)
            self.use_name_query_check.config(state="disabled")
            self.name_query_entry.config(state="normal")  # Enable entry
        else:
            # For other sort options, just enable the checkbox
            # Important: We do NOT automatically uncheck it anymore!
            self.use_name_query_check.config(state="normal")

            # Update the entry state based on checkbox value
            if self.use_name_query_var.get():
                self.name_query_entry.config(state="normal")
            else:
                self.name_query_entry.config(state="disabled")

    def _update_cluster_filters(self):
        """Update filter controls and refresh grid with pagination reset"""
        # Get current values from the UI
        membership = self.membership_var.get()
        filter_type = self.grid_filter_var.get()
        sort_option = self.sort_completion_var.get()
        use_name_query = self.use_name_query_var.get()
        name_query = self.name_query_var.get().strip() if use_name_query else ""
        rejection_filter = self.rejection_filter_var.get()

        # Save the current grid search parameters as last applied
        self.last_applied_grid_membership = membership
        self.last_applied_grid_filter = filter_type
        self.last_applied_grid_sort = sort_option
        self.last_applied_grid_use_name_query = use_name_query
        self.last_applied_grid_name_query = name_query
        self.last_applied_rejection_filter = rejection_filter

        # Reset to page 1 when changing search filters
        self.current_page[self.current_mode] = 1

        # FIXED: Reset lazy loading state when filters change
        if self.current_mode == "COMPLETION":
            self._reset_lazy_loading_state("COMPLETION")

        # Clear any cached search results
        if "COMPLETION" in self.full_signature_lists:
            self.full_signature_lists["COMPLETION"] = []

        self.selected_signatures = []
        self.last_selected_index = None

        # Refresh the grid to apply filters
        self._refresh_grid()

        # Update status
        self.status_var.set(f"Applied search filters: Membership: {membership}, " \
                            f"Completion: {filter_type}, Rejection: {rejection_filter}, " \
                            f"Sort: {sort_option}" + \
                            (f", Query: \"{name_query}\"" if use_name_query else ""))

    def _calculate_cluster_sizes(self):
        """
        Calculate and store all existing cluster sizes,
        excluding the reference cluster for more relevant filtering.
        """
        self.cluster_sizes = set()

        # For tracking displays that would actually appear in completion mode
        self.relevant_cluster_sizes = set()

        # Add size for unclustered (treat as size 1)
        if self.unclustered_signatures:
            self.cluster_sizes.add(1)
            self.relevant_cluster_sizes.add(1)

        # Add sizes for all clusters
        for cluster_id, signatures in self.clusters.items():
            size = len(signatures)
            self.cluster_sizes.add(size)

            # Only add to relevant sizes if it's not the reference cluster
            if cluster_id != self.current_reference_cluster:
                self.relevant_cluster_sizes.add(size)

        # Update min/max existing sizes
        if self.cluster_sizes:
            self.min_existing_size = min(self.cluster_sizes)
            self.max_existing_size = max(self.cluster_sizes)

        # Calculate min/max for relevant sizes (those that would appear in completion mode)
        if self.relevant_cluster_sizes:
            self.min_relevant_size = min(self.relevant_cluster_sizes)
            self.max_relevant_size = max(self.relevant_cluster_sizes)
        else:
            # Default to overall min/max if no relevant sizes found
            self.min_relevant_size = self.min_existing_size
            self.max_relevant_size = self.max_existing_size

    def _smooth_scroll_widget(self, event, scroller):
        """
        Handle scroll events using a smooth scroller.
        
        Args:
            event: The mousewheel event
            scroller: The SmoothScroller to use
        
        Returns:
            "break" to prevent event propagation
        """
        # Let the scroller handle the event
        return scroller.handle_scroll_event(event)

    def _setup_mousewheel_scrolling(self):
        """Set up mousewheel scrolling for all scrollable areas with smooth scrolling support"""
        print("Setting up smooth scrolling")

        # Initialize smooth scrollers dictionary if it doesn't exist
        if not hasattr(self, 'smooth_scrollers'):
            self.smooth_scrollers = {}

        # REMOVE all global bindings to prevent conflict
        try:
            self.canvas.unbind_all("<MouseWheel>")
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")
        except Exception:
            pass

        # Helper function to bind scrolling to a widget and all its children
        def bind_recursive(widget, canvas):
            # Create or get smooth scroller for this canvas
            if canvas not in self.smooth_scrollers:
                self.smooth_scrollers[canvas] = SmoothScroller(canvas)

            # Get the smooth scroller
            scroller = self.smooth_scrollers[canvas]

            # Bind directly to the widget
            widget.bind("<MouseWheel>", lambda e: self._smooth_scroll_widget(e, scroller), add="+")
            widget.bind("<Button-4>", lambda e: self._smooth_scroll_widget(e, scroller), add="+")
            widget.bind("<Button-5>", lambda e: self._smooth_scroll_widget(e, scroller), add="+")

            # Bind to all children recursively
            for child in widget.winfo_children():
                bind_recursive(child, canvas)

        # For main grid canvas - bind directly
        if hasattr(self, 'canvas') and self.canvas:
            # Create smooth scroller for main canvas
            self.smooth_scrollers[self.canvas] = SmoothScroller(self.canvas)

            # Bind events
            self.canvas.bind("<MouseWheel>", lambda e: self._smooth_scroll_widget(
                e, self.smooth_scrollers[self.canvas]), add="+")
            self.canvas.bind("<Button-4>", lambda e: self._smooth_scroll_widget(
                e, self.smooth_scrollers[self.canvas]), add="+")
            self.canvas.bind("<Button-5>", lambda e: self._smooth_scroll_widget(
                e, self.smooth_scrollers[self.canvas]), add="+")

            # Bind to scrollable frame and all its children
            if hasattr(self, 'scrollable_frame') and self.scrollable_frame:
                bind_recursive(self.scrollable_frame, self.canvas)

        # For cluster selector canvas (if it exists)
        if hasattr(self, 'cluster_canvas') and self.cluster_canvas:
            # Create smooth scroller for cluster canvas
            self.smooth_scrollers[self.cluster_canvas] = SmoothScroller(self.cluster_canvas)

            # Direct binding
            self.cluster_canvas.bind("<MouseWheel>", lambda e: self._smooth_scroll_widget(
                e, self.smooth_scrollers[self.cluster_canvas]), add="+")
            self.cluster_canvas.bind("<Button-4>", lambda e: self._smooth_scroll_widget(
                e, self.smooth_scrollers[self.cluster_canvas]), add="+")
            self.cluster_canvas.bind("<Button-5>", lambda e: self._smooth_scroll_widget(
                e, self.smooth_scrollers[self.cluster_canvas]), add="+")

            # Bind to scrollable frame and all its children
            if hasattr(self, 'cluster_scrollable_frame') and self.cluster_scrollable_frame:
                bind_recursive(self.cluster_scrollable_frame, self.cluster_canvas)

        print("Smooth scrolling setup complete")

    def _setup_widget_scrolling(self):
        """
        Set up widget-specific scrolling for nested containers with smooth scrolling.
        This approach makes scrolling context-aware - it only scrolls
        the container that the mouse is directly over.
        """
        # Make sure we have the smooth scrollers dictionary
        if not hasattr(self, 'smooth_scrollers'):
            self.smooth_scrollers = {}

        # Configure more precise scroll handling for the cluster selector
        def _bind_to_widget_and_children(widget, canvas):
            # Get or create smooth scroller for this canvas
            if canvas not in self.smooth_scrollers:
                self.smooth_scrollers[canvas] = SmoothScroller(canvas)

            scroller = self.smooth_scrollers[canvas]

            widget.bind("<MouseWheel>", lambda e: self._smooth_scroll_widget(e, scroller))
            widget.bind("<Button-4>", lambda e: self._smooth_scroll_widget(e, scroller))
            widget.bind("<Button-5>", lambda e: self._smooth_scroll_widget(e, scroller))

            for child in widget.winfo_children():
                _bind_to_widget_and_children(child, canvas)

        # Bind to cluster selector widgets if they exist
        if hasattr(self, 'cluster_scrollable_frame') and self.cluster_scrollable_frame:
            if hasattr(self, 'cluster_canvas') and self.cluster_canvas:
                _bind_to_widget_and_children(self.cluster_scrollable_frame, self.cluster_canvas)

        # Bind to signature grid widgets
        if hasattr(self, 'scrollable_frame') and self.scrollable_frame:
            if hasattr(self, 'canvas') and self.canvas:
                _bind_to_widget_and_children(self.scrollable_frame, self.canvas)

        # We also need to bind to the canvas itself
        if hasattr(self, 'cluster_canvas') and self.cluster_canvas:
            if self.cluster_canvas not in self.smooth_scrollers:
                self.smooth_scrollers[self.cluster_canvas] = SmoothScroller(self.cluster_canvas)

            scroller = self.smooth_scrollers[self.cluster_canvas]

            self.cluster_canvas.bind(
                "<MouseWheel>", lambda e: self._smooth_scroll_widget(e, scroller))
            self.cluster_canvas.bind(
                "<Button-4>", lambda e: self._smooth_scroll_widget(e, scroller))
            self.cluster_canvas.bind(
                "<Button-5>", lambda e: self._smooth_scroll_widget(e, scroller))

        if hasattr(self, 'canvas') and self.canvas:
            if self.canvas not in self.smooth_scrollers:
                self.smooth_scrollers[self.canvas] = SmoothScroller(self.canvas)

            scroller = self.smooth_scrollers[self.canvas]

            self.canvas.bind("<MouseWheel>", lambda e: self._smooth_scroll_widget(e, scroller))
            self.canvas.bind("<Button-4>", lambda e: self._smooth_scroll_widget(e, scroller))
            self.canvas.bind("<Button-5>", lambda e: self._smooth_scroll_widget(e, scroller))

    def _scroll_widget(self, event, widget, direction=None):
        """
        Enhanced version that ensures scrolling works reliably.
        
        Args:
            event: The mousewheel event
            widget: The canvas widget to scroll
            direction: Optional manual direction (-1 for up, 1 for down)
        """
        # Determine which canvas we're scrolling
        if widget not in [self.canvas, getattr(self, 'cluster_canvas', None)]:
            return "break"

        if direction is None:
            # Determine direction from the event
            if hasattr(event, 'delta') and event.delta:
                # Windows/macOS
                if event.delta > 0:
                    direction = -1  # Up
                else:
                    direction = 1   # Down
            elif hasattr(event, 'num') and event.num in (4, 5):
                # Linux
                direction = -1 if event.num == 4 else 1
            else:
                # Skip if we can't determine direction
                return "break"

        # Adjust scroll speed
        scroll_amount = 3  # Increased scroll speed

        # Scroll the widget
        widget.yview_scroll(direction * scroll_amount, "units")

        # Prevent event propagation to avoid double-scrolling
        return "break"

    def _populate_cluster_selector(self, search_text="", filter_type="Incomplete",
                                   sort_option="Visual Similarity"):
        """
        Populate the cluster selector with available clusters based on search criteria
        Using Canvas for better text control
        
        Args:
            search_text (str): Optional text to filter clusters by name
            filter_type (str): Completion filter type - "Both", "Complete", or "Incomplete"
            sort_option (str): "Visual Similarity", "Query Similarity",
                               "A→Z", "Z→A", "Size (↓)", "Size (↑)"
        """
        # Store the last displayed (applied) search parameters
        # This is different from the last entered parameters
        self.last_displayed_search_text = search_text
        self.last_displayed_filter_type = filter_type
        self.last_displayed_sort_option = sort_option

        # Clear previous items
        for widget in self.cluster_scrollable_frame.winfo_children():
            widget.destroy()

        # Create a list to hold all filtered clusters for display
        filtered_clusters = []

        # Create a list to hold filtered clusters with similarity scores
        scored_clusters = []

        # Convert search text to lowercase for case-insensitive matching
        search_text_lower = search_text.lower() if search_text else ""

        # First pass: apply completion filter and filter by search text
        for cluster_id, signatures in self.clusters.items():
            # Skip empty clusters
            if not signatures:
                continue

            # Apply completion filter (now using lowercase)
            if filter_type == "Complete" and cluster_id not in self.complete_clusters:
                continue
            elif filter_type == "Incomplete" and cluster_id in self.complete_clusters:
                continue

            # Convert cluster_id to string and lowercase for matching
            cluster_id_str = str(cluster_id).lower()

            # Filter by search text based on sort option
            if search_text_lower:
                if sort_option == "Query Similarity":
                    # For Query Similarity sorting, calculate similarity score
                    similarity_score = self._calculate_name_similarity(
                        search_text_lower, cluster_id_str
                    )

                    # Add to scored list with similarity score
                    scored_clusters.append((cluster_id, signatures, similarity_score))
                else:
                    # For other sort options, only include exact substring matches
                    if search_text_lower in cluster_id_str:
                        # Add to scored list with placeholder similarity (we'll sort differently)
                        scored_clusters.append((cluster_id, signatures, 0.0))
            else:
                # No search text, include all clusters
                scored_clusters.append((cluster_id, signatures, 0.0))

        # Sort based on selected option with enhanced logic
        if sort_option == "Visual Similarity":
            # Sort by visual similarity to current reference
            if self.current_reference:
                scored_with_similarity = []
                for cluster_id, signatures, _ in scored_clusters:
                    if signatures:
                        # Get representative signature
                        rep_sig = None
                        if cluster_id in self.cluster_displayed_signatures:
                            rep_sig = self.cluster_displayed_signatures[cluster_id]
                        else:
                            rep_sig = self._find_cluster_representative(signatures)

                        if rep_sig:
                            distance = self._calculate_distance(self.current_reference, rep_sig)
                            if distance is not None:
                                visual_similarity = self._convert_distance_to_similarity(distance)
                                scored_with_similarity.append(
                                    (cluster_id, signatures, visual_similarity))

                # Sort by visual similarity (highest first)
                scored_with_similarity.sort(key=lambda x: x[2], reverse=True)
                scored_clusters = scored_with_similarity
            # If no reference, fall back to alphabetical
            else:
                scored_clusters.sort(key=lambda x: str(x[0]).lower())
        elif sort_option == "Query Similarity" and search_text:
            # Two-tier sorting: exact matches first, then by similarity
            exact_matches = []
            non_matches = []

            for cluster_id, signatures, similarity_score in scored_clusters:
                cluster_id_str = str(cluster_id).lower()
                if search_text_lower in cluster_id_str:
                    exact_matches.append((cluster_id, signatures, similarity_score))
                else:
                    non_matches.append((cluster_id, signatures, similarity_score))

            # Sort exact matches by similarity score (highest first)
            exact_matches.sort(key=lambda x: x[2], reverse=True)
            # Sort non-matches by similarity score (highest first)
            non_matches.sort(key=lambda x: x[2], reverse=True)

            # Combine: exact matches first, then non-matches
            scored_clusters = exact_matches + non_matches
        elif sort_option == "A→Z":
            # Sort alphabetically (A→Z)
            scored_clusters.sort(key=lambda x: str(x[0]).lower())
        elif sort_option == "Z→A":
            # Sort alphabetically (Z→A)
            scored_clusters.sort(key=lambda x: str(x[0]).lower(), reverse=True)
        elif sort_option == "Size (↓)":
            # Size (↓) means smallest first (ascending)
            scored_clusters.sort(key=lambda x: len(x[1]))
        elif sort_option == "Size (↑)":
            # Size (↑) means largest first (descending)
            scored_clusters.sort(key=lambda x: len(x[1]), reverse=True)
        elif sort_option == "Path (↓)":
            if self.current_reference:
                clusters_with_ref_paths = []
                for cluster_id, signatures, _ in scored_clusters:
                    if signatures:
                        # Get representative signature
                        rep_sig = None
                        if cluster_id in self.cluster_displayed_signatures:
                            rep_sig = self.cluster_displayed_signatures[cluster_id]
                        else:
                            rep_sig = self._find_cluster_representative(signatures)

                        if rep_sig:
                            clusters_with_ref_paths.append((cluster_id, signatures, rep_sig))
                clusters_with_ref_paths.sort(key=lambda x: x[2])
                scored_clusters = clusters_with_ref_paths
            # If no reference, fall back to alphabetical
            else:
                scored_clusters.sort(key=lambda x: str(x[0]).lower())
        elif sort_option == "Path (↑)":
            if self.current_reference:
                clusters_with_ref_paths = []
                for cluster_id, signatures, _ in scored_clusters:
                    if signatures:
                        # Get representative signature
                        rep_sig = None
                        if cluster_id in self.cluster_displayed_signatures:
                            rep_sig = self.cluster_displayed_signatures[cluster_id]
                        else:
                            rep_sig = self._find_cluster_representative(signatures)

                        if rep_sig:
                            clusters_with_ref_paths.append((cluster_id, signatures, rep_sig))
                clusters_with_ref_paths.sort(key=lambda x: x[2], reverse=True)
                scored_clusters = clusters_with_ref_paths
            # If no reference, fall back to alphabetical
            else:
                scored_clusters.sort(key=lambda x: str(x[0]).lower())
        # Convert to filtered list for display
        filtered_clusters = [(cluster_id, signatures) \
                             for cluster_id, signatures, _ in scored_clusters]

        # Get the width of the canvas for layout calculations
        canvas_width = self.cluster_canvas.winfo_width()
        if canvas_width <= 1:
            canvas_width = 361  # Default fallback

        # Width for text area (canvas_width minus thumbnail and padding)
        text_width = canvas_width - 70  # 60px thumbnail + 10px padding

        # Create a frame for each filtered cluster
        for cluster_id, sigs in filtered_clusters:
            # Create a container for this cluster
            cluster_container = ttk.Frame(self.cluster_scrollable_frame)
            cluster_container.pack(fill=tk.X, expand=True, pady=1)

            # Add a border if this is the current cluster
            if cluster_id == self.current_reference_cluster:
                cluster_container.configure(style="Selected.TFrame")

            # Get a representative signature
            if sigs:
                representative_sig = None

                # FIXED: Prioritize user-selected reference signature
                # First check if the user has selected a reference signature for this cluster
                if hasattr(self, 'cluster_displayed_signatures') and \
                    cluster_id in self.cluster_displayed_signatures:

                    user_reference = self.cluster_displayed_signatures[cluster_id]
                    # Make sure the reference is part of our selection
                    if user_reference in sigs:
                        representative_sig = user_reference
                        print(f"Using user-selected reference "\
                              f"signature for cluster {cluster_id} in dialog")
                    elif user_reference in sigs:
                        # Reference is in the full set but not in selection - still use it
                        representative_sig = user_reference
                        print(f"Using user-selected reference from "\
                              f"full set for cluster {cluster_id} in dialog")

                # If no user-selected reference was found or it's
                # not in the selection, fall back to ordered list
                if not representative_sig:
                    # Try to find the most representative signature
                    ordered_sigs = self._get_cluster_signatures_by_similarity(cluster_id)
                    if ordered_sigs:
                        # Find first signature in ordered list that's in our selection
                        for sig in ordered_sigs:
                            if sig in sigs:
                                representative_sig = sig
                                print("Using most representative signature "\
                                      f"for cluster {cluster_id} in dialog")
                                break

                        # Fallback to first ordered signature if none found
                        if not representative_sig and ordered_sigs:
                            representative_sig = ordered_sigs[0]
                            print(f"Falling back to first ordered signature "\
                                  f"for cluster {cluster_id} in dialog")
                    else:
                        # Fallback to first signature in the selection
                        representative_sig = sigs[0] if sigs else None
                        print(f"Falling back to first selection "\
                              f"signature for cluster {cluster_id} in dialog")

                try:
                    # Create a thumbnail
                    img = self.preprocess_for_display(representative_sig)

                    if img is None:
                        # Error in preprocessing - try direct loading as fallback
                        img = Image.open(representative_sig)

                    img.thumbnail((60, 40))  # Original thumbnail size
                    img_tk = ImageTk.PhotoImage(img)

                    # Store reference to prevent garbage collection
                    cluster_container.image_tk = img_tk

                    # Create a canvas for the image
                    canvas = tk.Canvas(cluster_container, width=60, height=40, bg="#f0f0f0")

                    # CHANGE: Center the image in the canvas horizontally and vertically
                    # Calculate center position
                    x_center = 60 // 2
                    y_center = 40 // 2
                    # Calculate image position (centered)
                    x_pos = x_center - img_tk.width() // 2
                    y_pos = y_center - img_tk.height() // 2
                    # Create centered image
                    canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=img_tk)

                    canvas.pack(side=tk.LEFT, padx=2)
                except Exception:
                    # If image loading fails, show an error placeholder
                    canvas = tk.Canvas(cluster_container, width=60, height=40, bg="#f0f0f0")
                    # Center the error text
                    canvas.create_text(30, 20, text="Error", fill="red")
                    canvas.pack(side=tk.LEFT, padx=2)

            # Create a canvas for text with precise control
            text_canvas = tk.Canvas(cluster_container, width=text_width, \
                                    height=40, highlightthickness=0)
            text_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)

            # Get text components
            cluster_name = str(cluster_id)
            # Keep parentheses in the cluster list (only remove in signature count)
            size_text = f"({len(sigs)})"

            # MODIFIED: Add completion checkmark in parentheses if needed
            if cluster_id in self.complete_clusters:
                completion_mark = "(✓)"
                text_canvas.create_text(5, 20, text=completion_mark, anchor=tk.W, tags="text")
                x_offset = 30
            else:
                x_offset = 5  # No checkmark

            # Create size text at the right edge
            size_id = text_canvas.create_text(text_width-5, 20, text=size_text, \
                                              anchor=tk.E, tags="text")

            # Measure size text width
            size_bbox = text_canvas.bbox(size_id)
            if size_bbox:
                size_width = size_bbox[2] - size_bbox[0] + 5
            else:
                size_width = len(size_text) * 8  # Fallback estimate

            # Available width for cluster name
            name_width = text_width - x_offset - size_width

            # Calculate if we need to truncate
            # First create text with full name to measure
            name_id = text_canvas.create_text(x_offset, 20, text=cluster_name, \
                                              anchor=tk.W, tags="measure")
            name_bbox = text_canvas.bbox(name_id)

            if name_bbox and (name_bbox[2] - name_bbox[0]) > name_width:
                # Text is too long, need to truncate
                text_canvas.delete(name_id)  # Remove the measuring text

                # Try different truncation lengths until it fits
                for length in range(len(cluster_name), 3, -1):
                    truncated = cluster_name[:length] + "..."
                    text_canvas.delete("temp")  # Delete any previous test
                    temp_id = text_canvas.create_text(x_offset, 20, text=truncated, \
                                                      anchor=tk.W, tags="temp")
                    temp_bbox = text_canvas.bbox(temp_id)

                    if temp_bbox and (temp_bbox[2] - temp_bbox[0]) <= name_width:
                        # This length fits, use it
                        text_canvas.delete("temp")
                        text_canvas.create_text(x_offset, 20, text=truncated, \
                                                anchor=tk.W, tags="text")
                        break
                else:
                    # If no truncation worked, use minimal text
                    text_canvas.create_text(x_offset, 20, text=cluster_name[:3] + "...", \
                                            anchor=tk.W, tags="text")
            else:
                # Text fits without truncation, keep it
                text_canvas.itemconfig(name_id, tags="text")  # Change tag from "measure" to "text"

            # Make everything clickable
            cluster_container.bind("<Button-1>", lambda _, \
                                   cid=cluster_id: self._select_cluster(cid))
            canvas.bind("<Button-1>", lambda _, cid=cluster_id: self._select_cluster(cid))
            text_canvas.bind("<Button-1>", lambda _, \
                             cid=cluster_id: self._select_cluster(cid))

            # Highlight on hover
            cluster_container.bind("<Enter>", \
                                   lambda _: cluster_container.configure(style="Hover.TFrame"))
            cluster_container.bind("<Leave>", lambda _: cluster_container.configure(\
                style="Selected.TFrame" if \
                    (cluster_id == self.current_reference_cluster) else "TFrame"))

            # Store canvas for later reference
            cluster_container.text_canvas = text_canvas

        # Configure styles for hover and selection
        style = ttk.Style()
        style.configure("Hover.TFrame", background="#e0e0e0")
        style.configure("Selected.TFrame", background="#c0c0ff")

        # Show message if no clusters match the filters
        if not filtered_clusters:
            message_label = ttk.Label(self.cluster_scrollable_frame, \
                                      text="No matching clusters found", \
                                        foreground="gray", anchor=tk.CENTER)
            message_label.pack(fill=tk.X, expand=True, pady=20)

        self.root.after(100, self._setup_widget_scrolling)  # Refresh widget scrolling

    def _select_cluster(self, cluster_id):
        """Handle selection of a cluster from the selector panel with pagination reset"""
        # Skip if already on this cluster
        if cluster_id == self.current_reference_cluster:
            return

        print(f"Selecting cluster: {cluster_id}")

        # Get the signatures for this cluster
        if cluster_id in self.clusters and self.clusters[cluster_id]:

            # Reset shown signatures
            self.shown_signatures = set()

            # Reset verification page
            if hasattr(self, 'verification_page'):
                self.verification_page = 0

            # Reset pagination to page 1 when changing clusters
            self.current_page[self.current_mode] = 1

            # FIXED: Reset lazy loading state when switching clusters
            if self.current_mode == "COMPLETION":
                self._reset_lazy_loading_state("COMPLETION")

            # IMPORTANT: Use the user-selected reference if available
            if cluster_id in self.cluster_displayed_signatures and \
                self.cluster_displayed_signatures[cluster_id] in self.clusters[cluster_id]:

                # Use the user's previously selected reference for this cluster
                self.current_reference = self.cluster_displayed_signatures[cluster_id]
                print(f"Using user-selected reference: {os.path.basename(self.current_reference)}")
            else:
                # No user-selected reference exists, get ordered signatures
                ordered_signatures = self._get_cluster_signatures_by_similarity(cluster_id)

                if ordered_signatures:
                    # Use most representative signature as default
                    self.current_reference = ordered_signatures[0]
                    # Store this as the displayed signature
                    self.cluster_displayed_signatures[cluster_id] = self.current_reference
                    print("Using most representative signature as reference: " \
                        f"{os.path.basename(self.current_reference)}")
                else:
                    # If no ordering was possible, use the first signature
                    self.current_reference = self.clusters[cluster_id][0]
                    self.cluster_displayed_signatures[cluster_id] = self.current_reference
                    print("Using first signature as reference: " \
                        f"{os.path.basename(self.current_reference)}")

            self.current_reference_cluster = cluster_id
            self.current_displayed_signature = self.current_reference

            # Update displays
            self._update_reference_display()

            # Recalculate cluster sizes to update filter controls for the new reference
            if self.current_mode == "COMPLETION":
                # Apply last applied grid search parameters
                if hasattr(self, 'last_applied_grid_membership'):
                    self.membership_var.set(self.last_applied_grid_membership)
                    self.grid_filter_var.set(self.last_applied_grid_filter)
                    self.sort_completion_var.set(self.last_applied_grid_sort)
                    self.use_name_query_var.set(self.last_applied_grid_use_name_query)
                    self.name_query_var.set(self.last_applied_grid_name_query)
                    self.rejection_filter_var.set(self.last_applied_rejection_filter)

                    # Update state handlers
                    self._handle_membership_change()
                    self._update_name_query_entry_state()
                    self._update_name_query_checkbox_state()
                else:
                    self._apply_grid_search_parameters(use_defaults=True)

                self._calculate_cluster_sizes()

            # Clear cached signature lists for the modes
            if self.current_mode in self.full_signature_lists:
                self.full_signature_lists[self.current_mode] = []

            # Pre-extract features for completion mode
            if self.current_mode == "COMPLETION":
                self.status_var.set("Pre-processing reference and candidates...")
                self.root.update()

                # Extract reference features
                if self.current_reference not in self.features_cache:
                    self._extract_features_for_signatures([self.current_reference])

                # Pre-extract features for a batch of candidates
                sample_size = min(500, len(self.unclustered_signatures))
                if sample_size > 0:
                    batch = random.sample(self.unclustered_signatures, sample_size)
                    self._extract_features_for_signatures(batch)

                    # Create initial ranking
                    self._rank_all_candidates_for_reference()

                    # Set fresh reference state
                    self._fresh_reference = True

            # Important: Force refresh grid when changing clusters in verification mode
            if self.current_mode == "VERIFICATION":
                # Force a refresh of the grid
                self._refresh_grid()
            else:
                # For other modes, normal refresh
                self._refresh_grid()

            # When refreshing the cluster selector, use the last applied parameters
            if hasattr(self, 'last_applied_search_text') and hasattr(\
                self, 'last_applied_filter_type') and hasattr(self, 'last_applied_sort_option'):

                self._populate_cluster_selector(
                    self.last_applied_search_text,
                    self.last_applied_filter_type,
                    self.last_applied_sort_option
                )
            else:
                # Fallback to defaults if no applied values exist
                self._populate_cluster_selector("", "Incomplete", "Visual Similarity")

            # Update Next Cluster button state after changing cluster
            self._update_next_cluster_button_state()

            self.status_var.set(f"Switched to cluster {cluster_id}")

    def _handle_refresh_grid(self):
        """
        Handle refresh grid button, with special case for different modes.
        In discovery mode, maintains the current grid layout rather than changing it.
        """
        # Cache the current grid for discovery mode to allow true "refresh" behavior
        if self.current_mode == "DISCOVERY" and self.current_grid_signatures:
            # Store the current grid signatures for refresh
            if not hasattr(self, 'discovery_current_grid'):
                self.discovery_current_grid = []

            self.discovery_current_grid = self.current_grid_signatures.copy()
            print(f"Cached {len(self.discovery_current_grid)} signatures for discovery refresh")

        # For verification mode, set the refresh requested flag
        if self.current_mode == "VERIFICATION":
            self.refresh_requested = True
            print("DEBUG: Refresh requested in verification mode")

        # Call the regular refresh method
        self._refresh_grid()

    def _create_signature_frame(self, row, col):
        """Create a frame for displaying a signature in the grid"""
        index = row * self.grid_cols + col
        frame = ttk.Frame(self.scrollable_frame, borderwidth=2, relief="solid")

        # No longer using grid() - this frame will be packed in the update_grid_display method
        # Don't try to place it here

        # Canvas for the image
        canvas = tk.Canvas(frame, width=self.thumbnail_size[0], \
                           height=self.thumbnail_size[1], bg="#f0f0f0")
        canvas.pack(pady=(5, 0))

        # Label for the filename
        filename_label = ttk.Label(frame, text="No image", font=("TkDefaultFont", 8))
        filename_label.pack(pady=(2, 0))

        # Label for similarity score
        sim_label = ttk.Label(frame, text="", font=("TkDefaultFont", 8, "bold"))
        sim_label.pack(pady=(0, 2))

        # Store references to widgets
        frame.canvas = canvas
        frame.filename_label = filename_label
        frame.sim_label = sim_label
        frame.index = index
        frame.selected = False
        frame.signature_path = None
        frame.image_tk = None  # To prevent garbage collection

        # Add selection capability
        canvas.bind("<Button-1>", lambda e, idx=index: self._toggle_selection(idx, e))
        frame.bind("<Button-1>", lambda e, idx=index: self._toggle_selection(idx, e))
        filename_label.bind("<Button-1>", lambda e, idx=index: self._toggle_selection(idx, e))

        return frame

    def preprocess_for_display(self, img_path):
        """
        Preprocess a signature image for display using the exact same
        bounding box detection logic from the original feature extractor.
        
        Args:
            img_path: Path to the image file
            for_thumbnail: Whether this is for a small thumbnail (affects quality settings)
            
        Returns:
            PIL Image of the preprocessed signature or None on error
        """
        try:
            # Initialize feature extractor if needed
            if self.feature_extractor is None:
                self._initialize_feature_extractor()

            # Use the feature extractor's bounds detection method
            bounds = self.feature_extractor.get_signature_bounds(img_path=img_path)

            # If bounds detection fails, return the original image
            if bounds is None:
                return Image.open(img_path)

            # Unpack bounds
            x, y, w, h = bounds

            # Load the image with PIL for display (preserves original appearance)
            pil_img = Image.open(img_path)

            # Crop using the bounds from feature extractor
            cropped_img = pil_img.crop((x, y, x+w, y+h))

            return cropped_img
        except Exception as e:
            print(f"Error preprocessing image for display: {e}")
            # On error, try to return the original image
            try:
                return Image.open(img_path)
            except Exception:
                # If even that fails, return None
                return None

    def _weight_normalized_features(self, feature_dict):
        """
        Apply appropriate weights to normalized feature groups, matching the approach
        in SignatureClustering.compute_distances().
        
        Args:
            feature_dict: Dictionary with feature names as keys and feature arrays as values
            
        Returns:
            List of weighted feature arrays ready for concatenation
        """
        weighted_features = []

        # Add weighted Hu moments
        if 'hu' in feature_dict and feature_dict['hu'] is not None:
            weighted_features.append(feature_dict['hu'] * self.clustering_params['HU_WEIGHT'])

        # Add weighted LBP features
        if 'lbp' in feature_dict and feature_dict['lbp'] is not None:
            weighted_features.append(feature_dict['lbp'] * self.clustering_params['LBP_WEIGHT'])

        # Add weighted HOG features
        if 'hog' in feature_dict and feature_dict['hog'] is not None:
            weighted_features.append(feature_dict['hog'] * self.clustering_params['HOG_WEIGHT'])

        # Add weighted Zernike moments if available
        if 'zernike' in feature_dict and feature_dict['zernike'] is not None \
            and self.clustering_params.get('USE_ZERNIKE', False):

            weighted_features.append(\
                feature_dict['zernike'] * self.clustering_params['ZERNIKE_WEIGHT'])

        # Add weighted Gabor features if available
        if 'gabor' in feature_dict and feature_dict['gabor'] is not None \
            and self.clustering_params.get('USE_GABOR', False):

            weighted_features.append(\
                feature_dict['gabor'] * self.clustering_params.get('GABOR_WEIGHT', 0.0))

        # Add weighted stroke features if available
        if 'stroke' in feature_dict and feature_dict['stroke'] is not None \
            and self.clustering_params.get('USE_STROKE_FEATURES', False):

            weighted_features.append(feature_dict['stroke'] * \
                                     self.clustering_params.get('STROKE_FEATURE_WEIGHT', 0.0))

        return weighted_features

    def _combine_features(self, feature_tuple):
        """
        Combine features with appropriate normalization and weights,
        aligned with SignatureClustering.compute_distances().
        Now also caches the result for future use.
        
        Args:
            feature_tuple: Tuple of (hu_moments, lbp_hist, hog_feats,
            zernike_moments, gabor_features, stroke_features)
            
        Returns:
            Combined feature vector with normalization and weights applied
        """
        # First, check if this is a signature path rather than a feature tuple
        if isinstance(feature_tuple, str) and feature_tuple in self.combined_vectors_cache:
            # If we have the combined vector already cached, return it directly
            return self.combined_vectors_cache[feature_tuple]

        # If it's a signature path and in the features cache but not combined cache, get features
        if isinstance(feature_tuple, str) and feature_tuple in self.features_cache:
            sig_path = feature_tuple
            feature_tuple = self.features_cache[sig_path]
            # Process and cache the result
            combined = self._combine_features(feature_tuple)  # Recursive call with actual features
            self.combined_vectors_cache[sig_path] = combined
            return combined

        # Normal case - feature_tuple is actual features
        hu_moments, lbp_hist, hog_feats, zernike_moments, \
            gabor_features, stroke_features = feature_tuple

        # Convert tuple to dictionary format for consistency with original script
        feature_dict = {
            'hu': hu_moments.flatten().reshape(1, -1) if hu_moments is not None else None,
            'lbp': lbp_hist.flatten().reshape(1, -1) if lbp_hist is not None else None,
            'hog': hog_feats.flatten().reshape(1, -1) if hog_feats is not None else None,
            'zernike': zernike_moments.flatten().reshape(1, -1) if \
                zernike_moments is not None else None,
            'gabor': gabor_features.flatten().reshape(1, -1) if \
                gabor_features is not None else None,
            'stroke': stroke_features.flatten().reshape(1, -1) if \
                stroke_features is not None else None
        }

        # Apply normalization to each feature group if enabled
        normalize = self.clustering_params.get('NORMALIZE_FEATURES', True)
        normalize_method = self.clustering_params.get('NORMALIZE_METHOD', 'standard')

        if normalize:
            normalized_dict = {}
            for name, features in feature_dict.items():
                if features is not None and features.size > 0:
                    normalized_dict[name] = self.feature_extractor.normalize_feature_group(
                        features, method=normalize_method)
                else:
                    normalized_dict[name] = features
        else:
            normalized_dict = feature_dict

        # Apply weights using the same approach as the original script
        weighted_features = self._weight_normalized_features(normalized_dict)

        # Concatenate all weighted features
        if weighted_features:
            combined = np.hstack([f.flatten() for f in weighted_features])
        else:
            combined = np.array([])

        return combined

    def _build_hnsw_index_from_combined_vectors(self):
        """
        Build the HNSW index directly from combined vectors
        cache without requiring feature extraction.
        """
        try:
            # Use the same distance metric as in configuration
            self.hnsw_index = HNSWIndex(self.clustering_params['DISTANCE_METRIC'])

            # Set expected dataset size to help scale parameters appropriately
            total_signatures = self.total_signatures
            self.hnsw_index.set_expected_elements(total_signatures)

            # Add vectors to the index efficiently
            added_count = 0
            error_count = 0

            # Determine how many vectors we'll process
            total_count = len(self.combined_vectors_cache)
            if total_count == 0:
                print("No vectors to add to HNSW index")
                return

            start_time = datetime.now()

            # First prioritize reference signatures to ensure they're indexed
            for ref_sig in self.cluster_displayed_signatures.values():
                if ref_sig in self.combined_vectors_cache:
                    try:
                        # Get vector and ensure it's properly formatted
                        vector = self.combined_vectors_cache[ref_sig]
                        if vector is None or vector.size == 0:
                            print(f"Warning: Empty vector for "\
                                  f"reference {os.path.basename(ref_sig)}")
                            continue

                        # Ensure proper format before adding
                        vector = np.array(vector, dtype=np.float64, order='C').flatten()
                        self.hnsw_index.add_vector(ref_sig, vector)
                        added_count += 1
                    except Exception as e:
                        print(f"Error adding reference vector to HNSW index: {e}")
                        error_count += 1

            # Then add other signatures in smaller batches to limit memory pressure
            batch_size = 500  # Reduced batch size for better memory management
            signatures_to_add = list(self.combined_vectors_cache.keys())

            prev_percentage = 0
            self.status_var.set("Building HNSW index: 0%")
            self.root.update()

            sig_num = 1
            for i in range(0, len(signatures_to_add), batch_size):
                # Process in small batches
                batch = signatures_to_add[i:i+batch_size]

                for sig in batch:
                    if not self.hnsw_index.is_indexed(sig):  # Skip already indexed signatures
                        try:
                            vector = self.combined_vectors_cache[sig]
                            if vector is None or vector.size == 0:
                                continue

                            # Ensure proper format before adding
                            vector = np.array(vector, dtype=np.float64, order='C').flatten()
                            self.hnsw_index.add_vector(sig, vector)
                            added_count += 1
                        except Exception as e:
                            print(f"Error adding vector to HNSW index: {e}")
                            error_count += 1
                            # Continue processing other vectors instead of failing completely

                    # Update progress
                    cur_percentage = int((sig_num / total_count) * 100)
                    if cur_percentage > prev_percentage:
                        self.status_var.set(f"Building HNSW index: {cur_percentage}%")
                        prev_percentage = cur_percentage
                        self.root.update()

                    sig_num += 1

            time_diff = datetime.now() - start_time

            print(f"\nHNSW index built with {added_count} signatures "\
                  f"in {str(time_diff).split('.', maxsplit=1)[0]}\n")

            # If we had too many errors, warn but continue
            if error_count > total_count * 0.1:  # More than 10% error rate
                print(f"Warning: High error rate ({error_count}/{total_count}) "\
                      "when adding vectors to index")
        except Exception as e:
            print(f"Critical error building HNSW index: {e}")
            # Reset the index to prevent using an inconsistent state
            self.hnsw_index = None
            # Don't raise the exception - continue with a warning instead
            self.status_var.set("Warning: HNSW index build failed, some features may be limited")

    def _calculate_distance(self, sig1, sig2):
        """
        Calculate distance between two signatures using HNSW index or direct calculation.
        Prioritizes using combined vectors cache without relying on feature extraction.
        
        Args:
            sig1: Path to first signature
            sig2: Path to second signature
            
        Returns:
            Distance between signatures, or None if vectors not available
        """
        try:
            # First try to get the distance from HNSW index if it exists
            if hasattr(self, 'hnsw_index') and self.hnsw_index is not None:
                distance = self.hnsw_index.get_distance(sig1, sig2)
                if distance is not None:
                    return distance

            # Use combined vectors directly if available
            if sig1 in self.combined_vectors_cache and sig2 in self.combined_vectors_cache:
                combined1 = self.combined_vectors_cache[sig1]
                combined2 = self.combined_vectors_cache[sig2]

                # Validate vectors
                if combined1 is None or combined2 is None or \
                    combined1.size == 0 or combined2.size == 0:

                    return None

                # Ensure consistent format before calculations
                combined1 = np.array(combined1, dtype=np.float64, order='C').flatten()
                combined2 = np.array(combined2, dtype=np.float64, order='C').flatten()
            else:
                # Fall back to features cache if needed
                combined1 = None
                combined2 = None

                # Check if we have combined vectors cached
                if sig1 in self.combined_vectors_cache:
                    combined1 = self.combined_vectors_cache[sig1]
                elif sig1 in self.features_cache:
                    # Generate and cache combined vector
                    features1 = self.features_cache[sig1]
                    combined1 = self._combine_features(features1)
                    self.combined_vectors_cache[sig1] = combined1
                else:
                    # No vectors available for sig1
                    return None

                if sig2 in self.combined_vectors_cache:
                    combined2 = self.combined_vectors_cache[sig2]
                elif sig2 in self.features_cache:
                    # Generate and cache combined vector
                    features2 = self.features_cache[sig2]
                    combined2 = self._combine_features(features2)
                    self.combined_vectors_cache[sig2] = combined2
                else:
                    # No vectors available for sig2
                    return None

                # Validate vectors again
                if combined1 is None or combined2 is None \
                    or combined1.size == 0 or combined2.size == 0:

                    return None

                # Ensure consistent format
                combined1 = np.array(combined1, dtype=np.float64, order='C').flatten()
                combined2 = np.array(combined2, dtype=np.float64, order='C').flatten()

            # Check if vectors have same dimensions
            if combined1.shape != combined2.shape:
                print(f"Vector dimension mismatch: {combined1.shape} vs {combined2.shape}")
                return None

            # Stack vectors for pairwise distance calculation
            stacked_vectors = np.vstack([combined1, combined2])

            # Get the distance metric from configuration
            metric = self.clustering_params['DISTANCE_METRIC']

            # Calculate distance using scipy's pdist
            distance = pdist(stacked_vectors, metric=metric)[0]  # Get the single value from pdist

            # Add to HNSW index for future queries if it exists
            if hasattr(self, 'hnsw_index') and self.hnsw_index is not None:
                if not self.hnsw_index.is_indexed(sig1):
                    self.hnsw_index.add_vector(sig1, combined1)
                if not self.hnsw_index.is_indexed(sig2):
                    self.hnsw_index.add_vector(sig2, combined2)

            return distance
        except Exception as e:
            print(f"Error calculating distance between {os.path.basename(sig1)} "\
                  f"and {os.path.basename(sig2)}: {e}")
            return None

    def _open_directory(self):
        """Open directory containing signatures with save check if database already loaded"""
        directory = filedialog.askdirectory(title="Select Directory Containing Signatures")
        if directory:
            # Check if we already have a database loaded
            if self._has_loaded_database():
                # Show save dialog first
                self._show_save_before_open_dialog(directory)
            else:
                # No database loaded, proceed directly
                self._load_new_directory(directory)

    def _show_save_before_open_dialog(self, new_directory):
        """Show save before opening new database dialog
        when user opens a directory while one is loaded"""
        # Create modal dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Save Progress")
        dialog.geometry("450x150")
        dialog.transient(self.root)
        dialog.grab_set()  # Make it modal
        dialog.resizable(False, False)

        # Center the dialog on the parent window
        dialog.update_idletasks()
        parent_x = self.root.winfo_x()
        parent_y = self.root.winfo_y()
        parent_width = self.root.winfo_width()
        parent_height = self.root.winfo_height()

        dialog_width = dialog.winfo_width()
        dialog_height = dialog.winfo_height()

        x = parent_x + (parent_width // 2) - (dialog_width // 2)
        y = parent_y + (parent_height // 2) - (dialog_height // 2)
        dialog.geometry(f"+{x}+{y}")

        # Result variable to track user choice
        result = {'action': None}

        # Message
        message_frame = ttk.Frame(dialog, padding="20")
        message_frame.pack(expand=True, fill='both')

        ttk.Label(message_frame, text="Save progress before opening a new database?",
                font=("TkDefaultFont", 12)).pack(pady=(0, 20))

        # Button frame
        button_frame = ttk.Frame(message_frame)
        button_frame.pack(side='bottom')

        def on_cancel():
            result['action'] = 'cancel'
            dialog.destroy()

        def on_dont_save():
            result['action'] = 'dont_save'
            dialog.destroy()

        def on_save():
            result['action'] = 'save'
            dialog.destroy()

        # Buttons (in reverse order so Tab navigation goes Cancel -> Don't Save -> Save)
        save_btn = ttk.Button(button_frame, text="Save", command=on_save)
        save_btn.pack(side='right', padx=(5, 0))

        dont_save_btn = ttk.Button(button_frame, text="Don't Save", command=on_dont_save)
        dont_save_btn.pack(side='right', padx=(5, 0))

        cancel_btn = ttk.Button(button_frame, text="Cancel", command=on_cancel)
        cancel_btn.pack(side='right', padx=(5, 0))

        # Set focus to Cancel button by default
        cancel_btn.focus_set()

        # Handle window close button (X) - should act like Cancel
        dialog.protocol("WM_DELETE_WINDOW", on_cancel)

        # Wait for user to make a choice
        dialog.wait_window()

        # Process the result
        if result['action'] == 'cancel':
            # Do nothing, just return without opening new directory
            return
        elif result['action'] == 'dont_save':
            # Open new directory without saving
            self._load_new_directory(new_directory)
        elif result['action'] == 'save':
            # Try to save, then open new directory if successful
            if self._save_and_quit():
                self._load_new_directory(new_directory)
            else:
                # If save was cancelled or failed, show dialog again
                self._show_save_before_open_dialog(new_directory)

    def _load_new_directory(self, directory):
        """Load a new signature directory (internal method)"""
        self.base_directory = directory
        self._load_signatures()
        # Update status bar
        self.status_var.set(f"Loaded directory: {directory}")

    def _has_loaded_database(self):
        """Check if a signature database is currently loaded"""
        return (hasattr(self, 'total_signatures') and self.total_signatures > 0) or \
            (hasattr(self, 'clusters') and self.clusters) or \
            (hasattr(self, 'unclustered_signatures') and self.unclustered_signatures)

    def _initialize_cluster_references(self):
        """
        Initialize reference signatures for all clusters.
        This ensures each cluster has a proper reference image for discovery mode.
        IMPORTANT: This preserves any user-selected references.
        """
        if not hasattr(self, 'cluster_displayed_signatures'):
            self.cluster_displayed_signatures = {}

        # Initialize ordered signatures cache if needed
        if not hasattr(self, 'cluster_ordered_signatures'):
            self.cluster_ordered_signatures = {}

        # Initialize user-selected references set if needed
        if not hasattr(self, 'user_selected_references'):
            self.user_selected_references = set()

        # Process each cluster to ensure it has a reference signature
        print("Initializing reference signatures for all clusters...")
        for cluster_id, signatures in self.clusters.items():
            if not signatures:
                continue

            # Calculate ordered signatures for this cluster if not already cached
            if cluster_id not in self.cluster_ordered_signatures:
                ordered_signatures = self._get_cluster_signatures_by_similarity(cluster_id)
                if ordered_signatures:
                    self.cluster_ordered_signatures[cluster_id] = ordered_signatures.copy()
                    most_representative = ordered_signatures[0]
                    print(f"Calculated ordered signatures for cluster {cluster_id}")

                    # Only set reference if not already set by user
                    if cluster_id not in self.cluster_displayed_signatures:
                        self.cluster_displayed_signatures[cluster_id] = most_representative
                        print(f"Set initial reference for cluster {cluster_id}: " \
                              f"{os.path.basename(most_representative)}")
                else:
                    # Fallback to first signature if ordering fails
                    fallback = signatures[0]
                    # Only set reference if not already set by user
                    if cluster_id not in self.cluster_displayed_signatures:
                        self.cluster_displayed_signatures[cluster_id] = fallback
                        print(f"Set fallback reference for cluster " \
                              f"{cluster_id}: {os.path.basename(fallback)}")
            else:
                # We have ordered signatures but might not have a reference
                if cluster_id not in self.cluster_displayed_signatures \
                    and self.cluster_ordered_signatures[cluster_id]:

                    most_representative = self.cluster_ordered_signatures[cluster_id][0]
                    self.cluster_displayed_signatures[cluster_id] = most_representative
                    print(f"Set initial reference for cluster " \
                          f"{cluster_id}: {os.path.basename(most_representative)}")

        print(f"Initialized references for {len(self.cluster_displayed_signatures)} clusters")

    def _load_signatures(self):
        """Load signature files from the base directory"""
        # Reset application state
        self.clusters = {}
        self.unclustered_signatures = []
        self.cannot_link_constraints = []

        # Initialize tracking for displayed images
        self.current_displayed_signature = None  # Currently displayed signature in ref panel
        self.cluster_displayed_signatures = {}  # Track displayed signatures for each cluster

        # Initialize tracking for ordered signatures cache
        if not hasattr(self, 'cluster_ordered_signatures'):
            self.cluster_ordered_signatures = {}  # Cache for ordered signatures by cluster
        else:
            self.cluster_ordered_signatures = {}  # Clear any existing cache

        # Show loading indicator
        self.status_var.set("Loading signatures...")
        print(f"Loading signatures from directory: {self.base_directory}")
        self.root.update()

        if not self.base_directory or not os.path.exists(self.base_directory):
            messagebox.showerror("Error", "Please select a valid directory first")
            self.status_var.set("No valid directory selected")
            return

        try:

            # If a subdirectory was found, ask the user if they
            # want subdirectories to be interpreted as clusters.
            use_subdirs_as_clusters = False
            for item in os.listdir(self.base_directory):
                item_path = os.path.join(self.base_directory, item)
                if os.path.isdir(item_path):
                    use_subdirs_as_clusters = messagebox.askyesno(
                        "Subdirectory Interpretation", 
                        "Would you like subdirectories to be interpreted as preexisting clusters?")
                    break

            overall_start_time = datetime.now()

            # Initialize the feature extractor if not already done
            if self.feature_extractor is None:
                self.feature_extractor = SignatureFeatureExtractor(self.clustering_params)
                self.clustering = SignatureClustering(self.clustering_params)

            # Get all signature files
            all_signatures = []
            valid_extensions = self.clustering_params['VALID_FILE_ENDINGS']

            for root, _, files in os.walk(self.base_directory):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in valid_extensions):
                        sig_path = os.path.join(root, file)
                        all_signatures.append(sig_path)

            print(f"Found {len(all_signatures)} total signature files")

            if use_subdirs_as_clusters:
                # Setup cluster data structure based on directory hierarchy
                cluster_names_lower = set()
                for root, _, files in os.walk(self.base_directory):
                    rel_path = os.path.relpath(root, self.base_directory)
                    if rel_path != '.':  # It's in a subdirectory
                        # Sanitize cluster name
                        sanitized_path = self._sanitize_cluster_name(rel_path)

                        # Ensure that there are no cluster name collisions
                        if sanitized_path.lower() in cluster_names_lower:
                            i = 1
                            while f"{sanitized_path.lower()}_{i}" in cluster_names_lower:
                                i += 1
                            sanitized_path = f"{sanitized_path}_{i}"

                        # Treat subdirectory as a cluster
                        self.clusters[sanitized_path] = []
                        cluster_names_lower.add(sanitized_path.lower())

                        # Add signatures from this directory to the cluster
                        for file in files:
                            if any(file.lower().endswith(ext) for ext in valid_extensions):
                                sig_path = os.path.join(root, file)
                                self.clusters[sanitized_path].append(sig_path)

                print(f"Identified {len(self.clusters)} clusters")

                # Determine unclustered signatures (those not in any cluster)
                clustered_set = set()
                for sig_list in self.clusters.values():
                    clustered_set.update(sig_list)

                self.unclustered_signatures = \
                    [sig for sig in all_signatures if sig not in clustered_set]

                print(f"Found {len(self.unclustered_signatures)} unclustered signatures")
            else:
                # Treat all signatures as unclustered
                self.unclustered_signatures = all_signatures
                print(f"All {len(all_signatures)} signatures loaded as unclustered")

            # Count total signatures
            self.clustered_signatures = sum(len(cluster) for cluster in self.clusters.values())
            self.total_signatures = self.clustered_signatures + len(self.unclustered_signatures)

            # Update progress display
            self._update_progress_display()

            # Enable UI
            self._set_ui_enabled(True)

            # Handle mode-specific initialization
            self.current_reference = None
            self.current_reference_cluster = None

            self.features_cache = {}  # Reset feature cache for new dataset

            # IMPORTANT: Ensure grid_cols is set correctly for discovery mode
            if self.current_mode == "DISCOVERY" and hasattr(self, 'discovery_grid_cols'):
                print(f"Setting discovery mode to use {self.discovery_grid_cols} columns")
                self.grid_cols = self.discovery_grid_cols

            self._extract_features_for_signatures(all_signatures)

            self._build_hnsw_index_from_combined_vectors()

            # IMPORTANT: Initialize reference signatures for all clusters
            self._initialize_cluster_references()

            # CRITICAL FIX: Always select and process initial reference cluster
            # This ensures we have proper ordering of the first cluster regardless of starting mode
            first_reference_cluster = None
            first_reference = None

            if self.clusters:
                # Find first non-empty cluster
                for cluster_id, signatures in self.clusters.items():
                    if signatures:

                        # Force calculation of ordered signatures for this cluster
                        ordered = self._get_cluster_signatures_by_similarity(\
                            cluster_id, force_recalculate=True)

                        if ordered:
                            first_reference = ordered[0]
                            first_reference_cluster = cluster_id

                            # Store initial reference even if we don't set it as current yet
                            self.initial_reference = first_reference
                            self.initial_reference_cluster = first_reference_cluster

                            # Cache the ordered signatures
                            self.cluster_ordered_signatures[cluster_id] = ordered.copy()

                            # Store as displayed signature
                            self.cluster_displayed_signatures[cluster_id] = first_reference

                            break

            # After loading signatures and clusters, set initial reference based on current mode
            if self.current_mode in ["COMPLETION", "VERIFICATION"]:
                if first_reference_cluster and first_reference:
                    self.current_reference = first_reference
                    self.current_reference_cluster = first_reference_cluster
                    self.current_displayed_signature = first_reference

                    print(f"Set initial reference to {self.current_reference} " \
                          f"in cluster {self.current_reference_cluster}")

                    # For completion mode, we need to temporarily switch to discovery mode and back
                    # to ensure proper initialization
                    if self.current_mode == "COMPLETION":
                        # Save the current mode
                        saved_mode = self.current_mode

                        # Temporarily switch to discovery mode
                        self.mode_var.set("DISCOVERY")
                        self._change_mode()

                        # Switch back to completion mode
                        self.mode_var.set(saved_mode)
                        self._change_mode()
                    else:
                        # For verification mode, direct refresh is fine
                        # Update the reference display
                        self._update_reference_display()

                        sort_option = self.sort_var.get() if \
                            hasattr(self, 'sort_var') else "Visual Similarity"

                        self._populate_cluster_selector(sort_option=sort_option)

                        # If we're in verification mode, reset pagination
                        if self.current_mode == "VERIFICATION":
                            if hasattr(self, 'verification_page'):
                                self.verification_page = 0
                            else:
                                self.verification_page = 0

                        # Explicitly call refresh grid
                        self._refresh_grid()

            elif self.current_mode == "DISCOVERY":
                # CRITICAL FIX: For initial discovery mode, we need to ensure the grid is refreshed
                # after everything is fully loaded and initialized
                print("Initial discovery mode: Ensuring grid is properly loaded")
                # Force a new calculation (ignore any cached data)
                if hasattr(self, 'discovery_current_grid'):
                    self.discovery_current_grid = []
                # Clear mode grid cache
                if hasattr(self, 'mode_grid_cache') and "DISCOVERY" in self.mode_grid_cache:
                    del self.mode_grid_cache["DISCOVERY"]
                # Now refresh the grid
                self._refresh_grid()

            # Update UI based on current mode
            self._update_mode_specific_ui()

            # Update button state and keyboard shortcuts
            self._update_keyboard_shortcuts()

            # Update status message based on user's choice
            if use_subdirs_as_clusters:
                self.status_var.set(f"Loaded {self.total_signatures} signatures, "
                                f"{len(self.clusters)} clusters from subdirectories")
            else:
                self.status_var.set(f"Loaded {self.total_signatures} signatures as unclustered")

            overall_time_diff = datetime.now() - overall_start_time
            overall_time_diff_truncated = str(overall_time_diff).split('.', maxsplit=1)[0]
            print(f"\nTotal time to load signature database: {overall_time_diff_truncated}\n")

            self.status_var.set(f"Signature database loaded in " \
                                f"{overall_time_diff_truncated} from {self.base_directory}")

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to load signatures: {str(e)}")
            self.status_var.set("Error loading signatures")

    def _apply_grid_search_parameters(self, save_current=False, use_defaults=False):
        """
        Apply grid search parameters based on flags.
        
        Args:
            save_current: If True, save current UI values as last applied
            use_defaults: If True, reset UI to default values;
                          if False, set UI to last applied values
        """
        # If we should save current UI values
        if save_current:
            self.last_applied_grid_membership = self.membership_var.get() if \
                hasattr(self, 'membership_var') else "Both"
            self.last_applied_grid_filter = self.grid_filter_var.get() if \
                hasattr(self, 'grid_filter_var') else "Incomplete"
            self.last_applied_grid_sort = self.sort_completion_var.get() if \
                hasattr(self, 'sort_completion_var') else "Visual Similarity"
            self.last_applied_grid_use_name_query = self.use_name_query_var.get() if \
                hasattr(self, 'use_name_query_var') else False
            self.last_applied_grid_name_query = self.name_query_var.get() if \
                hasattr(self, 'name_query_var') else ""
            # NEW: Save rejection filter state
            self.last_applied_rejection_filter = self.rejection_filter_var.get() if \
                hasattr(self, 'rejection_filter_var') else "Non-rejected"

        # If UI widgets exist, update them
        if hasattr(self, 'membership_var') and hasattr(self, 'grid_filter_var') and \
            hasattr(self, 'sort_completion_var') and hasattr(self, 'use_name_query_var') and \
                hasattr(self, 'name_query_var') and hasattr(self, 'rejection_filter_var'):

            if use_defaults:
                # Use default values
                self.membership_var.set("Both")
                self.grid_filter_var.set("Incomplete")
                self.sort_completion_var.set("Visual Similarity")
                self.use_name_query_var.set(False)
                # NEW: Reset rejection filter to default
                self.rejection_filter_var.set("Non-rejected")
                # NEW: Also reset the last applied value to ensure actual filtering is updated
                self.last_applied_rejection_filter = "Non-rejected"

                # Leave name query as is - don't reset to cluster name
                # This preserves the last entered value

                # Update states for membership
                self._handle_membership_change()

                # Update name query entry state
                if hasattr(self, 'name_query_entry'):
                    self.name_query_entry.config(state="disabled")

                # Ensure Name Query checkbox state is correct based on sort option
                self._update_name_query_checkbox_state()
            else:
                # Use last applied values
                # Set these values directly to avoid triggering events
                self.membership_var.set(self.last_applied_grid_membership)
                self.grid_filter_var.set(self.last_applied_grid_filter)
                self.sort_completion_var.set(self.last_applied_grid_sort)
                self.use_name_query_var.set(self.last_applied_grid_use_name_query)
                self.name_query_var.set(self.last_applied_grid_name_query)
                # Use last applied value for rejection filter
                self.rejection_filter_var.set(self.last_applied_rejection_filter)

                # Now handle the state updates
                self._handle_membership_change()

                # Update name query entry state
                self._update_name_query_entry_state()

                # Ensure Name Query checkbox state is correct based on sort option
                self._update_name_query_checkbox_state()

    def _refresh_grid(self, force_reset_filters=False):
        """Rebuild the grid with available signatures, now with pagination support"""
        try:
            # Only reset to page 1 when explicitly resetting filters, not for normal navigation
            if force_reset_filters and self.current_mode == "COMPLETION":
                self.current_page["COMPLETION"] = 1
                self._reset_completion_grid_state()
            # For completion mode grid, use the last applied parameters
            if self.current_mode == "COMPLETION" and not force_reset_filters:
                if hasattr(self, 'last_applied_grid_membership'):
                    # Use last applied values
                    self.membership_var.set(self.last_applied_grid_membership)
                    self.grid_filter_var.set(self.last_applied_grid_filter)
                    self.sort_completion_var.set(self.last_applied_grid_sort)
                    self.use_name_query_var.set(self.last_applied_grid_use_name_query)
                    self.name_query_var.set(self.last_applied_grid_name_query)
                    self.rejection_filter_var.set(self.last_applied_rejection_filter)

                    # Update control states
                    self._handle_membership_change()
                    self._update_name_query_entry_state()
                    self._update_name_query_checkbox_state()

            # NEW: Force reset filters if requested
            if force_reset_filters and self.current_mode == "COMPLETION":
                self._apply_grid_search_parameters(use_defaults=True)

            self.status_var.set("Building scrollable grid...")
            self.root.update()

            # Clear current selection
            self.selected_signatures = []

            print(f"Refreshing grid in {self.current_mode} mode")

            # Discovery mode - use our new methods
            if self.current_mode == "DISCOVERY":
                # MODIFIED: Always get fresh signatures for discovery mode with proper pagination
                print("Getting fresh discovery mode signatures with pagination")
                try:
                    self.current_grid_signatures = self._select_discovery_signatures()
                    # Don't cache in discovery_current_grid anymore as it interferes with pagination
                except Exception as e:
                    print(f"ERROR selecting discovery signatures: {e}")
                    traceback.print_exc()
                    self.current_grid_signatures = []  # Set to empty list on error
            elif self.current_mode == "COMPLETION":
                # For completion mode, always get fresh signatures
                self.current_grid_signatures = self._select_completion_signatures()
            elif self.current_mode == "VERIFICATION":
                # For verification mode, always get fresh signatures for current cluster
                self.current_grid_signatures = self._select_verification_signatures()

            # Verify we have signatures to display
            if not self.current_grid_signatures:
                print(f"WARNING: No signatures to display in {self.current_mode} mode!")

                # Create empty grid
                for widget in self.scrollable_frame.winfo_children():
                    widget.destroy()

                # Add a message in the grid
                message_frame = ttk.Frame(self.scrollable_frame)
                message_frame.pack(fill=tk.BOTH, expand=True, pady=50)

                ttk.Label(
                    message_frame,
                    text=f"No signatures to display in {self.current_mode} mode",
                    font=("TkDefaultFont", 12)
                ).pack(pady=20)

                # Update pagination controls
                self._update_pagination_controls()

                return

            # Add debugging output to track grid_cols
            print(f"Grid columns before refresh: {self.grid_cols}")
            if self.current_mode == "DISCOVERY" and self.grid_cols != self.discovery_grid_cols:
                print(f"Correcting discovery mode columns from "\
                      f"{self.grid_cols} to {self.discovery_grid_cols}")
                self.grid_cols = self.discovery_grid_cols

            # Update the grid with the selected signatures
            self._update_grid_display()

            # Update reference display if needed
            if self.current_mode == "VERIFICATION":
                self._update_reference_display()

            # Refresh scrolling
            self._setup_mousewheel_scrolling()

            # Update pagination controls
            self._update_pagination_controls()

            sig_count = len(self.current_grid_signatures) if \
                hasattr(self, 'current_grid_signatures') else 0
            page_info = f" (Page {self.current_page[self.current_mode]} "\
                f"of {self.total_pages[self.current_mode]})"
            self.status_var.set(f"Grid built with {sig_count} signatures{page_info}")

        except Exception as e:
            traceback.print_exc()
            # Use a simpler error dialog to avoid nested errors
            self.status_var.set(f"Grid build error: {str(e)}")
            print(f"Failed to build grid: {str(e)}")

    def _get_representative_signatures(self, limit=None):
        """
        Get representative signatures from each cluster.
        Args:
            limit: Optional limit on total representatives to return 
        Returns:
            List of (signature_path, cluster_id) tuples
        """
        representatives = []

        # For each cluster, find the most representative signature
        for cluster_id, signatures in self.clusters.items():
            # Skip empty clusters
            if not signatures:
                continue

            # If feature extractor is available, use it to find the most representative signature
            if hasattr(self, 'feature_extractor') and self.features_cache:
                best_sig = None
                best_avg_distance = float('inf')

                # For each signature, calculate average distance to all others in the cluster
                for sig1 in signatures:
                    if sig1 not in self.features_cache:
                        continue

                    total_distance = 0.0
                    count = 0

                    for sig2 in signatures:
                        if sig1 != sig2 and sig2 in self.features_cache:
                            dist = self._calculate_distance(sig1, sig2)
                            if dist is not None:
                                total_distance += dist
                                count += 1

                    if count > 0:
                        avg_distance = total_distance / count
                        if avg_distance < best_avg_distance:
                            best_avg_distance = avg_distance
                            best_sig = sig1

                # If we found a representative, add it
                if best_sig:
                    representatives.append((best_sig, cluster_id))
                else:
                    # Fallback: use the first signature
                    representatives.append((signatures[0], cluster_id))
            else:
                # No feature cache available, just use the first signature
                representatives.append((signatures[0], cluster_id))

        # Shuffle to provide variety
        random.shuffle(representatives)

        # Limit if requested
        if limit and len(representatives) > limit:
            representatives = representatives[:limit]

        return representatives

    def _arrange_elements_by_similarity_hnsw_with_constraints(self, elements):
        """
        Arrange elements by similarity using HNSW for nearest neighbor search.
        Uses batched processing to handle large datasets, and respects cannot-link constraints.
        Now uses combined vectors cache for improved performance.
        
        Args:
            elements: List of signature paths to arrange
            
        Returns:
            List of signature paths arranged by similarity with constraint consideration
        """

        if not elements:
            return []

        if len(elements) == 1:
            return elements

        # Ensure features are extracted for all elements
        missing_elements = [elem for elem in elements if elem not in self.features_cache]
        if missing_elements:
            self._extract_features_for_signatures(missing_elements)

        # Get valid elements (those with features)
        valid_elements = [elem for elem in elements if elem in self.features_cache]

        # If too few valid elements, return what we have
        if len(valid_elements) <= 1:
            return valid_elements

        # Start with a random element
        arranged_elements = [random.choice(valid_elements)]
        remaining = set(valid_elements) - set(arranged_elements)

        # Process in batches for large datasets to avoid "ef or M is too small" errors
        batch_size = 1000  # Maximum batch size for knn_query

        # Set of elements that have already been attempted for placement
        attempted_elements = set()

        prev_percentage = 0
        self.status_var.set("Arranging elements using HNSW: 0%")
        self.root.update()

        num_elements_to_arrange = len(remaining)

        start_time = datetime.now()

        # Greedily add nearest neighbors
        while remaining:
            last_element = arranged_elements[-1]

            # Skip if this element was already attempted
            if last_element in attempted_elements:
                # If we've attempted all elements, just add a random one
                random_elem = random.choice(list(remaining))
                arranged_elements.append(random_elem)
                remaining.remove(random_elem)
                continue

            # Add to attempted elements
            attempted_elements.add(last_element)

            # Get combined vector for the last element
            last_vector = None
            if last_element in self.combined_vectors_cache:
                last_vector = self.combined_vectors_cache[last_element]
            elif last_element in self.features_cache:
                last_features = self.features_cache[last_element]
                last_vector = self._combine_features(last_features)
                self.combined_vectors_cache[last_element] = last_vector
            else:
                # No features available, skip to next element
                continue

            next_element = None

            # Determine batch size based on remaining elements
            current_batch_size = min(batch_size, len(remaining))

            try:
                # Find nearest neighbors among remaining elements
                nearest = self.hnsw_index.get_nearest_neighbors(last_vector, k=current_batch_size)

                # Filter nearest neighbors to respect cannot-link constraints
                valid_neighbors = []
                for sig, dist in nearest:
                    # Skip if not in remaining elements
                    if sig not in remaining:
                        continue

                    # Check if this signature has a cannot-link constraint with the last element
                    has_constraint = False

                    # First check direct cannot-link constraint
                    if self._has_cannot_link_constraint(last_element, sig):
                        has_constraint = True
                    else:
                        # Check if it belongs to a cluster that has
                        # a constraint with last_element's cluster
                        sig_cluster = None
                        last_element_cluster = None

                        # Find clusters for both elements
                        for cluster_id, cluster_sigs in self.clusters.items():
                            if sig in cluster_sigs:
                                sig_cluster = cluster_id
                            if last_element in cluster_sigs:
                                last_element_cluster = cluster_id

                        # Check if clusters have constraints
                        if sig_cluster and last_element_cluster and \
                            sig_cluster != last_element_cluster:

                            if self._is_cluster_rejected(sig_cluster, last_element_cluster):
                                has_constraint = True

                    # Only add if no constraint exists
                    if not has_constraint:
                        valid_neighbors.append((sig, dist))

                # Find first valid remaining element
                if valid_neighbors:
                    next_element = valid_neighbors[0][0]

            except Exception as e:
                print(f"Error in nearest neighbor search: {e}")
                # Fallback to direct calculation for this iteration
                next_element = None

            # If no valid nearest neighbor was found, try a direct approach
            if next_element is None:
                print("Using direct distance calculation as fallback")
                # Take a subset of remaining elements to avoid O(n²) complexity
                subset_size = min(50, len(remaining))
                subset = random.sample(list(remaining), subset_size)

                # Find nearest valid neighbor in the subset
                valid_candidates = []
                for elem in subset:
                    # Skip elements with cannot-link constraints
                    if self._has_cannot_link_constraint(last_element, elem):
                        continue

                    # Check cluster constraints
                    elem_cluster = None
                    last_element_cluster = None

                    # Find clusters for both elements
                    for cluster_id, cluster_sigs in self.clusters.items():
                        if elem in cluster_sigs:
                            elem_cluster = cluster_id
                        if last_element in cluster_sigs:
                            last_element_cluster = cluster_id

                    # Check if clusters have constraints
                    if elem_cluster and last_element_cluster and \
                        elem_cluster != last_element_cluster:

                        if self._is_cluster_rejected(elem_cluster, last_element_cluster):
                            continue

                    # Calculate distance for valid candidates using our improved method
                    distance = self._calculate_distance(last_element, elem)
                    if distance is not None:
                        valid_candidates.append((elem, distance))

                # Sort by distance and get the nearest
                if valid_candidates:
                    valid_candidates.sort(key=lambda x: x[1])
                    next_element = valid_candidates[0][0]

            # If we found a neighbor, add it
            if next_element is not None:
                arranged_elements.append(next_element)
                remaining.remove(next_element)
            else:
                # If no valid neighbor found, add a random one
                if remaining:
                    random_elem = random.choice(list(remaining))
                    arranged_elements.append(random_elem)
                    remaining.remove(random_elem)

            cur_percentage = int(((num_elements_to_arrange - len(remaining)) / \
                                  num_elements_to_arrange) * 100)
            if cur_percentage > prev_percentage:
                self.status_var.set(f"Arranging elements using HNSW: {cur_percentage}%")
                prev_percentage = cur_percentage
                self.root.update()

        # Add any elements that didn't have valid features at the end
        invalid_elements = [elem for elem in elements if elem not in valid_elements]
        arranged_elements.extend(invalid_elements)

        time_diff = datetime.now() - start_time
        print(f"\nArranged elements using HNSW in {str(time_diff).split('.', maxsplit=1)[0]}\n")

        return arranged_elements

    def _ensure_discovery_mode_isolation(self):
        """Ensure discovery mode state is isolated from completion mode changes"""
        if self.current_mode == "DISCOVERY":
            # Reset any completion mode specific state that might interfere
            if hasattr(self, 'completion_grid_signatures_cache'):
                self.completion_grid_signatures_cache = []
            if hasattr(self, 'completion_grid_length'):
                self.completion_grid_length = 0

    def _ensure_discovery_grid_layout(self):
        """Ensure the discovery grid layout exists and is properly initialized"""
        if not hasattr(self, 'discovery_grid_layout') or not self.discovery_grid_layout:
            print("Initializing discovery grid layout")
            self.discovery_grid_layout = []

            # Add cluster representatives
            for cluster_id, signatures in self.clusters.items():
                # Skip complete clusters
                if cluster_id in self.complete_clusters:
                    continue

                # Get the reference signature for the cluster
                if cluster_id in self.cluster_displayed_signatures and \
                    self.cluster_displayed_signatures[cluster_id] in signatures:

                    ref_sig = self.cluster_displayed_signatures[cluster_id]
                    self.discovery_grid_layout.append(ref_sig)
                    print(f"Added reference for cluster {cluster_id} to discovery grid layout")

            # Add unclustered signatures
            self.discovery_grid_layout.extend(self.unclustered_signatures)
            print(f"Added {len(self.unclustered_signatures)} "\
                  "unclustered signatures to discovery grid layout")

            # Freshly arrange the discovery mode grid
            print("Performing initial arrangement of discovery grid layout")
            self.discovery_grid_needs_fresh_arrangement = True
            sorted_layout = self._arrange_elements_by_similarity_hnsw_with_constraints(\
                self.discovery_grid_layout)
            self.discovery_grid_layout = sorted_layout
            self.discovery_grid_needs_fresh_arrangement = False

            # Update the full signature list cache
            self.full_signature_lists["DISCOVERY"] = self.discovery_grid_layout.copy()

    def _select_discovery_signatures(self):
        """
        Select signatures for discovery mode using lazy loading.
        Only calculates arrangement up to current page, extending as needed.
        
        Returns:
            List of signature paths for current page
        """
        current_page = self.current_page["DISCOVERY"]
        signatures_needed = current_page * self.signatures_per_page

        # Initialize lazy calculation state if needed
        if not hasattr(self, 'lazy_calculation_complete'):
            self.lazy_calculation_complete = False

        # Check if we have enough arranged signatures
        if len(self.lazy_discovery_arranged) >= signatures_needed:
            # We have enough, just return the current page
            start_idx = (current_page - 1) * self.signatures_per_page
            end_idx = min(start_idx + self.signatures_per_page, len(self.lazy_discovery_arranged))

            # Update total pages based on current arrangement
            total_signatures = len(self.lazy_discovery_arranged)
            if not self.lazy_calculation_complete:
                # Estimate total pages (we might have more signatures to arrange)
                estimated_total = total_signatures + len(self.unclustered_signatures)
                for cluster_id in self.clusters:
                    estimated_total += 1  # One representative per cluster
                total_pages = max(current_page, (estimated_total + self.signatures_per_page - 1) \
                                // self.signatures_per_page)
            else:
                total_pages = max(1, (total_signatures + self.signatures_per_page - 1) // \
                                self.signatures_per_page)

            self.total_pages["DISCOVERY"] = total_pages
            self._update_pagination_controls()

            return self.lazy_discovery_arranged[start_idx:end_idx]

        # Need to extend the arrangement
        print(f"Extending discovery arrangement to page {current_page}")

        # If this is the first calculation, build initial candidate pool
        if not self.lazy_discovery_arranged:
            print("Starting fresh discovery arrangement")

            # Initialize cluster references if needed
            if not hasattr(self, 'cluster_displayed_signatures') or \
                len(self.cluster_displayed_signatures) < len(self.clusters):
                self._initialize_cluster_references()

            # Build candidate pool with ALL unclustered signatures and ALL cluster references
            all_candidates_set = set()

            # Add ALL unclustered signatures (the main source of discovery signatures)
            all_candidates_set.update(self.unclustered_signatures)
            print(f"Added {len(self.unclustered_signatures)} unclustered signatures")

            # Add representative from EVERY incomplete cluster (for deterministic placement)
            added_cluster_reps = set()
            for cluster_id, signatures in self.clusters.items():
                if cluster_id in self.complete_clusters:
                    continue
                if signatures and cluster_id in self.cluster_displayed_signatures:
                    representative = self.cluster_displayed_signatures[cluster_id]
                    if representative in signatures and cluster_id not in added_cluster_reps:
                        all_candidates_set.add(representative)
                        added_cluster_reps.add(cluster_id)
                        print(f"Added cluster {cluster_id} representative")

            # Convert to list for processing
            all_candidates = list(all_candidates_set)
            print(f"Total discovery candidates: {len(all_candidates)}")

            if not all_candidates:
                self.total_pages["DISCOVERY"] = 1
                self._update_pagination_controls()
                return []

            # Extract features for initial batch
            initial_batch_size = min(signatures_needed + 100, len(all_candidates))
            initial_batch = all_candidates[:initial_batch_size]
            self._extract_features_for_signatures(initial_batch)

            # Find the most central/representative signature as deterministic starting point
            valid_candidates = [sig for sig in initial_batch if sig in self.features_cache]
            if not valid_candidates:
                self.total_pages["DISCOVERY"] = 1
                self._update_pagination_controls()
                return []

            # Calculate centroid to find most representative starting signature
            if len(valid_candidates) > 1:
                feature_vectors = []
                for sig in valid_candidates:
                    if sig in self.combined_vectors_cache:
                        feature_vectors.append(self.combined_vectors_cache[sig])
                    elif sig in self.features_cache:
                        combined = self._combine_features(self.features_cache[sig])
                        if combined is not None:
                            feature_vectors.append(combined)

                if feature_vectors:
                    centroid = np.mean(feature_vectors, axis=0)

                    # Find signature closest to centroid as deterministic starting point
                    best_distance = float('inf')
                    first_element = valid_candidates[0]  # fallback

                    for sig in valid_candidates:
                        if sig in self.combined_vectors_cache:
                            vector = self.combined_vectors_cache[sig]
                            distance = np.linalg.norm(vector - centroid)
                            if distance < best_distance:
                                best_distance = distance
                                first_element = sig
                else:
                    first_element = valid_candidates[0]
            else:
                first_element = valid_candidates[0]

            self.lazy_discovery_arranged = [first_element]
            self.remaining_candidates = [sig for sig in all_candidates if sig != first_element]

            print(f"Started with most central element: {os.path.basename(first_element)}")

        # Extend arrangement using greedy nearest neighbor approach
        target_signatures = min(signatures_needed, len(self.lazy_discovery_arranged) + \
                                len(self.remaining_candidates))

        # Extract features for current batch
        if self.remaining_candidates:
            batch_size = min(200, len(self.remaining_candidates))
            current_batch = self.remaining_candidates[:batch_size]
            self._extract_features_for_signatures(current_batch)

        while len(self.lazy_discovery_arranged) < target_signatures and self.remaining_candidates:
            last_element = self.lazy_discovery_arranged[-1]

            # Find the most similar candidate from ALL remaining candidates
            best_candidate = None
            best_distance = float('inf')

            # Check ALL remaining candidates to find the most similar one
            candidates_to_check = \
                self.remaining_candidates[:min(500, len(self.remaining_candidates))]

            for candidate in candidates_to_check:
                if candidate not in self.features_cache:
                    continue

                # Check cannot-link constraints
                has_constraint = False
                if self._has_cannot_link_constraint(last_element, candidate):
                    has_constraint = True
                else:
                    # Check cluster-level constraints
                    last_cluster = None
                    candidate_cluster = None

                    for cluster_id, cluster_sigs in self.clusters.items():
                        if last_element in cluster_sigs:
                            last_cluster = cluster_id
                        if candidate in cluster_sigs:
                            candidate_cluster = cluster_id

                    if last_cluster and candidate_cluster and last_cluster != candidate_cluster:
                        if self._is_cluster_rejected(last_cluster, candidate_cluster):
                            has_constraint = True

                if has_constraint:
                    continue

                # Calculate distance
                distance = self._calculate_distance(last_element, candidate)
                if distance is not None and distance < best_distance:
                    best_distance = distance
                    best_candidate = candidate

            if best_candidate:
                self.lazy_discovery_arranged.append(best_candidate)
                self.remaining_candidates.remove(best_candidate)
            else:
                # No valid candidate found, add first available one to avoid infinite loop
                if self.remaining_candidates:
                    random_candidate = self.remaining_candidates[0]
                    self.lazy_discovery_arranged.append(random_candidate)
                    self.remaining_candidates.remove(random_candidate)

        # Mark calculation as complete if we've arranged everything
        if not self.remaining_candidates:
            self.lazy_calculation_complete = True

        # Update pagination
        total_signatures = len(self.lazy_discovery_arranged)
        if not self.lazy_calculation_complete:
            estimated_total = total_signatures + len(self.remaining_candidates)
            total_pages = max(current_page, (estimated_total + self.signatures_per_page - 1) // \
                            self.signatures_per_page)
        else:
            total_pages = max(1, (total_signatures + self.signatures_per_page - 1) // \
                            self.signatures_per_page)

        self.total_pages["DISCOVERY"] = total_pages
        self._update_pagination_controls()

        # Return current page
        start_idx = (current_page - 1) * self.signatures_per_page
        end_idx = min(start_idx + self.signatures_per_page, len(self.lazy_discovery_arranged))

        return self.lazy_discovery_arranged[start_idx:end_idx]

    def _find_cluster_representative(self, cluster_signatures):
        """
        Find the most representative signature in a cluster.
        This is done by finding the signature with the
        minimum average distance to all other signatures.
        
        Args:
            cluster_signatures: List of signature paths in the cluster
            
        Returns:
            The signature path that best represents the cluster
        """
        if not cluster_signatures:
            return None

        # If there's only one signature, it's the representative
        if len(cluster_signatures) == 1:
            return cluster_signatures[0]

        # Ensure features are extracted for all signatures in the cluster
        missing_features = [sig for sig in cluster_signatures if sig not in self.features_cache]
        if missing_features:
            self._extract_features_for_signatures(missing_features)

        # Find the signature with the minimum average distance to all others
        min_avg_distance = float('inf')
        representative = None

        for sig1 in cluster_signatures:
            if sig1 not in self.features_cache:
                continue

            total_distance = 0.0
            valid_comparisons = 0

            for sig2 in cluster_signatures:
                if sig1 != sig2 and sig2 in self.features_cache:
                    distance = self._calculate_distance(sig1, sig2)
                    if distance is not None:
                        total_distance += distance
                        valid_comparisons += 1

            if valid_comparisons > 0:
                avg_distance = total_distance / valid_comparisons
                if avg_distance < min_avg_distance:
                    min_avg_distance = avg_distance
                    representative = sig1

        # Fallback to first signature if calculation fails
        if representative is None and cluster_signatures:
            representative = cluster_signatures[0]

        return representative

    def _calculate_name_similarity(self, name1, name2):
        """
        Calculate similarity between cluster names using weighted Damerau-Levenshtein
        with empirically-derived confusion weights for handwritten characters.
        
        Args:
            name1: First cluster name
            name2: Second cluster name
            
        Returns:
            Similarity score between 0 and 1, where 1 is most similar
        """
        # Handle None or empty values
        if not name1 or not name2:
            return 0.0

        # Normalize strings
        name1 = str(name1).upper()
        name2 = str(name2).upper()

        # Use rapidfuzz for efficient calculation.

        # Get confusion weights
        confusion_matrix = self._get_confusion_weights()

        # Use standard Damerau-Levenshtein as a base calculation
        base_distance = DamerauLevenshtein.normalized_distance(name1, name2)
        adjusted_distance = base_distance

        # Apply adjustments for character confusions
        i, j = 0, 0
        while i < len(name1) and j < len(name2):
            if name1[i] == name2[j]:
                i += 1
                j += 1
                continue

            # Check if characters are commonly confused
            char_pair = (name1[i], name2[j])
            if char_pair in confusion_matrix:
                # Reduce the distance based on confusion likelihood
                confusion_score = confusion_matrix[char_pair]
                # The adjustment should be proportional to the confusion score
                # Higher confusion score = smaller distance adjustment
                adjusted_distance -= 0.5 * confusion_score

            i += 1
            j += 1

        # Normalize to 0-1 similarity score
        max_len = max(len(name1), len(name2))
        similarity = 1.0 - (adjusted_distance / max_len if max_len > 0 else 0)

        # Ensure similarity is within valid range
        similarity = max(0.0, min(1.0, similarity))

        return similarity

    def _get_confusion_weights(self):
        """
        Returns an empirically-derived confusion matrix for handwritten characters.
        Based on CEDAR research data on handwriting recognition.
        
        Returns:
            Dictionary mapping character pairs to confusion likelihood (0-1)
        """
        # Confusion matrix based on CEDAR research
        # Higher values indicate greater likelihood of confusion
        confusion_matrix = {
            # Common confusions in handwriting based on research
            ('R', 'P'): 0.82,  # R and P are frequently confused
            ('R', 'B'): 0.78,
            ('P', 'B'): 0.75,
            ('L', 'I'): 0.85,
            ('L', 'J'): 0.65,
            ('I', 'J'): 0.70,
            ('O', 'Q'): 0.83,
            ('O', '0'): 0.90,
            ('Q', '0'): 0.75,
            ('M', 'N'): 0.72,
            ('N', 'W'): 0.68,
            ('M', 'W'): 0.65,
            ('F', 'T'): 0.75,
            ('F', 'J'): 0.68,
            ('S', '5'): 0.77,
            ('Z', '2'): 0.82,
            ('U', 'V'): 0.85,
            ('C', 'G'): 0.70,
            ('G', 'Q'): 0.72,
            ('D', 'O'): 0.65,
            ('K', 'R'): 0.68,
            ('Y', 'V'): 0.70,
            ('1', 'I'): 0.85,
            ('1', 'L'): 0.80,
            ('1', 'J'): 0.65,
            ('8', 'B'): 0.70,
            ('H', 'N'): 0.65,
            ('A', 'O'): 0.60,
            ('E', 'F'): 0.65,
            ('X', 'K'): 0.65,
        }

        # Make the matrix symmetric (if A is confused with B, B is confused with A)
        full_matrix = {}
        for (c1, c2), weight in confusion_matrix.items():
            full_matrix[(c1, c2)] = weight
            full_matrix[(c2, c1)] = weight  # Ensure symmetry

        return full_matrix

    def _remove_signatures_from_completion_grid(self, signatures_to_remove, cluster_updates=None):
        """
        Remove signatures from completion grid without full refresh
        
        Args:
            signatures_to_remove: List of signature paths to remove
            cluster_updates:
                Dict mapping cluster_id -> new_representative_signature for smart placement
        """
        if self.current_mode != "COMPLETION" \
            or not hasattr(self, 'completion_grid_signatures_cache'):
            return

        # Track removals and replacements
        new_grid = []
        removed_count = 0

        for sig in self.completion_grid_signatures_cache:
            if sig in signatures_to_remove:
                removed_count += 1

                # Check if this signature should be replaced by a cluster representative
                if cluster_updates:
                    for cluster_id, new_rep in cluster_updates.items():
                        # If this signature became the new reference for a cluster,
                        # replace it with the cluster's current representative
                        if sig == new_rep and cluster_id in self.cluster_displayed_signatures:
                            current_rep = self.cluster_displayed_signatures[cluster_id]
                            if current_rep not in new_grid \
                                and current_rep not in signatures_to_remove:
                                new_grid.append(current_rep)
                                break
                # If no replacement, skip this signature (remove it)
            else:
                new_grid.append(sig)

        # Update cached grid and length
        self.completion_grid_signatures_cache = new_grid
        self.completion_grid_length = len(new_grid)

        print(f"Removed {removed_count} signatures from completion "
              f"grid, new length: {self.completion_grid_length}")

        # Check if we need a full refresh due to insufficient length
        if self._should_refresh_completion_grid():
            print("Grid length insufficient, performing full refresh")
            # Save current page and furthest page before reset
            saved_page = self.current_page["COMPLETION"]
            saved_furthest_page = getattr(self, 'completion_furthest_page', 1)

            self._reset_completion_grid_state()

            # Restore the page and furthest page BEFORE calling refresh
            # so that _select_completion_signatures() uses the correct page
            self.current_page["COMPLETION"] = saved_page
            self.completion_furthest_page = max(saved_furthest_page, saved_page)

            self._refresh_grid()

            # After refresh, check if the page is still valid and adjust if needed
            total_pages = self.total_pages.get("COMPLETION", 1)
            if saved_page > total_pages and total_pages >= 1:
                # Page is no longer valid, go to the highest valid page
                self.current_page["COMPLETION"] = total_pages
                self.completion_furthest_page = max(saved_furthest_page, total_pages)
                self._update_pagination_controls()
                # Refresh again with the adjusted page
                self.current_grid_signatures = self._select_completion_signatures()
                self._update_grid_display()
        else:
            # Update current grid signatures and refresh display
            self.current_grid_signatures = self._get_current_page_from_cache()
            self._update_grid_display()
            self._update_pagination_controls()

    def _reset_completion_grid_state(self):
        """Reset completion mode grid state for full refresh"""
        if self.current_mode == "COMPLETION":
            self.completion_furthest_page = 1
            self.completion_grid_length = 0
            self.completion_grid_signatures_cache = []
            self.current_page["COMPLETION"] = 1
            # Reset lazy loading state
            self._reset_lazy_loading_state("COMPLETION")

    def _get_current_page_from_cache(self):
        """Get current page signatures from cached grid"""
        if not self.completion_grid_signatures_cache:
            return []

        current_page = self.current_page["COMPLETION"]
        start_idx = (current_page - 1) * self.signatures_per_page
        end_idx = min(start_idx + self.signatures_per_page,
                      len(self.completion_grid_signatures_cache))

        return self.completion_grid_signatures_cache[start_idx:end_idx]

    def _update_completion_grid_cache(self, new_signatures):
        """Update the completion grid cache with new signatures"""
        self.completion_grid_signatures_cache = new_signatures
        self.completion_grid_length = len(new_signatures)

    def _should_refresh_completion_grid(self):
        """Check if completion mode grid needs a full refresh based on length requirements"""
        if self.current_mode != "COMPLETION":
            return False

        # Calculate minimum required length for furthest visited page
        min_required_length = self.completion_furthest_page * self.signatures_per_page

        # If current grid length is below minimum, need refresh
        if self.completion_grid_length < min_required_length:
            # Check if we have enough candidates to replenish
            total_candidates = len(self.unclustered_signatures)
            for cluster_id in self.clusters:
                if cluster_id != self.current_reference_cluster:
                    total_candidates += 1  # Each cluster counts as one candidate

            # Calculate target length (furthest page + 2 additional pages)
            target_length = (self.completion_furthest_page + 2) * self.signatures_per_page

            # Only refresh if we have enough candidates to make it worthwhile
            return total_candidates >= target_length

        return False

    def _select_completion_signatures_lazy(self, membership_option, completion_type,
                                           name_query, use_name_query, rejection_filter):
        """True lazy loading for Visual Similarity sort with incremental processing
        TODO: I removed the signatures_needed argument, but it may be necessary
        for true lazy loading (I'm not sure). I will revisit this later."""
        current_page = self.current_page["COMPLETION"]

        # For Visual Similarity, we can use HNSW to get absolute ranking incrementally
        if not self.current_reference or not hasattr(self, 'hnsw_index') or self.hnsw_index is None:
            # Fall back to full calculation if no HNSW available
            return self._select_completion_signatures_full(membership_option, completion_type,
                                                        "Visual Similarity", name_query, 
                                                        use_name_query, rejection_filter)

        # Initialize lazy loading state if needed
        cache_key = f"{self.current_reference}_{membership_option}_" \
            f"{completion_type}_{name_query}_{rejection_filter}"

        if not hasattr(self, '_completion_lazy_state'):
            self._completion_lazy_state = {}

        # Reset state if this is a new reference/filter combination
        if cache_key not in self._completion_lazy_state:
            print("Initializing lazy loading state for completion mode...")

            # Build filtered candidate pool - but DON'T extract features yet
            candidates = []

            # Add filtered unclustered signatures
            if membership_option in ["Unclustered", "Both"] and not use_name_query:
                for sig in self.unclustered_signatures:
                    is_rejected = \
                        self._is_image_rejected_by_cluster(sig, self.current_reference_cluster)

                    # Apply rejection filter
                    if rejection_filter == "Rejected" and not is_rejected:
                        continue
                    elif rejection_filter == "Non-rejected" and is_rejected:
                        continue

                    candidates.append(sig)

            # Add filtered cluster representatives
            if membership_option in ["Clustered", "Both"]:
                for cluster_id, signatures in self.clusters.items():
                    if cluster_id == self.current_reference_cluster:
                        continue

                    # Apply completion filter
                    if completion_type == "Complete" and cluster_id not in self.complete_clusters:
                        continue
                    elif completion_type == "Incomplete" and cluster_id in self.complete_clusters:
                        continue

                    # Apply name filter
                    if name_query and not self._is_strict_substring_match(name_query, cluster_id):
                        continue

                    # Check rejection status
                    is_rejected = \
                        self._is_cluster_rejected(self.current_reference_cluster, cluster_id)
                    if rejection_filter == "Rejected" and not is_rejected:
                        continue
                    elif rejection_filter == "Non-rejected" and is_rejected:
                        continue

                    # Get representative signature
                    if cluster_id in self.cluster_displayed_signatures:
                        rep_sig = self.cluster_displayed_signatures[cluster_id]
                    elif signatures:
                        rep_sig = self._find_cluster_representative(signatures)
                    else:
                        continue

                    candidates.append(rep_sig)

            # Initialize state - NO pre-processing of candidates
            self._completion_lazy_state[cache_key] = {
                'candidates': set(candidates),
                'total_candidates': len(candidates),
                'sorted_signatures': [],  # Incrementally built sorted list
                'processed_candidates': set(),  # Track what we've already processed
                'search_exhausted': False  # Track if we've searched everything
            }

            print(f"Initialized lazy state with {len(candidates)} candidates (no pre-processing)")

        # Get state for current reference/filter combination
        state = self._completion_lazy_state[cache_key]

        if not state['candidates']:
            self.total_pages["COMPLETION"] = 1
            self._update_pagination_controls()
            return []

        try:
            # Get reference vector
            if self.current_reference not in self.combined_vectors_cache:
                self._extract_features_for_signatures([self.current_reference])

            ref_vector = self.combined_vectors_cache[self.current_reference]

            # Calculate how many signatures we need based
            # on furthest visited page + 2 additional pages
            target_page = self.completion_furthest_page + 2
            signatures_needed_for_extended = target_page * self.signatures_per_page

            # Only process more if we don't have enough signatures for the extended target
            while len(state['sorted_signatures']) < signatures_needed_for_extended \
                and not state['search_exhausted']:
                # Calculate how many more we need to find
                batch_size = max(self.signatures_per_page * 2, 200)  # Process in reasonable batches

                # Get next batch from HNSW, asking for more than we processed before
                k_to_request = min(
                    len(state['processed_candidates']) + batch_size,
                    self.hnsw_index.get_indexed_count()
                )

                # Get nearest neighbors from HNSW
                nearest = self.hnsw_index.get_nearest_neighbors(ref_vector, k=k_to_request)

                # Process signatures we haven't seen before
                new_signatures_found = []
                candidate_set = state['candidates']
                processed_set = state['processed_candidates']

                for sig, distance in nearest:
                    if sig in candidate_set and sig not in processed_set:
                        # Extract features for this signature if needed
                        if sig not in self.combined_vectors_cache:
                            self._extract_features_for_signatures([sig])
                            # Add to HNSW if extraction was successful
                            if sig in self.combined_vectors_cache:
                                self.hnsw_index.add_vector(sig, self.combined_vectors_cache[sig])

                        new_signatures_found.append((sig, distance))
                        processed_set.add(sig)

                # Add newly found signatures to our sorted list
                new_signatures_found.sort(key=lambda x: x[1])  # Sort by distance
                for sig, _ in new_signatures_found:
                    state['sorted_signatures'].append(sig)

                # Check if we've exhausted the search
                if len(processed_set) >= len(candidate_set) or len(new_signatures_found) == 0:
                    # Add any remaining unprocessed candidates
                    remaining = candidate_set - processed_set
                    if remaining:
                        # Extract features for remaining candidates
                        remaining_list = list(remaining)
                        self._extract_features_for_signatures(remaining_list)

                        # Calculate distances for remaining candidates
                        remaining_with_distances = []
                        for sig in remaining_list:
                            if sig in self.combined_vectors_cache:
                                distance = self._calculate_distance(self.current_reference, sig)
                                if distance is not None:
                                    remaining_with_distances.append((sig, distance))

                        # Sort and add remaining candidates
                        remaining_with_distances.sort(key=lambda x: x[1])
                        for sig, _ in remaining_with_distances:
                            state['sorted_signatures'].append(sig)

                    state['search_exhausted'] = True

            # Calculate pagination based on total available candidates, not just arranged signatures
            if state['search_exhausted']:
                total_signatures = len(state['sorted_signatures'])
                total_pages = max(1, (total_signatures + self.signatures_per_page - 1) // \
                                  self.signatures_per_page)
            else:
                # Estimate total pages based on total candidates, not just arranged signatures
                total_candidates = state['total_candidates']
                # Use actual total candidates for pagination, not just arranged signatures
                total_pages = max(current_page, (total_candidates + self.signatures_per_page - 1) \
                                  // self.signatures_per_page)

            self.total_pages["COMPLETION"] = total_pages
            self._update_pagination_controls()

            # Get signatures for the specific requested page
            start_idx = (current_page - 1) * self.signatures_per_page
            end_idx = min(start_idx + self.signatures_per_page, len(state['sorted_signatures']))
            page_signatures = state['sorted_signatures'][start_idx:end_idx]

            print(f"Completion mode lazy: page {current_page}/{total_pages}, "
                f"got {len(page_signatures)} signatures, "
                f"processed {len(state['processed_candidates'])}/"
                f"{state['total_candidates']} candidates, "
                f"arranged {len(state['sorted_signatures'])} total")

            return page_signatures

        except Exception as e:
            print(f"Error in HNSW lazy loading: {e}")
            # Fall back to full calculation
            return self._select_completion_signatures_full(membership_option, completion_type,
                                                        "Visual Similarity", name_query, 
                                                        use_name_query, rejection_filter)

    def _select_completion_signatures_full(self, membership_option, completion_type, sort_option,
                                         name_query, use_name_query, rejection_filter):
        """Full calculation version for non-Visual Similarity sorts"""
        current_page = self.current_page["COMPLETION"]

        # Build all candidates (existing logic)
        all_candidates = []

        # Process cluster representatives
        if membership_option in ["Clustered", "Both"]:
            for cluster_id, signatures in self.clusters.items():
                if cluster_id == self.current_reference_cluster:
                    continue

                # Apply completion filter
                if completion_type == "Complete" and cluster_id not in self.complete_clusters:
                    continue
                elif completion_type == "Incomplete" and cluster_id in self.complete_clusters:
                    continue

                # Apply name filter based on sort option
                if name_query:
                    if sort_option == "Query Similarity":
                        # For Query Similarity, include all and calculate similarity later
                        pass
                    else:
                        # For other sorts, strict substring matching
                        if not self._is_strict_substring_match(name_query, cluster_id):
                            continue

                # Get representative signature for this cluster
                if cluster_id in self.cluster_displayed_signatures:
                    rep_sig = self.cluster_displayed_signatures[cluster_id]
                elif signatures:
                    rep_sig = self._find_cluster_representative(signatures)
                else:
                    continue

                # Check rejection status (cluster-level)
                is_rejected = self._is_cluster_rejected(self.current_reference_cluster, cluster_id)

                # Apply rejection filter
                if rejection_filter == "Rejected" and not is_rejected:
                    continue
                elif rejection_filter == "Non-rejected" and is_rejected:
                    continue

                # Calculate visual similarity
                visual_similarity = 0.0
                if self.current_reference and rep_sig in self.features_cache:
                    distance = self._calculate_distance(self.current_reference, rep_sig)
                    if distance is not None:
                        visual_similarity = self._convert_distance_to_similarity(distance)

                # Calculate name similarity for Query Similarity sort
                name_similarity = 0.0
                if sort_option == "Query Similarity" and name_query:
                    name_similarity = \
                        self._calculate_name_similarity(name_query.lower(), str(cluster_id).lower())

                all_candidates.append({
                    'signature': rep_sig,
                    'cluster_id': cluster_id,
                    'is_rejected': is_rejected,
                    'cluster_size': len(signatures),
                    'type': 'cluster',
                    'visual_similarity': visual_similarity,
                    'name_similarity': name_similarity
                })

        # Process unclustered signatures
        if membership_option in ["Unclustered", "Both"]:
            include_unclustered = not use_name_query

            if include_unclustered:
                for sig in self.unclustered_signatures:
                    # Check rejection status (individual signature)
                    is_rejected = \
                        self._is_image_rejected_by_cluster(sig, self.current_reference_cluster)

                    # Apply rejection filter
                    if rejection_filter == "Rejected" and not is_rejected:
                        continue
                    elif rejection_filter == "Non-rejected" and is_rejected:
                        continue

                    # Calculate visual similarity
                    visual_similarity = 0.0
                    if self.current_reference and sig in self.features_cache:
                        distance = self._calculate_distance(self.current_reference, sig)
                        if distance is not None:
                            visual_similarity = self._convert_distance_to_similarity(distance)

                    all_candidates.append({
                        'signature': sig,
                        'cluster_id': None,
                        'is_rejected': is_rejected,
                        'cluster_size': 1,
                        'type': 'unclustered',
                        'visual_similarity': visual_similarity,
                        'name_similarity': 0.0  # Unclustered have no name similarity
                    })

        # Sort candidates with enhanced logic
        all_candidates = self._sort_completion_candidates(all_candidates, sort_option, name_query)

        # Calculate pagination
        total_pages = max(1, (len(all_candidates) + self.signatures_per_page - 1) //
                        self.signatures_per_page)
        self.total_pages["COMPLETION"] = total_pages
        self._update_pagination_controls()

        # Get current page subset
        start_idx = (current_page - 1) * self.signatures_per_page
        end_idx = min(start_idx + self.signatures_per_page, len(all_candidates))
        page_candidates = all_candidates[start_idx:end_idx]

        # Extract just the signature paths
        page_signatures = [candidate['signature'] for candidate in page_candidates]

        print(f"Completion mode: page {current_page}/{total_pages} "
              f"with {len(page_signatures)} signatures")

        return page_signatures

    def _select_completion_signatures(self):
        """
        Select signatures for completion mode with lazy loading for Visual Similarity sort
        """
        if not self.current_reference:
            self.total_pages["COMPLETION"] = 1
            self._update_pagination_controls()
            return []

        current_page = self.current_page["COMPLETION"]
        # TODO: I removed this, but it may be necessary for true lazy loading, revisit later.
        #signatures_needed = current_page * self.signatures_per_page

        # Get current filter values
        membership_option = getattr(self, 'last_applied_grid_membership', 'Both')
        completion_type = getattr(self, 'last_applied_grid_filter', 'Incomplete')
        sort_option = getattr(self, 'last_applied_grid_sort', 'Visual Similarity')
        use_name_query = getattr(self, 'last_applied_grid_use_name_query', False)
        name_query = getattr(self, 'last_applied_grid_name_query', '') if use_name_query else ''
        rejection_filter = getattr(self, 'last_applied_rejection_filter', 'Non-rejected')

        print(f"Completion mode filtering: membership={membership_option}, "
            f"completion={completion_type}, sort={sort_option}, "
            f"query='{name_query}', rejection={rejection_filter}")

        # Use lazy loading ONLY for Visual Similarity sort (which uses HNSW)
        if sort_option == "Visual Similarity":
            return self._select_completion_signatures_lazy(membership_option, completion_type,
                                                           name_query, use_name_query,
                                                           rejection_filter)
        else:
            return self._select_completion_signatures_full(membership_option, completion_type,
                                                        sort_option, name_query, use_name_query,
                                                        rejection_filter)

    def _sort_completion_candidates(self, candidates, sort_option, name_query):
        """Sort candidates with enhanced two-tier logic for Query Similarity"""
        if sort_option == "Query Similarity" and name_query:
            # Two-tier sorting: exact matches first, then by similarity
            exact_matches = []
            non_matches = []
            query_lower = name_query.lower()

            for candidate in candidates:
                if candidate['type'] == 'cluster':
                    name = str(candidate['cluster_id']).lower()
                    if query_lower in name:
                        exact_matches.append(candidate)
                    else:
                        non_matches.append(candidate)
                else:
                    # Unclustered signatures don't have names, so they go to non_matches
                    non_matches.append(candidate)

            # Sort exact matches by name similarity, then visual similarity
            exact_matches.sort(key=lambda x: (-x['name_similarity'], -x['visual_similarity']))
            # Sort non-matches by name similarity, then visual similarity
            non_matches.sort(key=lambda x: (-x['name_similarity'], -x['visual_similarity']))

            return exact_matches + non_matches
        elif sort_option == "Visual Similarity":
            candidates.sort(key=lambda x: -x['visual_similarity'])
        elif sort_option == "A→Z":
            def az_sort_key(x):
                if x['type'] == 'unclustered':
                    return (0, -x['visual_similarity'])
                else:
                    return (1, str(x['cluster_id']).lower(), -x['visual_similarity'])
            candidates.sort(key=az_sort_key)
        elif sort_option == "Z→A":
            def za_sort_key(x):
                if x['type'] == 'cluster':
                    return (1, str(x['cluster_id']).lower(), x['visual_similarity'])
                else:
                    return (0, x['visual_similarity'])
            candidates.sort(key=za_sort_key, reverse=True)
        elif sort_option == "Size (↓)":
            def size_down_sort_key(x):
                if x['type'] == 'unclustered':
                    return (0, -x['visual_similarity'])
                else:
                    return (1, x['cluster_size'], -x['visual_similarity'])
            candidates.sort(key=size_down_sort_key)
        elif sort_option == "Size (↑)":
            def size_up_sort_key(x):
                if x['type'] == 'cluster':
                    return (0, -x['cluster_size'], -x['visual_similarity'])
                else:
                    return (1, -x['visual_similarity'])
            candidates.sort(key=size_up_sort_key)
        elif sort_option == "Path (↓)":
            candidates.sort(key=lambda x: x["signature"])
        elif sort_option == "Path (↑)":
            candidates.sort(key=lambda x: x["signature"], reverse=True)
        elif sort_option == "Path Similarity":
            candidates = self._sort_by_path_similarity(candidates)

        return candidates

    def _sort_by_path_similarity(self, candidates):
        if not self.current_reference or not candidates:
            return candidates

        ref_path = self.current_reference
        ref_dir = os.path.dirname(ref_path)

        # Index all candidates by directory
        dir_to_candidates = {}
        for candidate in candidates:
            path = candidate["signature"]
            dirpath = os.path.dirname(path)
            dir_to_candidates.setdefault(dirpath, []).append((candidate, path))

        sorted_candidates = []
        processed_paths = set()
        visited_dirs = set()

        def radiating_sort(items, anchor_name):
            after = []
            before = []

            items.sort(key=lambda x: x[1])
            for obj, name in items:
                if name > anchor_name:
                    after.append((obj, name))
                elif name < anchor_name:
                    before.append((obj, name))
            before.reverse()

            result = []
            max_len = max(len(after), len(before))
            for i in range(max_len):
                if i < len(after):
                    result.append(after[i])
                if i < len(before):
                    result.append(before[i])
            return result

        def linear_sort(items, reverse=False):
            return sorted(items, key=lambda x: x[1], reverse=reverse)

        def process_subtree(current_dir, sort_reverse):
            if current_dir in visited_dirs:
                return
            visited_dirs.add(current_dir)

            file_items = []
            subdir_items = []

            # Collect files
            for candidate, path in dir_to_candidates.get(current_dir, []):
                filename = os.path.basename(path)
                file_items.append((candidate, filename))

            for candidate, filename in linear_sort(file_items, reverse=sort_reverse):
                sorted_candidates.append(candidate)
                processed_paths.add(candidate["signature"])

            # Collect subdirs
            try:
                entries = os.listdir(current_dir)
            except Exception:
                return

            for entry in entries:
                full_path = os.path.join(current_dir, entry)
                if os.path.isdir(full_path):
                    subdir_items.append((full_path, entry))

            for subdir_path, _ in linear_sort(subdir_items, reverse=sort_reverse):
                process_subtree(subdir_path, sort_reverse)

        def process_radiating_dir(current_dir, anchor_name, skip_subdir=None):
            if current_dir in visited_dirs:
                return
            visited_dirs.add(current_dir)

            file_items = []
            subdir_items = []

            # Collect and radiate-sort files
            for candidate, path in dir_to_candidates.get(current_dir, []):
                filename = os.path.basename(path)
                file_items.append((candidate, filename))

            for candidate, _ in radiating_sort(file_items, anchor_name):
                sorted_candidates.append(candidate)
                processed_paths.add(candidate["signature"])

            # Collect subdirs
            try:
                entries = os.listdir(current_dir)
            except Exception:
                return

            for entry in entries:
                full_path = os.path.join(current_dir, entry)
                if os.path.isdir(full_path) and full_path != skip_subdir:
                    subdir_items.append((full_path, entry))

            ranked_subdirs = radiating_sort(subdir_items, anchor_name)

            for subdir_path, subdir_name in ranked_subdirs:
                sort_reverse = subdir_name < anchor_name
                process_subtree(subdir_path, sort_reverse)

        # Step 1: Process the reference directory with radiation logic
        process_radiating_dir(ref_dir, os.path.basename(ref_path))

        # Step 2: Radiate outward by scoping up
        current_dir = ref_dir

        while current_dir != self.base_directory:
            parent_dir = os.path.dirname(current_dir)
            anchor_name = os.path.basename(current_dir)
            process_radiating_dir(parent_dir, anchor_name, skip_subdir=current_dir)
            current_dir = parent_dir

        # Step 3: Add any remaining unprocessed candidates
        for candidate in candidates:
            path = candidate["signature"]
            if path not in processed_paths:
                sorted_candidates.append(candidate)

        return sorted_candidates

    def _convert_distance_to_similarity(self, distance):
        """Convert a distance value to a similarity value (0-1)"""
        if distance is None:
            return 0.0

        if self.clustering_params['DISTANCE_METRIC'] in ['correlation', 'cosine']:
            # For metrics that are already 0-1, just invert
            similarity = 1.0 - distance
        else:
            # For other metrics, normalize based on threshold
            threshold = self.clustering_params.get('DISTANCE_THRESHOLD', 0.5)
            similarity = max(0, 1.0 - (distance / threshold))
            similarity = min(1.0, similarity)  # Cap at 1.0

        return similarity

    def _reset_lazy_loading_state(self, mode=None):
        """Reset lazy loading state for specified mode or all modes"""
        if mode is None or mode == "DISCOVERY":
            self.lazy_discovery_arranged = []
            self.lazy_calculation_complete = False
            if hasattr(self, 'remaining_candidates'):
                delattr(self, 'remaining_candidates')

        if mode is None or mode == "COMPLETION":
            self.lazy_completion_arranged = []
            self.completion_remaining_candidates = []
            self.completion_calculation_complete = False
            # Clear completion lazy state
            if hasattr(self, '_completion_lazy_state'):
                self._completion_lazy_state.clear()

    def _handle_grid_alteration_discovery(self, new_cluster_id, added_signatures,
                                        removed_signatures_by_cluster):
        """
        FIXED VERSION: Handle grid alterations for discovery mode with proper cluster placement.
        
        Args:
            new_cluster_id: ID of the newly created/modified cluster
            added_signatures: List of signatures that were added to the new cluster
            removed_signatures_by_cluster: Dict mapping cluster_id -> list of removed signatures
        """
        if not hasattr(self, 'lazy_discovery_arranged') or not self.lazy_discovery_arranged:
            return

        print(f"Handling discovery grid alteration: new_cluster={new_cluster_id}, "
              f"added={len(added_signatures)}, "
              f"removed_from={list(removed_signatures_by_cluster.keys())}")

        # Track the target position for the new cluster
        target_position = None
        cluster_reference = None

        # CRITICAL FIX: Update remaining_candidates for lazy loading
        # Remove any signatures that were added to clusters from the remaining candidates list
        if hasattr(self, 'remaining_candidates') and self.remaining_candidates:
            # Remove all added signatures from remaining candidates
            if added_signatures:
                original_remaining_count = len(self.remaining_candidates)
                self.remaining_candidates = [sig for sig in self.remaining_candidates
                                             if sig not in added_signatures]
                removed_from_remaining = original_remaining_count - len(self.remaining_candidates)
                if removed_from_remaining > 0:
                    print(f"Removed {removed_from_remaining} signatures from remaining candidates")

            # Also remove any signatures that were removed from other clusters
            for cluster_id, removed_sigs in removed_signatures_by_cluster.items():
                if removed_sigs:
                    original_remaining_count = len(self.remaining_candidates)
                    self.remaining_candidates = [sig for sig in self.remaining_candidates
                                                 if sig not in removed_sigs]
                    removed_from_remaining = \
                        original_remaining_count - len(self.remaining_candidates)
                    if removed_from_remaining > 0:
                        print(f"Removed {removed_from_remaining} signatures from "
                              f"remaining candidates (from cluster {cluster_id})")

        # Determine where the cluster should be placed
        if new_cluster_id and new_cluster_id in self.cluster_displayed_signatures:
            cluster_reference = self.cluster_displayed_signatures[new_cluster_id]

            # Check if the cluster reference is one of the added signatures
            if cluster_reference in added_signatures:
                # Find where this signature was in the discovery grid
                try:
                    target_position = self.lazy_discovery_arranged.index(cluster_reference)
                    print(f"New cluster reference {os.path.basename(cluster_reference)} "
                        f"found at position {target_position}")
                except ValueError:
                    print(f"New cluster reference {os.path.basename(cluster_reference)} "
                        "not found in discovery grid")
                    target_position = None
            else:
                # Reference didn't change, try to find existing cluster position
                try:
                    target_position = self.lazy_discovery_arranged.index(cluster_reference)
                    print(f"Existing cluster reference {os.path.basename(cluster_reference)} "
                        f"found at position {target_position}")
                except ValueError:
                    # Cluster reference not in grid, will be added at end
                    target_position = None

        # Remove all affected signatures from their current positions
        signatures_to_remove = set()

        # Collect all signatures that were moved to the new cluster
        signatures_to_remove.update(added_signatures)

        # Collect all signatures that were removed from other clusters
        for cluster_id, removed_sigs in removed_signatures_by_cluster.items():
            signatures_to_remove.update(removed_sigs)

        # Remove signatures from discovery grid (but track their positions)
        removed_positions = {}
        for sig in list(signatures_to_remove):
            try:
                pos = self.lazy_discovery_arranged.index(sig)
                removed_positions[sig] = pos
                self.lazy_discovery_arranged.remove(sig)
                print(f"Removed {os.path.basename(sig)} from position {pos}")
            except ValueError:
                pass  # Signature not in discovery grid

        # Place the new cluster at the appropriate position
        # CRITICAL FIX: Only add cluster representative if it doesn't already exist in the grid
        if cluster_reference and new_cluster_id:
            # Check if this cluster already has a representative in the discovery grid
            cluster_already_exists = cluster_reference in self.lazy_discovery_arranged

            if not cluster_already_exists:
                if target_position is not None:
                    # Adjust target position if signatures were removed before it
                    adjusted_position = target_position
                    for sig, pos in removed_positions.items():
                        if pos < target_position:
                            adjusted_position -= 1

                    # Ensure position is valid
                    adjusted_position = \
                        max(0, min(adjusted_position, len(self.lazy_discovery_arranged)))

                    # Insert cluster reference at the calculated position
                    self.lazy_discovery_arranged.insert(adjusted_position, cluster_reference)
                    print(f"Inserted cluster {new_cluster_id} at position {adjusted_position}")
                else:
                    # Add cluster at the end if no specific position determined
                    self.lazy_discovery_arranged.append(cluster_reference)
                    print(f"Added cluster {new_cluster_id} at end")
            else:
                print(f"Cluster {new_cluster_id} already exists in "
                      "discovery grid, not adding duplicate")

        # Handle removed signatures - place them after their original cluster
        for cluster_id, removed_sigs in removed_signatures_by_cluster.items():
            if cluster_id in self.cluster_displayed_signatures:
                cluster_ref = self.cluster_displayed_signatures[cluster_id]
                try:
                    cluster_pos = self.lazy_discovery_arranged.index(cluster_ref)
                    # Insert removed signatures after the cluster
                    for i, sig in enumerate(removed_sigs):
                        insert_pos = cluster_pos + 1 + i
                        if insert_pos <= len(self.lazy_discovery_arranged):
                            self.lazy_discovery_arranged.insert(insert_pos, sig)
                            print(f"Inserted removed signature {os.path.basename(sig)} "
                                f"after cluster {cluster_id}")
                except ValueError:
                    # Cluster not found, add removed signatures at end
                    self.lazy_discovery_arranged.extend(removed_sigs)

        # Update the full signature list cache
        self.full_signature_lists["DISCOVERY"] = self.lazy_discovery_arranged.copy()
        print(f"Updated discovery grid with {len(self.lazy_discovery_arranged)} signatures")

    def _handle_completion_grid_alteration(self, altered_signatures):
        """Remove signatures from completion arrangement without recalculating"""
        if not hasattr(self, 'lazy_completion_arranged'):
            return

        # Remove altered signatures from arrangement
        for sig in altered_signatures:
            if sig in self.lazy_completion_arranged:
                self.lazy_completion_arranged.remove(sig)

        # Remove from remaining candidates too
        self.completion_remaining_candidates = [
            candidate for candidate in self.completion_remaining_candidates
            if candidate['signature'] not in altered_signatures
        ]

    def _select_verification_signatures(self):
        """
        Select ALL signatures for verification mode - show EVERY signature from current cluster
        EXCLUDING the reference signature itself.
        Always use the CURRENT reference cluster, never cached data.
        Now supports pagination.
        """
        # If no reference cluster, return empty list
        if not self.current_reference_cluster or \
            self.current_reference_cluster not in self.clusters:

            print("No valid reference cluster for verification mode")
            self.total_pages["VERIFICATION"] = 1
            self._update_pagination_controls()
            return []

        print(f"Verification mode showing signatures from cluster {self.current_reference_cluster}")

        # Get ALL signatures from the current cluster
        cluster_sigs = self.clusters[self.current_reference_cluster].copy()

        # EXCLUDE the reference signature - we don't want to show it in the grid
        if self.current_reference in cluster_sigs:
            cluster_sigs.remove(self.current_reference)

        # If we have no signatures after removing the reference, return an empty list
        if not cluster_sigs:
            print("No signatures to display in verification mode after excluding reference")
            self.total_pages["VERIFICATION"] = 1
            self._update_pagination_controls()
            return []

        # Sort by similarity to centroid for consistent ordering
        if hasattr(self, 'cluster_ordered_signatures') and \
            self.current_reference_cluster in self.cluster_ordered_signatures:

            # Use cached ordering but remove reference signature if present
            ordered_sigs = self.cluster_ordered_signatures[self.current_reference_cluster].copy()
            if self.current_reference in ordered_sigs:
                ordered_sigs.remove(self.current_reference)

            # Make sure we're only returning signatures actually in the cluster
            ordered_sigs = [sig for sig in ordered_sigs if sig in cluster_sigs]

            # Add any signatures missing from ordered list (shouldn't happen but just in case)
            missing_sigs = [sig for sig in cluster_sigs if sig not in ordered_sigs]
            ordered_sigs.extend(missing_sigs)

            print(f"Using cached ordering for verification mode: {len(ordered_sigs)} signatures")
            sorted_signatures = ordered_sigs
        else:

            # Calculate ordering if not cached or attribute doesn't exist
            # Initialize the attribute if it doesn't exist
            if not hasattr(self, 'cluster_ordered_signatures'):
                self.cluster_ordered_signatures = {}

            # Calculate ordering if not cached
            ordered_sigs = \
                self._get_cluster_signatures_by_similarity(self.current_reference_cluster)

            # Remove reference and filter to cluster signatures
            if self.current_reference in ordered_sigs:
                ordered_sigs.remove(self.current_reference)

            ordered_sigs = [sig for sig in ordered_sigs if sig in cluster_sigs]

            # Add any missing signatures
            missing_sigs = [sig for sig in cluster_sigs if sig not in ordered_sigs]
            ordered_sigs.extend(missing_sigs)

            print(f"Created fresh ordering for verification mode: {len(ordered_sigs)} signatures")
            sorted_signatures = ordered_sigs

        # Cache the full sorted list for future pagination
        self.full_signature_lists["VERIFICATION"] = sorted_signatures

        # Calculate total pages
        total_pages = max(1, (len(sorted_signatures) + \
                              self.signatures_per_page - 1) // self.signatures_per_page)
        self.total_pages["VERIFICATION"] = total_pages

        # Ensure current page is valid
        if self.current_page["VERIFICATION"] > total_pages:
            self.current_page["VERIFICATION"] = total_pages

        # Get the current page's subset
        current_page = self.current_page["VERIFICATION"]
        start_idx = (current_page - 1) * self.signatures_per_page
        end_idx = min(start_idx + self.signatures_per_page, len(sorted_signatures))

        # Return just the signatures for the current page
        page_signatures = sorted_signatures[start_idx:end_idx]

        print(f"Verification mode: page {current_page}/{total_pages} " \
              f"with {len(page_signatures)} signatures")

        # Update pagination controls
        self._update_pagination_controls()

        return page_signatures

    def _update_grid_display(self):
        """
        Dynamically create and update the scrollable grid with strict ordering.
        Now strictly respects self.grid_cols in discovery mode.
        Ensures scrolling works after grid is updated.
        """
        try:
            # Clear previous grid
            self.signature_frames = []
            for widget in self.scrollable_frame.winfo_children():
                widget.destroy()

            # Skip if no signatures
            if not hasattr(self, 'current_grid_signatures') or not self.current_grid_signatures:
                print("No signatures to display in grid")
                return

            print(f"Creating grid with {len(self.current_grid_signatures)} signatures")

            # Determine the number of columns to use - ALWAYS use self.grid_cols
            # No dynamic adjustment for discovery mode anymore
            grid_cols = self.grid_cols

            # Calculate total rows needed
            total_signatures = len(self.current_grid_signatures)
            total_rows = (total_signatures + grid_cols - 1) // grid_cols

            print(f"Grid layout: {grid_cols} columns × {total_rows} rows")

            # Create all row frames first
            row_frames = []
            for _ in range(total_rows):
                row_frame = ttk.Frame(self.scrollable_frame)
                row_frame.pack(fill=tk.X, pady=2)
                row_frames.append(row_frame)

            # Create all signature frames
            for i in range(total_signatures):
                # Calculate row and column
                row = i // grid_cols
                col = i % grid_cols

                # Ensure row index is valid
                if row >= len(row_frames):
                    print(f"Warning: Row index {row} is out of range. Creating new row.")
                    row_frame = ttk.Frame(self.scrollable_frame)
                    row_frame.pack(fill=tk.X, pady=2)
                    row_frames.append(row_frame)

                # Get the row frame
                row_frame = row_frames[row]

                # Get signature path (with index check)
                if i < len(self.current_grid_signatures):
                    sig_path = self.current_grid_signatures[i]

                    # Create frame in this position
                    frame = self._create_signature_frame_dynamic(row_frame, i, sig_path)
                    if frame:
                        self.signature_frames.append(frame)

                        # Store grid position for debugging
                        frame.grid_index = i
                        frame.grid_row = row
                        frame.grid_col = col
                else:
                    print(f"Warning: Index {i} is out of range for current_grid_signatures")

            # After creating all frames, set up scrolling
            try:
                self.canvas.update_idletasks()
                self.canvas.configure(scrollregion=self.canvas.bbox("all"))

                # Reset scroll position to top
                self.canvas.yview_moveto(0)
            except tk.TclError:
                # Canvas might be destroyed
                pass

            # IMPORTANT: Set up scrolling again after creating the grid
            self._setup_mousewheel_scrolling()

        except Exception as e:
            traceback.print_exc()
            print(f"Error in _update_grid_display: {e}")

    def _create_signature_frame_dynamic(self, parent_row, index, sig_path=None):
        """
        Create a signature frame within a row for the dynamic grid
        
        Args:
            parent_row: The row frame to place this signature in
            index: Index for tracking
            sig_path: Path to signature image
        """
        try:
            # Create frame
            frame = ttk.Frame(parent_row, borderwidth=2, relief="solid")
            frame.pack(side=tk.LEFT, padx=5, pady=5)

            # Determine if this is a clustered signature
            frame.cluster_id = None
            if sig_path:
                for cluster_id, sigs in self.clusters.items():
                    if sig_path in sigs:
                        frame.cluster_id = cluster_id
                        break

            # Create a container for the canvas and the goto button
            canvas_container = ttk.Frame(frame)
            canvas_container.pack(fill=tk.X, pady=(5, 0))

            # Canvas for the image
            canvas = tk.Canvas(canvas_container, width=self.thumbnail_size[0], \
                               height=self.thumbnail_size[1], bg="#f0f0f0")
            canvas.pack(side=tk.BOTTOM)

            # Add the "go to" button for clustered signatures in discovery and completion mode
            is_clustered = frame.cluster_id is not None
            if is_clustered and self.current_mode in ["DISCOVERY", "COMPLETION"]:
                # Create an actual button
                goto_btn = ttk.Button(
                    canvas_container,
                    text="➚",
                    width=2,
                    command=lambda cid=frame.cluster_id: self._go_to_cluster_in_completion_mode(cid)
                )
                # Position the button at the top-right corner using place manager
                goto_btn.place(x=self.thumbnail_size[0]-52, y=0)
                frame.goto_btn = goto_btn

            # Show navigation in BOTH discovery and completion mode for clustered signatures
            # But not in verification mode
            needs_navigation = \
                (frame.cluster_id is not None and self.current_mode != "VERIFICATION")

            # First add the filename label (for all modes) - full width
            filename_label = ttk.Label(frame, text="No image", \
                                       font=("TkDefaultFont", 8), anchor=tk.CENTER)
            filename_label.pack(fill=tk.X, pady=(2, 0))

            if needs_navigation:
                # BOTH DISCOVERY AND COMPLETION MODES - Create navigation container
                nav_container = ttk.Frame(frame)
                nav_container.pack(fill=tk.X, pady=(0, 2))

                # Left navigation button
                prev_btn = ttk.Button(nav_container, text="←", width=2)
                prev_btn.pack(side=tk.LEFT, padx=1)
                frame.prev_btn = prev_btn

                # For completion mode, add centered similarity score
                if self.current_mode == "COMPLETION":
                    # Create a container to center the similarity label
                    sim_container = ttk.Frame(nav_container)
                    sim_container.pack(side=tk.LEFT, fill=tk.X, expand=True)

                    # Similarity score centered in its container
                    sim_label = ttk.Label(sim_container, text="", \
                                          font=("TkDefaultFont", 8, "bold"), anchor=tk.CENTER)
                    sim_label.pack(expand=True)
                else:
                    # For discovery mode, just add a spacer to push buttons apart
                    spacer = ttk.Frame(nav_container)
                    spacer.pack(side=tk.LEFT, fill=tk.X, expand=True)

                    # Create empty sim_label for consistency
                    sim_label = ttk.Label(frame, text="")

                # Right navigation button
                next_btn = ttk.Button(nav_container, text="→", width=2)
                next_btn.pack(side=tk.RIGHT, padx=1)
                frame.next_btn = next_btn
            else:
                # For verification mode or unclustered signatures
                if self.current_mode == "COMPLETION":
                    # Add similarity label for completion mode without navigation
                    sim_label = ttk.Label(frame, text="", \
                                          font=("TkDefaultFont", 8, "bold"), anchor=tk.CENTER)
                    sim_label.pack(pady=(0, 2))
                else:
                    # Create empty sim_label for consistency
                    sim_label = ttk.Label(frame, text="")

                # Set navigation button attributes to None for consistency
                frame.prev_btn = None
                frame.next_btn = None

            # Store references to widgets
            frame.canvas = canvas
            frame.filename_label = filename_label
            frame.sim_label = sim_label
            frame.index = index
            frame.selected = False
            frame.signature_path = sig_path
            frame.image_tk = None  # To prevent garbage collection
            frame.displayed_signature = sig_path  # Initially display the main signature
            frame.has_navigation = needs_navigation
            frame.cluster_sigs = []  # Initialize empty list for all frames
            frame.sig_index = 0  # Initialize for all frames
            # Add a flag to track if this is a reference signature
            frame.is_reference = sig_path == self.current_reference

            # Add selection capability
            canvas.bind("<Button-1>", lambda e, idx=index: self._toggle_selection(idx, e))
            frame.bind("<Button-1>", lambda e, idx=index: self._toggle_selection(idx, e))
            filename_label.bind("<Button-1>", lambda e, idx=index: self._toggle_selection(idx, e))

            # If signature path provided, load the image
            if sig_path:
                # Always use the actual signature to allow proper ordering in the grid
                frame.displayed_signature = sig_path

                # For discovery mode, the displayed signature
                # is already the reference if from a cluster
                if self.current_mode == "DISCOVERY" and frame.cluster_id is not None:
                    # Add "[C]" prefix for cluster reference
                    frame.filename_label.config(text=f"[C] {os.path.basename(sig_path)}")
                # For verification mode, just use the provided signature
                elif self.current_mode == "VERIFICATION":
                    # Add "[C]" prefix for verification mode signatures (all are from clusters)
                    frame.filename_label.config(text=f"[C] {os.path.basename(sig_path)}")
                else:
                    # For completion mode and other cases with clusters
                    if frame.cluster_id is not None:
                        # Add "[C]" prefix to indicate it's from a cluster
                        frame.filename_label.config(text=f"[C] {os.path.basename(sig_path)}")

                        # Get ordered signatures for clustered frames
                        if frame.cluster_id in self.cluster_ordered_signatures:
                            frame.cluster_sigs = \
                                self.cluster_ordered_signatures[frame.cluster_id].copy()
                        else:
                            frame.cluster_sigs = \
                                self._get_cluster_signatures_by_similarity(frame.cluster_id)
                            # Cache the ordered signatures
                            if hasattr(self, 'cluster_ordered_signatures'):
                                self.cluster_ordered_signatures[frame.cluster_id] = \
                                    frame.cluster_sigs.copy()

                        # Find index in ordered list if possible
                        try:
                            frame.sig_index = frame.cluster_sigs.index(sig_path)
                        except (ValueError, IndexError):
                            frame.sig_index = 0
                    else:
                        # Unclustered signature - just use as is
                        frame.filename_label.config(text=os.path.basename(sig_path))

                # CRITICAL: Set button commands AFTER frame is fully initialized
                if frame.has_navigation:
                    # Left button configuration
                    if hasattr(frame, 'prev_btn') and frame.prev_btn:
                        frame.prev_btn.config(
                            state=tk.NORMAL if frame.sig_index > 0 else tk.DISABLED,
                            command=lambda f=frame: self._navigate_grid_signature(f, -1)
                        )

                    # Right button configuration
                    if hasattr(frame, 'next_btn') and frame.next_btn:
                        frame.next_btn.config(
                            state=tk.NORMAL if (hasattr(frame, 'cluster_sigs') and
                                            frame.sig_index < len(frame.cluster_sigs) - 1)
                                        else tk.DISABLED,
                            command=lambda f=frame: self._navigate_grid_signature(f, 1)
                        )

                # Show similarity to reference in completion mode
                if self.current_reference and self.current_mode == "COMPLETION":
                    # Use the actual displayed signature for similarity calculation
                    target_sig = frame.displayed_signature
                    distance = self._calculate_distance(self.current_reference, target_sig)

                    if distance is not None:
                        # Convert distance to similarity (0-100%)
                        similarity = self._convert_distance_to_similarity(distance)

                        # Color code based on similarity
                        if similarity >= 0.8:
                            color = "#00FF00"  # Green
                        elif similarity >= 0.6:
                            color = "#2196F3"  # Blue
                        elif similarity >= 0.4:
                            color = "#FF9800"  # Orange
                        else:
                            color = "#F44336"  # Red

                        frame.sim_label.config(text=f"{similarity:.0%} similar", foreground=color)

                # Load and display the image
                self._update_frame_image(frame, frame.displayed_signature)

            return frame

        except Exception as e:
            traceback.print_exc()
            print(f"Error creating signature frame: {e}")
            return None  # Return None on failure

    def _update_frame_image(self, frame, sig_path):
        """
        Update the image displayed in a signature frame with proper positioning of labels
        
        Args:
            frame: The signature frame to update
            sig_path: Path to the signature image to display
        """
        # First, check if the frame still exists
        try:
            # Quick check if frame is still valid by accessing a known attribute
            _ = frame.winfo_children()
        except tk.TclError:
            # Frame has been destroyed, silently return
            return
        except Exception as e:
            # Some other error occurred
            print(f"Error checking frame: {e}")
            return

        # Update displayed signature
        frame.displayed_signature = sig_path

        try:
            # Check if file exists
            if not os.path.exists(sig_path):
                frame.canvas.create_text(self.thumbnail_size[0]//2, self.thumbnail_size[1]//2,
                                    text="File not found", fill="red")
                return

            # Clear canvas - with error handling
            try:
                frame.canvas.delete("all")
            except tk.TclError:
                # Canvas may have been destroyed
                return

            # Get canvas dimensions
            canvas_width = self.thumbnail_size[0]
            canvas_height = self.thumbnail_size[1]

            # Determine if we need the cluster name label (top) and/or reference indicator (bottom)
            is_clustered = hasattr(frame, 'cluster_id') and frame.cluster_id is not None

            # Check if this is a reference signature
            is_reference_signature = False

            if sig_path:  # Make sure we have a valid signature path
                # First check if this is exactly the current reference signature
                if sig_path == self.current_reference:
                    is_reference_signature = True
                # Then, for clustered signatures, check if they're the reference for their cluster
                elif is_clustered and frame.cluster_id:
                    # Skip this check in verification mode
                    if (self.current_mode != "VERIFICATION" and \
                        hasattr(self, 'cluster_displayed_signatures') and
                        frame.cluster_id in self.cluster_displayed_signatures and
                        self.cluster_displayed_signatures[frame.cluster_id] == sig_path):
                        is_reference_signature = True

            # FIXED: Check if rejection should be shown based on mode
            should_show_rejection = self.current_mode in ["COMPLETION", "VERIFICATION"]

            # FIXED: Check individual image rejection status
            # (direct constraint with ANY image in reference cluster)
            is_image_rejected = False
            if should_show_rejection and \
                self.current_reference_cluster in self.clusters and sig_path:

                # Check if this specific image has a constraint
                # with ANY image in the reference cluster
                for ref_sig in self.clusters[self.current_reference_cluster]:
                    if self._has_cannot_link_constraint(ref_sig, sig_path):
                        is_image_rejected = True
                        break

            # FIXED: Check cluster-level rejection status (any image in this cluster is rejected)
            is_cluster_rejected = False
            if should_show_rejection and is_clustered and \
                frame.cluster_id and self.current_reference_cluster:

                if frame.cluster_id != self.current_reference_cluster:
                    is_cluster_rejected = \
                        self._is_cluster_rejected(self.current_reference_cluster, frame.cluster_id)

            # Determine space needed for labels
            cluster_label_height = 20 if is_clustered else 0
            reference_label_height = 18 if is_reference_signature else 0

            # Check if this cell has a "go to" button
            # (only for clusters in discovery/completion mode)
            has_goto_button = (is_clustered and self.current_mode in ["DISCOVERY", "COMPLETION"])
            button_height = 29  # Estimated height of the button

            # Determine top margin based on which is taller - the cluster label or the button
            top_margin = max(cluster_label_height, button_height if has_goto_button else 0)

            # Calculate image area height
            image_area_height = canvas_height - top_margin - reference_label_height
            image_area_y = top_margin  # Start after cluster label

            # MODIFIED: Draw rejection background for both clustered AND unclustered rejected images
            if (is_cluster_rejected or is_image_rejected) and should_show_rejection:
                # Draw light red background for any kind of rejection
                frame.canvas.create_rectangle(
                    0, 0, canvas_width, canvas_height,
                    fill="#ffaaaa", outline="", tags="rejection_bg"
                )

            # Draw cluster name label if needed
            if is_clustered:
                # Get the cluster name
                cluster_display = str(frame.cluster_id)

                # NEW: Check if this cluster is marked as complete
                is_complete = frame.cluster_id in self.complete_clusters

                # Green if complete, Orange if incomplete
                box_color = "#2ECC71" if is_complete else "#FFA000"

                # NEW: Add cluster size for discovery and completion modes
                if self.current_mode in ["DISCOVERY", "COMPLETION"]:
                    # Get the number of signatures in this cluster
                    if frame.cluster_id in self.clusters:
                        cluster_size = len(self.clusters[frame.cluster_id])
                        # Don't modify cluster_display yet -
                        # we'll create the combined display text below
                    else:
                        cluster_size = 0
                else:
                    cluster_size = None  # No size shown for verification mode

                # Check if we need to account for "go to" button space
                button_width = 0
                if has_goto_button:
                    button_width = 52  # Width of "go to" button area

                # Available width for the cluster name badge
                available_width = canvas_width - button_width

                # Create a temporary canvas for text measurement
                measure_canvas = tk.Canvas(self.root, width=available_width, \
                                           height=cluster_label_height)

                # Standard padding around text
                text_padding = 10  # 5px on each side

                # NEW: Create display text with size if applicable
                if self.current_mode in ["DISCOVERY", "COMPLETION"] and cluster_size is not None:
                    full_display = f"{cluster_display} ({cluster_size})"
                else:
                    full_display = cluster_display

                # Create text for measurement using the full display text
                text_id = measure_canvas.create_text(
                    5, cluster_label_height//2,
                    text=full_display,
                    anchor="w",
                    font=("TkDefaultFont", 8, "bold")
                )

                # Get text dimensions
                text_bbox = measure_canvas.bbox(text_id)

                # Determine if truncation is needed
                if text_bbox and (text_bbox[2] - text_bbox[0] + text_padding) > available_width:

                    # Need to truncate - MODIFIED to handle combined name and size
                    if self.current_mode in \
                        ["DISCOVERY", "COMPLETION"] and cluster_size is not None:

                        # For discovery and completion modes with size
                        # Calculate space needed for size portion " (X)"
                        size_text = f" ({cluster_size})"
                        measure_canvas.delete("all")
                        measure_canvas.create_text(
                            0, cluster_label_height//2,
                            text=size_text,
                            font=("TkDefaultFont", 8, "bold")
                        )

                        # Truncate name only
                        for i in range(len(cluster_display) - 1, 2, -1):
                            # Try progressively shorter name versions
                            truncated_name = cluster_display[:i] + "..."
                            truncated_full = f"{truncated_name}{size_text}"

                            # Measure truncated full text
                            measure_canvas.delete("all")
                            trunc_id = measure_canvas.create_text(
                                5, cluster_label_height//2,
                                text=truncated_full,
                                anchor="w",
                                font=("TkDefaultFont", 8, "bold")
                            )

                            trunc_bbox = measure_canvas.bbox(trunc_id)

                            # Check if this fits
                            if trunc_bbox and \
                                (trunc_bbox[2] - trunc_bbox[0] + text_padding) <= available_width:

                                full_display = truncated_full
                                break
                    else:
                        # For verification mode - original truncation behavior
                        for i in range(len(cluster_display) - 1, 2, -1):
                            # Try progressively shorter versions
                            truncated = cluster_display[:i] + "..."

                            # Remove previous measurement
                            measure_canvas.delete("all")

                            # Measure truncated text
                            trunc_id = measure_canvas.create_text(
                                5, cluster_label_height//2,
                                text=truncated,
                                anchor="w",
                                font=("TkDefaultFont", 8, "bold")
                            )

                            trunc_bbox = measure_canvas.bbox(trunc_id)

                            # Check if this fits
                            if trunc_bbox and \
                                (trunc_bbox[2] - trunc_bbox[0] + text_padding) <= available_width:

                                full_display = truncated
                                break

                # Calculate badge width based on text width
                if text_bbox:
                    badge_width = text_bbox[2] - text_bbox[0] + text_padding
                else:
                    badge_width = len(full_display) * 6  # Approx. width based on character count

                # Ensure badge doesn't exceed available width
                badge_width = min(badge_width, available_width)

                # Cleanup measurement canvas
                measure_canvas.destroy()

                # Draw the cluster name badge with color based on completion status
                frame.canvas.create_rectangle(
                    0, 0, badge_width, cluster_label_height,
                    fill=box_color, outline=""
                )

                # Center text in badge
                frame.canvas.create_text(
                    badge_width/2, cluster_label_height/2,
                    text=full_display,
                    fill="white",
                    font=("TkDefaultFont", 8, "bold"),
                    anchor="center"  # Center alignment
                )

            # MODIFICATION: Preprocess the image instead of loading directly
            try:
                # Preprocess the image to crop to signature bounds
                img = self.preprocess_for_display(sig_path)

                if img is None:
                    # Error in preprocessing - try direct loading as fallback
                    img = Image.open(sig_path)

                # Calculate aspect ratio
                orig_width, orig_height = img.size
                img_aspect = orig_width / max(orig_height, 1)  # Avoid division by zero

                # Calculate dimensions to fit in the image area while preserving aspect ratio
                if img_aspect > canvas_width / image_area_height:  # Image is wider than tall
                    new_width = canvas_width
                    new_height = int(new_width / img_aspect)
                else:  # Image is taller than wide
                    new_height = image_area_height
                    new_width = int(new_height * img_aspect)

                # Resize image to fit available space
                img = img.resize((new_width, new_height), \
                                 Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS)

                # Convert to Tkinter-compatible format
                img_tk = ImageTk.PhotoImage(img)
                frame.image_tk = img_tk  # Store reference to prevent garbage collection

                # Calculate position to center the image in the image area
                x_pos = (canvas_width - new_width) // 2
                y_pos = image_area_y + (image_area_height - new_height) // 2

                # Display the image
                frame.canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=img_tk)
            except Exception as e:
                # Image processing error
                print(f"Error processing image {sig_path}: {e}")
                try:
                    # Display error message in the image area
                    frame.canvas.create_text(
                        canvas_width//2,
                        image_area_y + image_area_height//2,
                        text=f"Error: {str(e)[:20]}...",
                        fill="red"
                    )
                except tk.TclError:
                    # Canvas may have been destroyed
                    pass
                return  # Skip the rest of the processing

            # Update filename label with pixel-based truncation
            try:
                filename = os.path.basename(sig_path)

                # Create temporary canvas for text measurement
                temp_canvas = tk.Canvas(self.root, width=1, height=1)

                # Create text for measurement
                text_id = temp_canvas.create_text(0, 0, text=filename, font=("TkDefaultFont", 8))
                text_bbox = temp_canvas.bbox(text_id)

                # Check if truncation is needed
                if text_bbox and (text_bbox[2] - text_bbox[0]) > canvas_width:
                    # Try progressively shorter versions
                    for i in range(len(filename) - 1, 2, -1):
                        truncated = filename[:i] + "..."

                        # Measure truncated version
                        temp_canvas.delete("all")
                        trunc_id = \
                            temp_canvas.create_text(0, 0, text=truncated, font=("TkDefaultFont", 8))
                        trunc_bbox = temp_canvas.bbox(trunc_id)

                        # Check if this fits
                        if trunc_bbox and (trunc_bbox[2] - trunc_bbox[0]) <= canvas_width:
                            filename = truncated
                            break

                # Clean up temp canvas
                temp_canvas.destroy()

                frame.filename_label.config(text=filename)

            except tk.TclError:
                # Label may have been destroyed
                pass

            # Add the "R" indicator at the bottom-right for ANY reference signature
            if is_reference_signature:
                # Calculate position for R indicator - bottom right corner
                oval_size = 16
                margin = 0

                # Position in bottom-right corner with margin
                oval_x = canvas_width - oval_size - margin
                oval_y = canvas_height - oval_size - margin

                # MODIFICATION: Determine color based on whether the reference is user-selected
                is_user_selected = False
                if is_clustered and frame.cluster_id:
                    is_user_selected = frame.cluster_id in self.user_selected_references
                elif self.current_reference_cluster:
                    is_user_selected = \
                        self.current_reference_cluster in self.user_selected_references

                # Yellow for user-selected, blue for automatic
                fill_color = "#FFFF00" if is_user_selected else "#40C4FF"

                # Create the oval indicator
                frame.canvas.create_oval(
                    oval_x, oval_y,
                    oval_x + oval_size, oval_y + oval_size,
                    fill=fill_color, outline="#000000",
                    tags="reference_indicator"
                )

                # Add the "R" text
                frame.canvas.create_text(
                    oval_x + oval_size // 2, oval_y + oval_size // 2,
                    text="R", fill="#000000",
                    font=("TkDefaultFont", 8, "bold"),
                    tags="reference_indicator"
                )

            # FIXED: If this specific image is rejected, add
            # red border (only in completion/verification modes)
            if is_image_rejected and should_show_rejection:
                if frame.selected:
                    # Selected and rejected - use thicker border and add red rectangle in canvas
                    frame.config(relief="groove", borderwidth=4)
                else:
                    # Just rejected - use normal border and add red rectangle in canvas
                    frame.config(relief="solid", borderwidth=2)

                # Draw red rectangle along the inside edge of the canvas
                frame.canvas.create_rectangle(
                    3, 3, canvas_width-3, canvas_height-3,
                    outline="red", width=2, tags="rejection_indicator"
                )
            else:
                # Normal styling (not rejected)
                if frame.selected:
                    frame.config(relief="groove", borderwidth=4)
                else:
                    frame.config(relief="solid", borderwidth=2)

            # Update navigation buttons if they exist
            if (hasattr(frame, 'has_navigation') and frame.has_navigation and
                hasattr(frame, 'cluster_id') and frame.cluster_id):

                # Get ordered signatures for this cluster
                if hasattr(self, 'cluster_ordered_signatures') and \
                    frame.cluster_id in self.cluster_ordered_signatures:

                    ordered_sigs = self.cluster_ordered_signatures[frame.cluster_id]

                else:
                    ordered_sigs = self._get_cluster_signatures_by_similarity(frame.cluster_id)
                    if ordered_sigs and hasattr(self, 'cluster_ordered_signatures'):
                        self.cluster_ordered_signatures[frame.cluster_id] = ordered_sigs.copy()

                # Update navigation if we have ordered signatures
                if ordered_sigs:
                    try:
                        # Store the ordered signatures in the frame for navigation
                        frame.cluster_sigs = ordered_sigs

                        # Find current position in ordered list
                        frame.sig_index = ordered_sigs.index(sig_path)

                        # Update navigation buttons
                        if hasattr(frame, 'prev_btn') and frame.prev_btn:
                            # Check if button still exists
                            try:
                                frame.prev_btn.winfo_exists()
                                frame.prev_btn.config(state=tk.NORMAL if \
                                                      frame.sig_index > 0 else tk.DISABLED)
                            except tk.TclError:
                                # Button has been destroyed
                                pass

                        if hasattr(frame, 'next_btn') and frame.next_btn:
                            # Check if button still exists
                            try:
                                frame.next_btn.winfo_exists()
                                frame.next_btn.config(state=tk.NORMAL if frame.sig_index < \
                                                      len(ordered_sigs) - 1 else tk.DISABLED)
                            except tk.TclError:
                                # Button has been destroyed
                                pass

                    except ValueError:
                        # Signature not found in ordered list
                        print(f"WARNING: Signature {os.path.basename(sig_path)} not "\
                              f"found in ordered list for cluster {frame.cluster_id}")
                        if hasattr(frame, 'prev_btn') and frame.prev_btn:
                            frame.prev_btn.config(state=tk.DISABLED)
                        if hasattr(frame, 'next_btn') and frame.next_btn:
                            frame.next_btn.config(state=tk.DISABLED)
                    except Exception as e:
                        # Some other error
                        print(f"Error updating navigation buttons: {e}")

        except Exception as e:
            # Catch-all for any other errors
            print(f"Error updating frame image: {e}")
            try:
                frame.canvas.delete("all")
                frame.canvas.create_text(self.thumbnail_size[0]//2, self.thumbnail_size[1]//2,
                                    text=f"Error: {str(e)[:20]}...", fill="red")
            except tk.TclError:
                # Canvas may have been destroyed
                pass

    def _go_to_cluster_in_completion_mode(self, cluster_id):
        """
        Handle 'go to' button click - switch to completion mode and open the specified cluster
        
        Args:
            cluster_id: The ID of the cluster to open
        """
        # Make sure the cluster exists
        if cluster_id not in self.clusters:
            print(f"ERROR: Cluster {cluster_id} not found")
            return

        print(f"Going to cluster {cluster_id} in completion mode")

        # Set flag to prevent unnecessary calculations
        self._switching_to_specific_cluster = True
        self._target_cluster_id = cluster_id

        # Check if we're already in completion mode - if so, preserve search UI state
        already_in_completion_mode = self.current_mode == "COMPLETION"

        # Switch to completion mode if not already in it
        if not already_in_completion_mode:
            self.mode_var.set("COMPLETION")
            # This will call _change_mode() which handles mode switching
            # But we need to handle reference update after mode change
            self._change_mode()

        # IMPORTANT: Use the user-selected reference if available
        if cluster_id in self.cluster_displayed_signatures and \
            self.cluster_displayed_signatures[cluster_id] in self.clusters[cluster_id]:
            # Use the user's previously selected reference for this cluster
            self.current_reference = self.cluster_displayed_signatures[cluster_id]
            print(f"Using user-selected reference: {os.path.basename(self.current_reference)}")
        else:
            # No user-selected reference exists, get ordered signatures
            if cluster_id in self.cluster_ordered_signatures:
                ordered_sigs = self.cluster_ordered_signatures[cluster_id]
                if ordered_sigs:
                    # Use most representative signature as default
                    self.current_reference = ordered_sigs[0]
                    # Store this as the displayed signature
                    self.cluster_displayed_signatures[cluster_id] = self.current_reference
                    print(f"Using most representative signature as reference: " \
                          f"{os.path.basename(self.current_reference)}")
                else:
                    # If no ordering was possible, use the first signature
                    self.current_reference = self.clusters[cluster_id][0]
                    self.cluster_displayed_signatures[cluster_id] = self.current_reference
                    print(f"Using first signature as reference: " \
                          f"{os.path.basename(self.current_reference)}")
            else:
                # Calculate ordered signatures
                ordered_sigs = self._get_cluster_signatures_by_similarity(cluster_id)
                if ordered_sigs:
                    # Cache for future use
                    self.cluster_ordered_signatures[cluster_id] = ordered_sigs.copy()
                    # Use most representative signature as default
                    self.current_reference = ordered_sigs[0]
                    # Store this as the displayed signature
                    self.cluster_displayed_signatures[cluster_id] = self.current_reference
                    print(f"Using most representative signature as " \
                          f"reference: {os.path.basename(self.current_reference)}")
                else:
                    # If no ordering was possible, use the first signature
                    self.current_reference = self.clusters[cluster_id][0]
                    self.cluster_displayed_signatures[cluster_id] = self.current_reference
                    print(f"Using first signature as reference: " \
                          f"{os.path.basename(self.current_reference)}")

        self.current_reference_cluster = cluster_id
        self.current_displayed_signature = self.current_reference

        # Update reference display
        self._update_reference_display()

        # IMPORTANT: For the grid search, apply last applied values in appearance and function
        if hasattr(self, 'last_applied_grid_membership'):
            # Update the UI to match the last applied values
            self.membership_var.set(self.last_applied_grid_membership)
            self.grid_filter_var.set(self.last_applied_grid_filter)
            self.sort_completion_var.set(self.last_applied_grid_sort)
            self.use_name_query_var.set(self.last_applied_grid_use_name_query)
            self.name_query_var.set(self.last_applied_grid_name_query)
            self.rejection_filter_var.set(self.last_applied_rejection_filter)

            # Update states based on selected values
            self._handle_membership_change()
            self._update_name_query_entry_state()
            self._update_name_query_checkbox_state()
        else:
            # If no last applied values exist, use defaults
            self._apply_grid_search_parameters(use_defaults=True)

        # For completion mode, pre-extract features for consistency
        self.status_var.set("Pre-processing reference and candidates...")
        self.root.update()

        # First, ensure the reference signature has its features extracted
        if self.current_reference not in self.features_cache:
            self._extract_features_for_signatures([self.current_reference])

        # Pre-extract features for a batch of unclustered signatures
        sample_size = min(500, len(self.unclustered_signatures))
        if sample_size > 0:
            batch = random.sample(self.unclustered_signatures, sample_size)
            self._extract_features_for_signatures(batch)

            # Create initial ranking
            self._rank_all_candidates_for_reference()

        # If already in completion mode, use the last applied UI values
        if already_in_completion_mode:
            # For the cluster selector, apply last applied values in appearance and function
            if hasattr(self, 'last_applied_search_text') and \
                hasattr(self, 'last_applied_filter_type') and \
                    hasattr(self, 'last_applied_sort_option'):

                # Update the UI components to reflect the last applied values
                self.search_var.set(self.last_applied_search_text)
                self.search_filter_var.set(self.last_applied_filter_type)
                self.sort_var.set(self.last_applied_sort_option)

                # Apply the values to the cluster selector
                self._populate_cluster_selector(
                    self.last_applied_search_text,
                    self.last_applied_filter_type,
                    self.last_applied_sort_option
                )
            else:
                # For switching modes, use default sort option
                self._populate_cluster_selector("", "Incomplete", "Visual Similarity")

        # Refresh the grid to show related signatures
        self._refresh_grid()

        # Clear the flag after processing
        self._switching_to_specific_cluster = False
        self._target_cluster_id = None

        self.status_var.set(f"Switched to cluster {cluster_id} in completion mode")

    def _navigate_grid_signature(self, frame, direction):
        """
        Navigate to next/previous signature in a grid signature's cluster
        
        Args:
            frame: The signature frame
            direction: +1 for next, -1 for previous
        """
        try:
            # Basic validation
            if not hasattr(frame, 'cluster_id') or not frame.cluster_id:
                return

            if not hasattr(frame, 'cluster_sigs') or not frame.cluster_sigs:
                # If we don't have ordered signatures, get them
                if frame.cluster_id in self.cluster_ordered_signatures:
                    frame.cluster_sigs = self.cluster_ordered_signatures[frame.cluster_id].copy()
                else:
                    # Calculate and store ordered signatures
                    frame.cluster_sigs = \
                        self._get_cluster_signatures_by_similarity(frame.cluster_id)
                    if hasattr(self, 'cluster_ordered_signatures'):
                        self.cluster_ordered_signatures[frame.cluster_id] = \
                            frame.cluster_sigs.copy()

            if not frame.cluster_sigs:  # Still no signatures
                return

            # Calculate new index
            new_index = frame.sig_index + direction

            # Check bounds
            if new_index < 0 or new_index >= len(frame.cluster_sigs):
                return

            # Update index
            frame.sig_index = new_index

            # Get the new signature to display
            new_sig = frame.cluster_sigs[new_index]

            # Update navigation buttons if they exist
            if hasattr(frame, 'has_navigation') and frame.has_navigation:
                if hasattr(frame, 'prev_btn') and frame.prev_btn:
                    try:
                        frame.prev_btn.config(state=tk.NORMAL if new_index > 0 else tk.DISABLED)
                    except tk.TclError:
                        # Button may have been destroyed
                        pass

                if hasattr(frame, 'next_btn') and frame.next_btn:
                    try:
                        frame.next_btn.config(state=tk.NORMAL if new_index < \
                                              len(frame.cluster_sigs) - 1 else tk.DISABLED)
                    except tk.TclError:
                        # Button may have been destroyed
                        pass

            # Update displayed image
            self._update_frame_image(frame, new_sig)

        except Exception as e:
            # Log error and continue
            print(f"Error navigating grid signature: {e}")

    def _toggle_selection(self, index, event=None):
        """
        Toggle selection of a signature in the grid with support for shift-click
        
        Args:
            index: Index of the signature frame
            event: The mouse event (can be None if called programmatically)
        """
        print(f"Toggle selection called for index {index}, mode: {self.current_mode}")

        if index >= len(self.signature_frames):
            print(f"Error: Index {index} out of range for " \
                  f"signature_frames (len={len(self.signature_frames)})")
            return

        frame = self.signature_frames[index]
        if not hasattr(frame, 'signature_path') or not frame.signature_path:
            print(f"Error: Frame at index {index} has no signature_path")
            return

        # Ensure we can access the current signature list
        if not hasattr(self, 'current_grid_signatures') or not self.current_grid_signatures:
            print("Error: No current_grid_signatures available")
            return

        # Check for platform-specific modifier keys if event is provided
        has_control_modifier = False
        has_shift_modifier = False

        if event is not None:
            # Platform detection
            is_macos = sys.platform.startswith('darwin')

            if is_macos:
                # On macOS, check for Command key (⌘) and Shift key
                has_control_modifier = event.state & 0x8  # Command key on macOS
                has_shift_modifier = event.state & 0x1  # Shift key
            else:
                # On Windows/Linux, check for Control key and Shift key
                has_control_modifier = event.state & 0x4  # Control key on Windows/Linux
                has_shift_modifier = event.state & 0x1  # Shift key

        # Handle shift-click selection
        if has_shift_modifier and self.last_selected_index is not None:
            # Determine range of cells to select
            start_idx = min(self.last_selected_index, index)
            end_idx = max(self.last_selected_index, index)

            # Select all cells in the range
            for i in range(start_idx, end_idx + 1):
                if i < len(self.signature_frames):
                    range_frame = self.signature_frames[i]

                    # Skip frames that don't have a signature path
                    if not hasattr(range_frame, 'signature_path') or not range_frame.signature_path:
                        continue

                    # Select this cell if not already selected
                    if not range_frame.selected:
                        range_frame.selected = True

                        # Add to selected signatures if not already there
                        if range_frame.signature_path not in self.selected_signatures:
                            self.selected_signatures.append(range_frame.signature_path)

                        # Update visual appearance for selection
                        range_frame.config(relief="groove", borderwidth=4)

            # Don't update last_selected_index when shift-clicking
            # This allows for extending the selection in both directions

            print(f"Shift-clicked from index {self.last_selected_index} to {index}")
            print(f"Selected {len(self.selected_signatures)} signatures total")
            return

        # Regular click behavior (no shift key)
        if has_control_modifier or not frame.selected:
            # Toggle selection
            frame.selected = not frame.selected

            # Log selection state
            if frame.selected:
                print(f"Selected signature: {os.path.basename(frame.signature_path)}")
                # Add to selected signatures if not already there
                if frame.signature_path not in self.selected_signatures:
                    self.selected_signatures.append(frame.signature_path)
                # Update visual appearance for selection
                frame.config(relief="groove", borderwidth=4)  # More pronounced selection
            else:
                print(f"Deselected signature: {os.path.basename(frame.signature_path)}")
                # Remove from selected signatures
                if frame.signature_path in self.selected_signatures:
                    self.selected_signatures.remove(frame.signature_path)
                # Use consistent styling for deselected cells
                frame.config(relief="solid", borderwidth=2)

            # Update last_selected_index for regular clicks
            if frame.selected:
                self.last_selected_index = index
            elif self.last_selected_index == index:
                # If we deselected the last selected cell, reset the tracker
                self.last_selected_index = None

        # Log current selection count
        print(f"Total selected signatures: {len(self.selected_signatures)}")

    def _toggle_selection_of_focused(self):
        """Toggle selection of focused element (for keyboard shortcut)"""
        focused_widget = self.root.focus_get()
        for frame in self.signature_frames:
            if focused_widget in (frame, frame.canvas, frame.filename_label):
                # For spacebar, we want to toggle regardless of current state
                # This preserves the original spacebar behavior

                currently_selected = frame.selected
                frame.selected = not currently_selected

                # Update visual appearance
                if frame.selected:
                    frame.config(relief="groove", borderwidth=4)
                    if frame.signature_path not in self.selected_signatures:
                        self.selected_signatures.append(frame.signature_path)
                else:
                    # Reset to normal appearance
                    if hasattr(frame, 'cluster_id') and frame.cluster_id is not None:
                        frame.config(relief="ridge", borderwidth=2)
                    else:
                        frame.config(relief="solid", borderwidth=2)

                    if frame.signature_path in self.selected_signatures:
                        self.selected_signatures.remove(frame.signature_path)

                break

    def _group_selected(self):
        """
        FIXED VERSION: Group selected signatures with proper discovery grid placement
        """
        if not self.selected_signatures:
            messagebox.showinfo("No Selection", "Please select signatures to group")
            return

        try:

            # Capture current scroll position
            scroll_position = self._get_current_scroll_position()

            # Store current page to maintain position after operation
            current_page = self.current_page[self.current_mode]

            # For COMPLETION mode, use the current reference cluster as target
            target_cluster = self.current_reference_cluster

            # For completion mode, we need a target cluster
            if not target_cluster:
                # If no reference cluster, create a new one
                self._create_new_cluster()
                return

            # Track which clusters and unclustered signatures are involved
            involved_clusters = set()  # All clusters involved other than target
            unclustered_count = 0      # Count of unclustered signatures being added

            # Track the original position of signatures being moved
            signature_positions = {}
            if hasattr(self, 'lazy_discovery_arranged'):
                for sig in self.selected_signatures:
                    try:
                        pos = self.lazy_discovery_arranged.index(sig)
                        signature_positions[sig] = pos
                    except ValueError:
                        pass

            # Count how many clusters and unclustered signatures are involved
            for sig in self.selected_signatures:
                # Skip the reference itself
                if sig == self.current_reference:
                    continue

                # First check if this signature is part of another cluster
                found_in_other_cluster = False
                for cid, cluster_sigs in self.clusters.items():
                    # Skip current reference cluster
                    if cid == target_cluster:
                        continue

                    if sig in cluster_sigs:
                        # This is a clustered signature
                        found_in_other_cluster = True

                        # If we haven't processed this cluster yet
                        if cid not in involved_clusters:
                            involved_clusters.add(cid)
                        break

                # If not in another cluster, check if it's unclustered
                if not found_in_other_cluster and sig in self.unclustered_signatures:
                    unclustered_count += 1

            # CRITICAL FLAG: Determine if this is a multi-cluster merge
            # True when we're merging 2+ existing clusters
            merging_multiple_clusters = len(involved_clusters) >= 1

            # Check if we need to reconcile cluster names
            if source_clusters := involved_clusters:  # Python 3.8+ assignment expression
                # For completion mode with multiple clusters,
                # act as though the "New Cluster (N)" button was selected
                # UNLESS we're adding to the current cluster
                if len(source_clusters) > 0 and target_cluster != self.current_reference_cluster:
                    self.selected_signatures.insert(0, self.current_reference)
                    self._create_new_cluster()
                    return

            # Extended handling for completion mode when adding to current cluster
            # Now supports both unclustered signatures AND cluster merging
            if (self.current_mode == "COMPLETION" and \
                target_cluster == self.current_reference_cluster):
                # Note: Removed the involved_clusters == 0 condition to allow cluster merging

                # PERFORM THE ACTUAL CLUSTER MERGING AND SIGNATURE ADDITION
                # This replicates the logic from the main flow

                self._reset_lazy_loading_state("COMPLETION")

                # Add signatures to cluster
                if target_cluster not in self.clusters:
                    self.clusters[target_cluster] = []

                # Track which signatures were actually added and which should be removed from grid
                added_signatures = []  # All signatures added to target cluster
                signatures_to_remove_from_grid = []  # Only originally selected signatures
                merged_clusters = []

                # Check each selected signature (this is the core merging logic)
                for sig in self.selected_signatures:
                    # First check if this signature is part of another cluster
                    source_cluster = None
                    for cid, cluster_sigs in list(self.clusters.items()):
                        if cid != target_cluster and sig in cluster_sigs:
                            source_cluster = cid
                            break

                    if source_cluster:
                        # This signature is part of another cluster
                        # Merge that cluster into the target cluster
                        source_signatures = self.clusters[source_cluster].copy()

                        # Add all signatures from source cluster to target
                        for source_sig in source_signatures:
                            if source_sig not in self.clusters[target_cluster]:
                                self.clusters[target_cluster].append(source_sig)
                                added_signatures.append(source_sig)

                        # Only the originally selected signature should be removed from grid
                        signatures_to_remove_from_grid.append(sig)

                        # Record that we merged this cluster
                        if source_cluster not in merged_clusters:
                            merged_clusters.append(source_cluster)

                    else:

                        # Regular case - just add this signature
                        if sig not in self.clusters[target_cluster]:
                            self.clusters[target_cluster].append(sig)
                            added_signatures.append(sig)
                            signatures_to_remove_from_grid.append(sig)

                            # Remove from unclustered
                            if sig in self.unclustered_signatures:
                                self.unclustered_signatures.remove(sig)

                # Clear constraints between all members of the target cluster
                self._clear_constraints_between_members(self.clusters[target_cluster])

                # Remove the merged clusters and their completion status
                for cid in merged_clusters:
                    if cid in self.complete_clusters:
                        self.complete_clusters.remove(cid)

                    if cid in self.user_selected_references:
                        self.user_selected_references.remove(cid)

                    # Debug info for removed reference
                    if cid in self.cluster_displayed_signatures:
                        ref_sig = self.cluster_displayed_signatures[cid]
                        print("Removed reference from merged cluster "
                              f"{cid}: {os.path.basename(ref_sig)}")

                    # Delete the source cluster
                    del self.clusters[cid]

                # Always remove signatures from grid when they're grouped into a cluster
                if signatures_to_remove_from_grid:
                    # Remove signatures from grid without full refresh
                    self._remove_signatures_from_completion_grid(signatures_to_remove_from_grid, {})

                # Update discovery grid - handle both individual signatures and merged clusters
                if hasattr(self, 'lazy_discovery_arranged') and signatures_to_remove_from_grid:
                    # Track what signatures were moved (use the originally selected signatures)
                    removed_by_cluster = {}

                    # For merged clusters, track what was moved
                    if merged_clusters:
                        for cluster_id in merged_clusters:
                             # Everything was moved to new cluster
                            removed_by_cluster[cluster_id] = []

                    # Handle the grid alteration - this will
                    # remove the originally selected signatures
                    # from the discovery grid
                    self._handle_grid_alteration_discovery(target_cluster,
                                                           added_signatures,
                                                           removed_by_cluster)

                    # Handle completion mode grid alteration
                    self._handle_completion_grid_alteration(signatures_to_remove_from_grid)

                # Determine if we should recalculate the reference signature
                # Recalculate if we merged clusters OR added unclustered signatures,
                # and the cluster doesn't have a user-selected reference
                should_recalculate = (
                    len(merged_clusters) > 0 and target_cluster not in self.user_selected_references
                ) or (
                    len(merged_clusters) == 0 and len(added_signatures) > 0 and \
                    target_cluster not in self.user_selected_references
                )

                # Special case: If we're merging clusters, always recalculate
                # and remove user-selected reference status
                if len(merged_clusters) > 0:
                    if target_cluster in self.user_selected_references:
                        print("Removing user-selected reference status "
                              f"for merged cluster: {target_cluster}")
                        self.user_selected_references.remove(target_cluster)
                    should_recalculate = True

                if should_recalculate:
                    # Force recalculation of ordered signatures
                    ordered = self._get_cluster_signatures_by_similarity(target_cluster,
                                                                         force_recalculate=True)

                    # Set the most representative signature as default reference
                    if ordered:
                        # Update the reference
                        self.cluster_displayed_signatures[target_cluster] = ordered[0]

                        # Since this is the current reference cluster,
                        # update the current reference too
                        self.current_reference = ordered[0]
                        self.current_displayed_signature = self.current_reference

                # Update reference display for current cluster
                self._update_reference_display()

                # Update counts and status
                self.clustered_signatures = sum(len(cluster) for cluster in self.clusters.values())
                self._update_progress_display()

                # Update cluster selector if needed
                if self.current_mode in ["COMPLETION", "VERIFICATION"]:
                    self._populate_cluster_selector(self.last_applied_search_text,
                                                    self.last_applied_filter_type,
                                                    self.last_applied_sort_option)

                # Update status message and clear selection
                if merged_clusters:
                    self.status_var.set(f"Added signatures to cluster {target_cluster} "
                                        f"by merging {len(merged_clusters)} clusters")
                else:
                    self.status_var.set(f"Added {len(signatures_to_remove_from_grid)} "
                                        f"signatures to cluster {target_cluster}")

                # Restore scroll position after grid update
                if scroll_position is not None:
                    self._restore_scroll_position(scroll_position)

                self.selected_signatures = []
                self.last_selected_index = None

                return  # Exit early - no full refresh needed

            # If not in completion mode special case...

            # Add signatures to cluster
            if target_cluster not in self.clusters:
                self.clusters[target_cluster] = []

            # Track which signatures were actually added
            added_signatures = []
            merged_clusters = []

            # Check each selected signature
            for sig in self.selected_signatures:
                # First check if this signature is part of another cluster
                source_cluster = None
                for cid, cluster_sigs in list(self.clusters.items()):
                    if cid != target_cluster and sig in cluster_sigs:
                        source_cluster = cid
                        break

                if source_cluster:
                    # This signature is part of another cluster
                    # Merge that cluster into the target cluster
                    source_signatures = self.clusters[source_cluster].copy()

                    # Add all signatures from source cluster to target
                    for source_sig in source_signatures:
                        if source_sig not in self.clusters[target_cluster]:
                            self.clusters[target_cluster].append(source_sig)
                            added_signatures.append(source_sig)

                    # Record that we merged this cluster
                    if source_cluster not in merged_clusters:
                        merged_clusters.append(source_cluster)
                else:
                    # Regular case - just add this signature
                    if sig not in self.clusters[target_cluster]:
                        self.clusters[target_cluster].append(sig)
                        added_signatures.append(sig)

                        # Remove from unclustered
                        if sig in self.unclustered_signatures:
                            self.unclustered_signatures.remove(sig)

            # NEW: Clear constraints between all members of the target cluster
            self._clear_constraints_between_members(self.clusters[target_cluster])

            # Determine if we should recalculate the reference signature
            should_recalculate = (
                merging_multiple_clusters or  # Always recalculate for multi-cluster merges
                (unclustered_count > 0 and target_cluster not in self.user_selected_references)
            )

            if should_recalculate:
                # If we're forcing automatic reference due to multiple cluster merge,
                # always ensure target is not in user_selected_references
                if merging_multiple_clusters:
                    if target_cluster in self.user_selected_references:
                        print("Removing user-selected reference status "
                              f"for merged cluster: {target_cluster}")
                        self.user_selected_references.remove(target_cluster)

                # Force recalculation of ordered signatures
                ordered = self._get_cluster_signatures_by_similarity(target_cluster,
                                                                     force_recalculate=True)

                # Set the most representative signature as default reference
                if ordered:
                    # Update the reference
                    self.cluster_displayed_signatures[target_cluster] = ordered[0]

                    # If this is the current reference cluster, update the current reference too
                    if target_cluster == self.current_reference_cluster:
                        self.current_reference = ordered[0]
                        self.current_displayed_signature = self.current_reference

            # Remove the merged clusters and their completion status
            for cid in merged_clusters:
                if cid in self.complete_clusters:
                    self.complete_clusters.remove(cid)

                if cid in self.user_selected_references:
                    self.user_selected_references.remove(cid)

                # Still get the reference signature for debugging
                if cid in self.cluster_displayed_signatures:
                    ref_sig = self.cluster_displayed_signatures[cid]
                    print("Removed reference from merged cluster "
                          f"{cid}: {os.path.basename(ref_sig)}")

                # Now delete the source cluster
                del self.clusters[cid]

            # Update progress
            self.clustered_signatures = sum(len(cluster) for cluster in self.clusters.values())
            self._update_progress_display()

            # If this is the current reference cluster, update reference display
            if target_cluster == self.current_reference_cluster:
                self._update_reference_display()

            # *** FIXED: UPDATE DISCOVERY GRID LAYOUT WITH PROPER POSITIONING ***
            if hasattr(self, 'lazy_discovery_arranged'):
                # Track what signatures were moved
                removed_by_cluster = {}

                # For merged clusters, track what was moved
                if merged_clusters:
                    for cluster_id in merged_clusters:
                        removed_by_cluster[cluster_id] = []  # Everything was moved to new cluster

                # Handle the grid alteration with proper positioning
                self._handle_grid_alteration_discovery(target_cluster, added_signatures,
                                                       removed_by_cluster)

                # Handle completion mode grid alteration
                if self.current_mode == "COMPLETION":
                    self._handle_completion_grid_alteration(added_signatures)

            # For other modes, try to stay on the same page
            self.current_page[self.current_mode] = current_page

            # Clear cached signature lists for the current mode
            if self.current_mode in self.full_signature_lists:
                self.full_signature_lists[self.current_mode] = []

            # Reset lazy loading for completion mode
            if self.current_mode == "COMPLETION":
                self._reset_lazy_loading_state("COMPLETION")

            # Refresh the grid
            self._refresh_grid()

            # Update status message
            if merged_clusters:
                self.status_var.set(f"Added {len(added_signatures)} signatures to cluster "
                                    f"{target_cluster} by merging {len(merged_clusters)} clusters")
            else:
                self.status_var.set(f"Added {len(added_signatures)} "
                                    f"signatures to cluster {target_cluster}")

            if self.current_mode in ["COMPLETION", "VERIFICATION"]:
                self._populate_cluster_selector(self.last_applied_search_text,
                                                self.last_applied_filter_type,
                                                self.last_applied_sort_option)

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to group signatures: {str(e)}")
            self.status_var.set("Error grouping signatures")

    def _show_cluster_dialog(self, signatures, title="Cluster Creator"):
        """
        Show consolidated dialog for cluster creation, merging, or augmentation.
        
        This method replaces the previous separate methods:
        - _show_new_cluster_dialog
        - _show_add_to_existing_cluster_dialog
        - _show_cluster_merge_dialog
        
        The dialog allows users to:
        - Select/deselect signatures to include in the final cluster
        - Add more clusters from the selector
        - Choose or create a custom name for the resulting cluster
        
        Args:
            signatures: List of signature paths initially selected
            existing_cluster_id: Optional ID of an existing cluster for context
            title: Dialog title
            
        Returns:
            (cluster_name, selected_signatures) tuple or (None, None) if cancelled
        """
        # Identify involved clusters from the provided signatures
        involved_clusters = set()
        for sig in signatures:
            # Check if this signature belongs to an existing cluster
            for cluster_id, cluster_sigs in self.clusters.items():
                if sig in cluster_sigs:
                    involved_clusters.add(cluster_id)
                    break

        # Create name options based on involved clusters
        name_options = []
        if involved_clusters:
            # Add each cluster name as an option
            for cluster_id in sorted(involved_clusters):
                # Skip current reference cluster in verification mode
                if self.current_mode == "VERIFICATION" and \
                    cluster_id == self.current_reference_cluster:

                    continue
                name_options.append((cluster_id, f"Use name: {cluster_id}"))

        # Always add the custom name option
        name_options.append(("custom", "Use new name:"))

        # Call the enhanced dialog with the appropriate parameters
        return self._show_enhanced_cluster_dialog(
            title=title,
            message="Please finalize your cluster.",
            signatures=signatures,
            name_options=name_options
        )

    def _show_enhanced_cluster_dialog(self, title, message, signatures, name_options=None):
        """
        Show an enhanced dialog with horizontally scrollable preview and deselection checkboxes.
        
        Args:
            title: Title for the dialog window
            message: Message text to display at the top
            signatures: List of signature paths to display
            existing_cluster_id: Optional existing cluster ID when adding to existing cluster
            name_options: Optional list of (value, text) tuples for radio button options
                        If None, will show text entry field
        
        Returns:
            (cluster_name, selected_signatures) tuple or (None, None) if cancelled
        """

        def _bind_mousewheel_recursive(widget, canvas, event_handler):
            """Recursively bind mousewheel events to all children of a widget"""
            # Bind the mousewheel event to this widget
            widget.bind("<MouseWheel>", event_handler)
            widget.bind("<Button-4>", event_handler)
            widget.bind("<Button-5>", event_handler)

            # Bind to all children recursively
            for child in widget.winfo_children():
                _bind_mousewheel_recursive(child, canvas, event_handler)

        # Detect platform for hotkey modifiers
        is_mac = sys.platform.startswith('darwin')

        print(f"\n===== STARTING DIALOG: {title} =====")
        print(f"Input signatures: {len(signatures)}")

        if not signatures:
            return None, None, None

        # Create dialog window with larger default size
        dialog_window = tk.Toplevel(self.root)
        dialog_window.title(title)
        dialog_window.geometry("815x710")  # Increased height to accommodate all content
        dialog_window.transient(self.root)  # Set to be on top of the main window
        dialog_window.grab_set()  # Make the dialog modal

        # Create tracking dictionaries for radio buttons
        cluster_to_radio = {}       # Maps cluster IDs to radio button values
        radio_to_cluster = {}       # Maps radio values to cluster IDs

        # Store the result
        result = {'name': None, 'signatures': None}

        # Create main frame with minimal padding
        main_frame = ttk.Frame(dialog_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header message
        ttk.Label(main_frame, text=message, wraplength=750).pack(pady=(0, 0))

        # ==================== PREVIEW SECTION ====================
        # Create frame for signature preview images with horizontal scrolling
        preview_outer_frame = ttk.Frame(main_frame)
        preview_outer_frame.pack(fill=tk.X, pady=0)

        # Create canvas for horizontal scrolling - fixed height, expandable width
        preview_canvas = tk.Canvas(preview_outer_frame, height=220)
        preview_canvas.pack(side=tk.TOP, fill=tk.X, expand=True)

        # Add horizontal scrollbar
        h_scrollbar = ttk.Scrollbar(preview_outer_frame, orient=tk.HORIZONTAL, \
                                    command=preview_canvas.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        preview_canvas.configure(xscrollcommand=h_scrollbar.set)

        # Inner frame for previews - all items will be packed with side=tk.LEFT
        preview_frame = ttk.Frame(preview_canvas)
        preview_canvas.create_window((0, 0), window=preview_frame, anchor=tk.NW)

        # Dictionary to store all signatures for each cluster
        full_cluster_signatures = {}

        # Create a dictionary to store the checkboxes and their associated signatures
        ui_checkbox_vars = {}

        # Track clusters already added to the preview (for disabling "+" buttons)
        added_clusters = set()

        # Group signatures by their clusters
        cluster_signatures = {}
        unclustered = []

        # Identify mode-specific handling for verification mode
        in_verification_mode = self.current_mode == "VERIFICATION"

        # In verification mode, treat all signatures as individual selections
        if in_verification_mode:
            # All signatures should be treated as individual selections
            for sig in signatures:
                unclustered.append(sig)
        else:
            # Normal grouping by cluster for discovery and completion modes
            for sig in signatures:
                found_in_cluster = False
                for cluster_id, cluster_sigs in self.clusters.items():
                    if sig in cluster_sigs:
                        if cluster_id not in cluster_signatures:
                            cluster_signatures[cluster_id] = []
                            full_cluster_signatures[cluster_id] = []

                        # Add this signature to the cluster group if not already there
                        if sig not in cluster_signatures[cluster_id]:
                            cluster_signatures[cluster_id].append(sig)

                        # Add ALL signatures from this cluster to the full set
                        for cluster_sig in cluster_sigs:
                            if cluster_sig not in full_cluster_signatures[cluster_id]:
                                full_cluster_signatures[cluster_id].append(cluster_sig)

                        found_in_cluster = True

                        # Track this cluster as already added to preview
                        added_clusters.add(cluster_id)
                        break

                if not found_in_cluster and sig in self.unclustered_signatures:
                    unclustered.append(sig)

        # DEBUG: Print what we found
        print(f"Found {len(cluster_signatures)} clusters and " \
                f"{len(unclustered)} unclustered signatures")
        for cluster_id, sigs in cluster_signatures.items():
            print(f"  Cluster {cluster_id}: {len(sigs)} selected, " \
                    f"{len(full_cluster_signatures[cluster_id])} total")

        # Create name_var and custom_name_var early
        name_var = tk.StringVar()
        custom_name_var = tk.StringVar()
        custom_option = None
        radio_frame = None  # Initialize to avoid linting error, will be set properly later

        # List to keep track of radio buttons in order for cycling
        radio_buttons_list = []

        # Function to get the first available cluster for Ctrl+G
        def get_first_available_cluster():
            """Returns the first cluster in the selector that can be added (+ button enabled)"""
            for widget in cluster_selector_frame.winfo_children():
                if hasattr(widget, 'add_button') and hasattr(widget, 'cluster_id'):
                    if widget.add_button.cget('state') != 'disabled':
                        return widget.cluster_id, widget.add_button
            return None, None

        # Function to cycle through radio buttons for Ctrl+U
        def cycle_radio_buttons():
            """Cycles to the next radio button in the list"""
            if not radio_buttons_list:
                return

            current_value = name_var.get()

            # Find current index
            current_index = -1
            for i, (value, _) in enumerate(radio_buttons_list):
                if value == current_value:
                    current_index = i
                    break

            # Move to next radio button (wrap around to 0 if at end)
            next_index = (current_index + 1) % len(radio_buttons_list)
            next_value, _ = radio_buttons_list[next_index]
            name_var.set(next_value)

            # If we selected the custom option, focus on the entry field
            if next_value == "custom" and 'custom_entry' in locals():
                custom_entry.focus_set()

        # Function to add a cluster to the preview
        def add_cluster_to_preview(cluster_id, add_btn):
            # Skip if already added
            if cluster_id in added_clusters:
                return

            # Mark as added
            added_clusters.add(cluster_id)

            # Get cluster signatures
            if cluster_id not in self.clusters:
                return

            sigs = self.clusters[cluster_id]
            if not sigs:
                return

            # Create a container for this cluster
            cluster_container = ttk.Frame(preview_frame)
            cluster_container.pack(side=tk.LEFT, padx=10, pady=0)

            # Get the full set of signatures for this cluster
            full_sigs = sigs.copy()
            full_cluster_signatures[cluster_id] = full_sigs

            # Add cluster header - include actual count
            full_count = len(full_sigs)
            ttk.Label(cluster_container,
                    text=f"Cluster: {cluster_id} ({full_count})",
                    font=("TkDefaultFont", 10, "bold")).pack(pady=(0, 0))

            # For each cluster, we only want to show ONE representative signature
            representative_sig = None

            # Prioritize user-selected reference signature
            if hasattr(self, 'cluster_displayed_signatures') and \
                cluster_id in self.cluster_displayed_signatures:

                user_reference = self.cluster_displayed_signatures[cluster_id]
                # Make sure the reference exists in the cluster
                if user_reference in sigs:
                    representative_sig = user_reference
                    print(f"Using user-selected reference " \
                            f"signature for cluster {cluster_id} in dialog")

            # If no user-selected reference was found, fall back to ordered list
            if not representative_sig:
                # Try to find the most representative signature
                ordered_sigs = self._get_cluster_signatures_by_similarity(cluster_id)
                if ordered_sigs:
                    representative_sig = ordered_sigs[0]
                    print(f"Using most representative signature for cluster {cluster_id} in dialog")
                else:
                    # Fallback to first signature
                    representative_sig = sigs[0]
                    print(f"Falling back to first signature for cluster {cluster_id} in dialog")

            if representative_sig:
                # Create frame for the representative signature
                sig_frame = self._create_enhanced_preview_cell(
                    cluster_container,
                    representative_sig,
                    cluster_id,
                    allow_navigation=(not in_verification_mode)
                    # Only allow navigation in non-verification modes
                )
                sig_frame.pack(padx=5, pady=0)

                # Add a checkbox for including ALL signatures from this cluster
                var = tk.BooleanVar(value=True)  # Default to selected

                # Create an ID for this checkbox
                checkbox_id = f"CLUSTER:{cluster_id}"

                # Store checkbox info with ALL signatures from the cluster
                ui_checkbox_vars[checkbox_id] = {
                    'var': var,
                    'signatures': full_sigs,  # Use the COMPLETE set of signatures from this cluster
                    'cluster_id': cluster_id   # Store the cluster ID for easier access
                }

                # Add cluster to radio mapping
                if cluster_id not in cluster_to_radio:
                    cluster_to_radio[cluster_id] = cluster_id
                    radio_to_cluster[cluster_id] = cluster_id

                # Add checkbox below the signature with proper count
                cb = ttk.Checkbutton(sig_frame, \
                                        text=f"Include all {full_count} signatures", variable=var)
                cb.pack(side=tk.BOTTOM)

                # Add option to the name selection if radio_frame exists
                if radio_frame is not None:
                    # Add as radio button option if not already present
                    if cluster_id not in [radio_to_cluster.get(opt) for opt in \
                                            [rb.cget("value") for rb in \
                                            radio_frame.winfo_children() if \
                                            isinstance(rb, ttk.Radiobutton)]]:

                        radio_btn = ttk.Radiobutton(
                            radio_frame,
                            text=f"Use name: {cluster_id}",
                            variable=name_var,
                            value=cluster_id
                        )
                        radio_btn.pack(anchor=tk.W, padx=20, pady=2)

                        # Add to radio buttons list for cycling
                        radio_buttons_list.append((cluster_id, f"Use name: {cluster_id}"))

                        # AUTO-SELECT this newly added radio button
                        name_var.set(cluster_id)

                # Update the preview canvas to fit all content
                preview_frame.update_idletasks()
                preview_canvas.configure(scrollregion=preview_canvas.bbox("all"))

                # Disable all corresponding "+" buttons in the cluster selector
                for btn in cluster_selector_frame.winfo_children():
                    if hasattr(btn, "cluster_id") and btn.cluster_id == cluster_id:
                        if hasattr(btn, "add_button"):
                            btn.add_button.config(state=tk.DISABLED)

            # Disable the cluster's "add" button, since we just added it.
            add_btn.config(state="disabled")

            # Update display to ensure everything is visible
            dialog_window.update_idletasks()

            _bind_mousewheel_recursive(preview_frame, preview_canvas, _on_preview_scroll)

        # First add cluster frames - ONE CELL PER CLUSTER
        for cluster_id, sigs in cluster_signatures.items():
            # Create a container for this cluster
            cluster_container = ttk.Frame(preview_frame)
            cluster_container.pack(side=tk.LEFT, padx=10, pady=0)

            # Get the full set of signatures for this cluster
            full_sigs = full_cluster_signatures[cluster_id]

            # Add cluster header - include actual count
            actual_count = len(sigs)
            full_count = len(full_sigs)

            if full_count > actual_count:
                ttk.Label(cluster_container,
                        text=f"Cluster: {cluster_id} ({actual_count} of {full_count})",
                        font=("TkDefaultFont", 10, "bold")).pack(pady=(0, 0))
            else:
                ttk.Label(cluster_container,
                        text=f"Cluster: {cluster_id} ({full_count})",
                        font=("TkDefaultFont", 10, "bold")).pack(pady=(0, 0))

            # For each cluster, we only want to show ONE representative signature
            representative_sig = None

            # FIXED: Prioritize user-selected reference signature
            # First check if the user has selected a reference signature for this cluster
            if hasattr(self, 'cluster_displayed_signatures') and \
                cluster_id in self.cluster_displayed_signatures:

                user_reference = self.cluster_displayed_signatures[cluster_id]
                # Make sure the reference is part of our selection
                if user_reference in sigs:
                    representative_sig = user_reference
                    print(f"Using user-selected reference signature " \
                            f"for cluster {cluster_id} in dialog")
                elif user_reference in full_sigs:
                    # Reference is in the full set but not in selection - still use it
                    representative_sig = user_reference
                    print(f"Using user-selected reference from full " \
                            f"set for cluster {cluster_id} in dialog")

            # If no user-selected reference was found or it's
            # not in the selection, fall back to ordered list
            if not representative_sig:
                # Try to find the most representative signature
                ordered_sigs = self._get_cluster_signatures_by_similarity(cluster_id)
                if ordered_sigs:
                    # Find first signature in ordered list that's in our selection
                    for sig in ordered_sigs:
                        if sig in sigs:
                            representative_sig = sig
                            print(f"Using most representative signature " \
                                    f"for cluster {cluster_id} in dialog")
                            break

                    # Fallback to first ordered signature if none found
                    if not representative_sig and ordered_sigs:
                        representative_sig = ordered_sigs[0]
                        print(f"Falling back to first ordered " \
                                f"signature for cluster {cluster_id} in dialog")
                else:
                    # Fallback to first signature in the selection
                    representative_sig = sigs[0] if sigs else None
                    print(f"Falling back to first selection signature " \
                            f"for cluster {cluster_id} in dialog")

            if representative_sig:
                # Create frame for the representative signature
                sig_frame = self._create_enhanced_preview_cell(
                    cluster_container,
                    representative_sig,
                    cluster_id,
                    allow_navigation=(not in_verification_mode)
                    # Only allow navigation in non-verification modes
                )
                sig_frame.pack(padx=5, pady=0)

                # Add a checkbox for including ALL signatures from this cluster
                var = tk.BooleanVar(value=True)  # Default to selected

                # Create an ID for this checkbox
                checkbox_id = f"CLUSTER:{cluster_id}"

                # Store checkbox info with ALL signatures from the cluster
                ui_checkbox_vars[checkbox_id] = {
                    'var': var,
                    'signatures': full_sigs,  # Use the COMPLETE set of signatures from this cluster
                    'cluster_id': cluster_id   # Store the cluster ID for easier access
                }

                # Add cluster to radio mapping
                if cluster_id not in cluster_to_radio:
                    cluster_to_radio[cluster_id] = cluster_id
                    radio_to_cluster[cluster_id] = cluster_id

                # Add checkbox below the signature with proper count
                cb = ttk.Checkbutton(sig_frame, \
                                        text=f"Include all {full_count} signatures", variable=var)
                cb.pack(side=tk.BOTTOM)

        # Then add unclustered frames - ONE CELL PER SIGNATURE
        if unclustered:
            # Create a container for unclustered signatures
            unclustered_container = ttk.Frame(preview_frame)
            unclustered_container.pack(side=tk.LEFT, padx=10, pady=0)

            # Add header
            header_text = "Individual Signatures" if \
                in_verification_mode else "Unclustered Signatures"
            ttk.Label(unclustered_container, text=header_text, \
                        font=("TkDefaultFont", 10, "bold")).pack(pady=(0, 0))

            # Create a horizontal frame for all unclustered signatures
            unclustered_frames_container = ttk.Frame(unclustered_container)
            unclustered_frames_container.pack(fill=tk.X)

            # Add each unclustered signature side by side
            for sig in unclustered:
                # Create frame for this signature
                sig_frame = self._create_enhanced_preview_cell(
                    unclustered_frames_container,
                    sig,
                    None,
                    allow_navigation=False  # Never allow navigation for individual signatures
                )
                sig_frame.pack(side=tk.LEFT, padx=5, pady=0)

                # Add a checkbox for this signature
                var = tk.BooleanVar(value=True)  # Default to selected

                # Store this checkbox with just this signature
                ui_checkbox_vars[f"SIGNATURE:{sig}"] = {
                    'var': var,
                    'signatures': [sig]
                }

                # Add checkbox below the signature
                cb = ttk.Checkbutton(sig_frame, text="Include", variable=var)
                cb.pack(side=tk.BOTTOM)

        # ==================== SETUP SCROLLING CORRECTLY ====================
        # First wait for all widgets to be mapped so their sizes are calculated
        preview_frame.update_idletasks()

        # Now configure the scrolling properly
        # Get the total width needed for the preview frame
        preview_width = preview_frame.winfo_reqwidth()

        # Set the canvas scrollregion to the size of the frame
        preview_canvas.config(scrollregion=(0, 0, preview_width, preview_frame.winfo_reqheight()))

        # Set the canvas width to either the frame width or the window width, whichever is smaller
        dialog_window.update_idletasks()  # Ensure dialog window is updated
        canvas_width = min(preview_width, dialog_window.winfo_width() - 60)  # Leave some padding
        preview_canvas.config(width=canvas_width)

        # Add mousewheel scrolling for horizontal preview pane with smooth scrolling
        preview_scroller = SmoothScroller(preview_canvas, axis='x')

        def _on_preview_scroll(event):
            """Handle mousewheel events for horizontal scrolling using smooth scroller"""
            # Let the scroller handle the event
            return preview_scroller.handle_scroll_event(event)

        # Now bind to all children in the preview frame recursively
        _bind_mousewheel_recursive(preview_frame, preview_canvas, _on_preview_scroll)

        def _on_preview_scroll(event):
            """Handle mousewheel events for horizontal scrolling using smooth scroller"""
            # Let the scroller handle the event
            return preview_scroller.handle_scroll_event(event)

        # Bind mousewheel events to the preview canvas and frame
        preview_canvas.bind("<MouseWheel>", _on_preview_scroll)
        preview_canvas.bind("<Button-4>", _on_preview_scroll)
        preview_canvas.bind("<Button-5>", _on_preview_scroll)

        # Bind mousewheel events to the scrollbar too
        h_scrollbar.bind("<MouseWheel>", _on_preview_scroll)
        h_scrollbar.bind("<Button-4>", _on_preview_scroll)
        h_scrollbar.bind("<Button-5>", _on_preview_scroll)

        # ========== CREATE HORIZONTAL CONTAINER FOR CLUSTER SELECTOR AND NAME SELECTION ==========
        # Create a container frame to hold both panes side by side
        horizontal_container = ttk.Frame(main_frame)
        horizontal_container.pack(fill=tk.BOTH, expand=True, pady=5)

        # Left side - cluster selector
        left_side = ttk.Frame(horizontal_container, width=400)
        left_side.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        left_side.pack_propagate(False) # Prevent the frame from shrinking to fit its children


        # ==================== CLUSTER SELECTOR SECTION (LEFT SIDE) ====================
        # Create a frame for the cluster selector (similar to completion/verification mode)
        cluster_selector_container = ttk.LabelFrame(left_side, text="Add More Clusters")
        cluster_selector_container.pack(fill=tk.BOTH, expand=True, pady=0)

        # Create search and filter controls
        search_frame = ttk.Frame(cluster_selector_container)
        search_frame.pack(fill=tk.X, pady=0)

        # Search entry
        search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=search_var)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 0))

        # Create search button with magnifying glass icon
        search_btn = ttk.Button(search_frame, text="🔍", width=3)
        search_btn.pack(side=tk.LEFT, padx=(0, 0))

        # Create clear button with X icon
        clear_search_btn = ttk.Button(search_frame, text="✕", width=3)
        clear_search_btn.pack(side=tk.LEFT)

        # Filter and sort frame
        filter_sort_frame = ttk.Frame(cluster_selector_container)
        filter_sort_frame.pack(fill=tk.X, pady=0)

        # Completion dropdown
        ttk.Label(filter_sort_frame, text="Completion:").pack(side=tk.LEFT, padx=(0, 0))

        # Dropdown for filter options
        filter_var = tk.StringVar(value="Incomplete")
        filter_dropdown = ttk.Combobox(
            filter_sort_frame,
            textvariable=filter_var,
            values=["Incomplete", "Complete", "Both"],
            state="readonly",
            width=8
        )
        filter_dropdown.pack(side=tk.LEFT, padx=(0, 5))

        # Sort dropdown
        ttk.Label(filter_sort_frame, text="Sort:").pack(side=tk.LEFT, padx=(0, 0))

        # Dropdown for sort options
        sort_var = tk.StringVar(value="Visual Similarity")
        sort_dropdown = ttk.Combobox(
            filter_sort_frame,
            textvariable=sort_var,
            values=["Visual Similarity", "Query Similarity", "A→Z", "Z→A",
                    "Size (↓)", "Size (↑)", "Path (↓)", "Path (↑)"],
            state="readonly",
            width=11
        )
        sort_dropdown.pack(side=tk.LEFT)

        # Create scrollable frame for cluster list
        cluster_selector_scroll_frame = ttk.Frame(cluster_selector_container)
        cluster_selector_scroll_frame.pack(fill=tk.BOTH, expand=True, pady=0)

        # Create canvas with scrollbar for the cluster list
        cluster_selector_canvas = tk.Canvas(cluster_selector_scroll_frame)
        cluster_selector_scrollbar = ttk.Scrollbar(
            cluster_selector_scroll_frame,
            orient="vertical",
            command=cluster_selector_canvas.yview
        )
        cluster_selector_frame = ttk.Frame(cluster_selector_canvas)

        # Configure the canvas
        cluster_selector_frame.bind(
            "<Configure>",
            lambda _: cluster_selector_canvas.configure(\
                scrollregion=cluster_selector_canvas.bbox("all"))
        )

        cluster_selector_canvas.create_window((0, 0), window=cluster_selector_frame, anchor="nw")
        cluster_selector_canvas.configure(yscrollcommand=cluster_selector_scrollbar.set)

        # Pack the canvas and scrollbar
        cluster_selector_canvas.pack(side="left", fill="both", expand=True)
        cluster_selector_scrollbar.pack(side="right", fill="y")

        # Right side - name selection
        right_side = ttk.Frame(horizontal_container)
        right_side.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # ==================== NAME SELECTION SECTION (RIGHT SIDE) ====================
        # Create selection frame
        selection_frame = ttk.LabelFrame(right_side, text="Cluster Name")
        selection_frame.pack(fill=tk.BOTH, expand=True, pady=0)

        # Function to populate the cluster selector
        def populate_cluster_selector(search_text="", filter_type="Incomplete",
                                        sort_option="Visual Similarity"):
            """
            Populate the cluster selector with available clusters based on search criteria
            Using Canvas for better text control
            
            Args:
                search_text (str): Optional text to filter clusters by name
                filter_type (str): Completion filter type - "Both", "Complete", or "Incomplete"
                sort_option (str): "Visual Similarity", "Query Similarity", etc.
            """
            # Store the last displayed (applied) search parameters
            self.last_displayed_search_text = search_text
            self.last_displayed_filter_type = filter_type
            self.last_displayed_sort_option = sort_option

            # Clear previous items
            for widget in cluster_selector_frame.winfo_children():
                widget.destroy()

            # Create a list to hold all filtered clusters for display
            filtered_clusters = []

            # Create a list to hold filtered clusters with similarity scores
            scored_clusters = []

            # Convert search text to lowercase for case-insensitive matching
            search_text_lower = search_text.lower() if search_text else ""

            # First pass: apply completion filter and filter by search text
            for cluster_id, signatures_from_cluster in self.clusters.items():
                # Skip empty clusters
                if not signatures_from_cluster:
                    continue

                # Apply completion filter (now using lowercase)
                if filter_type == "Complete" and cluster_id not in self.complete_clusters:
                    continue
                elif filter_type == "Incomplete" and cluster_id in self.complete_clusters:
                    continue

                # Convert cluster_id to string and lowercase for matching
                cluster_id_str = str(cluster_id).lower()

                # Filter by search text based on sort option
                if search_text_lower:
                    if sort_option == "Query Similarity":
                        # For Query Similarity sorting, calculate similarity score
                        similarity_score = self._calculate_name_similarity(
                            search_text_lower, cluster_id_str
                        )

                        # Add to scored list with similarity score
                        scored_clusters.append(
                            (cluster_id, signatures_from_cluster, similarity_score))
                    else:
                        # For other sort options, only include exact substring matches
                        if search_text_lower in cluster_id_str:
                            # Add to scored list with placeholder
                            # similarity (we'll sort differently)
                            scored_clusters.append((cluster_id, signatures_from_cluster, 0.0))
                else:
                    # No search text, include all clusters
                    scored_clusters.append((cluster_id, signatures_from_cluster, 0.0))

            # Sort based on selected option with enhanced logic
            if sort_option == "Visual Similarity":
                # NEW: Calculate centroid for Visual Similarity in dialog
                reference_signatures = []

                # Collect signatures from checked items in the dialog
                for data in ui_checkbox_vars.values():
                    if data['var'].get():  # If checkbox is checked
                        reference_signatures.extend(data['signatures'])

                # If no signatures selected, use the initial signatures passed to dialog
                if not reference_signatures:
                    reference_signatures = signatures  # Use the initial signatures

                if reference_signatures:
                    # Calculate centroid of selected signatures
                    feature_vectors = []
                    for sig in reference_signatures:
                        if sig in self.combined_vectors_cache:
                            feature_vectors.append(self.combined_vectors_cache[sig])
                        elif sig in self.features_cache:
                            combined = self._combine_features(self.features_cache[sig])
                            if combined is not None:
                                feature_vectors.append(combined)

                    if feature_vectors:
                        # Calculate centroid - THIS IS THE KEY CHANGE
                        centroid = np.mean(feature_vectors, axis=0)

                        # MODIFIED: Use centroid directly instead of finding closest signature
                        scored_with_similarity = []
                        for cluster_id, signatures_list, _ in scored_clusters:
                            if signatures_list:
                                # Get representative signature
                                rep_sig = None
                                if cluster_id in self.cluster_displayed_signatures:
                                    rep_sig = self.cluster_displayed_signatures[cluster_id]
                                else:
                                    rep_sig = self._find_cluster_representative(signatures_list)

                                if rep_sig:
                                    # Get representative vector
                                    rep_vector = None
                                    if rep_sig in self.combined_vectors_cache:
                                        rep_vector = self.combined_vectors_cache[rep_sig]
                                    elif rep_sig in self.features_cache:
                                        rep_vector = \
                                            self._combine_features(self.features_cache[rep_sig])

                                    if rep_vector is not None:
                                        # Calculate distance from centroid to representative
                                        distance = np.linalg.norm(rep_vector - centroid)
                                        visual_similarity = \
                                            self._convert_distance_to_similarity(distance)
                                        scored_with_similarity.append(
                                            (cluster_id, signatures_list, visual_similarity))

                        # Sort by visual similarity (highest first)
                        scored_with_similarity.sort(key=lambda x: x[2], reverse=True)
                        scored_clusters = scored_with_similarity
                    else:
                        # Fallback to alphabetical if no feature vectors
                        scored_clusters.sort(key=lambda x: str(x[0]).lower())
                else:
                    # Fallback to alphabetical if no reference signatures
                    scored_clusters.sort(key=lambda x: str(x[0]).lower())

            elif sort_option == "Query Similarity" and search_text:
                # Two-tier sorting: exact matches first, then by similarity
                exact_matches = []
                non_matches = []

                for cluster_id, signatures, similarity_score in scored_clusters:
                    cluster_id_str = str(cluster_id).lower()
                    if search_text_lower in cluster_id_str:
                        exact_matches.append((cluster_id, signatures, similarity_score))
                    else:
                        non_matches.append((cluster_id, signatures, similarity_score))

                # Sort exact matches by similarity score (highest first)
                exact_matches.sort(key=lambda x: x[2], reverse=True)
                # Sort non-matches by similarity score (highest first)
                non_matches.sort(key=lambda x: x[2], reverse=True)

                # Combine: exact matches first, then non-matches
                scored_clusters = exact_matches + non_matches
            elif sort_option == "A→Z":
                # Sort alphabetically (A→Z)
                scored_clusters.sort(key=lambda x: str(x[0]).lower())
            elif sort_option == "Z→A":
                # Sort alphabetically (Z→A)
                scored_clusters.sort(key=lambda x: str(x[0]).lower(), reverse=True)
            elif sort_option == "Size (↓)":
                # Size (↓) means smallest first (ascending)
                scored_clusters.sort(key=lambda x: len(x[1]))
            elif sort_option == "Size (↑)":
                # Size (↑) means largest first (descending)
                scored_clusters.sort(key=lambda x: len(x[1]), reverse=True)
            elif sort_option == "Path (↓)":
                if self.current_reference:
                    clusters_with_ref_paths = []
                    for cluster_id, signatures, _ in scored_clusters:
                        if signatures:
                            # Get representative signature
                            rep_sig = None
                            if cluster_id in self.cluster_displayed_signatures:
                                rep_sig = self.cluster_displayed_signatures[cluster_id]
                            else:
                                rep_sig = self._find_cluster_representative(signatures)

                            if rep_sig:
                                clusters_with_ref_paths.append((cluster_id, signatures, rep_sig))
                    clusters_with_ref_paths.sort(key=lambda x: x[2])
                    scored_clusters = clusters_with_ref_paths
                # If no reference, fall back to alphabetical
                else:
                    scored_clusters.sort(key=lambda x: str(x[0]).lower())
            elif sort_option == "Path (↑)":
                if self.current_reference:
                    clusters_with_ref_paths = []
                    for cluster_id, signatures, _ in scored_clusters:
                        if signatures:
                            # Get representative signature
                            rep_sig = None
                            if cluster_id in self.cluster_displayed_signatures:
                                rep_sig = self.cluster_displayed_signatures[cluster_id]
                            else:
                                rep_sig = self._find_cluster_representative(signatures)

                            if rep_sig:
                                clusters_with_ref_paths.append((cluster_id, signatures, rep_sig))
                    clusters_with_ref_paths.sort(key=lambda x: x[2], reverse=True)
                    scored_clusters = clusters_with_ref_paths
                # If no reference, fall back to alphabetical
                else:
                    scored_clusters.sort(key=lambda x: str(x[0]).lower())

            # Convert to filtered list for display
            filtered_clusters = [(cluster_id, signatures) for \
                                    cluster_id, signatures, _ in scored_clusters]

            # Get the width of the canvas for layout calculations
            canvas_width = cluster_selector_canvas.winfo_width()
            if canvas_width <= 1:
                canvas_width = 361  # Default fallback

            # Width for text area (canvas_width minus thumbnail and padding)
            text_width = canvas_width - 135  # 60px thumbnail + 10px padding

            # Create a frame for each filtered cluster
            for cluster_id, sigs in filtered_clusters:
                # Create a container for this cluster
                cluster_container = ttk.Frame(cluster_selector_frame)
                cluster_container.pack(fill=tk.X, expand=True, pady=1)

                # Store cluster_id on the container for easy access
                cluster_container.cluster_id = cluster_id

                # Add a border if this is the current cluster
                if cluster_id == self.current_reference_cluster:
                    cluster_container.configure(style="Selected.TFrame")

                # Add "+" button at the left
                add_button = ttk.Button(
                    cluster_container,
                    text="＋",
                    width=2,
                    command=lambda cid=cluster_id: add_cluster_to_preview(cid, add_button)
                )
                add_button.pack(side=tk.LEFT, padx=(0, 2))

                # Store reference to the button on the container
                cluster_container.add_button = add_button

                # Disable button if cluster already in preview
                if cluster_id in added_clusters:
                    add_button.config(state=tk.DISABLED)

                # Disable the button for the current reference cluster in verification mode
                if self.current_mode == "VERIFICATION" and \
                    cluster_id == self.current_reference_cluster:

                    add_button.config(state=tk.DISABLED)

                # Get a representative signature
                if sigs:
                    representative_sig = None

                    # Prioritize user-selected reference signature
                    if hasattr(self, 'cluster_displayed_signatures') and \
                        cluster_id in self.cluster_displayed_signatures:

                        user_reference = self.cluster_displayed_signatures[cluster_id]
                        # Make sure the reference is part of our selection
                        if user_reference in sigs:
                            representative_sig = user_reference
                        elif user_reference in sigs:
                            # Reference is in the full set but not in selection - still use it
                            representative_sig = user_reference

                    # If no user-selected reference was found or it's
                    # not in the selection, fall back to ordered list
                    if not representative_sig:
                        # Try to find the most representative signature
                        ordered_sigs = self._get_cluster_signatures_by_similarity(cluster_id)
                        if ordered_sigs:
                            # Find first signature in ordered list that's in our selection
                            for sig in ordered_sigs:
                                if sig in sigs:
                                    representative_sig = sig
                                    break

                            # Fallback to first ordered signature if none found
                            if not representative_sig and ordered_sigs:
                                representative_sig = ordered_sigs[0]
                        else:
                            # Fallback to first signature in the selection
                            representative_sig = sigs[0] if sigs else None

                    try:
                        # Create a thumbnail
                        img = self.preprocess_for_display(representative_sig)

                        if img is None:
                            # Error in preprocessing - try direct loading as fallback
                            img = Image.open(representative_sig)

                        img.thumbnail((60, 40))  # Original thumbnail size
                        img_tk = ImageTk.PhotoImage(img)

                        # Store reference to prevent garbage collection
                        cluster_container.image_tk = img_tk

                        # Create a canvas for the image
                        canvas = tk.Canvas(cluster_container, width=60, height=40, bg="#f0f0f0")

                        # Center the image in the canvas horizontally and vertically
                        x_center = 60 // 2
                        y_center = 40 // 2
                        # Calculate image position (centered)
                        x_pos = x_center - img_tk.width() // 2
                        y_pos = y_center - img_tk.height() // 2
                        # Create centered image
                        canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=img_tk)

                        canvas.pack(side=tk.LEFT, padx=2)
                    except Exception:
                        # If image loading fails, show an error placeholder
                        canvas = tk.Canvas(cluster_container, width=60, height=40, bg="#f0f0f0")
                        # Center the error text
                        canvas.create_text(30, 20, text="Error", fill="red")
                        canvas.pack(side=tk.LEFT, padx=2)

                # Create a canvas for text with precise control
                text_canvas = tk.Canvas(cluster_container, width=text_width, \
                                        height=40, highlightthickness=0)
                text_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)

                # Get text components
                cluster_name = str(cluster_id)
                size_text = f"({len(sigs)})"
                # Keep parentheses in the cluster list (only remove in signature count)

                # Add completion checkmark in parentheses if needed
                if cluster_id in self.complete_clusters:
                    completion_mark = "(✓)"
                    text_canvas.create_text(5, 20, text=completion_mark, anchor=tk.W, tags="text")
                    x_offset = 30
                else:
                    x_offset = 5  # No checkmark

                # Create size text at the right edge
                size_id = text_canvas.create_text(text_width-5, 20, text=size_text, \
                                                anchor=tk.E, tags="text")

                # Measure size text width
                size_bbox = text_canvas.bbox(size_id)
                if size_bbox:
                    size_width = size_bbox[2] - size_bbox[0] + 5
                else:
                    size_width = len(size_text) * 10  # Increased fallback estimate

                # Available width for cluster name
                name_width = text_width - x_offset - size_width

                # Calculate if we need to truncate
                # First create text with full name to measure
                name_id = text_canvas.create_text(x_offset, 20, text=cluster_name, \
                                                    anchor=tk.W, tags="measure")
                name_bbox = text_canvas.bbox(name_id)

                if name_bbox and (name_bbox[2] - name_bbox[0]) > name_width:
                    # Text is too long, need to truncate
                    text_canvas.delete(name_id)  # Remove the measuring text

                    # Try different truncation lengths until it fits
                    for length in range(len(cluster_name), 3, -1):
                        truncated = cluster_name[:length] + "..."
                        text_canvas.delete("temp")  # Delete any previous test
                        temp_id = text_canvas.create_text(x_offset, 20, text=truncated, \
                                                            anchor=tk.W, tags="temp")
                        temp_bbox = text_canvas.bbox(temp_id)

                        if temp_bbox and (temp_bbox[2] - temp_bbox[0]) <= name_width:
                            # This length fits, use it
                            text_canvas.delete("temp")
                            text_canvas.create_text(x_offset, 20, text=truncated, \
                                                    anchor=tk.W, tags="text")
                            break
                    else:
                        # If no truncation worked, use minimal text
                        text_canvas.create_text(x_offset, 20, text=cluster_name[:3] + "...", \
                                                anchor=tk.W, tags="text")
                else:
                    # Text fits without truncation, keep it
                    text_canvas.itemconfig(name_id, tags="text")

                # Highlight on hover
                cluster_container.bind("<Enter>", lambda: \
                                        cluster_container.configure(style="Hover.TFrame"))
                cluster_container.bind("<Leave>", lambda: cluster_container.configure(\
                    style="Selected.TFrame" if (cluster_id == self.current_reference_cluster) \
                        else "TFrame"))

                # Store canvas for later reference
                cluster_container.text_canvas = text_canvas

            # Configure styles for hover and selection
            style = ttk.Style()
            style.configure("Hover.TFrame", background="#e0e0e0")
            style.configure("Selected.TFrame", background="#c0c0ff")

            # Show message if no clusters match the filters
            if not filtered_clusters:
                message_label = ttk.Label(cluster_selector_frame, \
                                            text="No matching clusters found", \
                                            foreground="gray", anchor=tk.CENTER)
                message_label.pack(fill=tk.X, expand=True, pady=5)

            # Rebind scrolling for all items after repopulating
            def _bind_scrolling_recursive(widget, canvas, handler):
                widget.bind("<MouseWheel>", handler)
                widget.bind("<Button-4>", handler)
                widget.bind("<Button-5>", handler)
                for child in widget.winfo_children():
                    _bind_scrolling_recursive(child, canvas, handler)

            # Define the handler function
            def _on_mousewheel(event):
                if event.delta > 0 or event.num == 4:
                    cluster_selector_canvas.yview_scroll(-1, "units")
                else:
                    cluster_selector_canvas.yview_scroll(1, "units")
                return "break"

            # Apply bindings to every widget
            _bind_scrolling_recursive(cluster_selector_frame, \
                                        cluster_selector_canvas, _on_mousewheel)

        # Different input based on name_options
        if name_options:
            # Separate regular options from the custom option
            regular_options = []

            for value, text in name_options:
                if value == "custom":
                    custom_option = (value, text)
                else:
                    regular_options.append((value, text))
        else:

            # For unclustered images, start with an empty list of options
            regular_options = []
            # Always provide a custom option
            custom_option = ("custom", "Use new name:")

        # Add custom option with inline entry (outside scroll area)
        if custom_option:
            custom_frame = ttk.Frame(selection_frame)
            custom_frame.pack(fill=tk.X, pady=(0, 0), padx=20)

            # Radio button for custom option
            custom_radio = ttk.Radiobutton(
                custom_frame,
                text=custom_option[1],
                variable=name_var,
                value=custom_option[0]
            )
            custom_radio.pack(side=tk.LEFT, padx=(0, 10))

            # Entry field immediately to the right
            custom_entry = ttk.Entry(custom_frame, textvariable=custom_name_var, width=40)
            custom_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

            # Add custom option to radio buttons list (first in the list)
            radio_buttons_list.insert(0, (custom_option[0], custom_option[1]))

            def update_custom_entry(*_):
                if name_var.get() == "custom":
                    custom_entry.focus_set()

            def on_custom_entry_focus(_):
                # When the user focuses the entry (click/tab), select the radio and enable
                name_var.set("custom")
                custom_entry.config(state="normal")

            # Track changes to radio selection
            name_var.trace_add("write", update_custom_entry)

            custom_entry.bind("<FocusIn>", on_custom_entry_focus)

        # Create frame for scrollable area
        radio_outer_frame = ttk.Frame(selection_frame)
        radio_outer_frame.pack(fill=tk.BOTH, expand=True, pady=0)

        # Create canvas for vertical scrolling with increased height
        # Increased height for vertical layout
        radio_canvas = tk.Canvas(radio_outer_frame, height=300)
        radio_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add vertical scrollbar
        v_scrollbar = ttk.Scrollbar(radio_outer_frame, orient=tk.VERTICAL, \
                                    command=radio_canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        radio_canvas.configure(yscrollcommand=v_scrollbar.set)

        # Inner frame for radio buttons - NOW PROPERLY SETTING radio_frame
        radio_frame = ttk.Frame(radio_canvas)

        # Configure the canvas
        radio_frame.bind(
            "<Configure>",
            lambda e: radio_canvas.configure(scrollregion=radio_canvas.bbox("all"))
        )

        radio_canvas.create_window((0, 0), window=radio_frame, anchor=tk.NW)

        # Add regular radio buttons to the scrollable area
        for value, text in regular_options:
            # Create the radio button
            radio_btn = ttk.Radiobutton(
                radio_frame,
                text=text,
                variable=name_var,
                value=value
            )
            radio_btn.pack(anchor=tk.W, padx=20, pady=2)

            # Add to radio buttons list
            radio_buttons_list.append((value, text))

        # Update scrollregion after radio buttons are added
        radio_frame.update_idletasks()
        radio_canvas.configure(scrollregion=radio_canvas.bbox("all"))

        # AUTO-SELECT the "Use new name:" radio button when dialog opens
        if custom_option:
            name_var.set(custom_option[0])
            # Focus on the custom entry field as well
            dialog_window.after(100, lambda: custom_entry.focus_set())

        # Add smooth scrolling to radio canvas (vertical)
        if regular_options and radio_frame:
            # Create a smooth scroller for the radio canvas
            radio_scroller = SmoothScroller(radio_canvas)

            def _on_radio_mousewheel(event):
                # Let the scroller handle the event
                return radio_scroller.handle_scroll_event(event)

            # Bind mousewheel events
            radio_canvas.bind("<MouseWheel>", _on_radio_mousewheel)
            radio_canvas.bind("<Button-4>", _on_radio_mousewheel)
            radio_canvas.bind("<Button-5>", _on_radio_mousewheel)
            radio_frame.bind("<MouseWheel>", _on_radio_mousewheel)
            radio_frame.bind("<Button-4>", _on_radio_mousewheel)
            radio_frame.bind("<Button-5>", _on_radio_mousewheel)

        # Error message label
        error_var = tk.StringVar()
        error_label = ttk.Label(main_frame, textvariable=error_var, foreground="red")
        error_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 0))

        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=0)

        # Flag to track if dialog was canceled
        dialog_cancelled = {'value': False}

        obsolete_cluster_names = set()

        # Create validate function
        def validate_and_set_result():
            print("\n=== PROCESSING DIALOG SELECTIONS ===")

            # Get the cluster name
            cluster_name = None

            #if name_options:
            selection = name_var.get()

            # Get the desired cluster name
            if selection == "custom" and custom_name_var is not None:
                cluster_name = custom_name_var.get().strip()
            elif selection in radio_to_cluster:
                cluster_name = radio_to_cluster[selection]
            else:
                cluster_name = selection

            # Check whether the user provided a name.
            if not cluster_name:
                error_var.set("Please enter a custom name for the cluster.")
                return

            # Check if name contains only allowed characters
            if not self._is_valid_cluster_name(cluster_name):
                error_var.set("Cluster names may only contain alphanumeric " \
                                "characters (a-z, A-Z, 0-9), underscores (_), " \
                                "hyphens (-), periods (.) and spaces.")
                return

            # If we are in verification mode, ensure that the desired cluster name is not
            # the same as that of the cluster from which we are trying to remove signatures.
            if self.current_mode == "VERIFICATION" and \
                cluster_name.lower() == self.current_reference_cluster.lower():

                error_var.set(f'"{self.current_reference_cluster}" is already the name of ' \
                                'the cluster from which you are trying to remove signatures.')
                return

            # Check if name matches a deselected cluster.
            deselected_cluster_match = None
            for checkbox_id, data in ui_checkbox_vars.items():
                if checkbox_id.startswith("CLUSTER:") and not data['var'].get():
                    # Extract cluster ID from the key format "CLUSTER:cluster_id"
                    key_parts = checkbox_id.split(":", 1)
                    if len(key_parts) == 2:
                        checkbox_cluster_id = key_parts[1]
                        if cluster_name.lower() == checkbox_cluster_id.lower():
                            deselected_cluster_match = checkbox_cluster_id

            # If name matches a deselected cluster, show error with improved message.
            if deselected_cluster_match is not None:
                error_var.set(f'"{deselected_cluster_match}" is a deselected cluster. ' \
                                'Either check the cluster or use a different name.')
                return

            # Check if name is not part of the merge
            existing_in_selection = False
            for checkbox_id, data in ui_checkbox_vars.items():
                if checkbox_id.lower() == f"cluster:{cluster_name.lower()}":
                    if data['var'].get():  # Only if checkbox is checked
                        existing_in_selection = True
                    break

            # Check if cluster name exists
            cluster_name_match = None
            for existing_cluster_name in self.clusters:
                if existing_cluster_name.lower() == cluster_name.lower():
                    cluster_name_match = existing_cluster_name
                    break

            # If name exists but cluster not in dialog selection, show error
            if cluster_name_match is not None and not existing_in_selection:
                error_var.set(f'A cluster with the name "{cluster_name_match}" already exists. ' \
                                'Either add the cluster or use a different name.')
                return

            # Get selected signatures based on checkbox states
            selected_sigs = []

            # Process each checkbox
            print("Processing selected checkboxes:")
            for checkbox_id, data in ui_checkbox_vars.items():
                var = data['var']
                sigs = data['signatures']

                if var.get():  # If checkbox is checked

                    if isinstance(checkbox_id, str) and checkbox_id.startswith("CLUSTER:"):
                        obsolete_cluster_names.add(checkbox_id[checkbox_id.find(':') + 1:])

                    print(f"  Selected: {checkbox_id} - adding {len(sigs)} signatures")
                    selected_sigs.extend(sigs)
                else:
                    print(f"  Not selected: {checkbox_id} - skipping {len(sigs)} signatures")

            obsolete_cluster_names.discard(cluster_name)

            # Remove duplicates while preserving order
            seen = set()
            selected_sigs = [x for x in selected_sigs if not (x in seen or seen.add(x))]

            print(f"Final selection: {len(selected_sigs)} unique signatures")

            if not selected_sigs:
                error_var.set("At least one signature must be selected")
                return

            # Set result and close
            result['name'] = cluster_name
            result['signatures'] = selected_sigs

            # IMPORTANT: Check if the chosen name is an existing
            # complete cluster and signatures are being added
            if selected_sigs and cluster_name in self.complete_clusters:
                # Get the current signatures in the cluster
                current_signatures = set(self.clusters.get(cluster_name, []))

                # Check if we're adding any new signatures
                adding_signatures = False
                for sig in selected_sigs:
                    if sig not in current_signatures:
                        adding_signatures = True
                        break

                # If adding new signatures to a complete cluster, mark it as incomplete
                if adding_signatures:
                    # Remove from complete clusters
                    self.complete_clusters.remove(cluster_name)
                    print(f"Cluster {cluster_name} marked as incomplete " \
                            "after adding signatures in merge dialog")

                    # Update checkbox state if it exists
                    if hasattr(self, 'complete_var') and \
                        self.current_reference_cluster == cluster_name:

                        self.complete_var.set(False)

            dialog_window.destroy()

        # Handle explicit cancel
        def on_cancel():
            dialog_cancelled['value'] = True
            dialog_window.destroy()

        # Cancel button - use on_cancel handler
        ttk.Button(
            button_frame,
            text="Cancel",
            command=on_cancel
        ).pack(side=tk.RIGHT, padx=5)

        # Create button
        create_btn = ttk.Button(
            button_frame,
            text="Create",
            command=validate_and_set_result
        )
        create_btn.pack(side=tk.RIGHT, padx=5)

        # Handle window close event (X button)
        def on_window_close():
            dialog_cancelled['value'] = True
            dialog_window.destroy()

        dialog_window.protocol("WM_DELETE_WINDOW", on_window_close)

        # Hook up search button
        def perform_search():
            search_text = search_var.get().strip()
            filter_type = filter_var.get()
            sort_option = sort_var.get()
            populate_cluster_selector(search_text, filter_type, sort_option)
            dialog_window.focus_set()
            return "break"

        search_btn.config(command=perform_search)

        # Bind Enter key directly to search entry for reliable functionality
        search_entry.bind("<Return>", lambda event: perform_search())

        # Hook up clear button
        def clear_search():
            search_var.set("")
            filter_var.set("Incomplete")
            sort_var.set("Visual Similarity")

        clear_search_btn.config(command=clear_search)

        # Populate initially
        populate_cluster_selector()

        # Add smooth scrolling to cluster selector
        SmoothScroller(cluster_selector_canvas)

        def _on_cluster_selector_scroll(event):
            # Create a handler that uses the cluster_selector_canvas
            cluster_selector_scroller = SmoothScroller(cluster_selector_canvas)
            return cluster_selector_scroller.handle_scroll_event(event)

        # Bind mousewheel events
        cluster_selector_canvas.bind("<MouseWheel>", _on_cluster_selector_scroll)
        cluster_selector_canvas.bind("<Button-4>", _on_cluster_selector_scroll)
        cluster_selector_canvas.bind("<Button-5>", _on_cluster_selector_scroll)
        cluster_selector_frame.bind("<MouseWheel>", _on_cluster_selector_scroll)
        cluster_selector_frame.bind("<Button-4>", _on_cluster_selector_scroll)
        cluster_selector_frame.bind("<Button-5>", _on_cluster_selector_scroll)

        # And bind to all children recursively
        _bind_mousewheel_recursive(cluster_selector_frame, cluster_selector_canvas, \
                                    _on_cluster_selector_scroll)

        # ==================== HOTKEY BINDINGS ====================

        # Define hotkey functions
        def focus_search_entry(_=None):
            """Focus on the search entry (Ctrl+F)"""
            search_entry.focus_set()
            return "break"

        def add_first_cluster(_=None):
            """Add the first available cluster (Ctrl+G)"""
            cluster_id, add_btn = get_first_available_cluster()
            if cluster_id and add_btn:
                add_cluster_to_preview(cluster_id, add_btn)
            return "break"

        def cycle_radio_selection(_=None):
            """Cycle through radio button selections (Ctrl+U)"""
            cycle_radio_buttons()
            return "break"

        # Bind hotkeys to the dialog window
        if is_mac:
            # macOS uses Command key
            dialog_window.bind("<Command-f>", focus_search_entry)
            dialog_window.bind("<Command-g>", add_first_cluster)
            dialog_window.bind("<Command-u>", cycle_radio_selection)
        else:
            # Windows/Linux use Control key
            dialog_window.bind("<Control-f>", focus_search_entry)
            dialog_window.bind("<Control-g>", add_first_cluster)
            dialog_window.bind("<Control-u>", cycle_radio_selection)

        # Set up hotkeys and focus management with callbacks
        search_entries = [search_entry]

        # Create a wrapper for search callback that ensures "break" is returned
        def search_callback_wrapper():
            perform_search()
            return "break"

        search_callbacks = {search_entry: search_callback_wrapper}

        if 'custom_entry' in locals():
            cluster_name_entry = custom_entry
            # Add callback reference for hotkey handler
            custom_entry.update_callback = validate_and_set_result
        else:
            cluster_name_entry = None

        # Set up existing hotkeys and focus management
        self._setup_dialog_hotkeys(dialog_window, create_btn, search_entries, cluster_name_entry)
        self._setup_dialog_focus_management(dialog_window, search_entries, search_callbacks)

        # Wait for the window to be closed
        dialog_window.wait_window()

        # Check if dialog was cancelled
        if dialog_cancelled['value'] or result['name'] is None or result['signatures'] is None:
            print("Dialog cancelled")
            return None, None, None

        # Print final result
        print(f"Dialog returned name: {result['name']} and {len(result['signatures'])} signatures")
        return result['name'], result['signatures'], obsolete_cluster_names

    def _setup_dialog_hotkeys(self, dialog_window, create_btn,
                              search_entries, cluster_name_entry=None):
        """
        Set up hotkey bindings for dialog windows
        
        Args:
            dialog_window: The dialog window
            create_btn: The create/action button
            search_entries: List of search entry widgets
            cluster_name_entry: Optional cluster name entry widget
        """
        def on_entry_key(event, entry_widget, action_callback):
            """Handle enter key press on entry widgets"""
            if event.widget == entry_widget:
                action_callback()
                return "break"
            return None

        def on_dialog_key(_):
            """Handle enter key press on dialog (when not on specific entries)"""
            focused = dialog_window.focus_get()

            # Check if focus is on any search entry
            for search_entry in search_entries:
                if focused == search_entry:
                    return None  # Let the search entry handle it

            # Check if focus is on cluster name entry
            if cluster_name_entry and focused == cluster_name_entry:
                return None  # Let the cluster name entry handle it

            # Otherwise, trigger create button
            create_btn.invoke()
            return "break"

        # Bind enter key to dialog for general case
        dialog_window.bind("<Return>", on_dialog_key)

        # Initial binding for search entries - these will be rebound by focus management
        for search_entry in search_entries:
            if hasattr(search_entry, 'search_callback'):
                search_entry.bind("<Return>",
                    lambda e, entry=search_entry: on_entry_key(e, entry, entry.search_callback))

        # Bind enter key to cluster name entry if provided
        if cluster_name_entry and hasattr(cluster_name_entry, 'update_callback'):
            cluster_name_entry.bind("<Return>",
                lambda e: on_entry_key(e, cluster_name_entry, cluster_name_entry.update_callback))

    def _setup_dialog_focus_management(self, dialog_window, entries, search_callbacks=None):
        """
        Set up focus management for dialog windows to handle clicking away from entries
        
        Args:
            dialog_window: The dialog window
            entries: List of entry widgets to manage focus for
            search_callbacks: Dict mapping entry widgets to their callback functions
        """
        if search_callbacks is None:
            search_callbacks = {}

        def on_dialog_click(event):
            """Handle clicks in dialog to manage focus"""
            clicked_widget = event.widget

            # Check if click was on an entry widget or its children
            clicked_on_entry = None
            for entry in entries:
                if clicked_widget == entry:
                    clicked_on_entry = entry
                    break

                # Check if clicked widget is a child of an entry
                parent = clicked_widget
                while parent:
                    if parent == entry:
                        clicked_on_entry = entry
                        break
                    try:
                        parent = parent.master
                    except Exception:
                        break
                if clicked_on_entry:
                    break

            if clicked_on_entry:
                # Click was on an entry - set focus and rebind hotkey
                clicked_on_entry.focus_set()
                if clicked_on_entry in search_callbacks:
                    # Rebind the enter key for this entry
                    clicked_on_entry.bind("<Return>",
                        lambda e: search_callbacks[clicked_on_entry]())
            else:
                # Click was not on any entry, set focus to dialog and unbind hotkeys
                dialog_window.focus_set()

                # Unbind enter key from all entries since focus moved away
                for entry in entries:
                    entry.unbind("<Return>")

        # Bind click event to dialog and all its children recursively
        def bind_recursive(widget):
            widget.bind("<Button-1>", on_dialog_click, add="+")
            for child in widget.winfo_children():
                bind_recursive(child)

        bind_recursive(dialog_window)

    def _create_enhanced_preview_cell(self, parent, sig_path, \
                                      cluster_id=None, allow_navigation=True):
        """
        Create an enhanced preview cell that mimics the grid display in discovery mode.
        
        Args:
            parent: Parent widget
            sig_path: Path to signature image
            cluster_id: Optional cluster ID this signature belongs to
            allow_navigation: Whether to enable navigation buttons
                
        Returns:
            The frame containing the preview cell
        """
        # Create frame
        frame = ttk.Frame(parent, borderwidth=2, relief="solid")

        # Set size and styling based on whether it's clustered
        if cluster_id is not None:
            frame.cluster_id = cluster_id
            frame.config(relief="ridge")
        else:
            frame.cluster_id = None

        # Canvas for the image - slightly smaller than the grid to fit in dialog
        canvas_width = 150
        canvas_height = 100
        canvas = tk.Canvas(frame, width=canvas_width, height=canvas_height, bg="#f0f0f0")
        canvas.pack(pady=(5, 0))

        # Label for the filename (center)
        filename_label = ttk.Label(frame, text="Loading...", \
                                   font=("TkDefaultFont", 8), anchor=tk.CENTER)
        filename_label.pack(fill=tk.X, expand=True)

        # Determine if this is a clustered signature AND navigation is allowed
        needs_navigation = (cluster_id is not None and allow_navigation)

        # Add navigation buttons if needed
        if needs_navigation:

            # Add a container for filename (and navigation controls if applicable)
            nav_container = ttk.Frame(frame)
            nav_container.pack(fill=tk.X, pady=(2, 0))

            # Left navigation button
            prev_btn = ttk.Button(nav_container, text="←", width=2)
            prev_btn.pack(side=tk.LEFT, padx=1)
            frame.prev_btn = prev_btn

            # Right navigation button
            next_btn = ttk.Button(nav_container, text="→", width=2)
            next_btn.pack(side=tk.RIGHT, padx=1)
            frame.next_btn = next_btn

            # Get ordered signatures for this cluster - FIXED to use ordered signatures
            if cluster_id in self.clusters:
                # FIX: Get ordered signatures by similarity instead of unordered cluster signatures
                if cluster_id in self.cluster_ordered_signatures:
                    # Use cached ordering if available
                    print(f"Using cached ordering for cluster {cluster_id}")
                    ordered_signatures = self.cluster_ordered_signatures[cluster_id]
                else:
                    # Calculate and cache ordered signatures if not available
                    print(f"Calculating ordered signatures for cluster {cluster_id}")
                    ordered_signatures = self._get_cluster_signatures_by_similarity(cluster_id)
                    self.cluster_ordered_signatures[cluster_id] = ordered_signatures.copy()

                frame.cluster_sigs = ordered_signatures  # Store ordered signatures

                # Find the index of the current signature
                try:
                    current_index = frame.cluster_sigs.index(sig_path)
                    frame.sig_index = current_index
                except (ValueError, IndexError):
                    frame.sig_index = 0

                # Set up navigation button commands
                prev_btn.config(
                    state=tk.NORMAL if frame.sig_index > 0 else tk.DISABLED,
                    command=lambda f=frame: self._navigate_preview_cell(f, -1)
                )
                next_btn.config(
                    state=tk.NORMAL if \
                        frame.sig_index < len(frame.cluster_sigs) - 1 else tk.DISABLED, \
                            command=lambda f=frame: self._navigate_preview_cell(f, 1)
                )
            else:
                # No cluster found (shouldn't happen)
                frame.cluster_sigs = [sig_path]
                frame.sig_index = 0
                prev_btn.config(state=tk.DISABLED)
                next_btn.config(state=tk.DISABLED)
        else:

            # Set navigation button attributes to None for consistency
            frame.prev_btn = None
            frame.next_btn = None
            frame.cluster_sigs = [sig_path]
            frame.sig_index = 0

        # Store references to widgets
        frame.canvas = canvas
        frame.filename_label = filename_label
        frame.signature_path = sig_path
        frame.image_tk = None  # To prevent garbage collection
        frame.displayed_signature = sig_path  # Initially display the main signature
        frame.has_navigation = needs_navigation

        # Check if this is a reference signature
        is_reference_signature = False
        if sig_path and sig_path == self.current_reference:
            is_reference_signature = True
        elif cluster_id and hasattr(self, 'cluster_displayed_signatures'):
            if cluster_id in self.cluster_displayed_signatures and \
                self.cluster_displayed_signatures[cluster_id] == sig_path:

                is_reference_signature = True

        # Load and display the image
        if sig_path:
            try:
                # Clear canvas
                canvas.delete("all")

                # Determine space needed for labels
                cluster_label_height = 20 if cluster_id is not None else 0
                reference_label_height = 18 if is_reference_signature else 0

                # Calculate image area height
                image_area_height = canvas_height - cluster_label_height - reference_label_height
                image_area_y = cluster_label_height  # Start after cluster label

                # Draw cluster name label if needed
                if cluster_id is not None:
                    # Get the cluster name
                    cluster_display = str(cluster_id)

                    # NEW: Check if this cluster is marked as complete
                    is_complete = cluster_id in self.complete_clusters

                    # NEW: Set box color based on completion status
                    box_color = "#2ECC71" if is_complete else "#FFA000"
                    # Green if complete, Orange if incomplete

                    # Add cluster size in parentheses for dialogs
                    # These are used in dialog boxes that can
                    # appear in both discovery and completion modes
                    if cluster_id in self.clusters:
                        cluster_size = len(self.clusters[cluster_id])
                        # Create the combined display text
                        full_display = f"{cluster_display} ({cluster_size})"
                    else:
                        full_display = cluster_display

                    # Available width for the cluster name badge -
                    # full width in popup (no "go to" button)
                    available_width = canvas_width

                    # Temporary canvas for text measurement
                    measure_canvas = tk.Canvas(self.root, width=available_width, \
                                               height=cluster_label_height)

                    # Standard padding around text
                    text_padding = 10  # 5px on each side

                    # Create text for measurement using the full display text
                    text_id = measure_canvas.create_text(
                        5, cluster_label_height//2,
                        text=full_display,
                        anchor="w",
                        font=("TkDefaultFont", 8, "bold")
                    )

                    # Get text dimensions
                    text_bbox = measure_canvas.bbox(text_id)

                    # Determine if truncation is needed
                    if text_bbox and (text_bbox[2] - text_bbox[0] + text_padding) > available_width:
                        # Need to truncate - MODIFIED to handle combined name and size
                        # Calculate space needed for size portion " (X)"
                        if cluster_id in self.clusters:
                            size_text = f" ({cluster_size})"
                            measure_canvas.delete("all")
                            measure_canvas.create_text(
                                0, cluster_label_height//2,
                                text=size_text,
                                font=("TkDefaultFont", 8, "bold")
                            )

                            # Truncate name only
                            for i in range(len(cluster_display) - 1, 2, -1):
                                # Try progressively shorter name versions
                                truncated_name = cluster_display[:i] + "..."
                                truncated_full = f"{truncated_name}{size_text}"

                                # Measure truncated full text
                                measure_canvas.delete("all")
                                trunc_id = measure_canvas.create_text(
                                    5, cluster_label_height//2,
                                    text=truncated_full,
                                    anchor="w",
                                    font=("TkDefaultFont", 8, "bold")
                                )

                                trunc_bbox = measure_canvas.bbox(trunc_id)

                                # Check if this fits
                                if trunc_bbox and (trunc_bbox[2] - trunc_bbox[0] + \
                                                   text_padding) <= available_width:
                                    full_display = truncated_full
                                    break
                        else:
                            # Original truncation behavior if no cluster size
                            for i in range(len(cluster_display) - 1, 2, -1):
                                # Try progressively shorter versions
                                truncated = cluster_display[:i] + "..."

                                # Remove previous measurement
                                measure_canvas.delete("all")

                                # Measure truncated text
                                trunc_id = measure_canvas.create_text(
                                    5, cluster_label_height//2,
                                    text=truncated,
                                    anchor="w",
                                    font=("TkDefaultFont", 8, "bold")
                                )

                                trunc_bbox = measure_canvas.bbox(trunc_id)

                                # Check if this fits
                                if trunc_bbox and (trunc_bbox[2] - trunc_bbox[0] + \
                                                   text_padding) <= available_width:
                                    full_display = truncated
                                    break

                    # Calculate badge width based on text width
                    if text_bbox:
                        badge_width = text_bbox[2] - text_bbox[0] + text_padding
                    else:
                        # Approximate width based on character count
                        badge_width = len(full_display) * 6

                    # Ensure badge doesn't exceed available width
                    badge_width = min(badge_width, available_width)

                    # Cleanup measurement canvas
                    measure_canvas.destroy()

                    # Draw the cluster name badge with color based on completion status
                    canvas.create_rectangle(
                        0, 0, badge_width, cluster_label_height,
                        fill=box_color, outline=""
                    )

                    # Center text in badge
                    canvas.create_text(
                        badge_width/2, cluster_label_height/2,
                        text=full_display,
                        fill="white",
                        font=("TkDefaultFont", 8, "bold"),
                        anchor="center"  # Center alignment
                    )

                # Open image and scale to fit the image area
                img = self.preprocess_for_display(sig_path)

                if img is None:
                    # Error in preprocessing - try direct loading as fallback
                    img = Image.open(sig_path)

                # Calculate aspect ratio
                orig_width, orig_height = img.size
                img_aspect = orig_width / max(orig_height, 1)  # Avoid division by zero

                # Calculate dimensions to fit in the image area while preserving aspect ratio
                if img_aspect > canvas_width / image_area_height:  # Image is wider than tall
                    new_width = canvas_width
                    new_height = int(new_width / img_aspect)
                else:  # Image is taller than wide
                    new_height = image_area_height
                    new_width = int(new_height * img_aspect)

                # Resize image
                img = img.resize((new_width, new_height), \
                                 Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS)

                # Convert to PhotoImage
                img_tk = ImageTk.PhotoImage(img)
                frame.image_tk = img_tk  # Keep reference

                # Calculate position to center the image in the image area
                x_pos = (canvas_width - new_width) // 2
                y_pos = image_area_y + (image_area_height - new_height) // 2

                # Display the image
                canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=img_tk)

                # Add reference indicator ("R") if this is a reference signature
                if is_reference_signature:
                    # Calculate position for R indicator -
                    # bottom right corner using FIXED positioning
                    oval_size = 16
                    h_margin = 0  # Horizontal margin
                    v_margin = 0  # Vertical margin

                    # Always use the same fixed positioning logic
                    oval_x = canvas_width - oval_size - h_margin
                    oval_y = canvas_height - oval_size - v_margin

                    # MODIFICATION: Determine color based on whether the reference is user-selected
                    is_user_selected = False
                    if cluster_id:
                        is_user_selected = cluster_id in self.user_selected_references
                    elif self.current_reference_cluster:
                        is_user_selected = \
                            self.current_reference_cluster in self.user_selected_references

                    # Yellow for user-selected, blue for automatic
                    fill_color = "#FFFF00" if is_user_selected else "#40C4FF"

                    # Create the oval indicator
                    canvas.create_oval(
                        oval_x, oval_y,
                        oval_x + oval_size, oval_y + oval_size,
                        fill=fill_color, outline="#000000"
                    )

                    # Add the "R" text
                    canvas.create_text(
                        oval_x + oval_size // 2, oval_y + oval_size // 2,
                        text="R", fill="#000000",
                        font=("TkDefaultFont", 8, "bold")
                    )

                # Update filename label with pixel-based truncation
                filename = os.path.basename(sig_path)

                # Create temporary canvas for text measurement
                temp_canvas = tk.Canvas(self.root, width=1, height=1)

                # Create text for measurement
                text_id = temp_canvas.create_text(0, 0, text=filename, font=("TkDefaultFont", 8))
                text_bbox = temp_canvas.bbox(text_id)

                # Check if truncation is needed
                if text_bbox and (text_bbox[2] - text_bbox[0]) > canvas_width:
                    # Try progressively shorter versions
                    for i in range(len(filename) - 1, 2, -1):
                        truncated = filename[:i] + "..."

                        # Measure truncated version
                        temp_canvas.delete("all")
                        trunc_id = \
                            temp_canvas.create_text(0, 0, text=truncated, font=("TkDefaultFont", 8))
                        trunc_bbox = temp_canvas.bbox(trunc_id)

                        # Check if this fits
                        if trunc_bbox and (trunc_bbox[2] - trunc_bbox[0]) <= canvas_width:
                            filename = truncated
                            break

                # Clean up temp canvas
                temp_canvas.destroy()

                filename_label.config(text=filename)

            except Exception as e:
                # Show error if image can't be loaded
                canvas.create_text(
                    canvas_width//2, canvas_height//2,
                    text=f"Error: {str(e)[:20]}...",
                    fill="red"
                )
                filename_label.config(text="Error loading image")

        return frame

    def _navigate_preview_cell(self, frame, direction):
        """
        Navigate to next/previous signature in a preview cell
        
        Args:
            frame: The preview cell frame
            direction: +1 for next, -1 for previous
        """
        # Basic validation
        if not hasattr(frame, 'cluster_id') or not frame.cluster_id:
            return

        if not hasattr(frame, 'cluster_sigs') or not frame.cluster_sigs:
            # If we don't have ordered signatures, get them - should rarely happen
            if frame.cluster_id in self.cluster_ordered_signatures:
                frame.cluster_sigs = self.cluster_ordered_signatures[frame.cluster_id].copy()
                print(f"Using cached ordered signatures " \
                      f"for navigation in cluster {frame.cluster_id}")
            else:
                # Calculate and cache ordered signatures
                frame.cluster_sigs = self._get_cluster_signatures_by_similarity(frame.cluster_id)
                self.cluster_ordered_signatures[frame.cluster_id] = frame.cluster_sigs.copy()
                print(f"Calculated ordered signatures for navigation in cluster {frame.cluster_id}")

        if not frame.cluster_sigs:  # Still no signatures
            return

        # Calculate new index
        new_index = frame.sig_index + direction

        # Check bounds
        if new_index < 0 or new_index >= len(frame.cluster_sigs):
            return

        # Update index
        frame.sig_index = new_index

        # Get the new signature to display
        new_sig = frame.cluster_sigs[new_index]

        # Update signature path
        frame.signature_path = new_sig
        frame.displayed_signature = new_sig

        # Update navigation buttons
        if hasattr(frame, 'prev_btn') and frame.prev_btn:
            frame.prev_btn.config(state=tk.NORMAL if new_index > 0 else tk.DISABLED)

        if hasattr(frame, 'next_btn') and frame.next_btn:
            frame.next_btn.config(state=tk.NORMAL if \
                                  new_index < len(frame.cluster_sigs) - 1 else tk.DISABLED)

        # Check if this is a reference signature
        is_reference_signature = False
        if new_sig and new_sig == self.current_reference:
            is_reference_signature = True
        elif frame.cluster_id and hasattr(self, 'cluster_displayed_signatures'):
            if frame.cluster_id in self.cluster_displayed_signatures and \
                self.cluster_displayed_signatures[frame.cluster_id] == new_sig:

                is_reference_signature = True

        # Update the image display with consistent formatting
        try:
            # Get canvas dimensions
            canvas_width = frame.canvas.winfo_width()
            canvas_height = frame.canvas.winfo_height()

            # Default dimensions if not available
            if canvas_width <= 1:
                canvas_width = 150
            if canvas_height <= 1:
                canvas_height = 100

            # Clear canvas
            frame.canvas.delete("all")

            # Determine space needed for labels
            cluster_label_height = 20 if frame.cluster_id is not None else 0
            reference_label_height = 23 if is_reference_signature else 0

            # Calculate image area height
            image_area_height = canvas_height - cluster_label_height - reference_label_height
            image_area_y = cluster_label_height  # Start after cluster label

            # Draw cluster name label if needed
            if frame.cluster_id is not None:
                # Get the cluster name
                cluster_display = str(frame.cluster_id)

                # NEW: Check if this cluster is marked as complete
                is_complete = frame.cluster_id in self.complete_clusters

                # NEW: Set box color based on completion status
                box_color = "#2ECC71" if is_complete else "#FFA000"
                # Green if complete, Orange if incomplete

                # NEW: Add size information
                if frame.cluster_id in self.clusters:
                    cluster_size = len(self.clusters[frame.cluster_id])
                    full_display = f"{cluster_display} ({cluster_size})"
                else:
                    full_display = cluster_display

                # Available width for the cluster name badge -
                # full width in popup (no "go to" button)
                available_width = canvas_width

                # Create a temporary canvas for text measurement
                measure_canvas = tk.Canvas(self.root, width=available_width, \
                                           height=cluster_label_height)

                # Standard padding around text
                text_padding = 10  # 5px on each side

                # Create text for measurement
                text_id = measure_canvas.create_text(
                    5, cluster_label_height//2,
                    text=full_display,
                    anchor="w",
                    font=("TkDefaultFont", 8, "bold")
                )

                # Get text dimensions
                text_bbox = measure_canvas.bbox(text_id)

                # Determine if truncation is needed
                if text_bbox and (text_bbox[2] - text_bbox[0] + text_padding) > available_width:
                    # Need to truncate - now handles combined name and size
                    if frame.cluster_id in self.clusters:
                        # Calculate space for size part
                        size_text = f" ({cluster_size})"
                        measure_canvas.delete("all")
                        measure_canvas.create_text(
                            0, cluster_label_height//2,
                            text=size_text,
                            font=("TkDefaultFont", 8, "bold")
                        )

                        # Truncate name only
                        for i in range(len(cluster_display) - 1, 2, -1):
                            # Try progressively shorter name versions
                            truncated_name = cluster_display[:i] + "..."
                            truncated_full = f"{truncated_name}{size_text}"

                            # Measure truncated full text
                            measure_canvas.delete("all")
                            trunc_id = measure_canvas.create_text(
                                5, cluster_label_height//2,
                                text=truncated_full,
                                anchor="w",
                                font=("TkDefaultFont", 8, "bold")
                            )

                            trunc_bbox = measure_canvas.bbox(trunc_id)

                            # Check if this fits
                            if trunc_bbox and \
                                (trunc_bbox[2] - trunc_bbox[0] + text_padding) <= available_width:

                                full_display = truncated_full
                                break
                    else:
                        # Original truncation behavior if no cluster size
                        for i in range(len(cluster_display) - 1, 2, -1):
                            # Try progressively shorter versions
                            truncated = cluster_display[:i] + "..."

                            # Remove previous measurement
                            measure_canvas.delete("all")

                            # Measure truncated text
                            trunc_id = measure_canvas.create_text(
                                5, cluster_label_height//2,
                                text=truncated,
                                anchor="w",
                                font=("TkDefaultFont", 8, "bold")
                            )

                            trunc_bbox = measure_canvas.bbox(trunc_id)

                            # Check if this fits
                            if trunc_bbox and \
                                (trunc_bbox[2] - trunc_bbox[0] + text_padding) <= available_width:

                                full_display = truncated
                                break

                # Calculate badge width based on text width
                if text_bbox:
                    badge_width = text_bbox[2] - text_bbox[0] + text_padding
                else:
                    # Approximate width based on character count
                    badge_width = len(full_display) * 6

                # Ensure badge doesn't exceed available width
                badge_width = min(badge_width, available_width)

                # Cleanup measurement canvas
                measure_canvas.destroy()

                # Draw the cluster name badge with updated color based on completion status
                frame.canvas.create_rectangle(
                    0, 0, badge_width, cluster_label_height,
                    fill=box_color, outline=""
                )

                # Center text in badge
                frame.canvas.create_text(
                    badge_width/2, cluster_label_height/2,
                    text=full_display,
                    fill="white",
                    font=("TkDefaultFont", 8, "bold"),
                    anchor="center"  # Center alignment
                )

            # Open and display the image
            img = self.preprocess_for_display(new_sig)

            if img is None:
                # Error in preprocessing - try direct loading as fallback
                img = Image.open(new_sig)

            # Calculate aspect ratio
            orig_width, orig_height = img.size
            img_aspect = orig_width / max(orig_height, 1)

            # Calculate dimensions to fit in the image area while preserving aspect ratio
            if img_aspect > canvas_width / image_area_height:  # Image is wider than tall
                new_width = canvas_width
                new_height = int(new_width / img_aspect)
            else:  # Image is taller than wide
                new_height = image_area_height
                new_width = int(new_height * img_aspect)

            # Resize image
            img = img.resize((new_width, new_height), \
                             Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS)

            # Convert to PhotoImage
            img_tk = ImageTk.PhotoImage(img)
            frame.image_tk = img_tk  # Keep reference

            # Calculate position to center the image in the image area
            x_pos = (canvas_width - new_width) // 2
            y_pos = image_area_y + (image_area_height - new_height) // 2

            # Display the image
            frame.canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=img_tk)

            # Add reference indicator ("R") if this is a reference signature
            if is_reference_signature:
                # Calculate position for R indicator - bottom right corner using FIXED positioning
                oval_size = 16
                h_margin = 5  # Horizontal margin
                v_margin = 5  # Vertical margin

                # Always use the same fixed positioning logic
                oval_x = canvas_width - oval_size - h_margin
                oval_y = canvas_height - oval_size - v_margin

                # MODIFICATION: Determine color based on whether the reference is user-selected
                is_user_selected = False
                if frame.cluster_id:
                    is_user_selected = frame.cluster_id in self.user_selected_references
                elif self.current_reference_cluster:
                    is_user_selected = \
                        self.current_reference_cluster in self.user_selected_references

                # Yellow for user-selected, blue for automatic
                fill_color = "#FFFF00" if is_user_selected else "#40C4FF"

                # Create the oval indicator
                frame.canvas.create_oval(
                    oval_x, oval_y,
                    oval_x + oval_size, oval_y + oval_size,
                    fill=fill_color, outline="#000000"
                )

                # Add the "R" text
                frame.canvas.create_text(
                    oval_x + oval_size // 2, oval_y + oval_size // 2,
                    text="R", fill="#000000",
                    font=("TkDefaultFont", 8, "bold")
                )

            # Update filename label with pixel-based truncation
            filename = os.path.basename(new_sig)

            # Create temporary canvas for text measurement
            temp_canvas = tk.Canvas(self.root, width=1, height=1)

            # Create text for measurement
            text_id = temp_canvas.create_text(0, 0, text=filename, \
                                              font=("TkDefaultFont", 8))
            text_bbox = temp_canvas.bbox(text_id)

            # Check if truncation is needed
            if text_bbox and (text_bbox[2] - text_bbox[0]) > canvas_width:
                # Try progressively shorter versions
                for i in range(len(filename) - 1, 2, -1):
                    truncated = filename[:i] + "..."

                    # Measure truncated version
                    temp_canvas.delete("all")
                    trunc_id = temp_canvas.create_text(0, 0, text=truncated, \
                                                       font=("TkDefaultFont", 8))
                    trunc_bbox = temp_canvas.bbox(trunc_id)

                    # Check if this fits
                    if trunc_bbox and (trunc_bbox[2] - trunc_bbox[0]) <= canvas_width:
                        filename = truncated
                        break

            # Clean up temp canvas
            temp_canvas.destroy()

            frame.filename_label.config(text=filename)

        except Exception as e:
            # Show error if image can't be loaded
            frame.canvas.create_text(
                canvas_width//2, canvas_height//2,
                text=f"Error: {str(e)[:20]}...",
                fill="red"
            )
            frame.filename_label.config(text="Error loading")

    def _create_new_cluster(self):
        """FIXED VERSION: Create a new cluster with proper discovery grid placement"""
        print(f"_create_new_cluster called in {self.current_mode} mode")
        print(f"Selected signatures: {len(self.selected_signatures)}")

        if not self.selected_signatures:
            messagebox.showinfo("No Selection", "Please select signatures to create a new group")
            return

        try:

            # Capture current scroll position
            scroll_position = self._get_current_scroll_position()

            # Store current page to maintain position after operation
            current_page = self.current_page[self.current_mode]

            # ADDED: Store original reference and cluster to preserve them
            preserve_reference = self.current_mode in ["COMPLETION", "VERIFICATION"]
            original_reference = self.current_reference if preserve_reference else None
            original_reference_cluster = \
                self.current_reference_cluster if preserve_reference else None

            # Variables to track new reference (if any)
            new_reference = None
            new_reference_cluster = None

            # Track original positions of selected signatures for discovery grid placement
            signature_positions = {}
            if hasattr(self, 'lazy_discovery_arranged'):
                for sig in self.selected_signatures:
                    try:
                        pos = self.lazy_discovery_arranged.index(sig)
                        signature_positions[sig] = pos
                    except ValueError:
                        pass

            # First, identify which clusters and unclustered signatures are involved
            involved_clusters = set()
            unclustered_signatures = []

            for sig in self.selected_signatures:
                # Check if this signature belongs to an existing cluster
                found_in_cluster = False
                for cluster_id, cluster_sigs in self.clusters.items():
                    if sig in cluster_sigs:
                        involved_clusters.add(cluster_id)
                        found_in_cluster = True
                        break

                # If not found in any cluster, it's unclustered
                if not found_in_cluster and sig in self.unclustered_signatures:
                    unclustered_signatures.append(sig)

            print(f"Involved clusters: {involved_clusters}")
            print(f"Unclustered signatures: {len(unclustered_signatures)}")

            # CRITICAL FLAG: Determine if this is a multi-cluster merge
            merging_multiple_clusters = len(involved_clusters) > 1

            # Use the consolidated dialog method
            dialog_title = "Move to New Group" \
                if self.current_mode == "VERIFICATION" else "Create New Group"
            new_cluster_id, selected_signatures, obsolete_cluster_names = self._show_cluster_dialog(
                self.selected_signatures,
                title=dialog_title
            )

            if new_cluster_id is None:  # User cancelled
                print("User cancelled new cluster dialog")
                return

            print(f"Selected new cluster ID: {new_cluster_id}")

            # Create or update the cluster
            if new_cluster_id not in self.clusters:
                self.clusters[new_cluster_id] = []

            # List of original clusters to track for debug and clean-up
            original_clusters = []
            if new_cluster_id in self.clusters and len(self.clusters[new_cluster_id]) > 0:
                original_clusters.append(new_cluster_id)

            # Track which signatures will determine cluster placement
            placement_determining_signatures = []

            # Process each signature in the selection
            processed_signatures = 0
            for sig in selected_signatures:
                # First check if it's already in another cluster
                source_cluster = None
                for cid, cluster_sigs in list(self.clusters.items()):
                    if cid != new_cluster_id and sig in cluster_sigs:
                        source_cluster = cid
                        break

                if source_cluster:
                    # Add to original clusters tracking if not already there
                    if source_cluster not in original_clusters:
                        original_clusters.append(source_cluster)

                    # Remove from original cluster
                    self.clusters[source_cluster].remove(sig)
                    print(f"Removed signature from cluster {source_cluster}")

                    # If this empties the cluster, remove it
                    if not self.clusters[source_cluster]:
                        del self.clusters[source_cluster]
                        print(f"Deleted empty cluster {source_cluster}")

                # Add to the new cluster if not already there
                if sig not in self.clusters[new_cluster_id]:
                    self.clusters[new_cluster_id].append(sig)
                    processed_signatures += 1
                    placement_determining_signatures.append(sig)
                    print(f"Added signature to cluster {new_cluster_id}")

                # Remove from unclustered if needed
                if sig in self.unclustered_signatures:
                    self.unclustered_signatures.remove(sig)

            # NEW: Clear constraints between all members of the new cluster
            self._clear_constraints_between_members(self.clusters[new_cluster_id])

            # Determine if this is from a single existing cluster
            single_source_cluster = None
            if len(original_clusters) == 1 and original_clusters[0] != new_cluster_id:
                single_source_cluster = original_clusters[0]

            if merging_multiple_clusters:
                print("Multi-cluster merge: Forcing automatic reference selection")

                # Always remove from user_selected_references for multi-cluster merges
                if new_cluster_id in self.user_selected_references:
                    print("Removing user-selected reference status "
                          f"for merged cluster: {new_cluster_id}")
                    self.user_selected_references.remove(new_cluster_id)

                # Calculate fresh ordered signatures
                ordered_signatures = self._get_cluster_signatures_by_similarity(
                    new_cluster_id, force_recalculate=True)

                # Set the first signature as reference (automatic selection)
                if ordered_signatures:
                    first_signature = ordered_signatures[0]
                    self.cluster_displayed_signatures[new_cluster_id] = first_signature
                    new_reference = first_signature
                    new_reference_cluster = new_cluster_id
                else:
                    # Fallback if no ordered signatures
                    first_signature = self.clusters[new_cluster_id][0]
                    self.cluster_displayed_signatures[new_cluster_id] = first_signature
                    new_reference = first_signature
                    new_reference_cluster = new_cluster_id

                # Remove obsolete cluster information
                for clust_name in obsolete_cluster_names:
                    self.complete_clusters.discard(clust_name)
                    self.cluster_displayed_signatures.pop(clust_name, None)
                    self.user_selected_references.discard(clust_name)

            else:
                # For non-multiple merges or fresh clusters
                should_inherit_reference = (
                    single_source_cluster is not None and
                    single_source_cluster in self.user_selected_references and
                    single_source_cluster in self.cluster_displayed_signatures
                )

                if should_inherit_reference:
                    # Inherit the user-selected reference from the source cluster
                    source_reference = self.cluster_displayed_signatures[single_source_cluster]
                    if source_reference in self.clusters[new_cluster_id]:
                        print("Inheriting user-selected reference "
                              f"from cluster {single_source_cluster}")
                        self.cluster_displayed_signatures[new_cluster_id] = source_reference
                        self.user_selected_references.add(new_cluster_id)
                        new_reference = source_reference
                        new_reference_cluster = new_cluster_id
                    else:
                        # Fallback to automatic reference if inherited
                        # reference is not in the new cluster
                        print("Cannot inherit reference - using automatic reference selection")
                        self._calculate_automatic_reference(new_cluster_id)
                        auto_ref = self.cluster_displayed_signatures.get(new_cluster_id)
                        if auto_ref:
                            new_reference = auto_ref
                            new_reference_cluster = new_cluster_id
                else:
                    # Use automatic reference selection
                    print(f"Using automatic reference selection for cluster {new_cluster_id}")
                    self._calculate_automatic_reference(new_cluster_id)
                    auto_ref = self.cluster_displayed_signatures.get(new_cluster_id)
                    if auto_ref:
                        new_reference = auto_ref
                        new_reference_cluster = new_cluster_id

            # ADDED: Decision about whether to use new reference or preserve original
            if self.current_mode != "DISCOVERY":
                if preserve_reference and original_reference_cluster is not None and \
                    original_reference is not None:
                    # Restore original reference instead of switching to the new cluster
                    self.current_reference = original_reference
                    self.current_reference_cluster = original_reference_cluster
                    self.current_displayed_signature = original_reference
                    print(f"Preserving original reference cluster: {original_reference_cluster}")
                elif new_reference is not None and new_reference_cluster is not None:
                    # Use the new reference if we didn't have one before or we're not preserving
                    self.current_reference = new_reference
                    self.current_reference_cluster = new_reference_cluster
                    self.current_displayed_signature = new_reference
                    print(f"Switching to new reference cluster: {new_reference_cluster}")

            # Update progress
            self.clustered_signatures = sum(len(cluster) for cluster in self.clusters.values())
            self._update_progress_display()

            # Update reference display if we switched to a new reference
            if self.current_mode != "DISCOVERY":
                self._update_reference_display()

            # *** FIXED: UPDATE DISCOVERY GRID LAYOUT WITH PROPER POSITIONING ***
            if hasattr(self, 'lazy_discovery_arranged'):
                # Track what signatures were moved
                removed_by_cluster = {}

                # For original clusters that were merged/moved, track the removals
                if original_clusters:
                    for cluster_id in original_clusters:
                        # Don't track the target cluster as "removed from"
                        if cluster_id != new_cluster_id:
                            # Everything was moved to new cluster
                            removed_by_cluster[cluster_id] = []

                # Handle the grid alteration with proper positioning
                self._handle_grid_alteration_discovery(new_cluster_id, selected_signatures,
                                                       removed_by_cluster)

            # *** MODE-SPECIFIC UPDATES ***
            if self.current_mode == "DISCOVERY":
                # Reset selection
                self.selected_signatures = []

                # Try to maintain current page
                self.current_page["DISCOVERY"] = current_page

                # FIXED: Call _refresh_grid() instead of _update_grid_display()
                # to ensure current_grid_signatures is updated
                # from the modified lazy_discovery_arranged
                self._refresh_grid()
            else:
                # For other modes, try to stay on the same page
                self.current_page[self.current_mode] = current_page

                # Clear cached signature lists for the current mode
                if self.current_mode in self.full_signature_lists:
                    self.full_signature_lists[self.current_mode] = []

                # Reset lazy loading for completion mode
                if self.current_mode == "COMPLETION":
                    self._reset_lazy_loading_state("COMPLETION")

            # If the operation removed the current cluster, go to the new cluster
            if original_reference_cluster is not None \
                and original_reference_cluster not in self.clusters:
                self._select_cluster(new_cluster_id)
            else:
                if self.current_mode == "DISCOVERY":
                    # For discovery mode, grid was already refreshed above
                    pass
                else:
                    if self.current_mode == "COMPLETION":
                        self._handle_completion_grid_alteration(selected_signatures)
                        self._remove_signatures_from_completion_grid(selected_signatures, {})
                    else:
                        self._refresh_grid()
                    if self.current_mode in ["COMPLETION", "VERIFICATION"]:
                        self._populate_cluster_selector(self.last_applied_search_text,
                                                        self.last_applied_filter_type,
                                                        self.last_applied_sort_option)

            # Restore scroll position after grid update
            if scroll_position is not None:
                self._restore_scroll_position(scroll_position)

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to create new group: {str(e)}")
            self.status_var.set("Error creating new group")

    def _calculate_automatic_reference(self, cluster_id):
        """Calculate and set the automatic reference for a cluster"""
        if cluster_id not in self.clusters or not self.clusters[cluster_id]:
            return

        # Calculate ordered signatures
        ordered_signatures = \
            self._get_cluster_signatures_by_similarity(cluster_id, force_recalculate=True)

        if ordered_signatures:
            # Set the most representative signature as reference
            self.cluster_displayed_signatures[cluster_id] = ordered_signatures[0]

            # Make it the current reference if it's the current reference cluster
            if self.current_reference_cluster == cluster_id:
                self.current_reference = ordered_signatures[0]
                self.current_displayed_signature = self.current_reference
        else:
            # Fallback to first signature
            self.cluster_displayed_signatures[cluster_id] = self.clusters[cluster_id][0]

            # Make it the current reference if it's the current reference cluster
            if self.current_reference_cluster == cluster_id:
                self.current_reference = self.clusters[cluster_id][0]
                self.current_displayed_signature = self.current_reference

        # Ensure it's not in user_selected_references
        if cluster_id in self.user_selected_references:
            self.user_selected_references.remove(cluster_id)

        print(f"Calculated automatic reference for cluster {cluster_id}")

    def _add_cannot_link_constraint(self, sig1, sig2):
        """
        Add a cannot-link constraint between two signatures.
        
        Args:
            sig1: First signature path
            sig2: Second signature path
        """
        # Skip if same signature
        if sig1 == sig2:
            return False

        # Check if constraint already exists
        if sig1 in self.cannot_link_map and sig2 in self.cannot_link_map[sig1]:
            return False

        # Add to the map in both directions for quick lookups
        if sig1 not in self.cannot_link_map:
            self.cannot_link_map[sig1] = set()
        self.cannot_link_map[sig1].add(sig2)

        if sig2 not in self.cannot_link_map:
            self.cannot_link_map[sig2] = set()
        self.cannot_link_map[sig2].add(sig1)

        # Also add to list for backward compatibility
        if (sig1, sig2) not in self.cannot_link_constraints and \
            (sig2, sig1) not in self.cannot_link_constraints:

            self.cannot_link_constraints.append((sig1, sig2))

        return True

    def _remove_cannot_link_constraint(self, sig1, sig2):
        """
        Remove a cannot-link constraint between two signatures.
        
        Args:
            sig1: First signature path
            sig2: Second signature path
        """
        # Remove from map
        if sig1 in self.cannot_link_map and sig2 in self.cannot_link_map[sig1]:
            self.cannot_link_map[sig1].remove(sig2)
            if not self.cannot_link_map[sig1]:  # Clean up empty sets
                del self.cannot_link_map[sig1]

        if sig2 in self.cannot_link_map and sig1 in self.cannot_link_map[sig2]:
            self.cannot_link_map[sig2].remove(sig1)
            if not self.cannot_link_map[sig2]:  # Clean up empty sets
                del self.cannot_link_map[sig2]

        # Remove from list
        for i, constraint in enumerate(self.cannot_link_constraints):
            if (constraint[0] == sig1 and constraint[1] == sig2) or \
            (constraint[0] == sig2 and constraint[1] == sig1):
                self.cannot_link_constraints.pop(i)
                break

    def _has_cannot_link_constraint(self, sig1, sig2):
        """
        Check if two signatures have a cannot-link constraint between them.
        
        Args:
            sig1: First signature path
            sig2: Second signature path
            
        Returns:
            True if the signatures have a cannot-link constraint, False otherwise
        """
        has_constraint = (sig1 in self.cannot_link_map and sig2 in self.cannot_link_map[sig1]) or \
                        (sig2 in self.cannot_link_map and sig1 in self.cannot_link_map[sig2])

        if has_constraint:
            print(f"Found constraint between {os.path.basename(sig1)} and {os.path.basename(sig2)}")

        return has_constraint

    def _is_cluster_rejected(self, cluster_id1, cluster_id2):
        """
        Check if any image in cluster1 has a cannot-link constraint with any image in cluster2.
        
        Args:
            cluster_id1: ID of the first cluster
            cluster_id2: ID of the second cluster
            
        Returns:
            bool: True if any image in cluster1 has a constraint with any image in cluster2
        """
        if cluster_id1 not in self.clusters or cluster_id2 not in self.clusters:
            return False

        # Get all signatures in each cluster
        signatures1 = self.clusters[cluster_id1]
        signatures2 = self.clusters[cluster_id2]

        # Check each pair of signatures - exit early if any constraint is found
        for sig1 in signatures1:
            for sig2 in signatures2:
                if self._has_cannot_link_constraint(sig1, sig2):
                    print(f"Found cluster rejection: {os.path.basename(sig1)} in " \
                          f"{cluster_id1} rejects {os.path.basename(sig2)} in {cluster_id2}")
                    return True

        return False

    def _is_image_rejected_by_cluster(self, image_path, cluster_id):
        """
        Check if an image has a cannot-link constraint with any image in a cluster.
        
        Args:
            image_path: Path to the image
            cluster_id: ID of the cluster
            
        Returns:
            bool: True if the image has a constraint with any image in the cluster
        """
        if cluster_id not in self.clusters:
            return False

        # Get all signatures in the cluster
        signatures = self.clusters[cluster_id]

        # Check if the image has a constraint with any signature in the cluster
        for sig in signatures:
            if self._has_cannot_link_constraint(image_path, sig):
                print(f"Found image rejection: {os.path.basename(image_path)} is " \
                      f"rejected by {os.path.basename(sig)} in {cluster_id}")
                return True

        return False

    def _clear_constraints_between_members(self, signatures):
        """
        Clear all cannot-link constraints between members of the given signature list.
        
        Args:
            signatures: List of signature paths
        """
        if not signatures or len(signatures) < 2:
            return

        print(f"Clearing constraints between {len(signatures)} signatures")

        # For each pair of signatures, remove any cannot-link constraints
        for i, sig_1 in enumerate(signatures):
            for j in range(i+1, len(signatures)):
                if self._has_cannot_link_constraint(sig_1, signatures[j]):
                    self._remove_cannot_link_constraint(sig_1, signatures[j])
                    print(f"Removed constraint between {os.path.basename(sig_1)} " \
                          f"and {os.path.basename(signatures[j])}")

    def _reject_selected(self):
        """
        Reject selected signatures with pagination support:
        - In verification mode: Remove from current cluster
        - In completion mode: Add cannot-link constraints with current reference
        - In discovery mode: Add constraints based on selection and reference
        """
        if not self.selected_signatures:
            messagebox.showinfo("No Selection", "Please select signatures to reject")
            return

        try:

            # Capture current scroll position
            scroll_position = self._get_current_scroll_position()

            # Store current page to maintain position after operation
            current_page = self.current_page[self.current_mode]

            # Different behavior based on mode
            if self.current_mode == "VERIFICATION":
                # Remove from current cluster
                if not self.current_reference_cluster:
                    messagebox.showinfo("No Cluster", "No active cluster to reject from")
                    return

                # Get current cluster
                if self.current_reference_cluster not in self.clusters:
                    messagebox.showinfo("Error", "Reference cluster not found")
                    return

                # CRITICAL: Save state before removal
                true_reference = self.current_reference
                left_pane_signature = self.current_displayed_signature
                is_viewing_reference = left_pane_signature == true_reference

                print(f"Reference: {os.path.basename(true_reference)}")
                print(f"Left pane displaying: {os.path.basename(left_pane_signature)}")
                print(f"Is viewing reference: {is_viewing_reference}")

                # NEW: Check if the cluster is marked as complete before removing images
                was_complete = self.current_reference_cluster in self.complete_clusters

                # Check if reference or displayed signature is among selected signatures
                reference_being_removed = true_reference in self.selected_signatures
                displayed_being_removed = left_pane_signature in self.selected_signatures

                # Remove selected signatures from the cluster
                rejected_count = 0
                for sig in self.selected_signatures:
                    if sig in self.clusters[self.current_reference_cluster]:
                        self.clusters[self.current_reference_cluster].remove(sig)
                        # Add back to unclustered signatures
                        self.unclustered_signatures.append(sig)
                        rejected_count += 1

                # *** UPDATE DISCOVERY GRID LAYOUT WITH LAZY LOADING ***
                if hasattr(self, 'lazy_discovery_arranged'):
                    removed_by_cluster = {self.current_reference_cluster: self.selected_signatures}
                    self._handle_grid_alteration_discovery(None, [], removed_by_cluster)

                # NEW: If images were removed and the cluster was
                # previously marked as complete, mark it as incomplete
                if rejected_count > 0 and was_complete:
                    if self.current_reference_cluster in self.complete_clusters:
                        self.complete_clusters.remove(self.current_reference_cluster)
                        # Update checkbox state if it exists
                        if hasattr(self, 'complete_var'):
                            self.complete_var.set(False)
                        print(f"Marked cluster {self.current_reference_cluster} " \
                            "as incomplete after removing images")

                        # Update next cluster button state after changing completion status
                        self._update_next_cluster_button_state()

                # Check if we still have signatures in the cluster
                remaining_sigs = self.clusters[self.current_reference_cluster]
                if not remaining_sigs:
                    print("WARNING: No signatures left in cluster after removal!")
                    # Handle empty cluster case
                    self.current_reference = None
                    self.current_reference_cluster = None
                    self.current_displayed_signature = None
                    self._update_reference_display()

                    # Reset to page 1 when removing all signatures
                    self.current_page["VERIFICATION"] = 1

                    self._refresh_grid()
                    self.status_var.set(f"Removed last {rejected_count} signatures from cluster")
                    return

                # Handle reference signature removal
                if reference_being_removed:
                    print("Reference was removed - selecting new reference")

                    # Save old reference for adding constraint later
                    old_reference = self.current_reference

                    # MODIFICATION: Always switch to automatic
                    # reference selection when reference is removed
                    if self.current_reference_cluster in self.user_selected_references:
                        self.user_selected_references.remove(self.current_reference_cluster)

                    # Get ordered signatures for this cluster
                    ordered_signatures = self._get_cluster_signatures_by_similarity(\
                        self.current_reference_cluster, force_recalculate=True)

                    if ordered_signatures:
                        # Set the most representative signature as new reference
                        self.current_reference = ordered_signatures[0]
                        print(f"Setting new reference after removal: " \
                            f"{os.path.basename(self.current_reference)}")
                    else:
                        # Fallback to first signature in the cluster
                        self.current_reference = remaining_sigs[0]
                        print(f"Setting fallback reference after removal: " \
                            f"{os.path.basename(self.current_reference)}")

                    # If we were viewing the reference, update
                    # the displayed signature to the new reference
                    if is_viewing_reference:
                        self.current_displayed_signature = self.current_reference
                        print(f"Updating displayed signature to new reference: " \
                            f"{os.path.basename(self.current_reference)}")

                    # Store in displayed signatures dictionary
                    self.cluster_displayed_signatures[self.current_reference_cluster] = \
                        self.current_reference
                    print(f"Stored new reference in cluster_displayed_signatures: " \
                        f"{os.path.basename(self.current_reference)}")

                    # REMOVED: No more constraint transfers from old reference to new reference
                elif displayed_being_removed:
                    # We didn't remove the reference, but we did remove the displayed signature
                    # Switch to displaying the reference
                    self.current_displayed_signature = self.current_reference
                    print(f"Displayed signature was removed, switching to reference: " \
                        f"{os.path.basename(self.current_reference)}")

                # NEW: Check if we need to update the automatic
                # reference after removing non-reference images
                if not reference_being_removed and \
                    self.current_reference_cluster not in self.user_selected_references:

                    print("Cluster uses automatic reference - checking if " \
                        "reference needs to be updated after removal")

                    # Recalculate ordered signatures with the new cluster composition
                    ordered_signatures = self._get_cluster_signatures_by_similarity(\
                        self.current_reference_cluster, force_recalculate=True)

                    if ordered_signatures and ordered_signatures[0] != self.current_reference:
                        # The most representative signature has changed - update the reference
                        old_reference = self.current_reference
                        self.current_reference = ordered_signatures[0]

                        print("Updating automatic reference from " \
                            f"{os.path.basename(old_reference)} to " \
                                f"{os.path.basename(self.current_reference)}")

                        # Update displayed signature if it was showing the reference
                        if self.current_displayed_signature == old_reference:
                            self.current_displayed_signature = self.current_reference

                        # Store in displayed signatures dictionary
                        self.cluster_displayed_signatures[self.current_reference_cluster] = \
                            self.current_reference

                # Update reference display
                self._update_reference_display()

                # Update counts
                self.clustered_signatures = sum(len(cluster) for cluster in self.clusters.values())
                self._update_progress_display()

                # Clear cached signature lists for verification mode
                self.full_signature_lists["VERIFICATION"] = []

                # Reset lazy loading for verification mode
                if hasattr(self, '_reset_lazy_loading_state'):
                    # Note: verification mode doesn't use lazy
                    # loading currently, but this is future-proof
                    pass

                # Check if we need to adjust current page
                total_remaining = len(self.clusters[self.current_reference_cluster]) - 1
                max_possible_page = max(1, (total_remaining + self.signatures_per_page - 1) // \
                                        self.signatures_per_page)

                if current_page > max_possible_page:
                    # If current page no longer exists, go to last page
                    self.current_page["VERIFICATION"] = max_possible_page
                    print(f"Adjusted to page {max_possible_page} " \
                        f"after removal (was on page {current_page})")
                else:
                    # Otherwise stay on current page
                    self.current_page["VERIFICATION"] = current_page

                self._refresh_grid()

                # Update message
                self.status_var.set(f"Removed {rejected_count} signatures " \
                                    f"from cluster {self.current_reference_cluster}")

                if self.current_mode in ["COMPLETION", "VERIFICATION"]:
                    self._populate_cluster_selector(self.last_applied_search_text, \
                                                    self.last_applied_filter_type, \
                                                        self.last_applied_sort_option)

            elif self.current_mode == "COMPLETION":
                # In completion mode: Add cannot-link constraints between reference signatures
                if not self.current_reference:
                    messagebox.showinfo("No Reference", "Please select a reference signature first")
                    return

                # Track signatures and clusters being processed
                signatures_to_remove = []
                cluster_updates = {}
                rejected_count = 0

                # Find all unique clusters in the selection
                selected_clusters = {}  # Maps cluster_id -> reference_signature

                for sig in self.selected_signatures:
                    # Skip the reference itself
                    if sig == self.current_reference:
                        continue

                    # First check if this is a clustered signature
                    found_cluster = False
                    for cluster_id, signatures in self.clusters.items():
                        # Skip current reference cluster
                        if cluster_id == self.current_reference_cluster:
                            continue

                        if sig in signatures:
                            # This is a clustered signature
                            found_cluster = True

                            # If we haven't processed this cluster yet
                            if cluster_id not in selected_clusters:
                                # Get the reference signature for this cluster
                                if cluster_id in self.cluster_displayed_signatures and \
                                    self.cluster_displayed_signatures[cluster_id] in signatures:
                                    ref_sig = self.cluster_displayed_signatures[cluster_id]
                                    selected_clusters[cluster_id] = ref_sig
                                else:
                                    selected_clusters[cluster_id] = sig

                                # Track for potential cluster reference updates
                                cluster_updates[cluster_id] = sig
                            break

                    # If this is an unclustered signature, track for removal
                    if not found_cluster:
                        signatures_to_remove.append(sig)
                        # Add direct constraint for unclustered signatures
                        if self._add_cannot_link_constraint(self.current_reference, sig):
                            rejected_count += 1

                # Process the cluster references
                for cluster_id, ref_sig in selected_clusters.items():
                    signatures_to_remove.append(ref_sig)
                    # Add constraint between reference signatures
                    if self._add_cannot_link_constraint(self.current_reference, ref_sig):
                        rejected_count += 1

                # Apply rejection filter check
                rejection_filter = getattr(self, 'last_applied_rejection_filter', 'Non-rejected')
                should_remove_from_grid = rejection_filter == "Non-rejected"

                if should_remove_from_grid and signatures_to_remove:
                    # Remove signatures from grid without full refresh
                    self._remove_signatures_from_completion_grid(
                        signatures_to_remove, cluster_updates)
                else:
                    # If not filtering or different filter, do minimal refresh
                    self.selected_signatures = []
                    self.last_selected_index = None
                    if hasattr(self, 'current_grid_signatures') and self.current_grid_signatures:
                        self._update_grid_display()

                # Update message
                self.status_var.set(f"Added {rejected_count} cannot-link constraints")

                # Refresh the grid to reflect rejected status
                if self.current_mode == "DISCOVERY":
                    # For discovery mode, just update the grid display
                    self._update_grid_display()
                else:
                    self._refresh_grid()

            # Restore scroll position after operations
            if scroll_position is not None:
                self._restore_scroll_position(scroll_position)

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to reject signatures: {str(e)}")
            self.status_var.set("Error rejecting signatures")

    def _unreject_selected(self):
        """
        Remove cannot-link constraints between selected signatures and the current cluster.
        For unclustered signatures, removes constraints with any image in the current cluster.
        For clustered signatures, removes constraints between all images in the selected cluster
        and all images in the current cluster.
        """
        if not self.selected_signatures:
            messagebox.showinfo("No Selection", "Please select signatures to unreject")
            return

        if not self.current_reference_cluster or \
            self.current_reference_cluster not in self.clusters:

            messagebox.showinfo("No Reference", "Please select a reference cluster first")
            return

        try:
            # Store current page to maintain position after operation
            current_page = self.current_page[self.current_mode]

            # Get all signatures in the current reference cluster
            current_cluster_signatures = self.clusters[self.current_reference_cluster]

            # Count how many constraints were removed
            total_constraints_removed = 0

            # Process each selected signature
            for sig in self.selected_signatures:
                # Skip the reference itself
                if sig == self.current_reference:
                    continue

                # Check if this is a clustered signature
                selected_cluster_id = None
                for cluster_id, cluster_sigs in self.clusters.items():
                    # Skip current reference cluster
                    if cluster_id == self.current_reference_cluster:
                        continue

                    if sig in cluster_sigs:
                        selected_cluster_id = cluster_id
                        break

                # Case 1: Selected signature is from another cluster
                if selected_cluster_id:
                    # Get all signatures in the selected cluster
                    selected_cluster_signatures = self.clusters[selected_cluster_id]

                    # Remove constraints between ALL signatures in both clusters
                    constraints_removed = 0

                    for current_sig in current_cluster_signatures:
                        for selected_sig in selected_cluster_signatures:
                            if self._has_cannot_link_constraint(current_sig, selected_sig):
                                self._remove_cannot_link_constraint(current_sig, selected_sig)
                                constraints_removed += 1

                    if constraints_removed > 0:
                        print(f"Removed {constraints_removed} constraints between cluster " \
                              f"{self.current_reference_cluster} and cluster {selected_cluster_id}")
                        total_constraints_removed += constraints_removed

                # Case 2: Selected signature is unclustered
                elif sig in self.unclustered_signatures:
                    # Remove constraints between this signature
                    # and ALL signatures in the current cluster
                    constraints_removed = 0

                    for current_sig in current_cluster_signatures:
                        if self._has_cannot_link_constraint(current_sig, sig):
                            self._remove_cannot_link_constraint(current_sig, sig)
                            constraints_removed += 1

                    if constraints_removed > 0:
                        print(f"Removed {constraints_removed} constraints between " \
                              f"unclustered signature {os.path.basename(sig)} " \
                                f"and cluster {self.current_reference_cluster}")
                        total_constraints_removed += constraints_removed

            # Clear cached signature lists for completion mode to force refresh
            self.full_signature_lists["COMPLETION"] = []

            # Try to maintain current page
            self.current_page[self.current_mode] = current_page

            # Refresh the grid to reflect these changes
            self._refresh_grid()

            if total_constraints_removed > 0:
                self.status_var.set(f"Removed {total_constraints_removed} rejection constraints")
            else:
                self.status_var.set("No rejection constraints found to remove")

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to unreject signatures: {str(e)}")
            self.status_var.set("Error unrejecting signatures")

    def _change_reference(self):
        """Change the reference signature"""
        # If any signature is selected, use the first one as reference
        if self.selected_signatures:
            self.current_reference = self.selected_signatures[0]

            # Find which cluster this signature belongs to
            for cluster_id, signatures in self.clusters.items():
                if self.current_reference in signatures:
                    self.current_reference_cluster = cluster_id
                    break
            else:
                # If not in any cluster, keep current cluster
                pass
        else:
            # No selection - pick another signature from current cluster
            if self.current_reference_cluster and self.current_reference_cluster in self.clusters:
                cluster_sigs = self.clusters[self.current_reference_cluster]

                if len(cluster_sigs) > 1:
                    # Find current reference index
                    try:
                        current_index = cluster_sigs.index(self.current_reference)
                        # Pick the next signature in the cluster
                        next_index = (current_index + 1) % len(cluster_sigs)
                        self.current_reference = cluster_sigs[next_index]
                    except ValueError:
                        # Reference not found in cluster, pick the first one
                        self.current_reference = cluster_sigs[0]

        # Reset shown signatures when reference changes
        self.shown_signatures = set()
        if hasattr(self, 'verification_index'):
            self.verification_index = 0

        if self.current_mode == "COMPLETION":
            self._calculate_cluster_sizes()

        # Update the displays
        self._update_reference_display()
        self._refresh_grid()

    def _next_cluster(self):
        """Move to the next incomplete cluster and reset pagination to page 1"""
        if not self.clusters:
            return

        # Get list of cluster IDs
        cluster_ids = list(self.clusters.keys())

        # Skip empty clusters
        cluster_ids = [cid for cid in cluster_ids if self.clusters[cid]]

        if not cluster_ids:
            return

        # Filter to only include incomplete clusters
        incomplete_clusters = [cid for cid in cluster_ids if cid not in self.complete_clusters]

        # If no incomplete clusters, disable the button
        # (should be handled by _update_next_cluster_button_state)
        if not incomplete_clusters:
            self.status_var.set("No more incomplete clusters to process")
            return

        # If the current cluster is not set, use the first incomplete cluster
        if not self.current_reference_cluster:
            next_cluster = incomplete_clusters[0]
        else:
            # Try to find the next incomplete cluster after the current one
            try:
                current_index = cluster_ids.index(self.current_reference_cluster)

                # Search for the next incomplete cluster after the current one
                found_next = False
                for i in range(current_index + 1, len(cluster_ids)):
                    candidate = cluster_ids[i]
                    if candidate in incomplete_clusters:
                        next_cluster = candidate
                        found_next = True
                        break

                # If not found, wrap around to the beginning
                if not found_next:
                    for i in range(0, current_index):
                        candidate = cluster_ids[i]
                        if candidate in incomplete_clusters:
                            next_cluster = candidate
                            found_next = True
                            break

                # If still not found, it means the current cluster is the only
                # incomplete one or there are no incomplete clusters, so don't change
                if not found_next:
                    # The button should be disabled in this
                    # case by _update_next_cluster_button_state
                    return

            except ValueError:
                # Current cluster not found in list (shouldn't happen),
                # fall back to first incomplete cluster
                if incomplete_clusters:
                    next_cluster = incomplete_clusters[0]
                else:
                    return

        # Set reference to the appropriate signature in the next cluster
        if self.clusters[next_cluster]:

            # Reset shown signatures when changing references
            # This is critical for ensuring we see top candidates
            self.shown_signatures = set()

            # Reset verification page counter when changing references
            if hasattr(self, 'verification_page'):
                self.verification_page = 0

            # Reset to page 1 when changing clusters
            self.current_page[self.current_mode] = 1

            # IMPORTANT: Use the user-selected reference if available
            if next_cluster in self.cluster_displayed_signatures and \
                self.cluster_displayed_signatures[next_cluster] in self.clusters[next_cluster]:

                # Use the user's previously selected reference for this cluster
                self.current_reference = self.cluster_displayed_signatures[next_cluster]
                print(f"Using user-selected reference: {os.path.basename(self.current_reference)}")
            else:
                # No user-selected reference exists, get ordered signatures
                ordered_signatures = self._get_cluster_signatures_by_similarity(next_cluster)

                if ordered_signatures:
                    # Use most representative signature as default
                    self.current_reference = ordered_signatures[0]
                    # Store this as the displayed signature
                    self.cluster_displayed_signatures[next_cluster] = self.current_reference
                    print("Using most representative signature as reference: " \
                          f"{os.path.basename(self.current_reference)}")
                else:
                    # If no ordering was possible, use the first signature
                    self.current_reference = self.clusters[next_cluster][0]
                    self.cluster_displayed_signatures[next_cluster] = self.current_reference
                    print("Using first signature as reference: " \
                          f"{os.path.basename(self.current_reference)}")

            # Update current cluster and displayed signature
            self.current_reference_cluster = next_cluster
            self.current_displayed_signature = self.current_reference

            if self.current_mode == "COMPLETION":
                # Apply last applied grid search parameters
                if hasattr(self, 'last_applied_grid_membership'):
                    self.membership_var.set(self.last_applied_grid_membership)
                    self.grid_filter_var.set(self.last_applied_grid_filter)
                    self.sort_completion_var.set(self.last_applied_grid_sort)
                    self.use_name_query_var.set(self.last_applied_grid_use_name_query)
                    self.name_query_var.set(self.last_applied_grid_name_query)
                    self.rejection_filter_var.set(self.last_applied_rejection_filter)

                    # Update state handlers
                    self._handle_membership_change()
                    self._update_name_query_entry_state()
                    self._update_name_query_checkbox_state()
                else:
                    self._apply_grid_search_parameters(use_defaults=True)

                self._calculate_cluster_sizes()

            # Reset verification index if it exists
            if hasattr(self, 'verification_index'):
                self.verification_index = 0

            # Update reference display
            self._update_reference_display()

            # NEW: Update Next Cluster button state
            self._update_next_cluster_button_state()

            # FOR COMPLETION MODE: Pre-extract features for consistency
            if self.current_mode == "COMPLETION":
                self.status_var.set("Pre-processing reference and candidates...")
                self.root.update()

                # First, ensure the reference signature has its features extracted
                if self.current_reference not in self.features_cache:
                    self._extract_features_for_signatures([self.current_reference])

                # Pre-extract features for a batch of unclustered signatures
                # Using a larger batch size to ensure comprehensive ranking
                sample_size = min(500, len(self.unclustered_signatures))
                if sample_size > 0:
                    batch = random.sample(self.unclustered_signatures, sample_size)
                    self._extract_features_for_signatures(batch)

                    # Now create a complete initial ranking with the extracted features
                    # This ensures the first grid shows top candidates
                    self._rank_all_candidates_for_reference()

                    # Track that we're in a fresh reference state
                    self._fresh_reference = True

            # Refresh the grid - this will show initial candidates
            self._refresh_grid()

            # When refreshing the cluster selector, use the last applied parameters
            if hasattr(self, 'last_applied_search_text') and \
                hasattr(self, 'last_applied_filter_type') and \
                    hasattr(self, 'last_applied_sort_option'):

                self._populate_cluster_selector(
                    self.last_applied_search_text,
                    self.last_applied_filter_type,
                    self.last_applied_sort_option
                )
            else:
                # Fallback to defaults if no applied values exist
                self._populate_cluster_selector("", "Incomplete", "Visual Similarity")

    def _change_mode(self):
        """Change the current operating mode with proper state persistence and button updates"""
        new_mode = self.mode_var.get()
        if new_mode != self.current_mode:
            old_mode = self.current_mode

            print(f"Changing mode from {old_mode} to {new_mode}")

            # Check if we're switching to a specific cluster to avoid unnecessary calculations
            skip_initial_calculations = getattr(self, '_switching_to_specific_cluster', False)
            target_cluster = getattr(self, '_target_cluster_id', None)

            # CRITICAL: Preserve current reference when switching modes
            if self.current_reference and self.current_reference_cluster:
                print("Preserving reference selection: " \
                      f"{os.path.basename(self.current_reference)} " \
                        f"for cluster {self.current_reference_cluster}")
                # Ensure the current reference is stored in displayed signatures
                self.cluster_displayed_signatures[self.current_reference_cluster] = \
                    self.current_reference

            # Store the current reference and cluster when leaving completion/verification mode
            # This will be used when returning back to these modes from discovery
            if old_mode in ["COMPLETION", "VERIFICATION"] and new_mode == "DISCOVERY":
                if not hasattr(self, 'last_cv_reference'):
                    self.last_cv_reference = None
                if not hasattr(self, 'last_cv_reference_cluster'):
                    self.last_cv_reference_cluster = None

                # Save the current reference and cluster
                if self.current_reference and self.current_reference_cluster:
                    self.last_cv_reference = self.current_reference
                    self.last_cv_reference_cluster = self.current_reference_cluster
                    print("Saving last completion/verification reference: " \
                          f"{os.path.basename(self.current_reference)} in " \
                            f"cluster {self.current_reference_cluster}")

            # Cache the current grid signatures for the old mode to allow returning to it
            # EXCEPT for completion mode, which should always recalculate
            if not hasattr(self, 'mode_grid_cache'):
                self.mode_grid_cache = {}

            # Save discovery mode grid state before changing modes
            if self.current_grid_signatures and old_mode == "DISCOVERY":
                self.mode_grid_cache["DISCOVERY"] = self.current_grid_signatures.copy()
                print(f"Cached {len(self.current_grid_signatures)} signatures for discovery mode")

            # NEW: When switching TO discovery mode, clear the cached grid to force recalculation
            # This ensures discovery mode utilizes all cached distances from completion mode
            if new_mode == "DISCOVERY" and "DISCOVERY" in self.mode_grid_cache:
                print("Clearing cached discovery grid to force recalculation with all distances")
                del self.mode_grid_cache["DISCOVERY"]

                # Also clear the full signature list for discovery mode
                if "DISCOVERY" in self.full_signature_lists:
                    self.full_signature_lists["DISCOVERY"] = []
                    print("Cleared discovery signature list to force fresh arrangement")

            # NEW: Reset pagination to page 1 when changing modes
            self.current_page[new_mode] = 1

            # CRITICAL: When switching to or from discovery mode, preserve the grid layout
            if old_mode == "DISCOVERY" or new_mode == "DISCOVERY":
                # Ensure we keep the persistent discovery grid layout
                if hasattr(self, 'discovery_grid_layout') and self.discovery_grid_layout:
                    # Update the full_signature_lists with our current layout
                    if "DISCOVERY" in self.full_signature_lists:
                        self.full_signature_lists["DISCOVERY"] = self.discovery_grid_layout.copy()

            # Remember last page in discovery mode
            if old_mode == "DISCOVERY":
                self.lazy_discovery_last_page = self.current_page["DISCOVERY"]
                print(f"Remembered discovery page: {self.lazy_discovery_last_page}")

            # Now change the mode
            self.current_mode = new_mode

            # Reset to page 1 when switching to completion mode
            if new_mode == "COMPLETION":
                self.current_page["COMPLETION"] = 1

            # Reset displayed signature when changing modes
            if self.current_reference:
                self.current_displayed_signature = self.current_reference

            # Ensure all clusters have reference signatures initialized when
            # entering discovery mode
            if new_mode == "DISCOVERY":
                self._initialize_cluster_references()

            # First, unpack all mode-specific frames to avoid ordering conflicts
            if hasattr(self, 'left_frame'):
                self.left_frame.pack_forget()  # Hide reference frame
            if hasattr(self, 'filter_frame'):
                self.filter_frame.pack_forget()  # Hide filter controls

            # Then re-configure the middle frame which contains both panes
            if hasattr(self, 'middle_frame'):
                self.middle_frame.pack_forget()
            self.middle_frame = ttk.Frame(self.main_frame)
            self.middle_frame.pack(fill=tk.BOTH, expand=True, pady=5)

            if new_mode == "DISCOVERY":
                # IMPORTANT: Set grid columns explicitly using discovery_grid_cols
                self.grid_cols = self.discovery_grid_cols

                # Ensure discovery mode isolation
                self._ensure_discovery_mode_isolation()

                # Return to last viewed page in discovery mode
                if hasattr(self, 'lazy_discovery_last_page') and self.lazy_discovery_last_page > 0:
                    self.current_page["DISCOVERY"] = self.lazy_discovery_last_page
                    print(f"Returning to discovery page: {self.lazy_discovery_last_page}")
                else:
                    self.current_page["DISCOVERY"] = 1

                # IMPORTANT: Reset state when entering discovery mode
                # Clear any discovery grid cache
                if hasattr(self, 'discovery_current_grid'):
                    self.discovery_current_grid = []

                # Clear mode grid cache for discovery to prevent using old cached grid
                if "DISCOVERY" in self.mode_grid_cache:
                    del self.mode_grid_cache["DISCOVERY"]

                # No left frame in discovery mode
                self.status_var.set("DISCOVERY MODE: Find and create new clusters")

                # Just add right frame (grid only)
                if hasattr(self, 'right_frame'):
                    self.right_frame.pack_forget()
                self.right_frame = ttk.LabelFrame(self.middle_frame, text="Signature Grid")
                self.right_frame.pack(fill=tk.BOTH, expand=True)

                # Create grid container with scrollbar for DISCOVERY mode
                grid_container = ttk.Frame(self.right_frame)
                grid_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                # Canvas for scrolling
                self.canvas = tk.Canvas(grid_container)
                self.grid_scrollbar = ttk.Scrollbar(grid_container, orient="vertical", \
                                                    command=self.canvas.yview)
                self.scrollable_frame = ttk.Frame(self.canvas)

                # Configure the scrollable frame to resize with its contents
                self.scrollable_frame.bind(
                    "<Configure>",
                    lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
                )

                # Create a window in the canvas for the scrollable frame
                self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
                self.canvas.configure(yscrollcommand=self.grid_scrollbar.set)

                # Pack the canvas and scrollbar
                self.canvas.pack(side="left", fill="both", expand=True)
                self.grid_scrollbar.pack(side="right", fill="y")
            else:
                # For other modes, use their standard column counts
                self.grid_cols = 4  # Default for completion and verification
                # For COMPLETION and VERIFICATION modes
                # First add the left frame
                if hasattr(self, 'left_frame'):
                    self.left_frame.pack_forget()
                self.left_frame = ttk.LabelFrame(self.middle_frame, text="Reference Signature")
                self.left_frame.pack(side=tk.LEFT, fill=tk.Y, expand=False, padx=(0, 5))

                # Rebuild reference frame contents
                self._rebuild_reference_frame()

                # Then add right frame
                if hasattr(self, 'right_frame'):
                    self.right_frame.pack_forget()
                self.right_frame = ttk.LabelFrame(self.middle_frame, text="Signature Grid")
                self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

                # Use 4 columns in other modes
                self.grid_cols = 4

                # CRITICAL FIX: When switching from discovery to completion/verification,
                # ensure we have a proper reference with correct ordering
                if old_mode == "DISCOVERY" and new_mode in ["COMPLETION", "VERIFICATION"]:
                    # First try to use the last completion/verification reference if available
                    if hasattr(self, 'last_cv_reference') and self.last_cv_reference and \
                    hasattr(self, 'last_cv_reference_cluster') and \
                        self.last_cv_reference_cluster and \
                            self.last_cv_reference_cluster in self.clusters:

                        cluster_id = self.last_cv_reference_cluster
                        print("Restoring previous completion/verification " \
                              f"reference from cluster {cluster_id}")

                        # Ensure the reference signature is still in the cluster
                        if self.last_cv_reference in self.clusters[cluster_id]:
                            self.current_reference = self.last_cv_reference
                        else:
                            # If reference is gone, use the most representative signature
                            ordered_signatures = \
                                self._get_cluster_signatures_by_similarity(cluster_id)
                            if ordered_signatures:
                                self.current_reference = ordered_signatures[0]
                            else:
                                self.current_reference = self.clusters[cluster_id][0]

                        self.current_reference_cluster = cluster_id
                        self.current_displayed_signature = self.current_reference

                        # Also update the displayed signatures
                        # dictionary to keep track of this selection
                        self.cluster_displayed_signatures[cluster_id] = self.current_reference

                        print(f"Restored reference to {os.path.basename(self.current_reference)} " \
                              f"in cluster {self.current_reference_cluster}")
                    # Fallback to initial reference if last_cv_reference isn't available
                    elif hasattr(self, 'initial_reference_cluster') and \
                        self.initial_reference_cluster:

                        cluster_id = self.initial_reference_cluster

                        # Ensure cluster has ordered signatures available (for navigation)
                        if cluster_id not in self.cluster_ordered_signatures:
                            print(f"Calculating ordered signatures for cluster {cluster_id}")
                            if cluster_id in self.clusters:
                                # Process ALL signatures in this cluster
                                self._extract_features_for_signatures(self.clusters[cluster_id])
                                # Force calculation and caching of ordered signatures
                                ordered = self._get_cluster_signatures_by_similarity(\
                                    cluster_id, force_recalculate=True)

                        # Check if we have a user-selected reference for this cluster
                        if cluster_id in self.cluster_displayed_signatures:
                            # Use the user-selected reference
                            self.current_reference = self.cluster_displayed_signatures[cluster_id]
                        else:
                            # Use the most representative signature as default
                            if cluster_id in self.cluster_ordered_signatures and \
                                self.cluster_ordered_signatures[cluster_id]:

                                self.current_reference = \
                                    self.cluster_ordered_signatures[cluster_id][0]

                            elif hasattr(self, 'initial_reference'):
                                self.current_reference = self.initial_reference

                        self.current_reference_cluster = cluster_id
                        self.current_displayed_signature = self.current_reference

                        print(f"Setting reference to {os.path.basename(self.current_reference)} " \
                              f"in cluster {self.current_reference_cluster}")

                # Auto-select a reference signature if none is selected
                if not self.current_reference or not self.current_reference_cluster:
                    # Find the first non-empty cluster
                    for cluster_id, signatures in self.clusters.items():
                        if signatures:
                            # Use ordered signatures if available
                            if cluster_id in self.cluster_ordered_signatures:
                                ordered = self.cluster_ordered_signatures[cluster_id]
                                self.current_reference = ordered[0]
                            else:
                                # Force calculation of ordered signatures
                                ordered = self._get_cluster_signatures_by_similarity(\
                                    cluster_id, force_recalculate=True)
                                if ordered:
                                    self.current_reference = ordered[0]
                                else:
                                    self.current_reference = signatures[0]

                            self.current_reference_cluster = cluster_id
                            self.current_displayed_signature = self.current_reference
                            break

                # Mode-specific configurations
                if self.current_mode == "COMPLETION":
                    # Add filter controls to the right frame for completion mode
                    self.filter_frame = ttk.Frame(self.right_frame)
                    self.filter_frame.pack(fill=tk.X, padx=10, pady=(5, 0))

                    # Recreate the filter controls
                    self._create_cluster_filter_controls()

                    # Apply last applied grid search parameters
                    if hasattr(self, 'last_applied_grid_membership'):
                        self.membership_var.set(self.last_applied_grid_membership)
                        self.grid_filter_var.set(self.last_applied_grid_filter)
                        self.sort_completion_var.set(self.last_applied_grid_sort)
                        self.use_name_query_var.set(self.last_applied_grid_use_name_query)
                        self.name_query_var.set(self.last_applied_grid_name_query)
                        self.rejection_filter_var.set(self.last_applied_rejection_filter)

                        # Update state handlers
                        self._handle_membership_change()
                        self._update_name_query_entry_state()
                        self._update_name_query_checkbox_state()

                    else:
                        # If no last applied values, use defaults
                        self._apply_grid_search_parameters(use_defaults=True)

                    # Calculate cluster sizes for filter controls
                    self._calculate_cluster_sizes()

                    # Update completion checkbox
                    if hasattr(self, 'complete_checkbox'):
                        self.complete_checkbox.config(state=tk.NORMAL)

                    # Check if we're switching to a specific
                    # cluster to avoid unnecessary calculations
                    skip_initial_calculations = \
                        getattr(self, '_switching_to_specific_cluster', False)
                    target_cluster = getattr(self, '_target_cluster_id', None)

                    if self.current_reference and not skip_initial_calculations:
                        # Only do initial calculations if we're not switching to a specific cluster
                        # Ensure features are extracted for the reference
                        if self.current_reference not in self.features_cache:
                            self._extract_features_for_signatures([self.current_reference])

                        # This will perform full ranking and cache results
                        self._rank_all_candidates_for_reference()
                    elif self.current_reference and skip_initial_calculations and target_cluster:
                        # We're switching to a specific cluster,
                        # skip unnecessary initial calculations
                        print("Skipping initial calculations - "
                              f"switching directly to cluster {target_cluster}")
                        # Still ensure features are extracted for
                        # the reference, but don't do full ranking
                        if self.current_reference not in self.features_cache:
                            self._extract_features_for_signatures([self.current_reference])

                    self.status_var.set("COMPLETION MODE: Find similar "\
                                        "signatures to add to the selected cluster")

                elif self.current_mode == "VERIFICATION":
                    # If switching to verification mode, reset pagination
                    if hasattr(self, 'verification_page'):
                        self.verification_page = 0
                    else:
                        self.verification_page = 0
                    print("DEBUG: Reset verification_page to 0 when switching to VERIFICATION mode")

                    # Update completion checkbox
                    if hasattr(self, 'complete_checkbox'):
                        self.complete_checkbox.config(state=tk.NORMAL)

                    self.status_var.set("VERIFICATION MODE: Check and remove " \
                                        "incorrect signatures from the selected cluster")

                # Update the reference display
                self._update_reference_display()

                self.root.update()

                # Use last applied values for the cluster selector
                if hasattr(self, 'last_applied_search_text') and \
                    hasattr(self, 'last_applied_filter_type') and \
                        hasattr(self, 'last_applied_sort_option'):

                    self._populate_cluster_selector(
                        self.last_applied_search_text,
                        self.last_applied_filter_type,
                        self.last_applied_sort_option
                    )
                else:
                    # Fallback to defaults if no applied values exist
                    self._populate_cluster_selector("", "Incomplete", "Visual Similarity")

                # Update grid size to match new dimensions
                self.grid_size = self.grid_cols * 3

                # Reset tracking for the new mode
                if hasattr(self, 'verification_index'):
                    self.verification_index = 0

                # Create grid container with scrollbar
                grid_container = ttk.Frame(self.right_frame)
                grid_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                # Canvas for scrolling
                self.canvas = tk.Canvas(grid_container)
                self.grid_scrollbar = ttk.Scrollbar(\
                    grid_container, orient="vertical", command=self.canvas.yview)
                self.scrollable_frame = ttk.Frame(self.canvas)

                # Configure the scrollable frame to resize with its contents
                self.scrollable_frame.bind(
                    "<Configure>",
                    lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
                )

                # Create a window in the canvas for the scrollable frame
                self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
                self.canvas.configure(yscrollcommand=self.grid_scrollbar.set)

                # Pack the canvas and scrollbar
                self.canvas.pack(side="left", fill="both", expand=True)
                self.grid_scrollbar.pack(side="right", fill="y")

            # Update mode-specific UI elements (buttons, etc.)
            self._update_mode_specific_ui()

            # Special handling for different modes
            if new_mode == "DISCOVERY":
                # Always get new signatures for discovery mode
                print("Forcing fresh discovery mode signatures")
                # Empty the current grid signatures to force refresh
                self.current_grid_signatures = []
                # Clear discovery mode cache to ensure fresh signatures
                if hasattr(self, 'mode_grid_cache'):
                    self.mode_grid_cache.pop("DISCOVERY", None)
                # Clear discovery current grid cache
                if hasattr(self, 'discovery_current_grid'):
                    self.discovery_current_grid = []
                # Force a grid refresh to get discovery signatures
                self._refresh_grid()
            elif new_mode == "COMPLETION":
                # For completion mode, always get fresh signatures
                print("Getting fresh completion mode signatures")
                self.full_signature_lists["COMPLETION"] = []
                self._refresh_grid()
            elif new_mode == "VERIFICATION" and self.current_reference_cluster:
                # For verification mode, always get fresh signatures for the current cluster
                print("Getting fresh verification signatures " \
                      f"for cluster {self.current_reference_cluster}")
                self._refresh_grid()

            # Update last search fields if they don't exist yet
            if not hasattr(self, 'last_applied_search_text'):
                self.last_applied_search_text = ""
            if not hasattr(self, 'last_applied_filter_type'):
                self.last_applied_filter_type = "Incomplete"
            if not hasattr(self, 'last_applied_sort_option'):
                self.last_applied_sort_option = "Visual Similarity"

            self.selected_signatures = []
            self.last_selected_index = None

            print(f"Changed to {self.current_mode} mode")

            # Setup mousewheel scrolling
            self._setup_mousewheel_scrolling()

            # After mode change, ensure focus is set correctly for keyboard shortcuts
            self.main_frame.focus_set()

    def _update_center_buttons(self):
        """Update center buttons based on current mode"""
        # Clear existing buttons from the center frame
        for widget in self.center_buttons.winfo_children():
            widget.destroy()

        if self.current_mode == "DISCOVERY":
            print("Creating discovery mode buttons")

            # Use no padding for discovery mode buttons
            button_padding = 0  # Removed external padding

            # Keep a reference to self.group_btn to avoid attribute errors
            self.group_btn = ttk.Button(self.root, text="Hidden") # Create but don't display

            # Discovery mode now only has the "Create New Group" button
            self.new_cluster_btn = ttk.Button(self.center_buttons, text="New Cluster (N)",
                                        command=self._create_new_cluster,
                                        style="Compact.TButton")
            self.new_cluster_btn.pack(side=tk.LEFT, padx=button_padding)

            # Create a hidden reference to reject_btn to avoid attribute errors
            self.reject_btn = ttk.Button(self.root, text="Hidden")  # Create but don't display

            # Create a hidden reference to unreject_btn to avoid attribute errors
            self.unreject_btn = ttk.Button(self.root, text="Hidden")  # Create but don't display

        elif self.current_mode == "COMPLETION":
            # Completion mode action buttons - remove padding between specific buttons
            self.group_btn = ttk.Button(self.center_buttons, text="Add to Cluster (G)",
                                    command=self._group_selected)
            self.group_btn.pack(side=tk.LEFT, padx=(0, 0))  # Remove padding

            self.new_cluster_btn = ttk.Button(self.center_buttons, text="New Cluster (N)",
                                            command=self._create_new_cluster)
            self.new_cluster_btn.pack(side=tk.LEFT, padx=0)

            self.reject_btn = ttk.Button(self.center_buttons, text="Reject (X)",
                                        command=self._reject_selected)
            self.reject_btn.pack(side=tk.LEFT, padx=0)

            # Add the Unreject button (only in completion mode)
            self.unreject_btn = ttk.Button(self.center_buttons, text="Unreject (U)",
                                        command=self._unreject_selected)
            self.unreject_btn.pack(side=tk.LEFT, padx=0)

        elif self.current_mode == "VERIFICATION":
            # Verification mode action buttons - remove padding between specific buttons

            # Create a hidden reference to self.group_btn to avoid attribute errors
            self.group_btn = ttk.Button(self.root, text="Hidden")  # Create but don't display

            # Move to New Cluster button
            self.new_cluster_btn = ttk.Button(self.center_buttons, text="Move to New Cluster (N)",
                                            command=self._create_new_cluster)
            self.new_cluster_btn.pack(side=tk.LEFT, padx=(0, 0))  # Remove padding

            # Remove from Cluster button
            self.reject_btn = ttk.Button(self.center_buttons, text="Remove from Cluster (X)",
                                        command=self._reject_selected)
            self.reject_btn.pack(side=tk.LEFT, padx=0)

            # Create a hidden reference to unreject_btn to avoid attribute errors
            self.unreject_btn = ttk.Button(self.root, text="Hidden")  # Create but don't display

    def _update_mode_specific_ui(self):
        """Update UI elements based on current mode"""
        # First, update button labels
        self._update_button_labels()

        # Clear all button containers first
        for widget in self.left_buttons.winfo_children():
            widget.pack_forget()

        # Recreate/refresh discovery mode buttons
        self.discovery_nav_buttons = ttk.Frame(self.left_buttons)
        self.discovery_nav_buttons.pack(side=tk.LEFT)

        # Base reload button
        reload_text = "Reload Grid (R)"
        reload_style = "Compact.TButton" if self.current_mode == "DISCOVERY" else ""
        self.refresh_btn = ttk.Button(self.discovery_nav_buttons, text=reload_text,
                                    command=self._refresh_grid,
                                    style=reload_style)
        self.refresh_btn.pack(side=tk.LEFT, padx=0)  # No padding

        # Add "Fresh Arrangement (F)" button for discovery mode
        # Only create and add it in discovery mode
        if self.current_mode == "DISCOVERY":
            self.fresh_arrangement_btn = ttk.Button(\
                self.discovery_nav_buttons, text="Fresh Arrangement (F)", \
                    command=self._fresh_arrangement_discovery, style="Compact.TButton")
            self.fresh_arrangement_btn.pack(side=tk.LEFT, padx=0)  # No padding

        # Update center buttons for the current mode
        self._update_center_buttons()

        for widget in self.right_buttons.winfo_children():
            widget.destroy()

        # Add the Deselect All button
        self.deselect_all_btn = ttk.Button(self.right_buttons, text="Deselect All (D)",
                                        command=self._deselect_all)
        self.deselect_all_btn.pack(side=tk.LEFT)

        # Update keyboard shortcuts to match the current actions
        self._update_keyboard_shortcuts()

    def _update_keyboard_shortcuts(self):
        """Update keyboard shortcuts based on current mode"""
        # First unbind any existing shortcuts
        for key in ['g', 'n', 'x', 'r', '<space>', 'd', 'u']:  # Add 'u' to the list
            try:
                self.root.unbind(key)
            except Exception:
                pass

        # Add common shortcuts for all modes
        self.root.bind("r", lambda _: \
                       self._handle_keyboard_shortcut(_, self._handle_refresh_grid))
        self.root.bind("<space>", lambda _: \
                       self._handle_keyboard_shortcut(_, self._toggle_selection_of_focused))
        self.root.bind("d", lambda _: \
                       self._handle_keyboard_shortcut(_, self._deselect_all))  # Add for all modes

        # Mode-specific shortcuts
        if self.current_mode == "DISCOVERY":
            # Discovery mode now only has the 'n' shortcut
            self.root.bind("n", lambda _: \
                           self._handle_keyboard_shortcut(_, self._create_new_cluster))
        elif self.current_mode == "COMPLETION":
            # Completion mode shortcuts
            self.root.bind("g", lambda _: self._handle_keyboard_shortcut(_, self._group_selected))
            self.root.bind("n", lambda _: \
                           self._handle_keyboard_shortcut(_, self._create_new_cluster))
            self.root.bind("x", lambda _: self._handle_keyboard_shortcut(_, self._reject_selected))
            self.root.bind("u", lambda _: \
                           self._handle_keyboard_shortcut(_, self._unreject_selected))
        else:  # Verification mode
            self.root.bind("n", lambda _: \
                           self._handle_keyboard_shortcut(_, self._create_new_cluster))
            self.root.bind("x", lambda _: self._handle_keyboard_shortcut(_, self._reject_selected))

    def _rebuild_reference_frame(self):
        """Rebuild the contents of the reference frame with updated layout"""
        # Clear any existing widgets
        for widget in self.left_frame.winfo_children():
            widget.destroy()

        # Set a fixed width for the left frame
        left_frame_width = 390  # Your preferred width
        image_area_width = 230  # Your preferred width
        image_area_height = 120  # Your preferred height
        self.left_frame.configure(width=left_frame_width)

        # Use propagate=False to FORCE the frame to maintain its width
        self.left_frame.pack_propagate(False)

        # Recreate the reference frame with less padding
        self.reference_frame = ttk.Frame(self.left_frame, width=left_frame_width)
        self.reference_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=3)

        # Make the frame focusable for better keyboard handling
        self.reference_frame.configure(takefocus=1)
        # Bind click to take focus away from entry widget
        self.reference_frame.bind("<Button-1>", self._take_focus_from_entry)

        # ========== IMAGE CONTAINER (MUST BE FIRST) ==========
        # Create a parent container for the image and nav buttons
        ref_display_container = ttk.Frame(self.reference_frame)
        ref_display_container.pack(fill=tk.X, pady=0)

        # Left navigation button
        self.ref_prev_btn = ttk.Button(ref_display_container, text="←", width=3,
                                    command=lambda: self._navigate_reference_image(-1))
        self.ref_prev_btn.pack(side=tk.LEFT)

        # Placeholder for reference image
        self.reference_canvas = tk.Canvas(ref_display_container, width=image_area_width, \
                                          height=image_area_height, bg="#f0f0f0")
        self.reference_canvas.pack(side=tk.LEFT, padx=5)
        # Bind click to take focus away from entry widget
        self.reference_canvas.bind("<Button-1>", self._take_focus_from_entry)

        # Right navigation button
        self.ref_next_btn = ttk.Button(ref_display_container, text="→", width=3,
                                    command=lambda: self._navigate_reference_image(1))
        self.ref_next_btn.pack(side=tk.RIGHT)

        # Image name below the container
        self.reference_info = ttk.Label(self.reference_frame, \
                                        text="No reference selected", anchor=tk.CENTER)
        self.reference_info.pack(fill=tk.X, pady=(0, 0))  # Minimal bottom padding

        # CRITICAL FIX: Don't pack the reference_cluster_info at all -
        # it's causing the spacing issue
        # Just create it but don't pack it
        self.reference_cluster_info = ttk.Label(self.reference_frame, text="")
        # self.reference_cluster_info.pack(pady=2)  # COMMENTED OUT - this is causing the gap

        # MODIFICATION: Replace "Update Reference" button with two side-by-side buttons
        ref_update_frame = ttk.Frame(self.reference_frame)
        ref_update_frame.pack(fill=tk.X, pady=(0, 0))  # Minimal top padding

        # "Use Automatic Reference" button
        self.use_auto_ref_btn = ttk.Button(ref_update_frame, text="Use Automatic Reference",
                                        command=self._use_automatic_reference)
        self.use_auto_ref_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 1))

        # "Set Custom Reference" button
        self.set_custom_ref_btn = ttk.Button(ref_update_frame, text="Set Custom Reference",
                                            command=self._set_custom_reference)
        self.set_custom_ref_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(1, 0))

        # Cluster name and info section
        cluster_info_frame = ttk.Frame(self.reference_frame)
        cluster_info_frame.pack(fill=tk.X, pady=0)
        # Bind click to take focus away from entry widget
        cluster_info_frame.bind("<Button-1>", self._take_focus_from_entry)

        # Cluster name section with entry and update button
        cluster_name_frame = ttk.Frame(cluster_info_frame)
        cluster_name_frame.pack(fill=tk.X, pady=0)

        ttk.Label(cluster_name_frame, text="Cluster:").pack(side=tk.LEFT, padx=(0, 2))

        # Entry widget for cluster name
        self.cluster_name_var = tk.StringVar()
        self.cluster_name_entry = ttk.Entry(cluster_name_frame,
                                            textvariable=self.cluster_name_var, width=12)
        self.cluster_name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        # Add hotkey binding for cluster name entry
        self.cluster_name_entry.bind("<Return>", lambda e: self._update_cluster_name())

        # Bind focus out event to handle entry completion
        self.cluster_name_entry.bind("<FocusOut>", self._on_entry_focus_out)

        # Update Name button
        self.update_cluster_name_btn = ttk.Button(cluster_name_frame, text="Update Name",
                                                command=self._update_cluster_name, width=10)
        self.update_cluster_name_btn.pack(side=tk.RIGHT)

        # NEW LAYOUT: Combine "Complete" checkbox, signature count, and "Next Cluster" button
        completion_frame = ttk.Frame(cluster_info_frame)
        completion_frame.pack(fill=tk.X, pady=0)

        # Left side: Complete label and checkbox
        complete_container = ttk.Frame(completion_frame)
        complete_container.pack(side=tk.LEFT)

        # Initialize checkbox variable
        self.complete_var = tk.BooleanVar(value=False)

        # Add label and checkbox
        ttk.Label(complete_container, text="Complete:").pack(side=tk.LEFT, padx=(0, 2))
        self.complete_checkbox = ttk.Checkbutton(
            complete_container,
            text="",
            variable=self.complete_var,
            command=self._toggle_cluster_completion
        )
        self.complete_checkbox.pack(side=tk.LEFT)

        # Set initial state based on current cluster
        if self.current_reference_cluster in self.complete_clusters:
            self.complete_var.set(True)
        else:
            self.complete_var.set(False)

        # Middle: Signature count label (now with parentheses)
        self.signature_count_label = ttk.Label(completion_frame, text="")
        self.signature_count_label.pack(side=tk.LEFT, padx=5)

        # Right: "Next Cluster" button - aligned to the right
        self.next_cluster_btn = ttk.Button(completion_frame, text="Next Cluster",
                                            command=self._next_cluster)
        self.next_cluster_btn.pack(side=tk.RIGHT)

        # Initialize Next Cluster button state
        if self.current_reference_cluster:
            self._update_next_cluster_button_state()
        else:
            self.next_cluster_btn.config(state=tk.DISABLED)

        # REMOVED LABEL: "Cluster Search" -> now just a regular frame
        search_frame = ttk.Frame(self.reference_frame)
        search_frame.pack(fill=tk.X, pady=0)

        # Search entry and buttons
        search_entry_frame = ttk.Frame(self.reference_frame)
        search_entry_frame.pack(fill=tk.X, pady=0)

        # Create search entry - MODIFIED: Use cached search text if available
        self.search_var = tk.StringVar(value=self.last_applied_search_text if \
                                       hasattr(self, 'last_applied_search_text') else "")
        self.search_entry = ttk.Entry(search_entry_frame, textvariable=self.search_var)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        # Add hotkey binding for search entry
        self.search_entry.bind("<Return>", lambda e: self._search_clusters())

        # Create search button with magnifying glass icon
        self.search_btn = ttk.Button(search_entry_frame, text="🔍", width=3,
                                    command=self._search_clusters)
        self.search_btn.pack(side=tk.LEFT, padx=(0, 1))

        # Create clear button with X icon
        self.clear_search_btn = ttk.Button(search_entry_frame, text="✕", width=3,
                                            command=self._clear_search)
        self.clear_search_btn.pack(side=tk.LEFT)

        # Create a frame to hold both sort dropdown and filter options on the same line
        filter_sort_frame = ttk.Frame(self.reference_frame)
        filter_sort_frame.pack(fill=tk.X, pady=(0, 2))

        # Completion dropdown first (changed from "Filter:" to "Completion:")
        ttk.Label(filter_sort_frame, text="Completion:").pack(side=tk.LEFT, padx=(0, 2))

        # Dropdown for filter options - MODIFIED: Use cached filter type if available
        self.search_filter_var = \
            tk.StringVar(value=self.last_applied_filter_type if \
                         hasattr(self, 'last_applied_filter_type') else "Incomplete")
        self.search_filter_dropdown = ttk.Combobox(
            filter_sort_frame,
            textvariable=self.search_filter_var,
            values=["Incomplete", "Complete", "Both"],
            state="readonly",
            width=8
        )
        self.search_filter_dropdown.pack(side=tk.LEFT, padx=(0, 10))  # Add padding after filter

        # Sort dropdown
        ttk.Label(filter_sort_frame, text="Sort:").pack(side=tk.LEFT, padx=(0, 2))

        # Dropdown for sort options - MODIFIED: Use cached sort option if available
        self.sort_var = \
            tk.StringVar(value=self.last_applied_sort_option if hasattr(
                self, 'last_applied_sort_option') else "Visual Similarity")

        self.sort_dropdown = ttk.Combobox(
            filter_sort_frame,
            textvariable=self.sort_var,
            values=["Visual Similarity", "Query Similarity", "A→Z", "Z→A",
                    "Size (↓)", "Size (↑)",  "Path (↓)", "Path (↑)", "Path Similarity"],
            state="readonly",
            width=11
        )
        self.sort_dropdown.pack(side=tk.LEFT)

        # REMOVED LABEL: "Available Clusters" -> now just a regular frame
        # Give it MORE HEIGHT due to freed vertical space
        cluster_selector_frame = ttk.Frame(self.reference_frame)
        cluster_selector_frame.pack(fill=tk.BOTH, expand=True, pady=0)  # FILL & EXPAND
        # Bind click to take focus away from entry widget
        cluster_selector_frame.bind("<Button-1>", self._take_focus_from_entry)

        # Create a canvas with scrollbar for the cluster list - WITH INCREASED HEIGHT
        self.cluster_canvas = tk.Canvas(cluster_selector_frame, \
                                        width=self.left_frame.winfo_width()-20, height=220)
        cluster_scrollbar = ttk.Scrollbar(cluster_selector_frame, \
                                          orient="vertical", command=self.cluster_canvas.yview)
        self.cluster_scrollable_frame = ttk.Frame(self.cluster_canvas)

        # Configure the canvas to update scrollregion when scrollable_frame changes size
        self.cluster_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.cluster_canvas.configure(scrollregion=self.cluster_canvas.bbox("all"))
        )

        # Create the window in the canvas - full width
        self.cluster_canvas.create_window((0, 0), window=self.cluster_scrollable_frame, anchor="nw")
        self.cluster_canvas.configure(yscrollcommand=cluster_scrollbar.set)

        # Pack the canvas and scrollbar - ENSURE FILL AND EXPAND
        self.cluster_canvas.pack(side="left", fill="both", expand=True)
        cluster_scrollbar.pack(side="right", fill="y")

    def _handle_automatic_reference_update(self, cluster_id):
        """Handle automatic reference updates without full refresh"""
        if (self.current_mode == "COMPLETION" and \
            cluster_id == self.current_reference_cluster and
            cluster_id not in self.user_selected_references and
            not getattr(self, 'completion_reference_changed_manually', False)):

            # This is an automatic reference update, don't refresh grid
            print(f"Automatic reference update for cluster {cluster_id}, skipping grid refresh")
            return True

        # Reset manual change flag after processing
        if hasattr(self, 'completion_reference_changed_manually'):
            self.completion_reference_changed_manually = False

        return False

    def _use_automatic_reference(self):
        """Set the cluster to use automatically calculated reference signatures."""
        if not self.current_reference_cluster or \
            self.current_reference_cluster not in self.clusters:
            return

        # Calculate the most representative signature for this cluster
        ordered_signatures = self._get_cluster_signatures_by_similarity(\
            self.current_reference_cluster, force_recalculate=True)

        if not ordered_signatures:
            return

        # Check if the reference would actually change
        new_reference = ordered_signatures[0]
        reference_actually_changing = self.current_reference != new_reference

        # Only mark as manual change if reference is actually changing
        if reference_actually_changing:
            self.completion_reference_changed_manually = True

        # Set the first (most representative) signature as the reference
        self.current_reference = new_reference
        self.current_displayed_signature = self.current_reference

        # Store in displayed signatures dictionary
        self.cluster_displayed_signatures[self.current_reference_cluster] = self.current_reference

        # Remove from user-selected references set
        if self.current_reference_cluster in self.user_selected_references:
            self.user_selected_references.remove(self.current_reference_cluster)

        # Update the display
        self._update_reference_display()

        # Only do full refresh if reference actually changed
        if reference_actually_changing and self.current_mode == "COMPLETION":
            self._reset_completion_grid_state()
            self._refresh_grid()
        elif self.current_mode == "COMPLETION":
            # Reference didn't change, just update display
            if hasattr(self, 'current_grid_signatures') and self.current_grid_signatures:
                self._update_grid_display()
        elif self.current_mode == "VERIFICATION":
            # Handle verification mode logic...
            current_left_pane = self.current_reference
            cluster_sigs = [sig for sig in self.clusters[self.current_reference_cluster]
                        if sig != current_left_pane]

            if self.current_reference_cluster in self.cluster_ordered_signatures:
                ordered_sigs = \
                    self.cluster_ordered_signatures[self.current_reference_cluster].copy()
                if current_left_pane in ordered_sigs:
                    ordered_sigs.remove(current_left_pane)
                ordered_sigs = [sig for sig in ordered_sigs if sig in cluster_sigs]
                missing_sigs = [sig for sig in cluster_sigs if sig not in ordered_sigs]
                ordered_sigs.extend(missing_sigs)
                self.current_grid_signatures = ordered_sigs
            else:
                self.current_grid_signatures = cluster_sigs

            self.selected_signatures = []
            self._update_grid_display()
        else:
            if hasattr(self, 'current_grid_signatures') and self.current_grid_signatures:
                self._update_grid_display()
            else:
                self._refresh_grid()

        self.status_var.set("Using automatic reference for " \
                            f"cluster {self.current_reference_cluster}")

    def _set_custom_reference(self):
        """Set the current displayed signature as a custom reference."""
        if not self.current_reference_cluster or \
            self.current_reference_cluster not in self.clusters:
            return

        if not self.current_displayed_signature:
            return

        # Check if current displayed signature is already the reference
        if self.current_displayed_signature == self.current_reference:
            if self.current_reference_cluster not in self.user_selected_references:
                self.user_selected_references.add(self.current_reference_cluster)
                self._update_reference_display()
                if hasattr(self, 'current_grid_signatures') and self.current_grid_signatures:
                    self._update_grid_display()
                else:
                    self._refresh_grid()
                self.status_var.set("Current reference marked as user-selected " \
                                    f"for cluster {self.current_reference_cluster}")
            return

        # Check if the reference would actually change
        reference_actually_changing = self.current_reference != self.current_displayed_signature

        # Only mark as manual change if reference is actually changing
        if reference_actually_changing:
            self.completion_reference_changed_manually = True

        # Update the reference signature
        self.current_reference = self.current_displayed_signature

        # Mark as user-selected
        self.user_selected_references.add(self.current_reference_cluster)

        # Store in displayed signatures dictionary
        self.cluster_displayed_signatures[self.current_reference_cluster] = self.current_reference

        # Update display
        self._update_reference_display()

        # Only do full refresh if reference actually changed
        if reference_actually_changing and self.current_mode == "COMPLETION":
            self._reset_completion_grid_state()
            self._refresh_grid()
        elif self.current_mode == "COMPLETION":
            # Reference didn't change, just update display
            if hasattr(self, 'current_grid_signatures') and self.current_grid_signatures:
                self._update_grid_display()
        elif self.current_mode == "VERIFICATION":
            # Handle verification mode logic...
            current_left_pane = self.current_reference
            cluster_sigs = [sig for sig in self.clusters[self.current_reference_cluster]
                        if sig != current_left_pane]

            if self.current_reference_cluster in self.cluster_ordered_signatures:
                ordered_sigs = \
                    self.cluster_ordered_signatures[self.current_reference_cluster].copy()
                if current_left_pane in ordered_sigs:
                    ordered_sigs.remove(current_left_pane)
                ordered_sigs = [sig for sig in ordered_sigs if sig in cluster_sigs]
                missing_sigs = [sig for sig in cluster_sigs if sig not in ordered_sigs]
                ordered_sigs.extend(missing_sigs)
                self.current_grid_signatures = ordered_sigs
            else:
                self.current_grid_signatures = cluster_sigs

            self._update_grid_display()
        else:
            if hasattr(self, 'current_grid_signatures') and self.current_grid_signatures:
                self._update_grid_display()
            else:
                self._refresh_grid()

        self.status_var.set(f"Custom reference set for cluster {self.current_reference_cluster}")

    def _search_clusters(self):
        """Search clusters based on current search text, completion filter, and sort options"""
        # Get search text
        search_text = self.search_var.get().strip().lower()

        # Get completion type (renamed from filter_type)
        completion_type = self.search_filter_var.get()

        # Get sort option
        sort_option = self.sort_var.get() if hasattr(self, 'sort_var') else "Visual Similarity"

        # Save the search parameters as last APPLIED values
        self.last_applied_search_text = search_text
        self.last_applied_filter_type = completion_type
        self.last_applied_sort_option = sort_option

        self.selected_signatures = []
        self.last_selected_index = None

        # Populate cluster list with filtered results - using all parameters
        self._populate_cluster_selector(search_text, completion_type, sort_option)

        # Update status bar with more detailed information
        if search_text:
            self.status_var.set(f"Showing clusters sorted by {sort_option} " \
                                f"(Completion: {completion_type}, Query: '{search_text}')")
        else:
            self.status_var.set(f"Showing all clusters with completion: " \
                                f"{completion_type}, sort: {sort_option}")

    def _clear_search(self):
        """Reset the search field values without performing a search"""
        # Clear search text
        self.search_var.set("")

        # Reset filter and sort to defaults
        self.search_filter_var.set("Incomplete")
        self.sort_var.set("Visual Similarity")

        # Update status
        self.status_var.set("Search values reset to defaults (not applied)")

    def _toggle_cluster_completion(self):
        """Toggle completion status of the current cluster"""
        if not self.current_reference_cluster:
            return

        # Update completion status based on checkbox
        if self.complete_var.get():
            # Add to complete clusters
            self.complete_clusters.add(self.current_reference_cluster)
            self.status_var.set(f"Cluster {self.current_reference_cluster} marked as complete")
        else:
            # Remove from complete clusters
            if self.current_reference_cluster in self.complete_clusters:
                self.complete_clusters.remove(self.current_reference_cluster)
                self.status_var.set(f"Cluster {self.current_reference_cluster} " \
                                    "marked as incomplete")

        # Update the Next Cluster button state after changing completion status
        self._update_next_cluster_button_state()

        # Update the progress display to reflect the change
        # in signatures that are part of complete clusters
        self._update_progress_display()

        # For verification mode, update all cells immediately
        if self.current_mode == "VERIFICATION":
            # Update all frames that belong to the current reference cluster
            for frame in self.signature_frames:
                if hasattr(frame, 'cluster_id') and \
                    frame.cluster_id == self.current_reference_cluster:

                    # Update the frame image to reflect the new completion status
                    self._update_frame_image(frame, frame.signature_path)

        # Use the last applied parameters
        if hasattr(self, 'last_applied_search_text') and \
            hasattr(self, 'last_applied_filter_type') and hasattr(self, 'last_applied_sort_option'):

            self._populate_cluster_selector(
                self.last_applied_search_text,
                self.last_applied_filter_type,
                self.last_applied_sort_option
            )
        else:
            # Fallback to defaults if no applied values exist
            self._populate_cluster_selector("", "Incomplete", "Visual Similarity")

    def _has_next_cluster_with_same_status(self):
        """
        Check if there are more clusters with the same completion status as the current cluster.
        
        Returns:
            bool: True if there is at least one more cluster with the same completion status,
                False otherwise
        """
        if not self.current_reference_cluster:
            return False

        # Determine if current cluster is complete
        is_current_complete = self.current_reference_cluster in self.complete_clusters

        # Get list of cluster IDs
        cluster_ids = list(self.clusters.keys())

        # Skip empty clusters
        cluster_ids = [cid for cid in cluster_ids if self.clusters[cid]]

        if not cluster_ids:
            return False

        # If the current cluster isn't in the list (shouldn't happen), return False
        if self.current_reference_cluster not in cluster_ids:
            return False

        # Find current cluster index
        current_index = cluster_ids.index(self.current_reference_cluster)

        # Check clusters after the current one
        for i in range(current_index + 1, len(cluster_ids)):
            next_cluster = cluster_ids[i]
            # Check if this cluster has the same completion status
            if (next_cluster in self.complete_clusters) == is_current_complete:
                return True

        # If we reach the end, check from the beginning
        for i in range(0, current_index):
            next_cluster = cluster_ids[i]
            # Check if this cluster has the same completion status
            if (next_cluster in self.complete_clusters) == is_current_complete:
                return True

        # No more clusters with the same completion status
        return False

    def _has_remaining_incomplete_clusters(self):
        """
        Check if there are any remaining incomplete clusters besides the current one.
        
        Returns:
            bool: True if there are other incomplete clusters, False otherwise
        """
        if not self.current_reference_cluster:
            # If no current cluster, check if any incomplete clusters exist
            return any(cid not in self.complete_clusters for cid in self.clusters)

        # Count incomplete clusters other than the current one
        for cluster_id in self.clusters:
            if cluster_id != self.current_reference_cluster and \
                cluster_id not in self.complete_clusters:

                # Found at least one other incomplete cluster
                return True

        # No other incomplete clusters found
        return False

    def _update_next_cluster_button_state(self):
        """
        Update the state of the Next Cluster button based on whether
        there are  more incomplete clusters besides the current one.
        """
        if not hasattr(self, 'next_cluster_btn'):
            return

        if self.current_mode not in ["COMPLETION", "VERIFICATION"]:
            # In discovery mode, always enable the button if there are clusters
            self.next_cluster_btn.config(state=tk.NORMAL if self.clusters else tk.DISABLED)
            return

        # In completion or verification mode, check for more incomplete clusters
        if self._has_remaining_incomplete_clusters():
            self.next_cluster_btn.config(state=tk.NORMAL)
        else:
            self.next_cluster_btn.config(state=tk.DISABLED)

    def _navigate_reference_image(self, direction):
        """
        Navigate through images in the current reference cluster
        
        Args:
            direction: +1 for next, -1 for previous
        """
        try:
            # Check if reference cluster exists
            if not self.current_reference_cluster or \
                self.current_reference_cluster not in self.clusters:
                return

            # Get all signatures in this cluster
            signatures = self.clusters[self.current_reference_cluster]
            if not signatures:
                return

            # Get current displayed signature
            current_sig = self.current_displayed_signature or self.current_reference

            # Get ordered signatures for this cluster
            ordered_signatures = None
            if self.current_reference_cluster in self.cluster_ordered_signatures:
                ordered_signatures = self.cluster_ordered_signatures[self.current_reference_cluster]
            else:
                # Calculate and cache ordered signatures if not available
                ordered_signatures = \
                    self._get_cluster_signatures_by_similarity(self.current_reference_cluster)
                self.cluster_ordered_signatures[self.current_reference_cluster] = \
                    ordered_signatures.copy()

            if not ordered_signatures:
                return

            # Find index of current signature
            try:
                current_index = ordered_signatures.index(current_sig)
            except ValueError:
                # If not found, start from the beginning
                current_index = 0

            # Calculate new index
            new_index = current_index + direction

            # Check bounds
            if new_index < 0 or new_index >= len(ordered_signatures):
                # Out of bounds - do nothing
                return

            # Get the new signature we're navigating to
            new_sig = ordered_signatures[new_index]

            # Special handling for verification mode - swap with grid image
            if self.current_mode == "VERIFICATION":
                # Check if new_sig is currently in the grid
                found_in_grid = False

                for frame in self.signature_frames:
                    if not hasattr(frame, 'signature_path'):
                        continue

                    if frame.signature_path == new_sig:
                        try:
                            # Save the current displayed signature in the reference pane
                            old_sig = self.current_displayed_signature

                            # Clear the frame's canvas to remove any existing indicators
                            frame.canvas.delete("all")

                            # Now swap the signatures
                            # The grid frame now shows what was in the reference pane
                            frame.signature_path = old_sig

                            # Update the reference pane to display what was in the grid
                            self.current_displayed_signature = new_sig

                            # Redraw the frame with the swapped signature
                            # Update frame image completely -
                            # this will only add "R" for actual reference
                            self._update_frame_image(frame, old_sig)

                            found_in_grid = True
                            break
                        except Exception as e:
                            print(f"Error during verification swap: {e}")

                # If we didn't find the signature in the grid, just update normally
                if not found_in_grid:
                    self.current_displayed_signature = new_sig
            else:
                # Normal mode - just update the displayed signature
                self.current_displayed_signature = new_sig

            # Update display with new image
            self._update_reference_display()

        except Exception as e:
            # Log error and continue
            print(f"Error navigating reference image: {e}")

    def _get_cluster_signatures_by_similarity(self, cluster_id, force_recalculate=False):
        """
        Get signatures from a cluster ordered by similarity to cluster centroid using HNSW.
        Uses batched processing for larger clusters.

        Args:
            cluster_id: The cluster ID to get signatures from
            force_recalculate: Force recalculation even if cached result exists

        Returns:
            List of signature paths ordered by similarity to centroid (most similar first)
        """
        # Check cache first (unless force_recalculate is True)
        if not force_recalculate and hasattr(self, 'cluster_ordered_signatures'):
            if cluster_id in self.cluster_ordered_signatures:
                cached_result = self.cluster_ordered_signatures[cluster_id]
                if cached_result and all(sig in self.clusters[cluster_id] for sig in cached_result):
                    if len(cached_result) == len(self.clusters[cluster_id]):
                        print(f"Using cached ordering for cluster {cluster_id}")
                        return cached_result

        if cluster_id not in self.clusters:
            return []

        signatures = self.clusters[cluster_id]
        if len(signatures) <= 1:
            if hasattr(self, 'cluster_ordered_signatures'):
                self.cluster_ordered_signatures[cluster_id] = signatures.copy()
            return signatures

        # Extract features for all signatures in this cluster
        self._extract_features_for_signatures(signatures)

        # Get valid signatures with features
        valid_signatures = [sig for sig in signatures if sig in self.features_cache]
        if len(valid_signatures) <= 1:
            if hasattr(self, 'cluster_ordered_signatures'):
                self.cluster_ordered_signatures[cluster_id] = signatures.copy()
            return signatures

        # Calculate feature vectors
        feature_vectors = {
            sig: self._combine_features(self.features_cache[sig])
            for sig in valid_signatures
        }

        # Ensure all vectors are the same shape
        vec_shapes = {vec.shape for vec in feature_vectors.values()}
        if len(vec_shapes) != 1:
            print("Warning: Inconsistent feature vector shapes. Falling back to direct distance.")
            return self._fallback_signature_ordering(feature_vectors, valid_signatures,
                                                     signatures, cluster_id)

        # Calculate centroid
        all_vectors = np.array(list(feature_vectors.values()))
        centroid = np.mean(all_vectors, axis=0)

        # Validate centroid
        if not isinstance(centroid, np.ndarray) or np.isnan(centroid).any() or centroid.size == 0:
            print("Invalid centroid vector. Falling back to direct distance.")
            return self._fallback_signature_ordering(feature_vectors, valid_signatures,
                                                     signatures, cluster_id)

        # Process in smaller batches for large clusters
        batch_size = 1000
        sorted_signatures = []

        try:
            if len(valid_signatures) > batch_size:
                remaining = set(valid_signatures)

                while remaining:
                    batch_count = min(batch_size, len(remaining))
                    batch_neighbors = self.hnsw_index.get_nearest_neighbors(centroid, k=batch_count)

                    matched_in_batch = 0
                    for sig, _ in batch_neighbors:
                        if sig in remaining:
                            sorted_signatures.append(sig)
                            remaining.remove(sig)
                            matched_in_batch += 1

                    # If nothing was matched, add some random signatures to prevent infinite loop
                    if matched_in_batch == 0:
                        print("Warning: No matches in batch. Adding random signatures to continue.")
                        random_count = min(50, len(remaining))
                        random_sigs = random.sample(list(remaining), random_count)
                        sorted_signatures.extend(random_sigs)
                        remaining.difference_update(random_sigs)

            else:
                # For smaller clusters, do it in one go
                neighbors = self.hnsw_index.get_nearest_neighbors(centroid, k=len(valid_signatures))
                sorted_signatures = [sig for sig, _ in neighbors if sig in valid_signatures]

            # Add any signatures that weren't returned by HNSW (should be rare)
            missing_signatures = [sig for sig in valid_signatures if sig not in sorted_signatures]
            sorted_signatures.extend(missing_signatures)

        except Exception as e:
            print(f"Error using HNSW for cluster ordering: {e}")
            print("Falling back to direct distance calculation.")
            return self._fallback_signature_ordering(feature_vectors, valid_signatures,
                                                     signatures, cluster_id)

        # Add any signatures that don't have features at the end
        missing_signatures = [sig for sig in signatures if sig not in valid_signatures]
        sorted_signatures.extend(missing_signatures)

        # Cache the result
        if hasattr(self, 'cluster_ordered_signatures'):
            self.cluster_ordered_signatures[cluster_id] = sorted_signatures.copy()

        return sorted_signatures

    def _fallback_signature_ordering(self, feature_vectors, valid_signatures,
                                     all_signatures, cluster_id):
        """
        Fallback method to order signatures by direct distance to centroid.
        """
        distances = {}

        try:
            all_vectors = np.array(list(feature_vectors.values()))
            centroid = np.mean(all_vectors, axis=0)

            for sig, vec in feature_vectors.items():
                stacked = np.vstack([centroid, vec])
                metric = self.clustering_params.get('DISTANCE_METRIC', 'euclidean')
                try:
                    distance = pdist(stacked, metric=metric)[0]
                    distances[sig] = distance
                except Exception:
                    pass

            sorted_signatures = sorted(
                [sig for sig in valid_signatures if sig in distances],
                key=lambda sig: distances.get(sig, float('inf'))
            )

        except Exception as e:
            print(f"Fallback distance calculation failed: {e}")
            sorted_signatures = list(valid_signatures)

        # Add missing valid signatures
        missing_valid = [sig for sig in valid_signatures if sig not in sorted_signatures]
        sorted_signatures.extend(missing_valid)

        # Add invalid (non-featured) signatures at the end
        missing_invalid = [sig for sig in all_signatures if sig not in sorted_signatures]
        sorted_signatures.extend(missing_invalid)

        # Cache if applicable
        if hasattr(self, 'cluster_ordered_signatures'):
            self.cluster_ordered_signatures[cluster_id] = sorted_signatures.copy()

        return sorted_signatures

    def _update_reference_display(self):
        """Update the reference signature display with navigation buttons"""
        # Case 1: No reference cluster selected
        if not self.current_reference_cluster:
            # Clear the display when no cluster is selected
            self.reference_canvas.delete("all")
            self.reference_info.config(
                text="No reference selected",
                foreground="black",  # Use foreground instead of fg
                cursor="",   # No hand cursor
                font=("TkDefaultFont", 13) # Remove underline when no file
            )
            self.reference_info.unbind("<Button-1>")

            # Disable navigation buttons
            if hasattr(self, 'ref_prev_btn'):
                self.ref_prev_btn.config(state=tk.DISABLED)
            if hasattr(self, 'ref_next_btn'):
                self.ref_next_btn.config(state=tk.DISABLED)

            # Disable reference buttons
            if hasattr(self, 'use_auto_ref_btn'):
                self.use_auto_ref_btn.config(state=tk.DISABLED)
            if hasattr(self, 'set_custom_ref_btn'):
                self.set_custom_ref_btn.config(state=tk.DISABLED)

            # Clear cluster name entry and disable update button
            if hasattr(self, 'cluster_name_var'):
                self.cluster_name_var.set("")
                self.update_cluster_name_btn.config(state=tk.DISABLED)

            # Update completion checkbox state
            if hasattr(self, 'complete_var'):
                # Set checkbox based on completion status
                self.complete_var.set(self.current_reference_cluster in self.complete_clusters)
                # Enable checkbox in both completion and verification modes
                if self.current_mode in ["COMPLETION", "VERIFICATION"]:
                    self.complete_checkbox.config(state=tk.NORMAL)
                else:
                    self.complete_checkbox.config(state=tk.DISABLED)

            # Clear signature count
            if hasattr(self, 'signature_count_label'):
                self.signature_count_label.config(text="")

            # Update Next Cluster button state
            self._update_next_cluster_button_state()

            return

        # Case 2: A reference cluster is selected

        # If no displayed signature is set, use the reference
        if not self.current_displayed_signature:
            self.current_displayed_signature = self.current_reference

        # Make sure the cluster has its features extracted first
        if self.current_reference_cluster in self.clusters:
            self._extract_features_for_signatures(\
                self.clusters[self.current_reference_cluster][:10])

        try:
            # Clear canvas
            self.reference_canvas.delete("all")

            # Get canvas dimensions
            canvas_width = self.reference_canvas.winfo_width()
            canvas_height = self.reference_canvas.winfo_height()

            # Set default dimensions if canvas not initialized
            if canvas_width <= 1:
                canvas_width = 230  # Same as image_area_width from _rebuild_reference_frame
            if canvas_height <= 1:
                canvas_height = 120  # Same as image_area_height from _rebuild_reference_frame

            # IMPORTANT: For image sizing, use a constrained width to leave room for buttons
            image_display_width = 230  # Same as image_area_width from _rebuild_reference_frame

            # Determine if we need space for the "R" label
            is_reference = self.current_displayed_signature == self.current_reference
            reference_label_height = 23 if is_reference else 0

            # Calculate image area height (leave space for reference label if needed)
            image_area_height = canvas_height - reference_label_height

            # MODIFICATION: Preprocess the image instead of loading directly
            try:
                # Preprocess the image to crop to signature bounds
                img = self.preprocess_for_display(self.current_displayed_signature)

                if img is None:
                    # Error in preprocessing - try direct loading as fallback
                    img = Image.open(self.current_displayed_signature)

                # Calculate aspect ratio
                orig_width, orig_height = img.size
                img_aspect = orig_width / max(orig_height, 1)  # Avoid division by zero

                # Calculate dimensions to fit in canvas while preserving aspect ratio
                if img_aspect > image_display_width / image_area_height:  # Image is wider than tall
                    new_width = image_display_width
                    new_height = int(new_width / img_aspect)
                else:  # Image is taller than wide
                    new_height = image_area_height
                    new_width = int(new_height * img_aspect)

                # Resize image to fit canvas while maintaining aspect ratio
                img = img.resize((new_width, new_height), Image.LANCZOS if \
                                 hasattr(Image, 'LANCZOS') else Image.ANTIALIAS)

                img_tk = ImageTk.PhotoImage(img)
                self.reference_tk = img_tk  # Store reference to prevent garbage collection

                # Calculate position to center the image in the image area
                x_pos = (canvas_width - new_width) // 2
                y_pos = (image_area_height - new_height) // 2  # Center in the image area

                # Display on canvas
                self.reference_canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=img_tk)

                # If this is the reference image, add the "R" label at the bottom-right
                if is_reference:
                    # Calculate position for "R" label - bottom right of the canvas
                    oval_size = 16
                    margin = 5

                    # Position in bottom-right, leaving margin
                    oval_x = canvas_width - oval_size - margin
                    oval_y = canvas_height - oval_size - margin

                    # Ensure it's fully visible
                    if oval_y + oval_size > canvas_height:
                        oval_y = canvas_height - oval_size

                    # MODIFICATION: Determine color based on whether the reference is user-selected
                    is_user_selected = \
                        self.current_reference_cluster in self.user_selected_references

                    # Yellow for user-selected, blue for automatic
                    fill_color = "#FFFF00" if is_user_selected else "#40C4FF"

                    # Create the indicator with background
                    self.reference_canvas.create_oval(
                        oval_x, oval_y,
                        oval_x + oval_size, oval_y + oval_size,
                        fill=fill_color, outline="#000000",
                        tags="reference_indicator"
                    )

                    self.reference_canvas.create_text(
                        oval_x + oval_size // 2, oval_y + oval_size // 2,
                        text="R", fill="#000000",
                        font=("TkDefaultFont", 8, "bold"),
                        tags="reference_indicator"
                    )

                # Update reference info with truncated filename to prevent panel expansion
                filename = os.path.basename(self.current_displayed_signature)

                # Create a temporary canvas for text measurement
                temp_canvas = tk.Canvas(self.root, width=1, height=1)

                # Get the actual width available in the pane
                actual_pane_width = self.left_frame.winfo_width()
                if actual_pane_width <= 1:  # Not yet rendered
                    actual_pane_width = 200  # Default width from _rebuild_reference_frame

                # Use minimal padding to allow maximum text display
                max_filename_width = actual_pane_width - 20  # Reduced padding

                # Create text for measurement
                text_id = temp_canvas.create_text(0, 0, text=filename, anchor="w")
                text_bbox = temp_canvas.bbox(text_id)

                # Check if truncation is needed using pure pixel-based approach
                if text_bbox and (text_bbox[2] - text_bbox[0]) > max_filename_width:
                    # Need to truncate the filename
                    for i in range(len(filename) - 1, 2, -1):
                        # Try progressively shorter versions with ellipsis
                        truncated = filename[:i] + "..."

                        # Remove previous text and create new for measurement
                        temp_canvas.delete("all")
                        trunc_id = temp_canvas.create_text(0, 0, text=truncated, anchor="w")
                        trunc_bbox = temp_canvas.bbox(trunc_id)

                        # Check if this fits
                        if trunc_bbox and (trunc_bbox[2] - trunc_bbox[0]) <= max_filename_width:
                            filename = truncated
                            break

                # Destroy temporary canvas
                temp_canvas.destroy()

                # Update the label with (possibly) truncated filename
                self.reference_info.config(
                    text=filename,
                    foreground="black",
                    cursor="hand2" if self.current_displayed_signature else "",
                    font=("TkDefaultFont", 13, "underline")  # Ensure underline is maintained
                )
                self.reference_info.bind("<Button-1>", \
                                         lambda _: self._open_signatures_in_file_explorer(\
                                             called_from_menu_option=False))

            except Exception as e:
                # Display error on canvas
                print(f"Error processing reference image: {e}")
                self.reference_canvas.create_text(canvas_width//2, canvas_height//2, \
                                                  text="Error loading image", fill="red")
                self.reference_info.config(text=f"Error: {str(e)[:20]}...")

                # Disable navigation buttons on error
                if hasattr(self, 'ref_prev_btn'):
                    self.ref_prev_btn.config(state=tk.DISABLED)
                if hasattr(self, 'ref_next_btn'):
                    self.ref_next_btn.config(state=tk.DISABLED)
                return

            # Get all signatures in the cluster ordered by similarity
            if self.current_reference_cluster in self.clusters:
                ordered_signatures = \
                    self._get_cluster_signatures_by_similarity(self.current_reference_cluster)

                # Enable/disable navigation buttons based on position
                if ordered_signatures:
                    try:
                        current_index = ordered_signatures.index(self.current_displayed_signature)
                        # Enable/disable prev button
                        if hasattr(self, 'ref_prev_btn') and self.ref_prev_btn:
                            self.ref_prev_btn.config(\
                                state=tk.NORMAL if current_index > 0 else tk.DISABLED)

                        # Enable/disable next button
                        if hasattr(self, 'ref_next_btn') and self.ref_next_btn:
                            self.ref_next_btn.config(\
                                state=tk.NORMAL if current_index < len(ordered_signatures) - 1 \
                                    else tk.DISABLED)

                    except ValueError:
                        # Signature not found in ordered list
                        if hasattr(self, 'ref_prev_btn') and self.ref_prev_btn:
                            self.ref_prev_btn.config(state=tk.DISABLED)
                        if hasattr(self, 'ref_next_btn') and self.ref_next_btn:
                            self.ref_next_btn.config(state=tk.DISABLED)

            # MODIFICATION: Update reference mode buttons
            if hasattr(self, 'use_auto_ref_btn') and hasattr(self, 'set_custom_ref_btn'):
                # Only enable the buttons in completion and verification modes
                if self.current_mode in ["COMPLETION", "VERIFICATION"]:
                    is_user_selected = \
                        self.current_reference_cluster in self.user_selected_references

                    # "Use Automatic Reference" button - enable only if currently user-selected
                    self.use_auto_ref_btn.config(
                        state=tk.NORMAL if is_user_selected else tk.DISABLED)

                    # "Set Custom Reference" button - enable if displayed signature isn't
                    # the reference or if it is the reference but not user-selected
                    can_set_custom = (self.current_displayed_signature != \
                                      self.current_reference) or not is_user_selected
                    self.set_custom_ref_btn.config(\
                        state=tk.NORMAL if can_set_custom else tk.DISABLED)
                else:
                    # Disable both buttons in discovery mode
                    self.use_auto_ref_btn.config(state=tk.DISABLED)
                    self.set_custom_ref_btn.config(state=tk.DISABLED)

            # Update cluster info and name entry
            if self.current_reference_cluster:
                cluster_size = len(self.clusters.get(self.current_reference_cluster, []))

                # Set cluster name in entry widget and enable update button
                if hasattr(self, 'cluster_name_var'):
                    self.cluster_name_var.set(self.current_reference_cluster)
                    self.update_cluster_name_btn.config(state=tk.NORMAL)

                # Update completion checkbox state
                if hasattr(self, 'complete_var'):
                    # Set checkbox based on completion status
                    self.complete_var.set(self.current_reference_cluster in self.complete_clusters)
                    # Enable/disable based on mode -
                    # enabled in both completion and verification modes
                    if self.current_mode in ["COMPLETION", "VERIFICATION"]:
                        self.complete_checkbox.config(state=tk.NORMAL)
                    else:
                        self.complete_checkbox.config(state=tk.DISABLED)

                # Update signature count
                if hasattr(self, 'signature_count_label'):
                    signature_count_text = f"({cluster_size} signatures)"
                    self.signature_count_label.config(text=signature_count_text)

                # CRITICAL FIX: Always save the current reference
                # to the displayed signatures dictionary
                # This ensures it persists even when switching modes or clusters
                if not hasattr(self, 'cluster_displayed_signatures'):
                    self.cluster_displayed_signatures = {}

                # Only update if the reference signature is in the cluster (it should be)
                if self.current_reference in self.clusters[self.current_reference_cluster]:
                    self.cluster_displayed_signatures[self.current_reference_cluster] = \
                        self.current_reference
                    print(f"Persisting reference for cluster {self.current_reference_cluster}: " \
                          f"{os.path.basename(self.current_reference)}")

            else:
                # No cluster
                if hasattr(self, 'cluster_name_var'):
                    self.cluster_name_var.set("")
                    self.update_cluster_name_btn.config(state=tk.DISABLED)

                # Disable completion checkbox
                if hasattr(self, 'complete_var'):
                    self.complete_var.set(False)
                    self.complete_checkbox.config(state=tk.DISABLED)

                if hasattr(self, 'signature_count_label'):
                    self.signature_count_label.config(text="Not in a cluster")

            # Update the Next Cluster button state
            self._update_next_cluster_button_state()

            if self.current_mode == "COMPLETION":
                self._calculate_cluster_sizes()

        except Exception as e:
            traceback.print_exc()
            # Display error on canvas
            self.reference_canvas.create_text(100, 75, text="Error loading image", fill="red")
            self.reference_info.config(text=f"Error: {str(e)}")

            # Disable navigation buttons on error
            if hasattr(self, 'ref_prev_btn'):
                self.ref_prev_btn.config(state=tk.DISABLED)
            if hasattr(self, 'ref_next_btn'):
                self.ref_next_btn.config(state=tk.DISABLED)

            # Disable reference buttons on error
            if hasattr(self, 'use_auto_ref_btn'):
                self.use_auto_ref_btn.config(state=tk.DISABLED)
            if hasattr(self, 'set_custom_ref_btn'):
                self.set_custom_ref_btn.config(state=tk.DISABLED)

            # Disable completion checkbox on error
            if hasattr(self, 'complete_var'):
                self.complete_var.set(False)
                self.complete_checkbox.config(state=tk.DISABLED)

    def _update_cluster_name(self):
        """Update the name of the current cluster"""
        if not self.current_reference_cluster or not hasattr(self, 'cluster_name_var'):
            return

        new_name = self.cluster_name_var.get().strip()

        # Validate new name
        if not new_name:
            messagebox.showerror("Invalid Name", "Cluster name cannot be empty")
            # Reset to current name
            self.cluster_name_var.set(self.current_reference_cluster)
            return

        # Check if name contains only allowed characters
        if not self._is_valid_cluster_name(new_name):
            messagebox.showerror("Invalid Name",
                            "Cluster names may only contain alphanumeric " \
                                "characters (a-z, A-Z, 0-9), underscores (_), " \
                                    "hyphens (-), periods (.) and spaces.")
            # Reset to current name
            self.cluster_name_var.set(self.current_reference_cluster)
            return

        # Check if cluster name exists
        cluster_name_match = None
        for existing_cluster_name in self.clusters:
            if existing_cluster_name.lower() == new_name.lower():
                cluster_name_match = existing_cluster_name
                break

        # Check if name already exists
        if cluster_name_match is not None and \
            new_name.lower() != self.current_reference_cluster.lower():

            messagebox.showerror("Name Exists", 'A cluster with the name ' \
                                 f'"{cluster_name_match}" already exists.')
            # Reset to current name
            self.cluster_name_var.set(self.current_reference_cluster)
            return

        # Get the cluster's signatures
        signatures = self.clusters[self.current_reference_cluster]

        # Check if the cluster is complete
        was_complete = self.current_reference_cluster in self.complete_clusters

        # MODIFICATION: Check if the cluster has a user-selected reference
        had_user_selected_reference = \
            self.current_reference_cluster in self.user_selected_references

        # Cache the current reference and displayed signature so we can restore them
        current_ref = self.current_reference
        current_displayed = self.current_displayed_signature

        # Store reference in cluster_displayed_signatures for the new name
        if self.current_reference and \
            self.current_reference_cluster in self.cluster_displayed_signatures:

            ref_sig = self.cluster_displayed_signatures[self.current_reference_cluster]
            # We'll restore this to the new cluster name later
        else:
            ref_sig = None

        # Remove old cluster
        del self.clusters[self.current_reference_cluster]

        # Remove old cluster from complete clusters if it was complete
        self.complete_clusters.discard(self.current_reference_cluster)

        # MODIFICATION: Remove old cluster from
        # user_selected_references if it had a user-selected reference
        self.user_selected_references.discard(self.current_reference_cluster)

        # Create cluster with new name
        self.clusters[new_name] = signatures

        # Add new cluster to complete clusters if old one was complete
        if was_complete:
            self.complete_clusters.add(new_name)

        # MODIFICATION: Add new cluster to user_selected_references
        # if old one had a user-selected reference
        if had_user_selected_reference:
            self.user_selected_references.add(new_name)

        # Restore reference signature for the new cluster name
        if ref_sig:
            self.cluster_displayed_signatures[new_name] = ref_sig

        # Update current reference cluster
        old_reference_cluster = self.current_reference_cluster
        self.current_reference_cluster = new_name

        # Restore current reference and displayed signature
        self.current_reference = current_ref
        self.current_displayed_signature = current_displayed

        # Update the display
        self._update_reference_display()

        self._populate_cluster_selector(self.last_applied_search_text, \
                                        self.last_applied_filter_type, \
                                            self.last_applied_sort_option)

        # Special handling for verification mode to avoid duplicate signatures in the grid
        if self.current_mode == "VERIFICATION":

            # Get the signature currently displayed in the left pane
            # This is the signature that should NOT appear in the grid
            current_left_pane = self.current_displayed_signature

            # CORRECTLY rebuild grid - exclude ONLY the signature showing in the left pane
            cluster_sigs = [sig for sig in self.clusters[self.current_reference_cluster]
                        if sig != current_left_pane]

            print(f"Building verification grid after rename with {len(cluster_sigs)} " \
                  "signatures (excluding only left pane signature)")

            # Get the ordered version if available
            if self.current_reference_cluster in self.cluster_ordered_signatures:
                ordered_sigs = \
                    self.cluster_ordered_signatures[self.current_reference_cluster].copy()
                # Remove left pane signature
                if current_left_pane in ordered_sigs:
                    ordered_sigs.remove(current_left_pane)
                # Keep only signatures in the cluster
                ordered_sigs = [sig for sig in ordered_sigs if sig in cluster_sigs]
                # Add any missing signatures
                missing_sigs = [sig for sig in cluster_sigs if sig not in ordered_sigs]
                ordered_sigs.extend(missing_sigs)
                # Use the ordered list
                self.current_grid_signatures = ordered_sigs
            else:
                # Just use the cluster signatures
                self.current_grid_signatures = cluster_sigs

            # Clear selected signatures
            self.selected_signatures = []

            # Rebuild the grid
            self._update_grid_display()
        else:
            # For other modes, a regular refresh works
            if hasattr(self, 'current_grid_signatures') and self.current_grid_signatures:
                self._update_grid_display()
            else:
                # If no signatures exist, do a full refresh
                self._refresh_grid()

        self.cluster_displayed_signatures.pop(old_reference_cluster, None)

        self.status_var.set(f"Updated cluster name to '{new_name}'")

    def _is_valid_cluster_name(self, name):
        """
        Check if a cluster name contains only allowed characters.
        Allowed characters are alphanumeric (a-z, A-Z, 0-9),
        underscore (_), hyphen (-), period (.), and space.
        
        Args:
            name: Cluster name to validate
            
        Returns:
            bool: True if name is valid, False otherwise
        """
        return bool(re.match(r'^[a-zA-Z0-9_\-\.\s]+$', name))

    def _sanitize_cluster_name(self, name):
        """
        Replace any disallowed characters in a cluster name with underscores.
        
        Args:
            name: Cluster name to sanitize
            
        Returns:
            str: Sanitized cluster name
        """
        return re.sub(r'[^a-zA-Z0-9_\-\.\s]', '_', name)

    def _update_progress_display(self):
        """Update the progress display"""
        if self.total_signatures > 0:
            # Calculate clustered progress
            progress = self.clustered_signatures / self.total_signatures
            self.clustered_label.config(\
                text=f"{self.clustered_signatures}/{self.total_signatures} ({progress:.1%})")

            # Calculate signatures in complete clusters
            signatures_in_complete_clusters = 0
            for cluster_id in self.complete_clusters:
                if cluster_id in self.clusters:
                    signatures_in_complete_clusters += len(self.clusters[cluster_id])

            # Update complete signatures display
            complete_progress = signatures_in_complete_clusters / self.total_signatures
            self.complete_label.config(\
                text=f"{signatures_in_complete_clusters}/{self.total_signatures} " \
                    f"({complete_progress:.1%})")

    def _update_button_labels(self):
        """Update button labels based on current mode"""
        if self.current_mode == "DISCOVERY":
            # We don't use these buttons in discovery mode
            # anymore, but keep references for compatibility

            self.new_cluster_btn.config(text="New Cluster (N)")

            # REMOVED: Update to reject_btn text - it no longer exists in discovery mode UI
            if hasattr(self, 'reject_btn'):
                self.reject_btn.config(text="Reject (X)")
        elif self.current_mode == "COMPLETION":
            self.group_btn.config(text="Add to Cluster (G)")
            self.new_cluster_btn.config(text="New Cluster (N)")
            self.reject_btn.config(text="Reject (X)")
        elif self.current_mode == "VERIFICATION":
            self.new_cluster_btn.config(text="Move to New Cluster (N)")
            self.reject_btn.config(text="Remove from Cluster (X)")

    def _set_ui_enabled(self, enabled):
        """Enable or disable UI elements"""
        state = tk.NORMAL if enabled else tk.DISABLED

        # Buttons
        self.group_btn.config(state=state)
        self.new_cluster_btn.config(state=state)
        self.reject_btn.config(state=state)
        self.refresh_btn.config(state=state)
        # view_clusters_btn removed
        self.change_ref_btn.config(state=state)
        self.next_cluster_btn.config(state=state)

    # The following methods are stubs for menu items

    def _get_relative_path(self, absolute_path):
        """
        Convert an absolute path to a path relative to the base_directory.
        
        Args:
            absolute_path: The absolute path to convert
            
        Returns:
            str: Path relative to base_directory or the original path if conversion fails
        """
        try:
            # Handle case where path could be None
            if not absolute_path:
                return absolute_path

            # Normalize paths to handle different path separators
            norm_abs_path = os.path.normpath(absolute_path)
            norm_base_dir = os.path.normpath(self.base_directory)

            # Check if the path starts with the base directory
            if norm_abs_path.startswith(norm_base_dir):
                # Get the relative path
                rel_path = os.path.relpath(norm_abs_path, norm_base_dir)
                return rel_path
            else:
                # If the path is not under the base directory, return the original
                # This could happen for paths outside the project directory
                return absolute_path
        except Exception as e:
            print(f"Error converting to relative path: {e}")
            return absolute_path

    def _get_absolute_path(self, relative_path, base_dir=None):
        """
        Convert a relative path to an absolute path based
        on the specified or current base_directory.
        
        Args:
            relative_path: The relative path to convert
            base_dir: Optional base directory to use instead of self.base_directory
            
        Returns:
            str: Absolute path or the original path if conversion fails
        """
        try:
            # Handle case where path could be None
            if not relative_path:
                return relative_path

            # Handle paths that are already absolute
            if os.path.isabs(relative_path):
                return relative_path

            # Use provided base_dir or default to self.base_directory
            base_directory = base_dir if base_dir is not None else self.base_directory

            # Create the absolute path
            abs_path = os.path.normpath(os.path.join(base_directory, relative_path))
            return abs_path
        except Exception as e:
            print(f"Error converting to absolute path: {e}")
            return relative_path

    def _create_zip_archive(self, zip_filename, temp_dir):
        """
        Create a ZIP archive from a prepared temporary directory.
        
        Args:
            zip_filename: Path to the ZIP file to create
            temp_dir: Path to the temporary directory containing prepared files
            show_progress: Whether to show progress in status bar
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:

            start_time = datetime.now()

            # Check whether a ZIP with the desired name already exists, and if it does,
            # store the saved progress in a ZIP with a different name and rename that ZIP
            # back to the original only after all progress has been written. This prevents
            # the old ZIP from getting deleted until the new one has been created (serving as
            # a protection against accidental deletion of previous saves in case of errors).
            temp_zip_filename = zip_filename
            if os.path.exists(zip_filename):
                zip_pref, zip_ext = os.path.splitext(zip_filename)
                i = 1
                while os.path.exists(f"{zip_pref}_{i}{zip_ext}"):
                    i += 1
                temp_zip_filename = f"{zip_pref}_{i}{zip_ext}"

            # Create ZIP file from the temp directory
            self.status_var.set("Creating ZIP archive...")
            self.root.update()
            shutil.make_archive(
                os.path.splitext(temp_zip_filename)[0],  # Base name without extension
                'zip',                              # Format
                temp_dir                           # Root directory to start from
            )

            # Clean up temporary directory
            self.status_var.set("Cleaning up temporary files...")
            self.root.update()
            shutil.rmtree(temp_dir)

            # If there is an existing ZIP, remove it and rename the new ZIP to the intended name.
            if temp_zip_filename != zip_filename:
                self.status_var.set("Deleting old ZIP archive...")
                self.root.update()
                os.remove(zip_filename)
                os.rename(temp_zip_filename, zip_filename)

            time_diff = datetime.now() - start_time
            print(f"Progress transferred to ZIP in {str(time_diff).split('.', maxsplit=1)[0]}\n")

            self.status_var.set(f"Archive created successfully: {zip_filename}")
            return True

        except Exception as e:
            traceback.print_exc()
            self.status_var.set(f"Error creating archive: {str(e)}")
            return False

    def _save_progress(self, filename=None):
        """Save current clustering progress with combined vectors and HNSW index in a ZIP archive"""
        # If we called this function from the "Save As..." menu option...
        if filename is None:
            filename = filedialog.asksaveasfilename(
                title="Save Progress",
                defaultextension=".zip",
                filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")],
                initialdir=os.path.expanduser("~")
            )

        # If we called this function from the "Save" menu option,
        # but the provided file path does not exist...
        elif not os.path.isfile(filename):
            messagebox.showerror("Save File Not Found",
                "No current save file could be found. Either save your current progress "
                "in a new file or load your progress from an existing file.")

        if not filename:
            return

        try:
            overall_start_time = datetime.now()

            # Block UI during save operation
            self._set_ui_enabled(False)
            self.status_var.set("Preparing to save progress...")
            self.root.update()

            self.status_var.set("Copying progress to dictionary...")
            self.root.update()

            start_time = datetime.now()

            # Create a serializable state object with relative paths
            state = {
                "clusters": {},
                "unclustered_signatures": [],
                "current_reference": None,
                "current_reference_cluster": self.current_reference_cluster,
                "cannot_link_constraints": [],
                "complete_clusters": list(self.complete_clusters),
                "user_selected_references": list(self.user_selected_references),
                "cluster_displayed_signatures": {},
            }

            # Convert all paths to relative paths
            self.status_var.set("Converting to relative paths...")
            self.root.update()

            # Serialize clusters with relative paths
            for cluster_id, signatures in self.clusters.items():
                state["clusters"][cluster_id] = [self._get_relative_path(sig) for sig in signatures]

            # Convert unclustered signatures to relative paths
            state["unclustered_signatures"] = \
                [self._get_relative_path(sig) for sig in self.unclustered_signatures]

            # Convert current reference to relative path
            if self.current_reference:
                state["current_reference"] = self._get_relative_path(self.current_reference)

            # Convert cannot-link constraints to relative paths
            for sig1, sig2 in self.cannot_link_constraints:
                rel_sig1 = self._get_relative_path(sig1)
                rel_sig2 = self._get_relative_path(sig2)
                state["cannot_link_constraints"].append((rel_sig1, rel_sig2))

            # Convert cannot_link_map to relative paths
            cannot_link_serialized = {}
            for sig, constraints in self.cannot_link_map.items():
                rel_sig = self._get_relative_path(sig)
                cannot_link_serialized[rel_sig] = [self._get_relative_path(s) for s in constraints]

            state["cannot_link_map"] = cannot_link_serialized

            # Convert cluster_displayed_signatures to relative paths
            for cluster_id, sig in self.cluster_displayed_signatures.items():
                state["cluster_displayed_signatures"][cluster_id] = self._get_relative_path(sig)

            # Add base directory info for reference
            state["original_base_directory"] = self.base_directory

            # NEW: Add discovery_grid_layout to state if it exists and is not empty
            if hasattr(self, 'discovery_grid_layout') and self.discovery_grid_layout:
                self.status_var.set(f"Saving discovery grid layout with " \
                                    f"{len(self.discovery_grid_layout)} signatures...")
                self.root.update()
                state["discovery_grid_layout"] = \
                    [self._get_relative_path(sig) for sig in self.discovery_grid_layout]
                print("Added discovery grid layout to save file: " \
                      f"{len(self.discovery_grid_layout)} signatures")

            self.status_var.set("Creating temp directory for vectors and HNSW index...")
            self.root.update()

            # Create temp directory for saving vectors and index
            temp_dir = os.path.join(os.path.dirname(filename), "temp_export")
            os.makedirs(temp_dir, exist_ok=True)

            self.status_var.set("Saving progress to JSON...")
            self.root.update()

            # Save the state to a JSON file in the temp directory
            progress_file = os.path.join(temp_dir, "progress.json")
            with open(progress_file, 'w', encoding="utf-8") as f:
                json.dump(state, f, indent=2)

            time_diff = datetime.now() - start_time
            print(f"\nProgress copied to JSON in {str(time_diff).split('.', maxsplit=1)[0]}\n")

            # Save combined vectors as NPZ file
            if self.combined_vectors_cache:
                prev_percentage = 0
                self.status_var.set("Copying combined vectors to dictionary: 0%")
                self.root.update()

                start_time = datetime.now()

                # Create a dictionary of relative paths -> numpy arrays
                vector_dict = {}
                for i, (sig, vector) in enumerate(self.combined_vectors_cache.items()):
                    rel_sig = self._get_relative_path(sig)
                    # Replace / with _ in keys since NPZ uses paths internally
                    safe_key = rel_sig.replace('/', '_').replace('\\', '_')
                    vector_dict[safe_key] = vector

                    # Update progress
                    cur_percentage = int(((i + 1) / len(self.combined_vectors_cache)) * 100)
                    if cur_percentage > prev_percentage:
                        self.status_var.set(\
                            f"Copying combined vectors to dictionary: {cur_percentage}%")
                        prev_percentage = cur_percentage
                        self.root.update()

                self.status_var.set("Mapping combined vector safe keys to relative paths...")
                self.root.update()

                # Create a mapping from safe_keys back to relative paths
                key_mapping = {
                    rel_sig.replace('/', '_').replace('\\', '_'): rel_sig
                    for rel_sig in \
                        [self._get_relative_path(sig) for sig in self.combined_vectors_cache]
                }

                self.status_var.set("Saving combined vector key mapping...")
                self.root.update()

                # Save the key mapping separately
                key_mapping_file = os.path.join(temp_dir, "vector_keys.json")
                with open(key_mapping_file, 'w', encoding="utf-8") as f:
                    json.dump(key_mapping, f)

                self.status_var.set("Saving vectors in NPZ file...")
                self.root.update()

                # Save vectors as NPZ file
                vectors_file = os.path.join(temp_dir, "combined_vectors.npz")
                np.savez_compressed(vectors_file, **vector_dict)

                time_diff = datetime.now() - start_time
                print("Combined vectors copied to NPZ in "
                      f"{str(time_diff).split('.', maxsplit=1)[0]}\n")

            # Save HNSW index if available
            if hasattr(self, 'hnsw_index') and \
                self.hnsw_index is not None and self.hnsw_index.index is not None:

                self.status_var.set("Saving HNSW index...")
                self.root.update()

                start_time = datetime.now()

                # Save index to temp directory
                index_file = os.path.join(temp_dir, "hnsw_index.bin")
                if self.hnsw_index.save_index(index_file):

                    # Add flag to state indicating index is saved
                    state["hnsw_index_saved"] = True

                    # Update progress.json with the updated state
                    with open(progress_file, 'w', encoding="utf-8") as f:
                        json.dump(state, f, indent=2)

                    time_diff = datetime.now() - start_time
                    print(f"HNSW index copied in {str(time_diff).split('.', maxsplit=1)[0]}\n")

                else:
                    print("\nFailed to save HNSW index\n")

            # Collect all signature paths that need to be copied
            self.status_var.set("Collecting signatures...")
            self.root.update()

            all_signatures = set()

            # Add unclustered signatures
            all_signatures.update(self.unclustered_signatures)

            # Add signatures from each cluster
            for cluster_signatures in self.clusters.values():
                all_signatures.update(cluster_signatures)

            # Add current reference if it exists
            if self.current_reference:
                all_signatures.add(self.current_reference)

            # Copy images while preserving directory structure
            self.status_var.set("Copying signatures: 0%")
            self.root.update()

            prev_percentage = 0

            start_time = datetime.now()

            # Create signatures directory in temp folder
            signatures_dir = os.path.join(temp_dir, "signatures")
            os.makedirs(signatures_dir, exist_ok=True)

            # Count for progress updates
            for i, sig_path in enumerate(all_signatures):
                # Get the relative path
                rel_path = self._get_relative_path(sig_path)

                # Create destination path in the signatures directory
                dest_path = os.path.join(signatures_dir, rel_path)

                # Create parent directories if they don't exist
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                # Copy the file
                try:
                    shutil.copy2(sig_path, dest_path)
                except Exception as e:
                    print(f"Error copying {sig_path}: {e}")

                # Update progress
                cur_percentage = int(((i + 1) / len(all_signatures)) * 100)
                if cur_percentage > prev_percentage:
                    self.status_var.set(f"Copying signatures: {cur_percentage}%")
                    prev_percentage = cur_percentage
                    self.root.update()

            time_diff = datetime.now() - start_time
            print(f"Signatures copied in {str(time_diff).split('.', maxsplit=1)[0]}\n")

            # Create the ZIP archive with all data
            success = self._create_zip_archive(filename, temp_dir)

            if success:
                self.save_file = filename

                overall_time_diff = datetime.now() - overall_start_time
                overall_time_diff_truncated = str(overall_time_diff).split('.', maxsplit=1)[0]
                print(f"Total time to save progress: {overall_time_diff_truncated}\n")

                self.status_var.set("Progress saved in " \
                                    f"{overall_time_diff_truncated} to {filename}")
            else:
                print("\nError creating ZIP archive\n")
                self.status_var.set("Error creating ZIP archive...")

            # Re-enable UI
            self._set_ui_enabled(True)

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to save progress: {str(e)}")
            self.status_var.set("Error saving progress")

            # Clean up temp directory if it exists
            temp_dir = os.path.join(os.path.dirname(filename), "temp_export")
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass

            # Re-enable UI
            self._set_ui_enabled(True)

    def _extract_zip_archive(self, zip_filename, target_base_dir):
        """
        Extract a ZIP archive and load progress JSON with combined vectors and HNSW index.
        """
        try:
            # Create temporary directory for extraction
            temp_dir = os.path.join(os.path.dirname(zip_filename), "temp_import")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)

            self.root.update()

            # Extract ZIP file
            prev_percentage = 0
            self.status_var.set("Extracting files from ZIP archive: 0%")
            self.root.update()

            start_time = datetime.now()

            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                # Get total size of all files in the ZIP
                total_size = sum(file_info.file_size for file_info in zip_ref.infolist())

                # Initialize extracted size counter
                extracted_size = 0

                # Extract files with progress updates
                for file_info in zip_ref.infolist():
                    zip_ref.extract(file_info.filename, temp_dir)

                    # Update extracted size
                    extracted_size += file_info.file_size

                    # Update progress
                    cur_percentage = int((extracted_size / total_size) * 100)
                    if cur_percentage > prev_percentage:
                        self.status_var.set(f"Extracting files from ZIP archive: {cur_percentage}%")
                        prev_percentage = cur_percentage
                        self.root.update()

            time_diff = datetime.now() - start_time
            print("\nExtracted files from ZIP archive in "
                  f"{str(time_diff).split('.', maxsplit=1)[0]}\n")

            self.status_var.set("Loading progress from JSON...")
            self.root.update()

            # Load progress data from JSON
            progress_file = os.path.join(temp_dir, "progress.json")
            if not os.path.exists(progress_file):
                raise Exception("Progress file not found in ZIP archive")

            with open(progress_file, 'r', encoding="utf-8") as f:
                progress_data = json.load(f)

            # Convert all relative paths in progress data to absolute paths

            self.status_var.set("Converting file paths from relative to absolute...")
            self.root.update()

            # Convert clusters
            clusters = {}
            for cluster_id, rel_signatures in progress_data.get("clusters", {}).items():
                clusters[cluster_id] = [self._get_absolute_path(\
                    rel_sig, target_base_dir) for rel_sig in rel_signatures]
            progress_data["clusters"] = clusters

            # Convert unclustered signatures
            progress_data["unclustered_signatures"] = [
                self._get_absolute_path(rel_sig, target_base_dir)
                for rel_sig in progress_data.get("unclustered_signatures", [])
            ]

            # Convert current reference
            if progress_data.get("current_reference"):
                progress_data["current_reference"] = self._get_absolute_path(
                    progress_data["current_reference"], target_base_dir
                )

            # Convert cannot-link constraints
            if "cannot_link_constraints" in progress_data:
                constraints = []
                for rel_sig1, rel_sig2 in progress_data["cannot_link_constraints"]:
                    abs_sig1 = self._get_absolute_path(rel_sig1, target_base_dir)
                    abs_sig2 = self._get_absolute_path(rel_sig2, target_base_dir)
                    constraints.append((abs_sig1, abs_sig2))
                progress_data["cannot_link_constraints"] = constraints

            # Convert cannot_link_map
            if "cannot_link_map" in progress_data:
                cannot_link_map = {}
                for rel_sig, rel_constraints in progress_data["cannot_link_map"].items():
                    abs_sig = self._get_absolute_path(rel_sig, target_base_dir)
                    cannot_link_map[abs_sig] = set(
                        self._get_absolute_path(rel_constraint, target_base_dir)
                        for rel_constraint in rel_constraints
                    )
                progress_data["cannot_link_map"] = cannot_link_map

            # Convert cluster_displayed_signatures
            if "cluster_displayed_signatures" in progress_data:
                displayed_sigs = {}
                for cluster_id, rel_sig in progress_data["cluster_displayed_signatures"].items():
                    displayed_sigs[cluster_id] = self._get_absolute_path(rel_sig, target_base_dir)
                progress_data["cluster_displayed_signatures"] = displayed_sigs

            start_time = datetime.now()

            # Check for and load NPZ combined vectors
            vectors_file = os.path.join(temp_dir, "combined_vectors.npz")
            if os.path.exists(vectors_file):
                self.status_var.set("Loading combined vectors from NPZ file: 0%")
                self.root.update()

                # Load key mapping
                key_mapping_file = os.path.join(temp_dir, "vector_keys.json")
                if os.path.exists(key_mapping_file):
                    with open(key_mapping_file, 'r', encoding="utf-8") as f:
                        key_mapping = json.load(f)

                    # Load vectors
                    with np.load(vectors_file) as data:

                        prev_percentage = 0

                        # Convert arrays using key mapping to get proper paths
                        combined_vectors = {}
                        for i, safe_key in enumerate(data.files):
                            if safe_key in key_mapping:
                                rel_sig = key_mapping[safe_key]
                                abs_sig = self._get_absolute_path(rel_sig, target_base_dir)
                                # Ensure proper memory layout and dtype when loading
                                # This prevents segmentation faults by
                                # ensuring vectors are C-contiguous and float64
                                try:
                                    vector_data = data[safe_key]
                                    if vector_data is not None and vector_data.size > 0:
                                        # Create a new array with specific layout and dtype
                                        combined_vectors[abs_sig] = np.array(\
                                            vector_data, dtype=np.float64, order='C').flatten()
                                    else:
                                        print(f"Warning: Empty vector for {abs_sig}, skipping")
                                except Exception as e:
                                    print(f"Error loading vector for {abs_sig}: {e}")

                            # Update progress
                            cur_percentage = int(((i + 1) / len(data.files)) * 100)
                            if cur_percentage > prev_percentage:
                                self.status_var.set("Loading combined vectors " \
                                                    f"from NPZ file: {cur_percentage}%")
                                prev_percentage = cur_percentage
                                self.root.update()

                        # Add vectors to progress data
                        progress_data["combined_vectors_loaded"] = True
                        # Store the actual vectors separately to avoid redundant conversion
                        progress_data["combined_vectors_dict"] = combined_vectors

                        print(f"Loaded {len(combined_vectors)} combined vectors from NPZ file")
                else:
                    print("Vector key mapping file not found, cannot load vectors")

            else:
                progress_data["combined_vectors_loaded"] = False
                print("No combined vectors found in save file")

            time_diff = datetime.now() - start_time
            print("\nLoaded combined vectors from NPZ file in "
                  f"{str(time_diff).split('.', maxsplit=1)[0]}\n")

            # Check for and load HNSW index
            index_file = os.path.join(temp_dir, "hnsw_index.bin")
            index_metadata = os.path.join(temp_dir, "hnsw_index.bin.metadata")
            if os.path.exists(index_file) and os.path.exists(index_metadata):
                self.status_var.set("Loading HNSW index...")
                self.root.update()

                # Create a new HNSW index instance
                hnsw_index = HNSWIndex()

                # Load the index
                if hnsw_index.load_index(index_file):
                    # Update the paths in the HNSW index mappings
                    original_base_dir = progress_data.get("original_base_directory", "")

                    if original_base_dir:
                        print("Updating paths in HNSW index mappings from " \
                              f"{original_base_dir} to {target_base_dir}")

                        # Update signature_to_id mapping
                        updated_signature_to_id = {}
                        for old_path, id_val in list(hnsw_index.signature_to_id.items()):
                            try:
                                # Try to compute relative path from original base directory
                                rel_path = os.path.relpath(old_path, original_base_dir)

                                # Construct new absolute path using target base directory
                                new_path = os.path.normpath(os.path.join(target_base_dir, rel_path))

                                # Update mapping
                                updated_signature_to_id[new_path] = id_val
                            except ValueError:
                                # If the old path is not relative to the original base directory,
                                # we can't compute a relative path
                                print(f"Warning: Could not compute relative path for {old_path}")
                                # Keep the old path in this case
                                updated_signature_to_id[old_path] = id_val

                        # Update id_to_signature mapping
                        updated_id_to_signature = {}
                        for id_val, old_path in list(hnsw_index.id_to_signature.items()):
                            try:
                                # Try to compute relative path from original base directory
                                rel_path = os.path.relpath(old_path, original_base_dir)

                                # Construct new absolute path using target base directory
                                new_path = os.path.normpath(os.path.join(target_base_dir, rel_path))

                                # Update mapping
                                updated_id_to_signature[id_val] = new_path
                            except ValueError:
                                # If the old path is not relative to the original base directory,
                                # we can't compute a relative path
                                print(f"Warning: Could not compute relative path for {old_path}")
                                # Keep the old path in this case
                                updated_id_to_signature[id_val] = old_path

                        # Update vector_cache mapping if it exists
                        if hasattr(hnsw_index, 'vector_cache') and hnsw_index.vector_cache:
                            updated_vector_cache = {}
                            for old_path, vector in list(hnsw_index.vector_cache.items()):
                                try:
                                    # Try to compute relative path from original base directory
                                    rel_path = os.path.relpath(old_path, original_base_dir)

                                    # Construct new absolute path using target base directory
                                    new_path = \
                                        os.path.normpath(os.path.join(target_base_dir, rel_path))

                                    # Update mapping
                                    updated_vector_cache[new_path] = vector
                                except ValueError:
                                    # If the old path is not relative to the original
                                    # base directory, we can't compute a relative path
                                    print("Warning: Could not compute " \
                                          f"relative path for {old_path}")
                                    # Keep the old path in this case
                                    updated_vector_cache[old_path] = vector

                            # Replace vector_cache with updated one
                            hnsw_index.vector_cache = updated_vector_cache

                        # Replace mappings with updated ones
                        hnsw_index.signature_to_id = updated_signature_to_id
                        hnsw_index.id_to_signature = updated_id_to_signature

                        print("Updated paths in HNSW index mappings: " \
                              f"{len(updated_signature_to_id)} signatures")
                    else:
                        print("Warning: Original base directory not available, " \
                              "could not update paths in HNSW index mappings")

                    # Add the loaded index to progress data
                    progress_data["hnsw_index_loaded"] = True
                    progress_data["hnsw_index"] = hnsw_index
                    print("HNSW index loaded successfully")
                else:
                    print("Failed to load HNSW index")
                    progress_data["hnsw_index_loaded"] = False
            else:
                print("HNSW index files not found, skipping index load")
                progress_data["hnsw_index_loaded"] = False

            # Clean up temporary directory
            self.status_var.set("Cleaning up temporary files...")
            self.root.update()

            # Keep the "signatures" subdirectory if using it as the new base directory
            if target_base_dir == os.path.join(temp_dir, "signatures"):
                # If we're using the extracted signatures folder as the base directory,
                # don't delete it - just remove the other files
                for item in os.listdir(temp_dir):
                    item_path = os.path.join(temp_dir, item)
                    if item != "signatures" and os.path.exists(item_path):
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
            else:
                # Otherwise, clean up everything
                shutil.rmtree(temp_dir)

            return progress_data

        except Exception as e:
            traceback.print_exc()
            self.status_var.set(f"Error extracting archive: {str(e)}")
            return None

    def _load_progress(self):
        """Load saved clustering progress from a ZIP archive with complete portability"""
        filename = filedialog.askopenfilename(
            title="Load Progress",
            filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")]
        )
        if not filename:
            return

        try:
            overall_start_time = datetime.now()

            # Block UI during load operation
            self._set_ui_enabled(False)
            self.status_var.set("Loading progress...")
            self.root.update()

            # Create temp directory to extract the ZIP
            temp_dir = os.path.join(os.path.dirname(filename), "temp_import")

            # Always use the extracted signatures as the base directory
            target_base_dir = os.path.join(temp_dir, "signatures")
            self.base_directory = target_base_dir

            # Extract ZIP and load progress data
            progress_data = self._extract_zip_archive(filename, target_base_dir)

            if not progress_data:
                self._set_ui_enabled(True)
                self.status_var.set("Error loading progress")
                return

            # Reset current state
            self.clusters = {}
            self.unclustered_signatures = []
            self.cannot_link_constraints = []
            self.cannot_link_map = {}
            self.complete_clusters = set()
            self.user_selected_references = set()
            self.cluster_displayed_signatures = {}

            # IMPORTANT: Initialize cluster_ordered_signatures dictionary
            self.cluster_ordered_signatures = {}

            # Reset feature and combined vector caches - we'll repopulate from saved data
            self.features_cache = {}  # Keep empty features cache for compatibility
            self.combined_vectors_cache = {}  # Will be populated from saved data

            # Clear other caches
            if hasattr(self, 'preprocessed_image_cache'):
                self.preprocessed_image_cache = {}

            # Clear any image-related caches for grid cells
            if hasattr(self, 'discovery_current_grid'):
                self.discovery_current_grid = []

            # Clear mode-specific grid caches to force fresh rendering
            if hasattr(self, 'mode_grid_cache'):
                self.mode_grid_cache = {}

            # Clear full signature lists to force recalculation
            self.full_signature_lists = {
                "DISCOVERY": [],
                "COMPLETION": [],
                "VERIFICATION": []
            }

            # Clear shown signatures to ensure fresh selection
            self.shown_signatures = set()

            # Apply loaded state
            self.clusters = progress_data.get("clusters", {})
            self.unclustered_signatures = progress_data.get("unclustered_signatures", [])
            self.current_reference = progress_data.get("current_reference")
            self.current_reference_cluster = progress_data.get("current_reference_cluster")
            self.cannot_link_constraints = progress_data.get("cannot_link_constraints", [])

            # Restore complete clusters
            self.complete_clusters = set(progress_data.get("complete_clusters", []))

            # Restore user-selected references
            self.user_selected_references = set(progress_data.get("user_selected_references", []))

            # Restore cluster displayed signatures
            if "cluster_displayed_signatures" in progress_data:
                self.cluster_displayed_signatures = \
                    progress_data.get("cluster_displayed_signatures", {})
            else:
                self.cluster_displayed_signatures = {}

            # Restore cannot_link_map
            if "cannot_link_map" in progress_data:
                self.cannot_link_map = progress_data["cannot_link_map"]
            else:
                # Backward compatibility: rebuild from constraints list
                for sig1, sig2 in progress_data.get("cannot_link_constraints", []):
                    self._add_cannot_link_constraint(sig1, sig2)

            # NEW: Restore discovery_grid_layout if it exists
            if "discovery_grid_layout" in progress_data and progress_data["discovery_grid_layout"]:
                # Convert relative paths to absolute
                self.discovery_grid_layout = [
                    self._get_absolute_path(rel_sig, target_base_dir)
                    for rel_sig in progress_data["discovery_grid_layout"]
                ]

                # Also update the full_signature_lists for discovery mode
                self.full_signature_lists["DISCOVERY"] = self.discovery_grid_layout.copy()

                # Set flag to prevent fresh arrangement on first discovery mode usage
                self.discovery_grid_needs_fresh_arrangement = False

                print("Loaded discovery grid layout with " \
                      f"{len(self.discovery_grid_layout)} signatures")
            else:
                # If no discovery grid layout in saved data, initialize empty
                self.discovery_grid_layout = []
                # Force a fresh arrangement when entering discovery mode
                self.discovery_grid_needs_fresh_arrangement = True
                print("No discovery grid layout found in saved data, will arrange fresh grid")

            # IMPROVED VECTOR HANDLING: Check if combined vectors were loaded
            if progress_data.get("combined_vectors_loaded", False) and \
                "combined_vectors_dict" in progress_data:

                prev_percentage = 0
                self.status_var.set("Transferring combined vectors to cache: 0%")
                self.root.update()

                start_time = datetime.now()

                # Carefully copy and validate each vector instead of direct dictionary assignment
                self.combined_vectors_cache = {}
                for i, (sig, vector) in enumerate(progress_data["combined_vectors_dict"].items()):
                    try:
                        if vector is not None and vector.size > 0:
                            # Create a new copy with proper layout to avoid memory issues
                            self.combined_vectors_cache[sig] = \
                                np.array(vector, dtype=np.float64, order='C').flatten()
                    except Exception as e:
                        print(f"Error processing vector for {os.path.basename(sig)}: {e}")

                    # Update status periodically
                    cur_percentage = int((i + 1) / len(\
                        progress_data["combined_vectors_dict"]) * 100)
                    if cur_percentage > prev_percentage:
                        self.status_var.set(\
                            f"Transferring combined vectors to cache: {cur_percentage}%")
                        prev_percentage = cur_percentage
                        self.root.update()

                time_diff = datetime.now() - start_time
                print("\nTransferred combined vectors to cache in " \
                      f"{str(time_diff).split('.', maxsplit=1)[0]}\n")

                # Remove the dictionary from progress_data to free memory
                del progress_data["combined_vectors_dict"]

                # Flag to indicate we've loaded vectors and don't need to extract features
                self.vectors_preloaded = True
            else:
                # Legacy or no vectors available
                print("\nNo combined vectors found in save file. " \
                      "Features will be computed as needed.\n")
                self.vectors_preloaded = False

            # Check if HNSW index was loaded
            if progress_data.get("hnsw_index_loaded", False) and "hnsw_index" in progress_data:
                self.status_var.set("Initializing loaded HNSW index...")
                self.root.update()

                # Use the loaded index
                self.hnsw_index = progress_data["hnsw_index"]

                # Remove the index from progress_data to free memory
                del progress_data["hnsw_index"]

                print("Using pre-loaded HNSW index with " \
                      f"{self.hnsw_index.get_indexed_count()} signatures")

                # Flag to indicate we've loaded the index
                self.index_preloaded = True
            else:
                # No index available
                print("No HNSW index found in save file. Index will be initialized as needed.")
                self.hnsw_index = None
                self.index_preloaded = False

            # Update counts
            self.clustered_signatures = sum(len(signatures) for \
                                            signatures in self.clusters.values())
            self.total_signatures = self.clustered_signatures + len(self.unclustered_signatures)

            # Update progress display
            self._update_progress_display()

            # Update mode-specific UI elements
            self._update_mode_specific_ui()

            # Update the reference display
            self._update_reference_display()

            # Reset to page 1 when loading
            self.current_page[self.current_mode] = 1

            # For completion mode, initialize filter controls
            if self.current_mode == "COMPLETION":
                # Apply last applied grid search parameters
                if hasattr(self, 'last_applied_grid_membership'):
                    self.membership_var.set(self.last_applied_grid_membership)
                    self.grid_filter_var.set(self.last_applied_grid_filter)
                    self.sort_completion_var.set(self.last_applied_grid_sort)
                    self.use_name_query_var.set(self.last_applied_grid_use_name_query)
                    self.name_query_var.set(self.last_applied_grid_name_query)
                    self.rejection_filter_var.set(self.last_applied_rejection_filter)

                    # Update state handlers
                    self._handle_membership_change()
                    self._update_name_query_entry_state()
                    self._update_name_query_checkbox_state()
                else:
                    self._apply_grid_search_parameters(use_defaults=True)

            # Update cluster selector if applicable
            if self.current_mode in ["COMPLETION", "VERIFICATION"]:
                if hasattr(self, 'last_applied_search_text') and \
                    hasattr(self, 'last_applied_filter_type') and \
                        hasattr(self, 'last_applied_sort_option'):

                    self._populate_cluster_selector(
                        self.last_applied_search_text,
                        self.last_applied_filter_type,
                        self.last_applied_sort_option
                    )
                else:
                    self._populate_cluster_selector("", "Incomplete", "Visual Similarity")

            # Initialize HNSW from combined vectors if needed (and not already loaded)
            if not self.index_preloaded and self.combined_vectors_cache:
                self.status_var.set("Initializing HNSW index from combined vectors...")
                self.root.update()
                self._build_hnsw_index_from_combined_vectors()

            # Final step: refresh the grid
            self.status_var.set("Finalizing display...")
            self.root.update()
            self._refresh_grid()

            # Setup mousewheel scrolling for the refreshed UI
            self._setup_mousewheel_scrolling()

            # Set save file path
            self.save_file = filename

            # Re-enable UI
            self._set_ui_enabled(True)

            overall_time_diff = datetime.now() - overall_start_time
            overall_time_diff_truncated = str(overall_time_diff).split('.', maxsplit=1)[0]
            print(f"\nTotal time to load progress: {overall_time_diff_truncated}\n")

            self.status_var.set(f"Progress loaded in {overall_time_diff_truncated} from {filename}")

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to load progress: {str(e)}")
            self.status_var.set("Error loading progress")

            # CRITICAL: Reset state to avoid segmentation fault
            self.clusters = {}
            self.unclustered_signatures = []
            self.cannot_link_constraints = []
            self.cannot_link_map = {}
            self.complete_clusters = set()
            self.user_selected_references = set()
            self.cluster_displayed_signatures = {}
            self.cluster_ordered_signatures = {}
            self.features_cache = {}
            self.combined_vectors_cache = {}
            self.hnsw_index = None

            # Clear caches
            if hasattr(self, 'preprocessed_image_cache'):
                self.preprocessed_image_cache = {}
            if hasattr(self, 'discovery_current_grid'):
                self.discovery_current_grid = []
            if hasattr(self, 'mode_grid_cache'):
                self.mode_grid_cache = {}

            # Reset full signature lists
            self.full_signature_lists = {
                "DISCOVERY": [],
                "COMPLETION": [],
                "VERIFICATION": []
            }

            # Re-enable UI
            self._set_ui_enabled(True)

    def _show_save_before_quit_dialog(self):
        """Show save before quit dialog when user tries to exit"""
        # Create modal dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Save Progress")
        dialog.geometry("400x150")
        dialog.transient(self.root)
        dialog.grab_set()  # Make it modal
        dialog.resizable(False, False)

        # Center the dialog on the parent window
        dialog.update_idletasks()
        parent_x = self.root.winfo_x()
        parent_y = self.root.winfo_y()
        parent_width = self.root.winfo_width()
        parent_height = self.root.winfo_height()

        dialog_width = dialog.winfo_width()
        dialog_height = dialog.winfo_height()

        x = parent_x + (parent_width // 2) - (dialog_width // 2)
        y = parent_y + (parent_height // 2) - (dialog_height // 2)
        dialog.geometry(f"+{x}+{y}")

        # Result variable to track user choice
        result = {'action': None}

        # Message
        message_frame = ttk.Frame(dialog, padding="20")
        message_frame.pack(expand=True, fill='both')

        ttk.Label(message_frame, text="Save progress before quitting?",
                font=("TkDefaultFont", 12)).pack(pady=(0, 20))

        # Button frame
        button_frame = ttk.Frame(message_frame)
        button_frame.pack(side='bottom')

        def on_cancel():
            result['action'] = 'cancel'
            dialog.destroy()

        def on_dont_save():
            result['action'] = 'dont_save'
            dialog.destroy()

        def on_save():
            result['action'] = 'save'
            dialog.destroy()

        # Buttons (in reverse order so Tab navigation goes Cancel -> Don't Save -> Save)
        save_btn = ttk.Button(button_frame, text="Save", command=on_save)
        save_btn.pack(side='right', padx=(5, 0))

        dont_save_btn = ttk.Button(button_frame, text="Don't Save", command=on_dont_save)
        dont_save_btn.pack(side='right', padx=(5, 0))

        cancel_btn = ttk.Button(button_frame, text="Cancel", command=on_cancel)
        cancel_btn.pack(side='right', padx=(5, 0))

        # Set focus to Cancel button by default
        cancel_btn.focus_set()

        # Handle window close button (X) - should act like Cancel
        dialog.protocol("WM_DELETE_WINDOW", on_cancel)

        # Wait for user to make a choice
        dialog.wait_window()

        # Process the result
        if result['action'] == 'cancel':
            # Do nothing, just return
            return
        elif result['action'] == 'dont_save':
            # Quit without saving
            self.root.quit()
        elif result['action'] == 'save':
            # Try to save, then quit if successful
            if self._save_and_quit():
                self.root.quit()
            else:
                # If save was cancelled or failed, show dialog again
                self._show_save_before_quit_dialog()

    def _save_and_quit(self):
        """Save progress and return True if successful, False if cancelled"""
        try:
            # Check if we have a valid save file that actually exists
            if (hasattr(self, 'save_file') and self.save_file and \
                os.path.isfile(self.save_file)):
                # We have an existing save file, use it
                self._save_progress(self.save_file)
                return True
            else:
                # No valid save file exists, do Save As by calling _save_progress() without args
                # Store the current save_file value to detect if it changed
                old_save_file = getattr(self, 'save_file', None)

                self._save_progress()

                # If save_file was updated, save was successful
                new_save_file = getattr(self, 'save_file', None)
                if new_save_file and new_save_file != old_save_file:
                    return True
                else:
                    # Save was cancelled (user closed Save As dialog without selecting a file)
                    return False
        except Exception:
            # Save failed with exception - error message already shown by _save_progress
            return False

    def _change_thumbnail_size(self, change):
        """Change the thumbnail size"""
        new_width = self.thumbnail_size[0] + change
        new_height = self.thumbnail_size[1] + change

        if new_width < 100 or new_height < 80:
            return

        if new_width > 400 or new_height > 300:
            return

        self.thumbnail_size = (new_width, new_height)

        # Update canvases
        for frame in self.signature_frames:
            frame.canvas.config(width=new_width, height=new_height)

        # Refresh the grid
        self._refresh_grid()

    def _initialize_feature_extractor(self):
        """Initialize the feature extractor with current parameters"""
        if self.feature_extractor is None:
            self.feature_extractor = SignatureFeatureExtractor(self.clustering_params)
            self.clustering = SignatureClustering(self.clustering_params)

        # Initialize feature cache if not already done
        if not hasattr(self, 'features_cache'):
            self.features_cache = {}

    def _extract_features_for_signatures(self, signatures, callback=None):
        """
        Extract features for a list of signatures and cache them.
        FIXED VERSION: Reduces redundant processing and improves performance.
        
        Args:
            signatures: List of signature paths
            callback: Optional function to call when extraction is complete
        """
        # Filter out None values and ensure we have strings
        valid_signatures = []
        for sig in signatures:
            if sig and isinstance(sig, (str, bytes, os.PathLike)):
                valid_signatures.append(sig)
            elif sig:
                print(f"Warning: Invalid signature type: {type(sig)} - {sig}")

        if not valid_signatures:
            return []

        # Initialize feature extractor if needed
        self._initialize_feature_extractor()

        # Track signatures that need processing
        signatures_needing_extraction = []
        processed_signatures = []

        # Check what we already have to avoid redundant work
        for sig_path in valid_signatures:
            # If we have combined vector, we're done
            if sig_path in self.combined_vectors_cache:
                processed_signatures.append(sig_path)
                # Ensure features_cache has placeholder if needed
                if sig_path not in self.features_cache:
                    self.features_cache[sig_path] = (None, None, None, None, None, None)
            # If we have features but no combined vector, create combined vector
            elif sig_path in self.features_cache:
                if sig_path not in self.combined_vectors_cache:
                    try:
                        feature_tuple = self.features_cache[sig_path]
                        combined = self._combine_features(feature_tuple)
                        self.combined_vectors_cache[sig_path] = combined
                    except Exception as e:
                        print("Error creating combined vector for "
                              f"{os.path.basename(sig_path)}: {e}")
                processed_signatures.append(sig_path)
            else:
                # Need to extract features
                signatures_needing_extraction.append(sig_path)

        # If no extraction needed, return early (but only print message if substantial number)
        if not signatures_needing_extraction:
            if len(processed_signatures) > 10:  # Only log for substantial batches
                print(f"No feature extraction needed - all {len(processed_signatures)} "
                      "signatures already have vectors")
            if callback:
                callback(processed_signatures)
            return processed_signatures

        # Extract features for signatures that need it
        if signatures_needing_extraction:
            self.status_var.set(f"Extracting features for {len(signatures_needing_extraction)} "
                                "signatures...")
            self.root.update()

            prev_percentage = 0
            start_time = datetime.now()

            for i, sig_path in enumerate(signatures_needing_extraction):
                try:
                    # Extract features
                    feature_tuple = self.feature_extractor.extract_features(sig_path)

                    # Cache the features
                    if any(f is not None for f in feature_tuple):
                        self.features_cache[sig_path] = feature_tuple
                        processed_signatures.append(sig_path)

                        # Calculate and cache combined vector
                        combined = self._combine_features(feature_tuple)
                        self.combined_vectors_cache[sig_path] = combined

                    cur_percentage = int((i + 1) / len(signatures_needing_extraction) * 100)
                    if cur_percentage > prev_percentage:
                        self.status_var.set(f"Extracting features: {cur_percentage}%")
                        prev_percentage = cur_percentage
                        self.root.update()

                except Exception as e:
                    print(f"Error extracting features for {sig_path}: {e}")

            time_diff = datetime.now() - start_time
            if len(signatures_needing_extraction) > 10:  # Only log timing for substantial batches
                print(f"Feature extraction completed for {len(signatures_needing_extraction)} "
                      f"signatures in {str(time_diff).split('.', maxsplit=1)[0]}")

        # Update status
        self.status_var.set(f"Processed features for {len(processed_signatures)} signatures")

        # Call callback if provided
        if callback:
            callback(processed_signatures)

        return processed_signatures

    def _rank_all_candidates_for_reference(self):
        """
        Modified version: Comprehensive ranking of ALL unclustered signatures
        for the current reference using HNSW with improved caching.
        """
        if not self.current_reference:
            return []

        # Ensure reference has features
        if self.current_reference not in self.features_cache:
            self._extract_features_for_signatures([self.current_reference])
            if self.current_reference not in self.features_cache:
                self.status_var.set("Cannot extract features for reference signature")
                return []  # Cannot rank without reference features

        self.status_var.set("Ranking candidates for completion mode...")
        self.root.update()

        # For optimization, track all candidates
        total_signatures = len(self.unclustered_signatures)
        all_candidates = []

        # Optimize for memory usage by processing signatures in batches
        batch_size = 500  # Process at most 500 signatures at a time to limit memory usage

        # For very small datasets, process everything directly
        if total_signatures <= batch_size:
            # Extract features for all unclustered signatures
            self._extract_features_for_signatures(self.unclustered_signatures)

            # Calculate distances directly using HNSW when possible
            for sig in self.unclustered_signatures:
                if sig in self.features_cache:
                    # Calculate distance using updated method
                    distance = self._calculate_distance(self.current_reference, sig)

                    if distance is not None:
                        all_candidates.append((sig, distance))
        else:
            # For larger datasets, use batched processing
            remaining = set(self.unclustered_signatures)
            processed = set()

            # First, check HNSW index for already processed signatures
            if hasattr(self, 'hnsw_index') and self.hnsw_index is not None:
                # Get reference vector
                ref_features = self.features_cache[self.current_reference]
                ref_vector = self._combine_features(ref_features)

                # Add reference to HNSW if not already indexed
                if not self.hnsw_index.is_indexed(self.current_reference):
                    self.hnsw_index.add_vector(self.current_reference, ref_vector)

                # Check for existing candidates
                for sig in list(remaining):
                    if self.hnsw_index.is_indexed(sig):
                        distance = self.hnsw_index.get_distance(self.current_reference, sig)
                        if distance is not None:
                            all_candidates.append((sig, distance))
                            processed.add(sig)

            # Remove processed signatures from remaining
            remaining -= processed

            # Process remaining signatures in batches
            prev_percentage = 0
            while remaining:
                # Get a batch of unprocessed signatures
                batch = list(remaining)[:batch_size]

                # Extract features for this batch
                self._extract_features_for_signatures(batch)

                # Calculate distances
                for sig in batch:
                    if sig in self.features_cache:
                        distance = self._calculate_distance(self.current_reference, sig)
                        if distance is not None:
                            all_candidates.append((sig, distance))

                    # Mark as processed
                    remaining.remove(sig)

                    # Update progress periodically
                    cur_percentage = \
                        int((total_signatures - len(remaining)) / total_signatures * 100)
                    if cur_percentage > prev_percentage:
                        self.status_var.set(f"Ranking candidates: {cur_percentage}% complete")
                        prev_percentage = cur_percentage
                        self.root.update()

        # Sort by similarity (ascending distance)
        all_candidates.sort(key=lambda x: x[1])

        # Update status with the results
        self.status_var.set(f"Ranked {len(all_candidates)} candidates for completion mode")

        return all_candidates

    def _export_clusters(self):
        """Export final clusters"""
        if not self.clusters:
            messagebox.showinfo("No Clusters", "No clusters to export")
            return

        if not self.output_directory:
            # Ask for output directory if not set
            output_dir = filedialog.askdirectory(\
                title="Select Output Directory for Exported Clusters")
            if not output_dir:
                return
            self.output_directory = output_dir

        try:
            self.status_var.set("Exporting clusters...")
            self.root.update()

            # Create main output directory if it doesn't exist
            if not os.path.exists(self.output_directory):
                os.makedirs(self.output_directory)

            # Create a timestamped subfolder for this export
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_path = os.path.join(self.output_directory, f"exported_clusters_{timestamp}")
            os.makedirs(export_path, exist_ok=True)

            # Export each cluster using actual cluster names
            for cluster_id, signatures in self.clusters.items():
                # Sanitize cluster name for use as folder name
                sanitized_name = self._sanitize_cluster_name(str(cluster_id))
                cluster_dir = os.path.join(export_path, sanitized_name)

                # Handle potential directory name conflicts
                original_dir = cluster_dir
                counter = 1
                while os.path.exists(cluster_dir):
                    cluster_dir = f"{original_dir}_{counter}"
                    counter += 1

                os.makedirs(cluster_dir, exist_ok=True)

                # Copy all signatures to the cluster directory
                for sig_path in signatures:
                    # Create the destination path
                    dest_name = os.path.basename(sig_path)
                    dest_path = os.path.join(cluster_dir, dest_name)

                    # Handle potential name conflicts
                    if os.path.exists(dest_path):
                        # Add a unique suffix to the filename
                        base, ext = os.path.splitext(dest_name)
                        dest_name = f"{base}_{datetime.now().strftime('%H%M%S%f')}{ext}"
                        dest_path = os.path.join(cluster_dir, dest_name)

                    # Copy the file
                    shutil.copy2(sig_path, dest_path)

            # Export unclustered signatures to the top level
            for sig_path in self.unclustered_signatures:
                dest_name = os.path.basename(sig_path)
                dest_path = os.path.join(export_path, dest_name)

                # Handle potential name conflicts
                if os.path.exists(dest_path):
                    # Add a unique suffix to the filename
                    base, ext = os.path.splitext(dest_name)
                    dest_name = f"{base}_{datetime.now().strftime('%H%M%S%f')}{ext}"
                    dest_path = os.path.join(export_path, dest_name)

                # Copy the file
                shutil.copy2(sig_path, dest_path)

            # Create metadata file with clustering information
            metadata_path = os.path.join(export_path, "clustering_metadata.json")
            with open(metadata_path, 'w', encoding="utf-8") as f:
                metadata = {
                    "export_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "num_clusters": len(self.clusters),
                    "num_unclustered": len(self.unclustered_signatures),
                    "total_signatures": sum(len(sigs) for sigs in self.clusters.values()) + \
                        len(self.unclustered_signatures),
                    "original_directory": self.base_directory,
                    "clusters": {},
                    "unclustered_filenames": \
                        [os.path.basename(sig) for sig in self.unclustered_signatures]
                }

                # Add cluster information using actual cluster names
                for cluster_id, signatures in self.clusters.items():
                    metadata["clusters"][str(cluster_id)] = {
                        "size": len(signatures),
                        "signature_filenames": [os.path.basename(sig) for sig in signatures]
                    }

                # Save as JSON
                json.dump(metadata, f, indent=2)

            # Show success message
            total_clustered = sum(len(sigs) for sigs in self.clusters.values())
            messagebox.showinfo("Export Complete", f"Exported {len(self.clusters)} "
                                f"clusters ({total_clustered} signatures) "
                                f"and {len(self.unclustered_signatures)} "
                                f"unclustered signatures to {export_path}")
            self.status_var.set(f"Exported {len(self.clusters)} clusters "
                                f"and {len(self.unclustered_signatures)} "
                                "unclustered signatures successfully")

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Export Error", f"Failed to export clusters: {str(e)}")
            self.status_var.set("Error exporting clusters")

    def _show_help(self):
        """Show help information"""
        help_text = """
Signature Clustering Assistant

Operating Modes:
- Discovery: Find and create new clusters
- Completion: Ensure all signatures for a cluster are found
- Verification: Check and fix existing clusters

Keyboard Shortcuts:
- G: Group selected signatures
- N: Create new cluster from selection
- X: Reject selected signatures
- R: Refresh grid
- Space: Toggle selection of focused item

For more information, see documentation.
        """
        messagebox.showinfo("Help", help_text)

    def _show_about(self):
        """Show about information"""
        about_text = """
Signature Clustering Assistant

A tool for human-guided clustering of historical signatures.

Version: 1.0
        """
        messagebox.showinfo("About", about_text)


# Main application entry point
if __name__ == "__main__":
    tk_root = tk.Tk()
    app = SignatureClusteringApp(tk_root)
    tk_root.mainloop()
