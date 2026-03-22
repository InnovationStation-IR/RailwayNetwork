#!/usr/bin/env python3
"""
Comprehensive Testing Framework for Railway Simulation

This module provides unit tests, integration tests, and performance tests
for the enhanced railway simulation system.
"""

import unittest
import tempfile
import os
import json
import time
import threading
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock

# Import the simulation modules
try:
    from simulation.enhanced import (
        EnhancedSimulation, SimulationConfig, SimulationMode,
        EnhancedTrain, EnhancedTrackBlock, EnhancedSignal,
        ODPair, TrainType, Priority, SignalAspect,
        Colors, create_sample_od_pairs, run_enhanced_simulation
    )
    SIMULATION_AVAILABLE = True
except ImportError:
    SIMULATION_AVAILABLE = False

try:
    from utils.performance import (
        PerformanceManager, PerformanceConfig, PerformanceMonitor,
        SpatialIndex, PathCache, MemoryManager, LRUCache
    )
    PERFORMANCE_AVAILABLE = True
except ImportError:
    PERFORMANCE_AVAILABLE = False

# ---------------------------
# Test Data Generation
# ---------------------------

def create_test_graph_data() -> Dict[str, Any]:
    """Create test railway graph data"""
    return {
        "nodes": [
            {"id": 1, "label": "Station A", "pos_x": 0.0, "pos_y": 0.0},
            {"id": 2, "label": "Station B", "pos_x": 100.0, "pos_y": 0.0},
            {"id": 3, "label": "Station C", "pos_x": 50.0, "pos_y": 50.0},
            {"id": 4, "label": "Junction 1", "pos_x": 25.0, "pos_y": 25.0},
        ],
        "edges": [
            {"source": 1, "target": 4, "distance": 35.36},
            {"source": 4, "target": 2, "distance": 35.36},
            {"source": 1, "target": 3, "distance": 70.71},
            {"source": 3, "target": 2, "distance": 70.71},
            {"signals": [
                {"id": "sig_1_to_4_starter", "position": {"x": 12.5, "y": 12.5}},
                {"id": "sig_4_to_2_starter", "position": {"x": 37.5, "y": 12.5}},
            ]}
        ]
    }

def create_test_od_pairs() -> List[ODPair]:
    """Create test OD pairs"""
    return [
        ODPair("1", "2", 0.0, Priority.NORMAL, TrainType.PASSENGER),
        ODPair("1", "3", 10.0, Priority.HIGH, TrainType.EXPRESS),
        ODPair("2", "1", 20.0, Priority.NORMAL, TrainType.FREIGHT),
    ]

# ---------------------------
# Base Test Classes
# ---------------------------

class SimulationTestCase(unittest.TestCase):
    """Base test case for simulation tests"""
    
    def setUp(self):
        """Set up test environment"""
        if not SIMULATION_AVAILABLE:
            self.skipTest("Simulation modules not available")
        
        # Create temporary test data file
        self.test_data = create_test_graph_data()
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(self.test_data, self.temp_file)
        self.temp_file.close()
        
        # Create test configuration
        self.test_config = SimulationConfig(
            episode_duration=10.0,
            initial_trains=3,
            snapshot_interval=1.0,
            seed=42
        )
    
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'temp_file'):
            os.unlink(self.temp_file.name)

class PerformanceTestCase(unittest.TestCase):
    """Base test case for performance tests"""
    
    def setUp(self):
        """Set up test environment"""
        if not PERFORMANCE_AVAILABLE:
            self.skipTest("Performance modules not available")
        
        self.perf_config = PerformanceConfig(
            enable_performance_monitoring=True,
            enable_spatial_indexing=True,
            enable_path_caching=True,
            enable_memory_management=True
        )

# ---------------------------
# Unit Tests
# ---------------------------

class TestSimulationConfig(SimulationTestCase):
    """Test simulation configuration"""
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Valid configuration
        config = SimulationConfig()
        self.assertIsInstance(config, SimulationConfig)
        
        # Invalid configuration should raise error
        with self.assertRaises(ValueError):
            SimulationConfig(simulation_speed_factor=-1.0)
        
        with self.assertRaises(ValueError):
            SimulationConfig(train_speed_pps=0.0)
    
    def test_config_defaults(self):
        """Test default configuration values"""
        config = SimulationConfig()
        self.assertEqual(config.simulation_speed_factor, 150.0)
        self.assertEqual(config.train_speed_pps, 12.0)
        self.assertEqual(config.initial_trains, 20)
        self.assertTrue(config.collision_detection)

class TestODPair(SimulationTestCase):
    """Test OD pair functionality"""
    
    def test_od_pair_creation(self):
        """Test OD pair creation"""
        od_pair = ODPair("A", "B", 100.0, Priority.HIGH, TrainType.EXPRESS)
        
        self.assertEqual(od_pair.origin, "A")
        self.assertEqual(od_pair.destination, "B")
        self.assertEqual(od_pair.depart_time, 100.0)
        self.assertEqual(od_pair.priority, Priority.HIGH)
        self.assertEqual(od_pair.train_type, TrainType.EXPRESS)
    
    def test_od_pair_metadata(self):
        """Test OD pair metadata"""
        od_pair = ODPair("A", "B", 0.0)
        od_pair.metadata["test_key"] = "test_value"
        
        self.assertEqual(od_pair.metadata["test_key"], "test_value")

class TestEnhancedSignal(SimulationTestCase):
    """Test enhanced signal functionality"""
    
    def test_signal_creation(self):
        """Test signal creation"""
        signal_data = {
            "id": "test_signal",
            "position": {"x": 10.0, "y": 20.0},
            "initial_aspect": "green"
        }
        
        signal = EnhancedSignal(signal_data)
        
        self.assertEqual(signal.id, "test_signal")
        self.assertEqual(signal.pos, (10.0, 20.0))
        self.assertEqual(signal.aspect, SignalAspect.GREEN)
    
    def test_signal_aspect_change(self):
        """Test signal aspect changes"""
        signal_data = {
            "id": "test_signal",
            "position": {"x": 0.0, "y": 0.0},
            "initial_aspect": "green"
        }
        
        signal = EnhancedSignal(signal_data)
        
        # Test aspect change
        changed = signal.change_aspect(SignalAspect.RED, 10.0)
        self.assertTrue(changed)
        self.assertEqual(signal.aspect, SignalAspect.RED)
        
        # Test timing constraints
        changed = signal.change_aspect(SignalAspect.GREEN, 11.0)  # Too soon
        self.assertFalse(changed)
        self.assertEqual(signal.aspect, SignalAspect.RED)

class TestEnhancedTrackBlock(SimulationTestCase):
    """Test enhanced track block functionality"""
    
    def test_track_block_creation(self):
        """Test track block creation"""
        import simpy
        env = simpy.Environment()
        config = SimulationConfig()
        
        block = EnhancedTrackBlock(env, ("A", "B"), config)
        
        self.assertEqual(block.id, ("A", "B"))
        self.assertFalse(block.is_occupied)
        self.assertFalse(block._disrupted)
    
    def test_track_block_safety(self):
        """Test track block safety checks"""
        import simpy
        env = simpy.Environment()
        config = SimulationConfig()
        
        block = EnhancedTrackBlock(env, ("A", "B"), config)
        
        # Test safe entry
        self.assertTrue(block.is_safe_for_entry(10.0, 100.0))
        
        # Test unsafe entry (too close)
        self.assertFalse(block.is_safe_for_entry(10.0, 5.0))
        
        # Test unsafe entry (occupied)
        block.is_occupied = True
        self.assertFalse(block.is_safe_for_entry(10.0, 100.0))

class TestEnhancedSimulation(SimulationTestCase):
    """Test enhanced simulation functionality"""
    
    def test_simulation_creation(self):
        """Test simulation creation"""
        sim = EnhancedSimulation(
            datafile=self.temp_file.name,
            config=self.test_config,
            mode=SimulationMode.HEADLESS
        )
        
        self.assertTrue(sim.is_running)
        self.assertEqual(len(sim.stations), 3)  # 3 stations in test data
        self.assertEqual(len(sim.track_blocks), 4)  # 4 edges in test data
    
    def test_train_spawning(self):
        """Test train spawning"""
        sim = EnhancedSimulation(
            datafile=self.temp_file.name,
            config=self.test_config,
            mode=SimulationMode.HEADLESS
        )
        
        od_pairs = create_test_od_pairs()
        initial_count = len(sim.trains)
        
        sim.spawn_train_from_od(od_pairs[0])
        
        self.assertEqual(len(sim.trains), initial_count + 1)
        self.assertEqual(sim.trains[-1].id, f"T{sim.train_counter - 1:04d}")
    
    def test_simulation_advance(self):
        """Test simulation time advancement"""
        sim = EnhancedSimulation(
            datafile=self.temp_file.name,
            config=self.test_config,
            mode=SimulationMode.HEADLESS
        )
        
        initial_time = sim.current_sim_time
        sim.advance(5.0)
        
        self.assertEqual(sim.current_sim_time, initial_time + 5.0)
    
    def test_network_metrics(self):
        """Test network metrics calculation"""
        sim = EnhancedSimulation(
            datafile=self.temp_file.name,
            config=self.test_config,
            mode=SimulationMode.HEADLESS
        )
        
        # Test node utilization
        utilization = sim.get_node_utilization("1")
        self.assertGreaterEqual(utilization, 0.0)
        self.assertLessEqual(utilization, 1.0)
        
        # Test edge utilization
        edge_key = ("1", "4")
        utilization = sim.get_edge_utilization(edge_key)
        self.assertGreaterEqual(utilization, 0.0)
        self.assertLessEqual(utilization, 1.0)
        
        # Test network utilization
        network_util = sim.get_network_utilization()
        self.assertGreaterEqual(network_util, 0.0)
        self.assertLessEqual(network_util, 1.0)

# ---------------------------
# Performance Tests
# ---------------------------

class TestPerformanceMonitor(PerformanceTestCase):
    """Test performance monitoring"""
    
    def test_performance_monitor_creation(self):
        """Test performance monitor creation"""
        monitor = PerformanceMonitor()
        
        self.assertIsInstance(monitor.current_metrics, type(monitor.current_metrics))
        self.assertEqual(monitor.frame_count, 0)
    
    def test_performance_monitor_update(self):
        """Test performance monitor updates"""
        monitor = PerformanceMonitor()
        
        monitor.update_frame(10, 20, 5)
        
        self.assertEqual(monitor.current_metrics.train_count, 10)
        self.assertEqual(monitor.current_metrics.edge_count, 20)
        self.assertEqual(monitor.current_metrics.signal_count, 5)
        self.assertEqual(monitor.frame_count, 1)
    
    def test_performance_degradation_detection(self):
        """Test performance degradation detection"""
        monitor = PerformanceMonitor()
        
        # Simulate good performance
        for _ in range(10):
            monitor.update_frame(10, 20, 5)
        
        self.assertFalse(monitor.is_performance_degraded(30.0, 1024.0))
        
        # Simulate degraded performance (low FPS)
        monitor.current_metrics.fps = 20.0
        self.assertTrue(monitor.is_performance_degraded(30.0, 1024.0))

class TestSpatialIndex(PerformanceTestCase):
    """Test spatial indexing"""
    
    def test_spatial_index_creation(self):
        """Test spatial index creation"""
        index = SpatialIndex(100.0)
        
        self.assertEqual(index.cell_size, 100.0)
        self.assertEqual(len(index.cells), 0)
        self.assertEqual(len(index.objects), 0)
    
    def test_spatial_index_operations(self):
        """Test spatial index operations"""
        index = SpatialIndex(100.0)
        
        # Add objects
        obj1 = "train1"
        obj2 = "train2"
        
        index.add_object(obj1, (50.0, 50.0))
        index.add_object(obj2, (150.0, 150.0))
        
        self.assertEqual(len(index.objects), 2)
        
        # Test radius query
        objects_in_radius = index.get_objects_in_radius((100.0, 100.0), 200.0)
        self.assertIn(obj1, objects_in_radius)
        self.assertIn(obj2, objects_in_radius)
        
        # Test update
        index.update_object(obj1, (250.0, 250.0))
        self.assertEqual(len(index.objects), 2)
        
        # Test removal
        index.remove_object(obj1)
        self.assertEqual(len(index.objects), 1)

class TestPathCache(PerformanceTestCase):
    """Test path caching"""
    
    def test_path_cache_creation(self):
        """Test path cache creation"""
        cache = PathCache(100)
        
        self.assertEqual(cache.hit_count, 0)
        self.assertEqual(cache.miss_count, 0)
    
    def test_path_cache_operations(self):
        """Test path cache operations"""
        cache = PathCache(100)
        
        # Test cache miss
        path = cache.get_path("A", "B")
        self.assertIsNone(path)
        self.assertEqual(cache.miss_count, 1)
        
        # Test cache hit
        test_path = ["A", "C", "B"]
        cache.cache_path("A", "B", test_path)
        
        cached_path = cache.get_path("A", "B")
        self.assertEqual(cached_path, test_path)
        self.assertEqual(cache.hit_count, 1)
    
    def test_path_cache_hit_rate(self):
        """Test path cache hit rate calculation"""
        cache = PathCache(100)
        
        # Add some hits and misses
        cache.hit_count = 7
        cache.miss_count = 3
        
        hit_rate = cache.get_hit_rate()
        self.assertEqual(hit_rate, 0.7)

class TestLRUCache(PerformanceTestCase):
    """Test LRU cache"""
    
    def test_lru_cache_creation(self):
        """Test LRU cache creation"""
        cache = LRUCache(3)
        
        self.assertEqual(cache.max_size, 3)
        self.assertEqual(len(cache.cache), 0)
    
    def test_lru_cache_operations(self):
        """Test LRU cache operations"""
        cache = LRUCache(3)
        
        # Add items
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        
        self.assertEqual(len(cache.cache), 3)
        
        # Test get
        value = cache.get("key1")
        self.assertEqual(value, "value1")
        
        # Test LRU eviction
        cache.put("key4", "value4")
        self.assertEqual(len(cache.cache), 3)
        self.assertIsNone(cache.get("key2"))  # Should be evicted
        self.assertIsNotNone(cache.get("key1"))  # Should still be there
    
    def test_lru_cache_clear(self):
        """Test LRU cache clear"""
        cache = LRUCache(3)
        
        cache.put("key1", "value1")
        cache.clear()
        
        self.assertEqual(len(cache.cache), 0)
        self.assertEqual(len(cache.access_order), 0)

# ---------------------------
# Integration Tests
# ---------------------------

class TestSimulationIntegration(SimulationTestCase):
    """Test simulation integration"""
    
    def test_full_simulation_run(self):
        """Test full simulation run"""
        config = SimulationConfig(
            episode_duration=5.0,
            initial_trains=2,
            snapshot_interval=1.0,
            seed=42
        )
        
        results = run_enhanced_simulation(
            datafile=self.temp_file.name,
            config=config,
            mode=SimulationMode.HEADLESS
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn("episode_duration", results)
        self.assertIn("total_trains", results)
        self.assertIn("completed_trains", results)
        self.assertGreaterEqual(results["episode_duration"], 5.0)
    
    def test_od_pair_generation(self):
        """Test OD pair generation"""
        od_pairs = create_sample_od_pairs(["A", "B", "C"], 5)
        
        self.assertEqual(len(od_pairs), 5)
        
        for od_pair in od_pairs:
            self.assertIsInstance(od_pair, ODPair)
            self.assertIn(od_pair.origin, ["A", "B", "C"])
            self.assertIn(od_pair.destination, ["A", "B", "C"])
            self.assertGreaterEqual(od_pair.depart_time, 0.0)

class TestPerformanceIntegration(PerformanceTestCase):
    """Test performance integration"""
    
    def test_performance_manager_creation(self):
        """Test performance manager creation"""
        manager = PerformanceManager(self.perf_config)
        
        self.assertIsNotNone(manager.monitor)
        self.assertIsNotNone(manager.spatial_index)
        self.assertIsNotNone(manager.path_cache)
        self.assertIsNotNone(manager.memory_manager)
    
    def test_performance_manager_update(self):
        """Test performance manager updates"""
        manager = PerformanceManager(self.perf_config)
        
        manager.update_frame(10, 20, 5)
        
        stats = manager.get_performance_stats()
        self.assertIn("is_degraded", stats)
        self.assertIn("optimization_level", stats)
        self.assertIn("profiler_stats", stats)

# ---------------------------
# Stress Tests
# ---------------------------

class TestSimulationStress(SimulationTestCase):
    """Test simulation under stress"""
    
    def test_large_number_of_trains(self):
        """Test simulation with large number of trains"""
        config = SimulationConfig(
            episode_duration=10.0,
            initial_trains=50,
            snapshot_interval=0.5,
            max_trains=100
        )
        
        sim = EnhancedSimulation(
            datafile=self.temp_file.name,
            config=config,
            mode=SimulationMode.HEADLESS
        )
        
        # Generate many OD pairs
        od_pairs = create_sample_od_pairs(sim.stations, 50)
        sim.spawn_trains_from_ods(od_pairs)
        
        # Run simulation
        sim.run_headless_episode(5.0)
        
        # Verify simulation handled the load
        self.assertLessEqual(len(sim.trains), config.max_trains)
        self.assertTrue(sim.is_running)

class TestPerformanceStress(PerformanceTestCase):
    """Test performance under stress"""
    
    def test_memory_pressure(self):
        """Test performance under memory pressure"""
        manager = PerformanceManager(self.perf_config)
        
        # Simulate memory pressure
        for i in range(1000):
            manager.update_frame(i, i * 2, i // 2)
        
        # Check if performance degraded
        stats = manager.get_performance_stats()
        self.assertIsInstance(stats["is_degraded"], bool)
    
    def test_cache_performance(self):
        """Test cache performance under load"""
        cache = PathCache(1000)
        
        # Add many paths
        for i in range(500):
            cache.cache_path(f"station_{i}", f"station_{i+1}", [f"station_{i}", f"station_{i+1}"])
        
        # Test cache hit rate
        hit_rate = cache.get_hit_rate()
        self.assertGreaterEqual(hit_rate, 0.0)
        self.assertLessEqual(hit_rate, 1.0)

# ---------------------------
# Mock Tests
# ---------------------------

class TestMockSimulation(SimulationTestCase):
    """Test simulation with mocks"""
    
    @patch('simul8_enhanced.load_enhanced_railway_graph')
    def test_simulation_with_mock_graph(self, mock_load):
        """Test simulation with mocked graph loading"""
        # Mock graph data
        mock_graph = Mock()
        mock_graph.nodes.return_value = [("1", {"pos": (0, 0), "label": "A"})]
        mock_graph.edges.return_value = [("1", "2")]
        
        mock_load.return_value = (mock_graph, [])
        
        sim = EnhancedSimulation(
            datafile="mock_file.json",
            config=self.test_config,
            mode=SimulationMode.HEADLESS
        )
        
        self.assertTrue(sim.is_running)
        mock_load.assert_called_once_with("mock_file.json")
    
    @patch('simul8_enhanced.pygame')
    def test_visual_simulation_with_mock_pygame(self, mock_pygame):
        """Test visual simulation with mocked pygame"""
        mock_pygame.init.return_value = None
        mock_pygame.display.set_mode.return_value = Mock()
        mock_pygame.display.set_caption.return_value = None
        mock_pygame.font.SysFont.return_value = Mock()
        mock_pygame.time.Clock.return_value = Mock()
        
        config = SimulationConfig()
        sim = EnhancedSimulation(
            datafile=self.temp_file.name,
            config=config,
            mode=SimulationMode.VISUAL
        )
        
        self.assertTrue(sim.visual)

# ---------------------------
# Test Suite
# ---------------------------

def create_test_suite() -> unittest.TestSuite:
    """Create comprehensive test suite"""
    suite = unittest.TestSuite()
    
    # Unit tests
    suite.addTest(unittest.makeSuite(TestSimulationConfig))
    suite.addTest(unittest.makeSuite(TestODPair))
    suite.addTest(unittest.makeSuite(TestEnhancedSignal))
    suite.addTest(unittest.makeSuite(TestEnhancedTrackBlock))
    suite.addTest(unittest.makeSuite(TestEnhancedSimulation))
    
    # Performance tests
    suite.addTest(unittest.makeSuite(TestPerformanceMonitor))
    suite.addTest(unittest.makeSuite(TestSpatialIndex))
    suite.addTest(unittest.makeSuite(TestPathCache))
    suite.addTest(unittest.makeSuite(TestLRUCache))
    
    # Integration tests
    suite.addTest(unittest.makeSuite(TestSimulationIntegration))
    suite.addTest(unittest.makeSuite(TestPerformanceIntegration))
    
    # Stress tests
    suite.addTest(unittest.makeSuite(TestSimulationStress))
    suite.addTest(unittest.makeSuite(TestPerformanceStress))
    
    # Mock tests
    suite.addTest(unittest.makeSuite(TestMockSimulation))
    
    return suite

# ---------------------------
# Test Runner
# ---------------------------

def run_tests(verbose: bool = True) -> unittest.TestResult:
    """Run all tests"""
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    return result

def run_performance_tests() -> Dict[str, Any]:
    """Run performance-specific tests"""
    performance_tests = [
        TestPerformanceMonitor,
        TestSpatialIndex,
        TestPathCache,
        TestLRUCache,
        TestPerformanceIntegration,
        TestPerformanceStress
    ]
    
    suite = unittest.TestSuite()
    for test_class in performance_tests:
        suite.addTest(unittest.makeSuite(test_class))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0
    }

# ---------------------------
# Main Test Execution
# ---------------------------

if __name__ == "__main__":
    print("Running Railway Simulation Tests...")
    print("=" * 50)
    
    # Run all tests
    result = run_tests(verbose=True)
    
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    # Run performance tests separately
    print("\n" + "=" * 50)
    print("Running Performance Tests...")
    perf_result = run_performance_tests()
    print(f"Performance Tests - Success Rate: {perf_result['success_rate']:.1%}")
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    exit(exit_code)
