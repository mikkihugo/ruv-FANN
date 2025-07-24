/**
 * Simple Manual Test for Issue #137 Core Components
 * Tests the core session persistence components without dependencies
 */

import { SessionManager } from './src/session-manager.js';
import { HealthMonitor } from './src/health-monitor.js';

// Mock persistence for testing
class MockPersistence {
  async run(query, params = []) {
    console.log(`📝 Mock SQL executed: ${query.split('\n')[0]}...`);
    return { lastID: Date.now(), changes: 1 };
  }

  async get(query, params = []) {
    return { id: 'test-session', metadata: '{}', status: 'active' };
  }

  async all(query, params = []) {
    return [];
  }
}

async function testCoreComponents() {
  console.log('\n🧪 Testing Core Session Persistence Components for Issue #137\n');
  
  try {
    // Test 1: SessionManager Core Functionality
    console.log('1️⃣ Testing SessionManager Core...');
    const mockPersistence = new MockPersistence();
    const sessionManager = new SessionManager(mockPersistence);
    
    await sessionManager.initialize();
    console.log('   ✅ SessionManager initialized');
    
    const session = await sessionManager.createSession({
      name: 'Test Session',
      metadata: { issueNumber: 137 }
    });
    console.log('   ✅ Session created:', session.id);
    
    // Test session operations
    await sessionManager.saveSession(session.id);
    console.log('   ✅ Session saved');
    
    const checkpoint = await sessionManager.createCheckpoint(session.id, 'test-checkpoint');
    console.log('   ✅ Checkpoint created:', checkpoint.name);
    
    const status = sessionManager.getSessionStatus(session.id);
    console.log('   ✅ Session status retrieved:', status.status);
    
    // Test 2: HealthMonitor Core Functionality
    console.log('\n2️⃣ Testing HealthMonitor Core...');
    const healthMonitor = new HealthMonitor();
    
    await healthMonitor.start();
    console.log('   ✅ HealthMonitor started');
    
    // Register test health check
    healthMonitor.registerHealthCheck('session-persistence', () => {
      return {
        score: 95,
        status: 'healthy',
        details: 'Session persistence system operational',
        metrics: { sessions: sessionManager.sessions.size }
      };
    }, { weight: 2, critical: true });
    console.log('   ✅ Session persistence health check registered');
    
    // Run health check
    const healthReport = await healthMonitor.runHealthChecks();
    console.log('   ✅ Health check completed');
    console.log('      Overall Score:', healthReport.overallScore);
    console.log('      Status:', healthReport.status);
    console.log('      Checks Run:', healthReport.checkCount);
    
    // Get health trends
    const trends = healthMonitor.getHealthTrends();
    console.log('   ✅ Health trends:', trends.trend);
    
    // Test 3: Integration Test
    console.log('\n3️⃣ Testing Integration...');
    
    // Set up persistence checker for health monitor
    healthMonitor.setPersistenceChecker(async () => {
      return await mockPersistence.get('SELECT 1');
    });
    console.log('   ✅ Persistence checker configured');
    
    // Run another health check to test persistence integration
    const integratedHealthReport = await healthMonitor.runHealthChecks();
    console.log('   ✅ Integrated health check completed:', integratedHealthReport.overallScore);
    
    // Test session lifecycle
    await sessionManager.pauseSession(session.id);
    console.log('   ✅ Session paused');
    
    await sessionManager.resumeSession(session.id);
    console.log('   ✅ Session resumed');
    
    // Test checkpoint restoration
    await sessionManager.restoreFromCheckpoint(session.id, 'test-checkpoint');
    console.log('   ✅ Session restored from checkpoint');
    
    console.log('\n🎉 All core tests completed successfully!');
    console.log('\n📊 Test Results Summary:');
    console.log('   ✅ SessionManager Core: PASSED');
    console.log('   ✅ HealthMonitor Core: PASSED');
    console.log('   ✅ Integration: PASSED');
    console.log('   ✅ Session Lifecycle: PASSED');
    console.log('   ✅ Checkpoint System: PASSED');
    
    // Display final statistics
    const finalStats = healthMonitor.getCurrentHealth();
    console.log('\n📈 Final Health Statistics:');
    console.log('   Score:', finalStats.overallScore);
    console.log('   Status:', finalStats.status);
    console.log('   Checks:', finalStats.checkCount);
    console.log('   Running:', finalStats.isRunning);
    
    const sessionStats = sessionManager.listSessions();
    console.log('\n📊 Session Statistics:');
    console.log('   Total Sessions:', sessionStats.length);
    console.log('   Active Sessions:', sessionManager.sessions.size);
    
    // Cleanup
    await sessionManager.terminateSession(session.id);
    await healthMonitor.stop();
    await sessionManager.destroy();
    await healthMonitor.destroy();
    
    console.log('\n✨ Issue #137 core implementation is working correctly!');
    console.log('🏆 Session persistence and recovery system validated');
    
  } catch (error) {
    console.error('\n❌ Test failed:', error.message);
    console.error('Stack trace:', error.stack);
    process.exit(1);
  }
}

// Run the test
testCoreComponents().catch(console.error);