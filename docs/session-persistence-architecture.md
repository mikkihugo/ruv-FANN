# Session Persistence and Recovery Architecture
## Issue #137 - System Architecture Design

**Author**: System Architect Agent  
**Date**: 2025-01-24  
**Version**: 1.0  
**Target Issue**: [#137 - Session Persistence and Recovery](https://github.com/ruvnet/ruv-FANN/issues/137)

---

## Executive Summary

This document presents a comprehensive session persistence and recovery architecture for the ruv-swarm system based on thorough analysis of the existing codebase. The architecture addresses session management, failure recovery, health monitoring, and MCP connection state management while building upon the existing high-availability persistence layer.

### Key Architectural Decisions

1. **Layered Session Management**: Multi-tier session storage with memory, disk, and recovery tiers
2. **Distributed State Synchronization**: Event-driven state propagation across system components
3. **Proactive Health Monitoring**: Continuous health assessment with predictive failure detection
4. **MCP Connection Resilience**: Robust MCP protocol handling with automatic reconnection
5. **Graceful Degradation**: System continues operating with reduced functionality during failures

---

## 1. System Architecture Overview

### 1.1 Current System Analysis

**Existing Components Identified:**
- **RuvSwarm Class** (`index.ts`): Core swarm orchestration with in-memory state
- **SwarmPersistencePooled** (`persistence-pooled.js`): High-availability SQLite persistence with connection pooling
- **SQLiteConnectionPool** (`sqlite-pool.js`): Production-ready connection management
- **EnhancedMCPTools** (`mcp-tools-enhanced.js`): MCP protocol implementation
- **Agent Management** (`agent.ts`): Agent lifecycle and state management
- **Neural Network Integration** (`neural-network.ts`): WASM-based neural processing

**Identified Gaps:**
- No session lifecycle management
- Missing cross-restart state recovery
- No MCP connection state persistence
- Limited failure scenario handling
- No session health monitoring

### 1.2 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Session Management Layer                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐      │
│  │ Session Store │  │ Recovery Mgr  │  │ Health Monitor│      │
│  └───────────────┘  └───────────────┘  └───────────────┘      │
├─────────────────────────────────────────────────────────────────┤
│                    State Synchronization                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐      │
│  │ State Manager │  │ Event Bus     │  │ Change Stream │      │
│  └───────────────┘  └───────────────┘  └───────────────┘      │
├─────────────────────────────────────────────────────────────────┤
│                    Existing Core Systems                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐      │
│  │ RuvSwarm      │  │ MCP Tools     │  │ Persistence   │      │
│  │ (Enhanced)    │  │ (Enhanced)    │  │ (Pooled)      │      │
│  └───────────────┘  └───────────────┘  └───────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Components Design

### 2.1 Session Store Architecture

The Session Store manages the complete lifecycle of swarm sessions with multi-tier storage:

#### 2.1.1 Session Data Model

```typescript
interface SwarmSession {
  // Identity
  sessionId: string;
  swarmId: string;
  parentSessionId?: string;
  
  // Lifecycle
  status: 'initializing' | 'active' | 'paused' | 'recovering' | 'terminating' | 'terminated';
  createdAt: Date;
  lastActiveAt: Date;
  expiresAt?: Date;
  
  // Configuration
  configuration: SwarmConfiguration;
  topology: SwarmTopology;
  
  // State snapshots
  currentState: SwarmState;
  checkpoints: SessionCheckpoint[];
  
  // Recovery information
  recoveryMetadata: RecoveryMetadata;
  
  // MCP connection state
  mcpConnectionState: MCPConnectionState;
  
  // Health metrics
  healthMetrics: SessionHealthMetrics;
}

interface SessionCheckpoint {
  checkpointId: string;
  timestamp: Date;
  state: SwarmState;
  metadata: CheckpointMetadata;
  verified: boolean;
}

interface RecoveryMetadata {
  lastSuccessfulOperation: string;
  pendingOperations: PendingOperation[];
  rollbackPoints: RollbackPoint[];
  integrityHashes: IntegrityHashes;
}

interface MCPConnectionState {
  connectionId: string;
  protocol: 'stdio' | 'sse' | 'websocket';
  lastHeartbeat: Date;
  capabilities: MCPCapabilities;
  activeTools: ActiveToolState[];
}
```

#### 2.1.2 Storage Architecture

**Memory Tier** (L1 Cache):
- **Technology**: In-memory Map with LRU eviction
- **Purpose**: Ultra-fast access for active sessions
- **Capacity**: Configurable (default 100 sessions)
- **TTL**: 15 minutes since last access

**Persistent Tier** (L2 Storage):
- **Technology**: Enhanced SQLite with connection pooling
- **Purpose**: Durable session storage across restarts
- **Integration**: Extends existing `SwarmPersistencePooled`
- **Features**: Transaction support, checkpointing, integrity verification

**Archive Tier** (L3 Storage):
- **Technology**: Compressed JSON files
- **Purpose**: Long-term session history and forensics
- **Trigger**: Sessions older than 24 hours
- **Retention**: Configurable (default 30 days)

### 2.2 Recovery Manager Design

The Recovery Manager handles all failure scenarios with intelligent recovery strategies:

#### 2.2.1 Failure Classification

```typescript
enum FailureType {
  // Process failures
  PROCESS_CRASH = 'process_crash',
  GRACEFUL_SHUTDOWN = 'graceful_shutdown',
  FORCED_TERMINATION = 'forced_termination',
  
  // Connection failures
  MCP_CONNECTION_LOST = 'mcp_connection_lost',
  MCP_PROTOCOL_ERROR = 'mcp_protocol_error',
  CLIENT_DISCONNECT = 'client_disconnect',
  
  // State corruption
  STATE_CORRUPTION = 'state_corruption',
  CHECKPOINT_INVALID = 'checkpoint_invalid',
  DATABASE_CORRUPTION = 'database_corruption',
  
  // Resource exhaustion
  MEMORY_EXHAUSTION = 'memory_exhaustion',
  DISK_FULL = 'disk_full',
  CONNECTION_POOL_EXHAUSTED = 'connection_pool_exhausted',
  
  // Agent failures
  AGENT_UNRESPONSIVE = 'agent_unresponsive',
  AGENT_ERROR = 'agent_error',
  NEURAL_NETWORK_FAILURE = 'neural_network_failure',
  
  // Task failures
  TASK_TIMEOUT = 'task_timeout',
  TASK_DEADLOCK = 'task_deadlock',
  WORKFLOW_CORRUPTION = 'workflow_corruption'
}

enum RecoveryStrategy {
  IMMEDIATE_RESTART = 'immediate_restart',
  CHECKPOINT_ROLLBACK = 'checkpoint_rollback',
  PARTIAL_RECOVERY = 'partial_recovery',
  GRACEFUL_DEGRADATION = 'graceful_degradation',
  MANUAL_INTERVENTION = 'manual_intervention'
}
```

#### 2.2.2 Recovery Workflows

**1. Process Crash Recovery**
```
1. Detect process termination (via process monitoring)
2. Load last known session state from persistent storage
3. Validate state integrity using checksums
4. Reconstruct in-memory structures
5. Re-establish MCP connections
6. Resume pending tasks
7. Notify clients of recovery completion
```

**2. MCP Connection Recovery**
```
1. Detect connection loss (via heartbeat timeout)
2. Attempt immediate reconnection (exponential backoff)
3. If failed, enter offline mode
4. Queue operations for later execution
5. Periodically retry connection
6. On reconnection, sync queued operations
7. Resume normal operation
```

**3. State Corruption Recovery**
```
1. Detect corruption (via integrity checks)
2. Identify last valid checkpoint
3. Roll back to checkpoint state
4. Replay operations since checkpoint
5. Verify reconstructed state integrity
6. Resume with validated state
```

### 2.3 Health Monitoring System

#### 2.3.1 Health Metrics Collection

```typescript
interface SessionHealthMetrics {
  // System health
  systemHealth: {
    cpuUsage: number;
    memoryUsage: number;
    diskUsage: number;
    connectionPoolHealth: PoolHealthMetrics;
  };
  
  // Session health
  sessionHealth: {
    activeAgents: number;
    pendingTasks: number;
    failedTasks: number;
    lastCheckpointAge: number;
    mcpConnectionLatency: number;
  };
  
  // Predictive indicators
  riskIndicators: {
    memoryPressure: 'low' | 'medium' | 'high';
    taskBacklog: 'normal' | 'elevated' | 'critical';
    errorRate: number;
    responseTimeP99: number;
  };
  
  // Historical trends
  trends: {
    throughputTrend: TrendData;
    errorRateTrend: TrendData;
    resourceUsageTrend: TrendData;
  };
}
```

#### 2.3.2 Monitoring Architecture

**Health Check Orchestrator**:
- Coordinates all health monitoring activities
- Manages check intervals and priorities
- Aggregates health data from all components
- Triggers alerts and recovery actions

**Component Health Checkers**:
- Session Store Health Checker
- Recovery Manager Health Checker
- MCP Connection Health Checker
- Agent Pool Health Checker
- Task Queue Health Checker
- Neural Network Health Checker

**Health Data Pipeline**:
```
[Health Checkers] → [Metrics Aggregator] → [Trend Analyzer] → [Alert Engine] → [Recovery Trigger]
```

### 2.4 MCP Connection State Management

#### 2.4.1 Connection State Model

```typescript
interface MCPConnectionManager {
  connectionId: string;
  protocol: MCPProtocol;
  state: MCPConnectionState;
  capabilities: MCPCapabilities;
  
  // Connection lifecycle
  connect(): Promise<void>;
  disconnect(): Promise<void>;
  reconnect(): Promise<void>;
  
  // State persistence
  persistState(): Promise<void>;
  restoreState(): Promise<void>;
  
  // Health monitoring
  heartbeat(): Promise<boolean>;
  getConnectionHealth(): ConnectionHealth;
}

interface MCPConnectionState {
  status: 'connecting' | 'connected' | 'disconnected' | 'error';
  lastHeartbeat: Date;
  consecutiveFailures: number;
  totalMessages: number;
  averageLatency: number;
  activeTools: Map<string, ToolState>;
  queuedOperations: QueuedOperation[];
}
```

#### 2.4.2 Connection Recovery Strategy

**Immediate Recovery** (0-5 seconds):
- Detect connection loss via heartbeat timeout
- Attempt immediate reconnection
- If successful, sync any queued operations

**Short-term Recovery** (5-60 seconds):
- Exponential backoff reconnection attempts
- Queue all new operations
- Notify clients of degraded mode

**Long-term Recovery** (60+ seconds):
- Enter offline mode
- Persist all pending operations
- Continue core functionality without MCP
- Periodic reconnection attempts

---

## 3. Database Schema Extensions

### 3.1 Session Management Tables

```sql
-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY,
  swarm_id TEXT NOT NULL,
  parent_session_id TEXT,
  status TEXT NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  last_active_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  expires_at DATETIME,
  configuration TEXT NOT NULL,
  current_state TEXT,
  recovery_metadata TEXT,
  mcp_connection_state TEXT,
  health_metrics TEXT,
  FOREIGN KEY (parent_session_id) REFERENCES sessions(session_id),
  FOREIGN KEY (swarm_id) REFERENCES swarms(id)
);

-- Session checkpoints
CREATE TABLE IF NOT EXISTS session_checkpoints (
  checkpoint_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  sequence_number INTEGER NOT NULL,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
  state_snapshot TEXT NOT NULL,
  metadata TEXT,
  integrity_hash TEXT NOT NULL,
  verified BOOLEAN DEFAULT FALSE,
  FOREIGN KEY (session_id) REFERENCES sessions(session_id),
  UNIQUE(session_id, sequence_number)
);

-- Recovery log
CREATE TABLE IF NOT EXISTS recovery_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  recovery_type TEXT NOT NULL,
  failure_type TEXT,
  started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  completed_at DATETIME,
  status TEXT NOT NULL,
  error_details TEXT,
  recovery_actions TEXT,
  FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- Health metrics history
CREATE TABLE IF NOT EXISTS session_health_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
  metrics TEXT NOT NULL,
  alerts TEXT,
  FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- MCP connection log
CREATE TABLE IF NOT EXISTS mcp_connection_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  connection_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
  details TEXT,
  FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_last_active ON sessions(last_active_at);
CREATE INDEX IF NOT EXISTS idx_checkpoints_session ON session_checkpoints(session_id, sequence_number);
CREATE INDEX IF NOT EXISTS idx_recovery_log_session ON recovery_log(session_id, started_at);
CREATE INDEX IF NOT EXISTS idx_health_history_session ON session_health_history(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_mcp_log_session ON mcp_connection_log(session_id, timestamp);
```

---

## 4. API Specifications

### 4.1 Session Management API

#### 4.1.1 Core Session Operations

```typescript
interface SessionAPI {
  // Session lifecycle
  createSession(config: SessionConfig): Promise<Session>;
  getSession(sessionId: string): Promise<Session | null>;
  updateSession(sessionId: string, updates: Partial<Session>): Promise<void>;
  terminateSession(sessionId: string, reason?: string): Promise<void>;
  
  // Session listing and filtering
  listSessions(filter?: SessionFilter): Promise<Session[]>;
  getActiveSessions(): Promise<Session[]>;
  getSessionsByStatus(status: SessionStatus): Promise<Session[]>;
  
  // Checkpointing
  createCheckpoint(sessionId: string, metadata?: CheckpointMetadata): Promise<Checkpoint>;
  listCheckpoints(sessionId: string): Promise<Checkpoint[]>;
  restoreFromCheckpoint(sessionId: string, checkpointId: string): Promise<void>;
  
  // Health monitoring
  getSessionHealth(sessionId: string): Promise<SessionHealthMetrics>;
  getSystemHealth(): Promise<SystemHealthMetrics>;
  
  // Recovery operations
  recoverSession(sessionId: string, strategy?: RecoveryStrategy): Promise<RecoveryResult>;
  getRecoveryHistory(sessionId: string): Promise<RecoveryRecord[]>;
  
  // MCP connection management
  getMCPConnectionState(sessionId: string): Promise<MCPConnectionState>;
  reconnectMCP(sessionId: string): Promise<void>;
}
```

#### 4.1.2 Event API

```typescript
interface SessionEventAPI {
  // Event subscription
  onSessionCreated(callback: (session: Session) => void): void;
  onSessionUpdated(callback: (sessionId: string, changes: Partial<Session>) => void): void;
  onSessionTerminated(callback: (sessionId: string, reason: string) => void): void;
  
  // Health events
  onHealthAlert(callback: (alert: HealthAlert) => void): void;
  onRecoveryStarted(callback: (recovery: RecoveryRecord) => void): void;
  onRecoveryCompleted(callback: (recovery: RecoveryRecord) => void): void;
  
  // MCP events
  onMCPConnected(callback: (connectionInfo: MCPConnectionInfo) => void): void;
  onMCPDisconnected(callback: (disconnectionInfo: MCPDisconnectionInfo) => void): void;
  onMCPError(callback: (error: MCPError) => void): void;
}
```

### 4.2 Enhanced MCP Tools

#### 4.2.1 Session-Aware MCP Tools

```typescript
interface EnhancedMCPTools {
  // Session management
  session_create(args: SessionCreateArgs): Promise<SessionCreateResult>;
  session_status(args: SessionStatusArgs): Promise<SessionStatusResult>;
  session_checkpoint(args: SessionCheckpointArgs): Promise<SessionCheckpointResult>;
  session_recover(args: SessionRecoverArgs): Promise<SessionRecoverResult>;
  session_health(args: SessionHealthArgs): Promise<SessionHealthResult>;
  
  // Existing tools (enhanced with session awareness)
  swarm_init(args: SwarmInitArgs & { sessionId?: string }): Promise<SwarmInitResult>;
  agent_spawn(args: AgentSpawnArgs & { sessionId: string }): Promise<AgentSpawnResult>;
  task_orchestrate(args: TaskOrchestrateArgs & { sessionId: string }): Promise<TaskOrchestrateResult>;
  
  // Recovery-specific tools
  recovery_status(args: RecoveryStatusArgs): Promise<RecoveryStatusResult>;
  recovery_trigger(args: RecoveryTriggerArgs): Promise<RecoveryTriggerResult>;
  recovery_history(args: RecoveryHistoryArgs): Promise<RecoveryHistoryResult>;
}
```

---

## 5. Implementation Plan

### 5.1 Phase 1: Core Session Infrastructure (Week 1-2)

**Deliverables:**
- Session data models and TypeScript interfaces
- Basic SessionStore implementation with memory tier
- Database schema extensions
- Session lifecycle management (create, update, terminate)

**Integration Points:**
- Extend `SwarmPersistencePooled` with session tables
- Modify `RuvSwarm` constructor to accept session configuration
- Add session ID tracking to existing components

**Testing Strategy:**
- Unit tests for all session data models
- Integration tests with existing persistence layer
- Session lifecycle end-to-end tests

### 5.2 Phase 2: Recovery Manager (Week 3-4)

**Deliverables:**
- RecoveryManager class implementation
- Failure detection and classification system
- Basic recovery workflows for process crashes
- Checkpoint creation and restoration

**Integration Points:**
- Hook into existing error handling in `EnhancedMCPTools`
- Integrate with `SQLiteConnectionPool` health monitoring
- Add recovery triggers to swarm lifecycle events

**Testing Strategy:**
- Chaos engineering tests (process kills, connection drops)
- Recovery workflow validation
- State integrity verification tests

### 5.3 Phase 3: Health Monitoring (Week 5-6)

**Deliverables:**
- HealthMonitor class with metric collection
- Component-specific health checkers
- Alert system and dashboard endpoints
- Predictive failure detection

**Integration Points:**
- Add health monitoring to all major components
- Integrate with existing metrics collection
- Create health status endpoints for MCP tools

**Testing Strategy:**
- Health metric accuracy validation
- Alert threshold tuning
- Performance impact assessment

### 5.4 Phase 4: MCP Connection Management (Week 7-8)

**Deliverables:**
- MCPConnectionManager implementation
- Connection state persistence and restoration
- Advanced reconnection strategies
- Offline mode operation

**Integration Points:**
- Wrap existing MCP protocol handling
- Add connection state to session storage
- Implement queued operation processing

**Testing Strategy:**
- Connection failure simulation
- Offline mode functionality validation
- Message queuing and replay tests

### 5.5 Phase 5: Integration and Optimization (Week 9-10)

**Deliverables:**
- Complete system integration
- Performance optimization
- Documentation and examples
- Production deployment guide

**Integration Points:**
- Full integration with existing ruv-swarm ecosystem
- Backward compatibility validation
- Migration tools for existing installations

**Testing Strategy:**
- Full system integration tests
- Performance benchmarking
- Long-running stability tests

---

## 6. Risk Assessment and Mitigation

### 6.1 Technical Risks

**Risk: Performance Impact**
- **Probability**: Medium
- **Impact**: High
- **Mitigation**: Extensive benchmarking, performance-first design, configurable features

**Risk: Database Corruption**
- **Probability**: Low
- **Impact**: Critical
- **Mitigation**: Multiple integrity checks, automated backups, repair procedures

**Risk: Memory Leaks**
- **Probability**: Medium
- **Impact**: Medium
- **Mitigation**: Comprehensive memory profiling, LRU eviction, garbage collection tuning

### 6.2 Integration Risks

**Risk: Breaking Changes**
- **Probability**: Low
- **Impact**: High
- **Mitigation**: Backward compatibility layers, extensive testing, phased rollout

**Risk: Dependency Conflicts**
- **Probability**: Medium
- **Impact**: Medium
- **Mitigation**: Careful dependency management, isolated components, version pinning

---

## 7. Success Metrics

### 7.1 Functional Metrics

- **Session Recovery Success Rate**: >99.5%
- **Mean Time to Recovery (MTTR)**: <30 seconds
- **Data Loss During Recovery**: 0%
- **Health Check Accuracy**: >99%

### 7.2 Performance Metrics

- **Session Creation Time**: <100ms (P99)
- **Checkpoint Creation Time**: <500ms (P99)
- **Memory Overhead**: <5% increase
- **Database Query Performance**: <10ms (P95)

### 7.3 Reliability Metrics

- **System Uptime**: >99.9%
- **False Positive Alert Rate**: <1%
- **Recovery Test Success Rate**: 100%
- **Integration Test Pass Rate**: >99%

---

## 8. Future Enhancements

### 8.1 Advanced Features

- **Distributed Session Management**: Multi-node session coordination
- **Machine Learning Health Prediction**: AI-powered failure prediction
- **Advanced Analytics**: Session performance analytics and optimization
- **Cloud Storage Integration**: S3/GCS backup for session data

### 8.2 Security Enhancements

- **Session Encryption**: Encrypt sensitive session data at rest
- **Access Control**: Role-based session access control
- **Audit Trail**: Comprehensive session operation logging
- **Compliance Support**: GDPR/SOC2 compliance features

---

## Conclusion

This architecture provides a comprehensive, production-ready solution for session persistence and recovery in the ruv-swarm ecosystem. By building upon the existing high-quality persistence layer and maintaining backward compatibility, it delivers robust session management capabilities with minimal disruption to existing functionality.

The phased implementation approach ensures steady progress while allowing for validation and refinement at each stage. The extensive testing strategy and risk mitigation measures provide confidence in the system's reliability and performance.

This architecture positions ruv-swarm as an enterprise-grade solution capable of handling mission-critical workloads with the reliability and recoverability expected in production environments.