# Component Interaction Diagrams
## Session Persistence and Recovery Architecture - Issue #137

**Author**: System Architect Agent  
**Date**: 2025-01-24  
**Version**: 1.0  
**Parent Document**: [Session Persistence Architecture](./session-persistence-architecture.md)

---

## System Component Overview

### 1. High-Level Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Session Management Layer                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │   SessionStore  │  │ RecoveryManager │  │ HealthMonitor   │              │
│  │                 │  │                 │  │                 │              │
│  │ • Memory Tier   │  │ • Failure Detect│  │ • Metric Collect│              │
│  │ • Persistent    │  │ • Recovery Exec │  │ • Alert Manager │              │
│  │ • Archive Tier  │  │ • State Validation│ • Trend Analysis│              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
├─────────────────────────────────────────────────────────────────────────────┤
│                        State Synchronization Layer                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │  StateManager   │  │   EventBus      │  │ ChangeStream    │              │
│  │                 │  │                 │  │                 │              │
│  │ • State Sync    │  │ • Event Routing │  │ • Change Detect │              │
│  │ • Conflict Res  │  │ • Subscription  │  │ • Delta Compute │              │
│  │ • Consistency   │  │ • Broadcasting  │  │ • Stream Process│              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
├─────────────────────────────────────────────────────────────────────────────┤
│                           Enhanced Core Systems                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │ RuvSwarm        │  │ EnhancedMCP     │  │ SwarmPersistence│              │
│  │ (Enhanced)      │  │ Tools           │  │ Pooled          │              │
│  │                 │  │                 │  │                 │              │
│  │ • Session Aware │  │ • Session APIs  │  │ • Connection    │              │
│  │ • Auto Recovery │  │ • Connection Mgr│  │   Pool          │              │
│  │ • Health Report │  │ • Offline Mode  │  │ • HA Features   │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Component Interactions

### 2. Session Lifecycle Flow

```mermaid
sequenceDiagram
    participant Client as Client/Claude
    participant MCP as EnhancedMCPTools
    parameter SS as SessionStore
    participant RS as RuvSwarm
    participant DB as SwarmPersistence
    participant HM as HealthMonitor
    participant RM as RecoveryManager

    Client->>MCP: session_create(config)
    MCP->>SS: createSession(sessionConfig)
    SS->>DB: INSERT session record
    DB-->>SS: session saved
    SS->>RS: initializeSwarm(sessionId, config)
    RS-->>SS: swarm initialized
    SS->>HM: registerSession(sessionId)
    HM-->>SS: monitoring started
    SS-->>MCP: Session created
    MCP-->>Client: SessionCreateResult

    Note over Client,RM: Session is now active and monitored

    Client->>MCP: agent_spawn(sessionId, agentConfig)
    MCP->>RS: addAgent(agentConfig)
    RS->>DB: store agent state
    RS->>SS: updateSessionState(sessionId, newState)
    SS->>HM: updateMetrics(sessionId, agentAdded)
    RS-->>MCP: Agent spawned
    MCP-->>Client: AgentSpawnResult

    Note over Client,RM: Continuous health monitoring

    HM->>HM: collectMetrics()
    HM->>SS: getSessionHealth(sessionId)
    SS-->>HM: health metrics
    HM->>RM: checkThresholds(metrics)
    
    alt Health issue detected
        RM->>RM: triggerRecovery(sessionId, issue)
        RM->>SS: createCheckpoint(sessionId)
        RM->>Client: notifyHealthIssue(details)
    end
```

### 3. Recovery Process Flow

```mermaid
flowchart TD
    A[Failure Detected] --> B{Classify Failure Type}
    B -->|Process Crash| C[Process Recovery Path]
    B -->|MCP Disconnect| D[Connection Recovery Path]  
    B -->|State Corruption| E[State Recovery Path]
    B -->|Resource Exhaustion| F[Resource Recovery Path]

    C --> C1[Load Session State from DB]
    C1 --> C2{State Integrity OK?}
    C2 -->|Yes| C3[Restore In-Memory State]
    C2 -->|No| C4[Load Latest Checkpoint]
    C3 --> C5[Re-establish MCP Connections]
    C4 --> C6[Replay Operations Since Checkpoint]
    C5 --> C7[Notify Recovery Success]
    C6 --> C5

    D --> D1[Attempt Immediate Reconnection]
    D1 --> D2{Connection Successful?}
    D2 -->|Yes| D3[Sync Queued Operations]
    D2 -->|No| D4[Exponential Backoff]
    D3 --> D5[Resume Normal Operation]
    D4 --> D6{Max Attempts Reached?}
    D6 -->|No| D1
    D6 -->|Yes| D7[Enter Offline Mode]
    D7 --> D8[Periodic Reconnection Attempts]
    D8 --> D2

    E --> E1[Validate Current State]  
    E1 --> E2{Validation Result}
    E2 -->|Minor Issues| E3[Graceful Degradation]
    E2 -->|Major Issues| E4[Checkpoint Rollback]
    E2 -->|Critical Issues| E5[Fresh State Initialization]
    E3 --> E6[Continue with Warnings]
    E4 --> E7[Restore from Checkpoint]
    E5 --> E8[Notify Data Loss]
    E7 --> E6
    E8 --> E6

    F --> F1[Identify Resource Bottleneck]
    F1 --> F2{Resource Type}
    F2 -->|Memory| F3[Trigger Memory Cleanup]
    F2 -->|Disk| F4[Archive Old Data]
    F2 -->|Connections| F5[Scale Connection Pool]
    F3 --> F6[Monitor Resource Usage]
    F4 --> F6
    F5 --> F6
    F6 --> F7{Resources Available?}
    F7 -->|Yes| F8[Resume Normal Operation]
    F7 -->|No| F9[Enter Degraded Mode]
```

### 4. Health Monitoring Architecture

```mermaid
graph TB
    subgraph "Health Monitoring System"
        HM[Health Monitor Orchestrator]
        MA[Metrics Aggregator]
        TA[Trend Analyzer]
        AE[Alert Engine]
        RT[Recovery Trigger]
    end

    subgraph "Component Health Checkers"
        SSC[SessionStore Checker]
        RMC[RecoveryManager Checker]
        MCC[MCP Connection Checker]
        APC[Agent Pool Checker]
        TQC[Task Queue Checker]
        NNC[Neural Network Checker]
        DBC[Database Checker]
    end

    subgraph "Data Flow"
        CM[Collected Metrics]
        TD[Trend Data]
        AL[Alerts]
        RA[Recovery Actions]
    end

    SSC --> CM
    RMC --> CM
    MCC --> CM
    APC --> CM
    TQC --> CM
    NNC --> CM
    DBC --> CM

    HM --> SSC
    HM --> RMC
    HM --> MCC
    HM --> APC
    HM --> TQC  
    HM --> NNC
    HM --> DBC

    CM --> MA
    MA --> TA
    TA --> TD
    MA --> AE
    AE --> AL
    AL --> RT
    RT --> RA

    style HM fill:#e1f5fe
    style MA fill:#f3e5f5
    style TA fill:#e8f5e8
    style AE fill:#fff3e0
    style RT fill:#ffebee
```

### 5. MCP Connection State Management

```mermaid
stateDiagram-v2
    [*] --> Initializing
    Initializing --> Connected : Connection established
    Initializing --> Failed : Connection failed
    
    Connected --> Heartbeating : Start heartbeat
    Heartbeating --> Connected : Heartbeat OK
    Heartbeating --> Reconnecting : Heartbeat failed
    
    Reconnecting --> Connected : Reconnection successful
    Reconnecting --> OfflineMode : Max retries exceeded
    
    OfflineMode --> QueueOperations : Queue new operations
    QueueOperations --> OfflineMode : Continue queuing
    OfflineMode --> PeriodicRetry : Timer triggered
    PeriodicRetry --> Connected : Reconnection successful
    PeriodicRetry --> OfflineMode : Still disconnected
    
    Connected --> SyncMode : Reconnected from offline
    SyncMode --> Connected : Sync completed
    
    Failed --> Initializing : Retry initialization
    Connected --> Terminating : Shutdown requested
    OfflineMode --> Terminating : Shutdown requested
    Terminating --> [*]

    note right of Heartbeating
        Heartbeat every 5s
        Timeout after 15s
    end note

    note right of OfflineMode
        Queue operations
        Periodic retry every 60s
        Max offline time: 5min
    end note
```

### 6. Data Flow Architecture

```mermaid
flowchart LR
    subgraph "Client Layer"
        C[Claude Client]
        WS[WebSocket/SSE]
        ST[stdio]
    end

    subgraph "MCP Protocol Layer"
        MCP[Enhanced MCP Tools]
        CM[Connection Manager]
        OM[Offline Mode Handler]
    end

    subgraph "Session Management Layer"
        SS[Session Store]
        L1[Memory Tier - L1]
        L2[Persistent Tier - L2]
        L3[Archive Tier - L3]
    end

    subgraph "Core Business Logic"
        RS[RuvSwarm Core]
        AP[Agent Pool]
        TQ[Task Queue]
        NN[Neural Networks]
    end

    subgraph "Persistence Layer"
        SP[SwarmPersistence Pooled]
        RC[Reader Connections]
        WC[Writer Connection]
        WT[Worker Threads]
    end

    subgraph "Storage Layer"
        DB[(SQLite Database)]
        WAL[(WAL Files)]
        ARC[(Archive Files)]
    end

    C --> MCP
    WS --> CM
    ST --> CM
    CM --> MCP
    CM --> OM

    MCP --> SS
    SS --> L1
    L1 --> L2
    L2 --> L3

    SS --> RS
    RS --> AP
    RS --> TQ
    RS --> NN

    L2 --> SP
    SP --> RC
    SP --> WC
    SP --> WT

    RC --> DB
    WC --> DB
    WT --> DB
    DB --> WAL
    L3 --> ARC

    style L1 fill:#e3f2fd
    style L2 fill:#e8f5e8
    style L3 fill:#fff3e0
    style DB fill:#f3e5f5
```

---

## Component Integration Points

### 7. RuvSwarm Core Integration

```typescript
// Enhanced RuvSwarm with session awareness
class EnhancedRuvSwarm extends RuvSwarm {
  constructor(options: SwarmOptions & { sessionId?: string }) {
    super(options);
    this.sessionId = options.sessionId;
    this.sessionStore = new SessionStore();
    this.recoveryManager = new RecoveryManager(this);
    this.healthMonitor = new HealthMonitor(this);
    
    // Register for recovery events
    this.recoveryManager.on('recovery-needed', this.handleRecovery.bind(this));
    this.healthMonitor.on('health-issue', this.handleHealthIssue.bind(this));
  }
  
  async addAgent(config: AgentConfig): Promise<string> {
    const agentId = await super.addAgent(config);
    
    // Session-aware persistence
    if (this.sessionId) {
      await this.sessionStore.updateSessionState(this.sessionId, {
        agents: this.state.agents,
        lastModified: new Date()
      });
      
      // Update health metrics
      await this.healthMonitor.recordMetric('agent_added', {
        sessionId: this.sessionId,
        agentId,
        timestamp: Date.now()
      });
    }
    
    return agentId;
  }
  
  private async handleRecovery(event: RecoveryEvent): Promise<void> {
    // Implement recovery logic specific to this swarm instance
  }
  
  private async handleHealthIssue(issue: HealthIssue): Promise<void> {
    // Implement health issue handling
  }
}
```

### 8. Enhanced MCP Tools Integration

```typescript
class SessionAwareMCPTools extends EnhancedMCPTools {
  constructor() {
    super();
    this.sessionManager = new SessionManager();
    this.connectionManager = new MCPConnectionManager();
  }
  
  async swarm_init(args: SwarmInitArgs & { sessionId?: string }): Promise<SwarmInitResult> {
    // Create or restore session
    let session: Session;
    
    if (args.sessionId) {
      // Restore existing session
      session = await this.sessionManager.getSession(args.sessionId);
      if (!session) {
        throw new Error(`Session ${args.sessionId} not found`);
      }
      
      // Trigger recovery if needed
      if (session.status === 'terminated' || session.status === 'recovering') {
        await this.recoveryManager.recoverSession(args.sessionId);
        session = await this.sessionManager.getSession(args.sessionId);
      }
    } else {
      // Create new session
      session = await this.sessionManager.createSession({
        configuration: args,
        parentSessionId: args.parentSessionId
      });
    }
    
    // Initialize swarm with session context
    const swarm = new EnhancedRuvSwarm({
      ...args,
      sessionId: session.sessionId
    });
    
    await swarm.init();
    
    // Store swarm reference
    this.activeSwarms.set(session.sessionId, swarm);
    
    return {
      success: true,
      sessionId: session.sessionId,
      swarmId: swarm.id,
      status: session.status
    };
  }
  
  async session_checkpoint(args: { sessionId: string }): Promise<SessionCheckpointResult> {
    const session = await this.sessionManager.getSession(args.sessionId);
    if (!session) {
      throw new Error(`Session ${args.sessionId} not found`);
    }
    
    const swarm = this.activeSwarms.get(args.sessionId);
    if (!swarm) {
      throw new Error(`No active swarm for session ${args.sessionId}`);
    }
    
    // Create checkpoint
    const checkpoint = await this.sessionManager.createCheckpoint(args.sessionId, {
      state: swarm.getState(),
      metadata: {
        timestamp: new Date(),
        trigger: 'manual',
        agentCount: swarm.getAgentCount(),
        taskCount: swarm.getTaskCount()
      }
    });
    
    return {
      success: true,
      checkpointId: checkpoint.checkpointId,
      timestamp: checkpoint.timestamp,
      size: checkpoint.size
    };
  }
}
```

### 9. Database Integration Schema

```sql
-- Enhanced tables for session management integration

-- Extend existing swarms table
ALTER TABLE swarms ADD COLUMN session_id TEXT;
ALTER TABLE swarms ADD COLUMN session_status TEXT DEFAULT 'active';
CREATE INDEX IF NOT EXISTS idx_swarms_session ON swarms(session_id);

-- Extend existing agents table  
ALTER TABLE agents ADD COLUMN session_id TEXT;
CREATE INDEX IF NOT EXISTS idx_agents_session ON agents(session_id);

-- Extend existing tasks table
ALTER TABLE tasks ADD COLUMN session_id TEXT;
CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id);

-- Health monitoring integration
CREATE TABLE IF NOT EXISTS session_component_health (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  component_name TEXT NOT NULL,
  health_status TEXT NOT NULL,
  metrics TEXT,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_component_health_session ON session_component_health(session_id, timestamp);

-- MCP connection state integration
CREATE TABLE IF NOT EXISTS session_mcp_connections (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  connection_id TEXT NOT NULL,
  protocol_type TEXT NOT NULL,
  status TEXT NOT NULL,
  capabilities TEXT,
  last_heartbeat DATETIME,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (session_id) REFERENCES sessions(session_id),
  UNIQUE(session_id, connection_id)
);

CREATE INDEX IF NOT EXISTS idx_mcp_connections_session ON session_mcp_connections(session_id);
```

---

## Performance Considerations

### 10. Memory Management Strategy

```mermaid
graph TD
    A[Session Created] --> B[L1 Memory Cache]
    B --> C{Cache Full?}
    C -->|No| D[Store in Memory]
    C -->|Yes| E[LRU Eviction]
    E --> F[Move to L2 Persistent Storage]
    F --> G[Update L1 Cache]
    
    H[Session Access] --> I{In L1 Cache?}
    I -->|Yes| J[Return from Memory]
    I -->|No| K[Load from L2]
    K --> L[Add to L1 Cache]
    L --> M[Return Data]
    
    N[Session Archived] --> O{Age > 24h?}
    O -->|Yes| P[Compress and Move to L3]
    O -->|No| Q[Keep in L2]
    P --> R[Update Indexes]
    
    style B fill:#e3f2fd
    style F fill:#e8f5e8  
    style P fill:#fff3e0
```

### 11. Connection Pool Optimization

```mermaid
graph LR
    subgraph "Read Operations"
        R1[Reader 1] --> DB[(Database)]
        R2[Reader 2] --> DB
        R3[Reader 3] --> DB
        R4[Reader 4] --> DB
    end
    
    subgraph "Write Operations"
        W[Writer] --> DB
        WQ[Write Queue] --> W
    end
    
    subgraph "Heavy Operations"
        WT1[Worker Thread 1] --> DB
        WT2[Worker Thread 2] --> DB
        WTQ[Worker Queue] --> WT1
        WTQ --> WT2
    end
    
    subgraph "Health Monitoring"
        HM[Health Monitor] --> R1
        HM --> R2
        HM --> W
        HM --> WT1
    end
    
    style DB fill:#f3e5f5
    style W fill:#ffebee
    style WT1 fill:#e8f5e8
    style WT2 fill:#e8f5e8
```

---

## Conclusion

These component interaction diagrams provide a comprehensive view of how the session persistence and recovery architecture integrates with the existing ruv-swarm system. The diagrams illustrate:

1. **Clear separation of concerns** between session management, recovery, and core functionality
2. **Robust data flow patterns** that ensure consistency and reliability
3. **Comprehensive integration points** that extend existing components without breaking changes
4. **Performance optimization strategies** that maintain system responsiveness
5. **Health monitoring architecture** that provides proactive issue detection

The architecture is designed to be:
- **Minimally invasive**: Builds upon existing components with minimal changes
- **Highly performant**: Uses multi-tier caching and connection pooling
- **Fault tolerant**: Provides multiple recovery mechanisms for different failure types
- **Observable**: Comprehensive health monitoring and alerting
- **Scalable**: Designed to handle enterprise workloads

This design ensures that the session persistence and recovery features integrate seamlessly with the existing ruv-swarm ecosystem while providing robust, production-ready capabilities.