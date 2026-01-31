# Voice AI Assistant: PersonaPlex + Grok + Openclaw

**Date:** 2026-01-30
**Status:** Brainstorm / Design Phase

---

## A) Clarifying Questions (Highest-Leverage)

1. **Latency Target:** What's the maximum acceptable end-to-end latency (user stops speaking → starts hearing response)? <500ms is "magical," 500-1000ms is acceptable, >1500ms feels sluggish.

2. **Offline Capability:** Must the assistant work when there's no internet? If yes, do you expect full functionality or degraded mode (basic commands, no Grok)?

3. **Privacy Tier:** Which data can leave the device?
   - Never: raw audio
   - Maybe: transcripts
   - OK: anonymized intent summaries
   - Full cloud: everything goes to Grok

4. **iOS Relationship:** Is iOS a companion (audio I/O only, processing on Mac/cloud) or must it work standalone when Mac is unavailable?

5. **Tool Calling Scope:** What can the assistant do? Examples:
   - Read-only queries (weather, calendar lookup)
   - Write actions (send email, control HomeKit)
   - Code execution (run shell commands via Openclaw)

6. **Conversation Memory:** How much history? Per-session only, 24-hour rolling window, or persistent user profile?

7. **Multi-User:** Single user per device, or household with voice identification?

8. **Wake Word:** Always-on listening with wake word, or push-to-talk only?

9. **Personas as Product:** Is switching personas a core feature (e.g., "Talk to Chef mode"), or is there one fixed assistant personality?

10. **Grok Commitment:** Is Grok 4.1 Fast Reasoning the required backend, or would you consider alternatives (Claude, GPT-4o) if Grok proves limiting?

11. **Revenue Model Clarity:** Subscription monthly, one-time purchase, or usage-based? This affects architecture (metering, quotas, offline fallbacks).

12. **Model Size Constraint:** PersonaPlex is 7B params. For MLX on M-series Macs, this is feasible but tight on 8GB machines. Minimum supported Mac?

---

## B) Assumptions & Success Metrics

### Assumptions

| Assumption | Rationale |
|------------|-----------|
| Target device: M1+ Mac with 16GB+ RAM | 7B model + audio pipeline needs headroom |
| Network always available for MVP | Grok cloud is required; offline is v2 |
| Single user per device | Simplifies auth, memory, persona |
| English only for v1 | PersonaPlex trained primarily on English |
| Push-to-talk or simple VAD for v1 | True wake word requires separate model |
| Grok API has <300ms p50 latency | Without this, budget blows |
| PersonaPlex can run with MPS backend | Needs validation; CUDA path is Docker |
| MIT license for PersonaPlex is commercial-friendly | Confirm weights license (NVIDIA Open Model) |

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Voice Latency (user stop → response start)** | <800ms p50, <1500ms p95 | Client-side instrumentation |
| **Turn-taking naturalness** | <5% false interrupts | User study / dogfooding |
| **Grok reasoning quality** | 4.5/5 user satisfaction | In-app feedback |
| **Reliability** | 99.5% successful conversations | Error rate logging |
| **Cost per user/month** | <$5 at 30 min/day usage | Grok API billing |
| **Memory usage (macOS)** | <8GB resident | Activity Monitor |
| **iOS companion latency** | <200ms additional vs local | Network RTT |

---

## C) Architecture Options

### Option 1: Host-Audio / Container-Inference

```
┌─────────────────────────────────────────────────────────────┐
│                     macOS Host                              │
│  ┌─────────────────┐                                        │
│  │  Native Swift   │  Opus/PCM                              │
│  │  Audio Capture  │◄────────────────┐                      │
│  │  & Playback     │                 │                      │
│  └────────┬────────┘                 │                      │
│           │ WebSocket                │                      │
│           ▼                          │                      │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                   Docker                               │ │
│  │  ┌──────────────────────────────────────────────────┐  │ │
│  │  │            PersonaPlex Container                 │  │ │
│  │  │  ┌─────────┐  ┌─────────┐  ┌──────────────────┐  │  │ │
│  │  │  │  Mimi   │  │   LM    │  │ Grok Mediator    │  │  │ │
│  │  │  │ Encoder │→ │  (7B)   │→ │ (sends to cloud) │  │  │ │
│  │  │  └─────────┘  └─────────┘  └──────────────────┘  │  │ │
│  │  │       │            │              │              │  │ │
│  │  │       ▼            ▼              ▼              │  │ │
│  │  │  ┌─────────┐  ┌─────────┐  ┌──────────────────┐  │  │ │
│  │  │  │  Mimi   │  │Response │  │ Grok Response    │  │  │ │
│  │  │  │ Decoder │← │ Fusion  │← │ (injected safe)  │  │  │ │
│  │  │  └─────────┘  └─────────┘  └──────────────────┘  │  │ │
│  │  └──────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ HTTPS
                    ┌──────────────────┐
                    │   Grok 4.1 API   │
                    └──────────────────┘
```

**Pros:**
- GPU acceleration via Docker CUDA passthrough (if NVIDIA GPU) or CPU fallback
- Clean separation: Swift handles audio permissions/devices, Python handles ML
- Existing PersonaPlex Dockerfile works with minimal changes
- iOS connects to same Docker service over LAN

**Cons:**
- No CUDA on Apple Silicon; Docker runs CPU-only or needs Rosetta hacks
- Docker Desktop licensing ($5/user/month for >250 employees, but you're solo)
- Audio latency penalty: host→container WebSocket adds 10-50ms
- MPS (Metal) not accessible from Docker

**When to choose:** You have a Linux server or NVIDIA eGPU, or you accept CPU inference (~10x slower).

---

### Option 2: Mostly-On-Device (MLX Port)

```
┌─────────────────────────────────────────────────────────────┐
│                     macOS Host                              │
│  ┌─────────────────┐                                        │
│  │  Native Swift   │                                        │
│  │  Audio I/O      │                                        │
│  └────────┬────────┘                                        │
│           │ Shared Memory / Unix Socket                     │
│           ▼                                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              PersonaPlex-MLX                         │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐              │   │
│  │  │  Mimi   │  │  LM-MLX │  │ Mediator│──────────────┼───┼─► Grok API
│  │  │  MLX    │  │  (7B)   │  │         │              │   │
│  │  └─────────┘  └─────────┘  └─────────┘              │   │
│  │       Metal GPU Acceleration                         │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Openclaw Gateway                        │   │
│  │              (Docker or Native)                      │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Pros:**
- Full Metal GPU acceleration (M1 Pro: ~30 tokens/sec, M3 Max: ~80 t/s)
- No Docker required for inference (Openclaw can still use Docker)
- Lowest possible audio latency (shared memory)
- True offline mode possible (skip Grok for simple queries)

**Cons:**
- MLX port is substantial engineering: Mimi codec, transformer, streaming state
- PersonaPlex uses custom ops (RoPE, FlashAttention variants) needing rewrite
- Model weights need conversion (safetensors → MLX format)
- Risk: some architectures don't port cleanly

**When to choose:** This is the endgame for a premium macOS product. Start prototyping early.

---

### Option 3: Split Duplex (Simpler, Higher Latency)

```
┌─────────────────────────────────────────────────────────────┐
│                     macOS Host                              │
│  ┌─────────────────┐                                        │
│  │  Swift Audio    │                                        │
│  │  + Apple Speech │──── Transcript ────┐                   │
│  └─────────────────┘                    │                   │
│                                         ▼                   │
│                              ┌──────────────────┐           │
│                              │  Orchestrator    │           │
│                              │  (TypeScript)    │           │
│                              └────────┬─────────┘           │
│                                       │                     │
│         ┌─────────────────────────────┼─────────────────┐   │
│         │                             │                 │   │
│         ▼                             ▼                 │   │
│  ┌──────────────┐          ┌──────────────────┐         │   │
│  │  Grok API    │          │  PersonaPlex     │         │   │
│  │  (reasoning) │          │  (TTS only)      │         │   │
│  └──────────────┘          │  Docker/Native   │         │   │
│         │                  └──────────────────┘         │   │
│         │ Response text            ▲                    │   │
│         └──────────────────────────┘                    │   │
│                        Speak this text                  │   │
└─────────────────────────────────────────────────────────┘   │
```

**Pros:**
- Uses Apple's built-in SpeechRecognizer (free, private, low latency)
- PersonaPlex only does TTS, not full duplex (simpler, less compute)
- Easier to debug (text pipeline visible)
- Can swap in other TTS (Eleven Labs, OpenAI TTS) as fallback

**Cons:**
- Loses PersonaPlex's killer feature: natural interruptions/backchannels
- Apple Speech has ~500ms latency; combined with Grok, expect 1-2s total
- Two models (ASR + TTS) vs one integrated model
- iOS Speech framework has stricter background limits

**When to choose:** MVP if MLX port is blocked. Ship something, learn, iterate.

---

### iOS Architecture (All Options)

iOS cannot run Docker. Three paths:

**A) Thin Client (Recommended for MVP)**
```
iOS App ──WebSocket──► macOS Gateway ──► PersonaPlex + Grok
```
- iOS captures audio, streams to Mac over LAN/Tailscale
- Mac returns audio stream
- Works when Mac is on; graceful degradation otherwise

**B) Cloud Gateway**
```
iOS App ──WebSocket──► Cloud VM (PersonaPlex + Grok)
```
- Always works, but adds hosting cost (~$100/mo for GPU VM)
- Privacy concerns (audio leaves device)
- Could use serverless GPU (Modal, Replicate) for burst

**C) On-Device iOS (Future)**
- Wait for Core ML / MLX-iOS maturity
- A17 Pro can run 7B models slowly (~5 t/s)
- Not viable for real-time voice today

**Recommendation:** Start with (A) Thin Client. Add (B) as paid tier.

---

## D) Integration Design: Openclaw <> PersonaPlex

### Contract: Events & Messages

```typescript
// Shared types (TypeScript + Python pydantic)

interface AudioFrame {
  type: "audio";
  codec: "opus" | "pcm";
  sampleRate: 24000;
  channels: 1;
  data: Uint8Array;
  timestamp: number; // ms since session start
}

interface PartialTranscript {
  type: "transcript";
  text: string;
  isFinal: boolean;
  confidence: number;
  speaker: "user" | "assistant";
}

interface Interruption {
  type: "interruption";
  source: "user" | "system";
  timestamp: number;
}

interface ToolCallIntent {
  type: "tool_call";
  id: string;
  name: string; // e.g., "send_email", "run_command"
  arguments: Record<string, unknown>;
  confidence: number;
}

interface ToolCallResult {
  type: "tool_result";
  id: string;
  success: boolean;
  output?: string;
  error?: string;
}

interface PersonaState {
  type: "persona";
  voicePrompt: string; // e.g., "NATF2.pt"
  textPrompt: string;
  activeSkills: string[]; // Openclaw skills enabled
}

interface SessionControl {
  type: "control";
  action: "start" | "stop" | "pause" | "resume";
  config?: Partial<PersonaState>;
}
```

### Transport

**Primary:** WebSocket (bidirectional, low latency)
- `wss://localhost:8998/api/chat` for local
- `wss://gateway.example.com/api/chat` for remote

**Fallback:** gRPC-Web (if firewall blocks WS upgrades)

**Local-only by default:** No internet exposure without explicit config.

### Backpressure & Streaming Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    Full Duplex Flow                         │
│                                                             │
│  User Audio ────► [Ring Buffer 500ms] ────► Encoder         │
│       ▲                                         │           │
│       │                                         ▼           │
│  Interruption ◄── [VAD Detection] ◄──── LM Streaming        │
│  Detection                                      │           │
│       │                                         ▼           │
│       └────────────────────────────────── Decoder           │
│                                                 │           │
│  Assistant Audio ◄── [Jitter Buffer 100ms] ◄───┘           │
└─────────────────────────────────────────────────────────────┘
```

**Rules:**
1. Audio frames are fire-and-forget (UDP semantics over WS)
2. Ring buffer smooths input jitter; drops oldest if overflow
3. Jitter buffer on output prevents choppy playback
4. Interruption signal immediately flushes output buffer
5. Tool calls block further generation until result arrives (configurable)

### Observability

```yaml
# Structured logging (JSON lines)
{
  "ts": "2026-01-30T10:15:30.123Z",
  "level": "info",
  "event": "audio_frame",
  "session_id": "abc123",
  "direction": "inbound",
  "latency_ms": 42,
  "buffer_depth": 3
}

# Metrics (Prometheus format)
personaplex_audio_latency_ms{direction="inbound"} 42
personaplex_buffer_depth{buffer="input"} 3
personaplex_interruptions_total 7
personaplex_tool_calls_total{tool="send_email"} 2

# Traces (OpenTelemetry)
- Span: session
  - Span: utterance
    - Span: encode
    - Span: lm_inference
    - Span: grok_call
    - Span: decode
```

---

## E) Grok Mediation Layer (Critical)

### What We Send to Grok

**NOT raw audio.** Grok is text-only.

```json
{
  "conversation_id": "uuid",
  "turn_id": 42,
  "user_message": {
    "text": "What's the weather in Tokyo?",
    "confidence": 0.95,
    "detected_intent": "weather_query",
    "entities": {"location": "Tokyo"}
  },
  "context": {
    "persona": "helpful assistant",
    "active_tools": ["weather", "calendar"],
    "recent_turns": 5,
    "user_preferences": {"units": "metric"}
  },
  "system_prompt": "You are a voice assistant. Be concise. Respond in 1-3 sentences."
}
```

### What We Receive from Grok

```json
{
  "response": {
    "text": "It's currently 15 degrees Celsius in Tokyo with partly cloudy skies.",
    "tool_calls": [],
    "reasoning_trace": "User asked about weather. Retrieved current conditions."
  },
  "metadata": {
    "model": "grok-4.1-fast",
    "latency_ms": 180,
    "tokens_used": {"prompt": 120, "completion": 35}
  }
}
```

**Validation Schema (Zod):**

```typescript
const GrokResponseSchema = z.object({
  response: z.object({
    text: z.string().max(2000), // Hard limit
    tool_calls: z.array(z.object({
      name: z.string().regex(/^[a-z_]+$/), // Allowlist pattern
      arguments: z.record(z.unknown())
    })).max(3), // Max 3 tool calls per turn
    reasoning_trace: z.string().optional()
  }),
  metadata: z.object({
    model: z.string(),
    latency_ms: z.number(),
    tokens_used: z.object({
      prompt: z.number(),
      completion: z.number()
    })
  })
});
```

### Injection Safety: Control Channel Architecture

**NEVER concatenate Grok output directly into PersonaPlex system prompt.**

```
┌─────────────────────────────────────────────────────────────┐
│                   Injection-Safe Architecture               │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                PersonaPlex System Prompt             │    │
│  │  (IMMUTABLE - set once at session start)            │    │
│  │                                                      │    │
│  │  "You are a helpful assistant named Alex.            │    │
│  │   Speak naturally. Use the [RESPONSE] below."       │    │
│  └─────────────────────────────────────────────────────┘    │
│                            │                                │
│                            ▼                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           Structured Response Slot                   │    │
│  │  (Grok output injected here, sanitized)             │    │
│  │                                                      │    │
│  │  [RESPONSE]                                          │    │
│  │  Text: "It's 15 degrees in Tokyo."                  │    │
│  │  Emotion: neutral                                    │    │
│  │  Pace: normal                                        │    │
│  │  [/RESPONSE]                                         │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

**Sanitization Pipeline:**

```python
def sanitize_grok_response(raw: str) -> str:
    # 1. Schema validation (already done by Zod)
    # 2. Length limit
    text = raw[:2000]

    # 3. Strip any <system> tags (injection attempt)
    text = re.sub(r'</?system>', '', text, flags=re.IGNORECASE)

    # 4. Strip control characters
    text = ''.join(c for c in text if c.isprintable() or c in '\n ')

    # 5. Escape special tokens
    text = text.replace('[RESPONSE]', '').replace('[/RESPONSE]', '')

    # 6. Validate UTF-8
    text = text.encode('utf-8', errors='ignore').decode('utf-8')

    return text
```

**Allowlist Fields Only:**

```python
ALLOWED_RESPONSE_FIELDS = {'text', 'emotion', 'pace', 'volume'}

def build_response_slot(grok_output: dict) -> str:
    lines = ['[RESPONSE]']
    for field in ALLOWED_RESPONSE_FIELDS:
        if field in grok_output:
            value = sanitize_grok_response(str(grok_output[field]))
            lines.append(f"{field.title()}: {value}")
    lines.append('[/RESPONSE]')
    return '\n'.join(lines)
```

### Memory Strategy

| Data | Location | Retention |
|------|----------|-----------|
| Raw audio | Never persisted | Ephemeral (stream) |
| Transcripts | Local SQLite | 7 days, encrypted |
| Grok conversation | Local + Grok API | Session only at Grok; local 30 days |
| User preferences | Local config | Permanent |
| Tool call logs | Local | 30 days |

**What goes to Grok:**
- Current turn text
- Last N turns (configurable, default 5)
- Summarized context (not raw history)
- No audio, no PII unless user opts in

---

## F) PersonaPlex MLX Port Brainstorm

### Current Stack Analysis

PersonaPlex components needing port:

| Component | PyTorch Usage | MLX Difficulty | Notes |
|-----------|---------------|----------------|-------|
| **Mimi Encoder** | Conv1D, ResNet-style | Medium | Standard ops |
| **Mimi Decoder** | Conv1DTranspose | Medium | MLX has transpose conv |
| **VQ Codebook** | Embedding + Lookup | Easy | Direct equivalent |
| **Transformer LM** | Attention, RoPE, Norm | Medium | mlx-lm exists |
| **Depformer** | Small transformer | Medium | Same as above |
| **Streaming State** | Custom caching | Hard | No MLX equivalent |
| **Opus Codec** | C library (sphn) | N/A | Keep as-is (C binding) |

### Fast Path: PyTorch MPS (No Docker)

**Goal:** Run PersonaPlex on macOS Metal without full MLX rewrite.

```bash
# Test MPS support
python -c "import torch; print(torch.backends.mps.is_available())"
```

**Changes needed:**
1. Remove CUDA-specific code (`torch.cuda.*`)
2. Replace `device="cuda"` with `device="mps"`
3. Handle MPS limitations (some ops fallback to CPU)

**Estimated effort:** 2-3 days

**Risk:** MPS is slower than CUDA, and some ops (FlashAttention) don't exist.

### MLX Path: Full Port

**Phase 1: Mimi Codec (Week 1-2)**
- Port SEncoderDecoder (SEANet architecture)
- Port VQ (ResidualVectorQuantizer)
- Test: encode→decode round-trip matches PyTorch

**Phase 2: LM Transformer (Week 3-4)**
- Use `mlx-lm` as base
- Port custom RoPE implementation
- Port Depformer (depth-first transformer)
- Port streaming state management

**Phase 3: Integration (Week 5-6)**
- Wire up audio pipeline
- Benchmark latency
- Optimize hot paths

### Weight Conversion

```python
# PyTorch → MLX safetensors
import torch
import mlx.core as mx
from safetensors.torch import load_file

# Load PyTorch weights
pt_weights = load_file("personaplex-7b.safetensors")

# Convert to MLX
mlx_weights = {}
for name, tensor in pt_weights.items():
    # Transpose certain layers (Conv1D weight layout differs)
    if 'conv' in name and 'weight' in name:
        tensor = tensor.transpose(-1, -2)
    mlx_weights[name] = mx.array(tensor.numpy())

# Save
mx.savez("personaplex-7b-mlx.npz", **mlx_weights)
```

### "Good Enough" Alternatives

If MLX port is too costly:

1. **llama.cpp with Mimi wrapper:** llama.cpp has Metal support. Port only Mimi to MLX, use llama.cpp for LM.

2. **Hybrid: Apple Neural Engine for Mimi, CPU for LM:** ANE is fast for conv-heavy models. 7B LM on CPU is slow but works.

3. **Smaller model:** Fine-tune a 1-3B variant of PersonaPlex. Faster inference, lower quality.

4. **Cloud fallback:** Run PersonaPlex on Replicate/Modal for <$0.01/minute. Local Mimi only.

### 2-Week Spike Plan

**Goal:** De-risk the MLX port with minimal investment.

| Day | Task | Success Criteria |
|-----|------|------------------|
| 1-2 | Run PersonaPlex with MPS backend | Server starts, basic inference works |
| 3-4 | Benchmark MPS vs CPU latency | Measure tokens/sec, audio latency |
| 5-6 | Port Mimi encoder to MLX | Encodes audio, outputs match PyTorch |
| 7-8 | Port Mimi decoder to MLX | Round-trip audio test passes |
| 9-10 | Prototype LM inference in MLX | Single forward pass, logits match |
| 11-12 | Streaming state prototype | 10 consecutive tokens work |
| 13-14 | Integration test, latency benchmark | End-to-end audio with MLX |

**Exit criteria:**
- MPS path: <2x latency vs CUDA → proceed with MPS
- MLX path: encoder/decoder working, LM feasible → commit to full port
- Neither: fall back to cloud or Split Duplex architecture

### 6-Week MVP Plan

**Week 1-2: Foundation**
- [ ] MPS backend working (fallback)
- [ ] Mimi codec ported to MLX
- [ ] Audio I/O via Swift (Swabble integration)

**Week 3-4: Core LM**
- [ ] LM transformer ported (use mlx-lm patterns)
- [ ] Streaming state management
- [ ] Depformer ported

**Week 5: Integration**
- [ ] End-to-end voice loop working
- [ ] Grok mediator integrated
- [ ] Openclaw skill execution

**Week 6: Polish**
- [ ] Latency optimization (<800ms target)
- [ ] Error handling, graceful degradation
- [ ] iOS thin client prototype

**Definition of Done:**
- User can have a 5-minute conversation with <1s latency
- Grok provides reasoning, PersonaPlex speaks
- At least one Openclaw skill works (e.g., "send a message")

---

## G) Packaging, UX, and Payments

### macOS Packaging with Docker

**Approach:** Native Swift app wraps Docker.

```
┌─────────────────────────────────────────────────────────────┐
│                    PersonaVoice.app                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  SwiftUI Menubar App                                │    │
│  │  - Audio permissions                                 │    │
│  │  - Microphone selection                              │    │
│  │  - Settings UI                                       │    │
│  └─────────────────────────────────────────────────────┘    │
│                            │                                │
│                            ▼                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Embedded Docker Compose                            │    │
│  │  - PersonaPlex container                            │    │
│  │  - Openclaw container                                │    │
│  │  - Auto-start on app launch                          │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

**Installer:**
1. DMG with drag-to-Applications
2. First launch: prompt for Docker Desktop (or Colima) install
3. Download containers in background (~5GB)
4. Grant microphone permission

**Updates:**
- App updates via Sparkle (standard macOS pattern)
- Container updates via `docker pull` in background

**Permissions:**
- Microphone (required)
- Accessibility (optional, for keyboard shortcuts)
- Automation (optional, for Openclaw tool calling)

### iOS App Scope

**V1: Companion App**
- Audio capture/playback
- WebSocket to Mac gateway
- Settings sync
- Push notifications for async responses

**V2: Standalone (stretch)**
- On-device VAD (Voice Activity Detection)
- Direct Grok API connection
- Cached responses for common queries
- Graceful offline mode

**Code sharing:**
- Swift Package for audio handling (Swabble already exists)
- Shared WebSocket protocol
- Shared settings schema

### Pricing Model

| Tier | Price | Features | Infra Cost |
|------|-------|----------|------------|
| **Free Trial** | $0 for 7 days | 30 min/day | $0.50/user |
| **Personal** | $15/month | Unlimited local, 2 hrs cloud | $3/user |
| **Pro** | $30/month | Unlimited, iOS companion, priority | $8/user |
| **BYO Key** | $10/month | Bring your own Grok API key | $0.50/user |

**Why these numbers:**
- Grok API: ~$0.10 per 10 min conversation (estimated)
- 2 hrs/month = $1.20 Grok cost
- PersonaPlex compute: ~$0.50/hr on cloud GPU
- Margin: 60-70% at Personal tier

**Payment integration:**
- RevenueCat for iOS + Mac
- Stripe for web fallback
- License key validation for offline mode

---

## H) Risk Register

| # | Risk | P | I | Detection | Mitigation |
|---|------|---|---|-----------|------------|
| 1 | **MLX port takes >3 months** | High | High | Week 2 spike failure | Fall back to Split Duplex; cloud PersonaPlex |
| 2 | **Grok latency >500ms** | Med | High | Early API testing | Cache common responses; local fallback for simple queries |
| 3 | **Docker on Mac is clunky** | Med | Med | User complaints | Colima as lightweight alternative; native build long-term |
| 4 | **MPS backend too slow** | Med | Med | Benchmark in spike | CPU fallback; quantization (INT8) |
| 5 | **PersonaPlex license blocks commercial use** | Low | Critical | Lawyer review | NVIDIA Open Model License allows commercial; confirm |
| 6 | **iOS Audio background restrictions** | Med | Med | iOS 17 testing | Push notification fallback; companion-only mode |
| 7 | **Prompt injection via Grok** | Med | High | Security testing | Control channel architecture; sanitization |
| 8 | **User privacy concerns (audio to cloud)** | Med | Med | User feedback | Local-first default; clear privacy policy |
| 9 | **Grok API pricing changes** | Low | Med | Monitor pricing page | Multi-provider support (Claude/GPT fallback) |
| 10 | **Memory pressure on 8GB Macs** | High | Med | Activity Monitor | Minimum spec: 16GB; quantized models for 8GB |

---

## I) Next Actions Checklist

### Immediate (This Week)

1. **[1 day] Validate PersonaPlex license**
   - Read NVIDIA Open Model License
   - Confirm commercial use allowed
   - Document any attribution requirements

2. **[1 day] Get Grok API access**
   - Sign up for xAI API
   - Test basic completions
   - Measure latency from your location

3. **[2 days] Run PersonaPlex locally with MPS**
   - Clone repo, install deps
   - Modify `server.py` for `device="mps"`
   - Run `offline.py` test, measure speed

4. **[1 day] Set up dev environment**
   - Docker Desktop or Colima
   - Build PersonaPlex container
   - Test WebSocket connection from Swift

### Week 2

5. **[3 days] Mimi MLX port spike**
   - Port encoder Conv1D layers
   - Test single audio frame encoding
   - Compare outputs to PyTorch

6. **[2 days] Grok mediator prototype**
   - Python script: transcript → Grok → response
   - Implement sanitization
   - Measure round-trip latency

7. **[2 days] Swift audio harness**
   - Extend Swabble for WebSocket audio
   - Test opus encode/decode
   - Bidirectional stream working

### Week 3-4

8. **[5 days] Complete Mimi MLX port**
   - Decoder
   - VQ codebook
   - Round-trip test

9. **[5 days] LM transformer MLX port**
   - Base transformer
   - RoPE
   - Streaming state

### Week 5-6

10. **[5 days] Integration**
    - Wire everything together
    - Grok injection working
    - First real conversation

11. **[5 days] Polish + iOS prototype**
    - Latency optimization
    - iOS thin client
    - Basic UI

---

## What I Need From You (For Next Phase)

Once you provide the repos, I'll:

1. **Inspect PersonaPlex:**
   - `moshi/models/lm.py` → model architecture details
   - `moshi/modules/` → custom ops needing port
   - `requirements.txt` → dependency audit
   - Run `python -m moshi.offline --help` to understand CLI

2. **Inspect Openclaw:**
   - `src/provider-web.ts` → WebSocket patterns
   - `extensions/` → plugin architecture
   - `Swabble/` → Swift audio handling
   - `docker-compose.yml` → container orchestration

3. **Run commands:**
   ```bash
   # PersonaPlex
   pip install moshi/.
   python -m moshi.server --help
   python -m moshi.offline --input-wav test.wav --output-wav out.wav

   # Check model size
   du -sh ~/.cache/huggingface/hub/models--nvidia--personaplex*
   ```

4. **Architecture clues to look for:**
   - How streaming state is managed (critical for MLX port)
   - Audio frame size and sample rate
   - Text tokenizer details
   - Any hardcoded CUDA ops
