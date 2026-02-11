"""
Layer 6: Cognitive Core - Memory System

Implements episodic, semantic, and working memory for Brian-QARI.
Allows the robot to learn from experience, recall past events,
and build up knowledge about its environment and tasks.

Memory Architecture:
    - Working Memory: Short-term buffer for current task context (seconds)
    - Episodic Memory: Autobiographical events with temporal context (hours/days)
    - Semantic Memory: Generalized knowledge and learned facts (permanent)
    - Skill Memory: Motor skill parameters from successful actions (permanent)

All memories are stored as quantum-encoded semantic vectors (2048-D)
for efficient similarity search and associative recall.

Integrates with BrianMind's reflect() pipeline to consolidate
short-term experiences into long-term knowledge.
"""

import hashlib
import json
import math
import os
import time
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

class MemoryType(Enum):
    """Types of memory in Brian's cognitive architecture."""
    WORKING = auto()    # Current task context, very short lived
    EPISODIC = auto()   # Specific events with time and place
    SEMANTIC = auto()   # General knowledge and facts
    SKILL = auto()      # Motor skill parameters and policies


class MemoryImportance(Enum):
    """How important a memory is (affects retention)."""
    TRIVIAL = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Memory:
    """A single memory entry in Brian's memory system."""
    memory_id: str
    memory_type: MemoryType
    content: Dict[str, Any]
    semantic_vector: np.ndarray     # 2048-D encoding
    importance: MemoryImportance = MemoryImportance.MEDIUM
    timestamp_created: float = 0.0
    timestamp_accessed: float = 0.0
    access_count: int = 0
    emotional_valence: float = 0.0   # -1 (negative) to 1 (positive)
    confidence: float = 1.0
    associations: List[str] = field(default_factory=list)  # linked memory IDs
    tags: List[str] = field(default_factory=list)
    decay_rate: float = 0.01         # how fast this memory fades
    strength: float = 1.0            # current memory strength (0-1)


@dataclass
class EpisodicEvent:
    """A specific event in episodic memory."""
    event_type: str          # "grasped_object", "navigated_to", "spoke_to_human"
    location: Optional[np.ndarray] = None  # where it happened
    participants: List[str] = field(default_factory=list)  # who was involved
    objects: List[str] = field(default_factory=list)  # what objects were involved
    outcome: str = "unknown"  # success, failure, partial
    duration_s: float = 0.0
    sensory_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticFact:
    """A piece of general knowledge."""
    subject: str
    predicate: str
    obj: str
    confidence: float = 1.0
    source: str = "experience"   # experience, instruction, inference
    contradictions: int = 0


@dataclass
class SkillMemory:
    """Parameters for a learned motor skill."""
    skill_name: str
    joint_trajectory: Optional[np.ndarray] = None  # typical trajectory
    success_rate: float = 0.0
    avg_duration: float = 0.0
    avg_force: float = 0.0
    context_requirements: Dict[str, Any] = field(default_factory=dict)
    last_executed: float = 0.0
    execution_count: int = 0
    best_parameters: Dict[str, float] = field(default_factory=dict)


@dataclass
class RecallResult:
    """Result of a memory recall query."""
    memory: Memory
    similarity: float       # how well it matches the query
    relevance: float        # similarity * importance * recency
    time_since_access: float


# ============================================================================
# Working Memory
# ============================================================================

class WorkingMemory:
    """
    Short-term memory buffer for the current task context.
    Limited capacity (~7 items), fast access, automatic decay.
    """

    def __init__(self, capacity: int = 7, decay_time: float = 30.0):
        self.capacity = capacity
        self.decay_time = decay_time
        self._items: deque = deque(maxlen=capacity)
        self._attention_weights: Dict[str, float] = {}

    def push(self, key: str, value: Any, priority: float = 1.0) -> None:
        """Add an item to working memory."""
        self._items.append({
            "key": key,
            "value": value,
            "priority": priority,
            "timestamp": time.time(),
        })
        self._attention_weights[key] = priority

    def get(self, key: str) -> Optional[Any]:
        """Retrieve an item from working memory."""
        for item in reversed(self._items):
            if item["key"] == key:
                item["timestamp"] = time.time()  # refresh
                return item["value"]
        return None

    def get_all(self) -> List[Dict[str, Any]]:
        """Get all current working memory items."""
        now = time.time()
        active = []
        for item in self._items:
            age = now - item["timestamp"]
            if age < self.decay_time:
                item["strength"] = max(0, 1.0 - age / self.decay_time)
                active.append(item)
        return active

    def get_context_vector(self, encoder) -> np.ndarray:
        """Encode current working memory as a single semantic vector."""
        items = self.get_all()
        if not items:
            return np.zeros(2048)

        vectors = []
        weights = []
        for item in items:
            text = f"{item['key']}: {item['value']}"
            vec = encoder.quantum_encode(text)
            vectors.append(vec)
            weights.append(item["strength"] * item["priority"])

        if not vectors:
            return np.zeros(2048)

        weights = np.array(weights)
        weights /= weights.sum() + 1e-10
        combined = sum(w * v for w, v in zip(weights, vectors))
        norm = np.linalg.norm(combined)
        return combined / norm if norm > 0 else combined

    def clear(self) -> None:
        self._items.clear()
        self._attention_weights.clear()

    @property
    def size(self) -> int:
        return len(self._items)


# ============================================================================
# Memory Index (Vector Similarity Search)
# ============================================================================

class MemoryIndex:
    """
    Efficient similarity search over memory vectors.
    Uses a flat index with cosine similarity for simplicity.
    For large memory stores, can be upgraded to FAISS or Annoy.
    """

    def __init__(self, dimension: int = 2048):
        self.dimension = dimension
        self._vectors: List[np.ndarray] = []
        self._memory_ids: List[str] = []
        self._matrix: Optional[np.ndarray] = None  # cached matrix for batch search
        self._dirty = True

    def add(self, memory_id: str, vector: np.ndarray) -> None:
        """Add a vector to the index."""
        norm = np.linalg.norm(vector)
        normalized = vector / norm if norm > 0 else vector
        self._vectors.append(normalized)
        self._memory_ids.append(memory_id)
        self._dirty = True

    def remove(self, memory_id: str) -> None:
        """Remove a vector from the index."""
        try:
            idx = self._memory_ids.index(memory_id)
            self._vectors.pop(idx)
            self._memory_ids.pop(idx)
            self._dirty = True
        except ValueError:
            pass

    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """
        Find the k most similar memories to the query vector.

        Returns:
            List of (memory_id, similarity_score) tuples, sorted by similarity.
        """
        if not self._vectors:
            return []

        # Normalize query
        norm = np.linalg.norm(query_vector)
        query = query_vector / norm if norm > 0 else query_vector

        # Build matrix if dirty
        if self._dirty or self._matrix is None:
            self._matrix = np.array(self._vectors)
            self._dirty = False

        # Cosine similarity via dot product (vectors are pre-normalized)
        similarities = self._matrix @ query
        k = min(k, len(self._memory_ids))
        top_indices = np.argsort(similarities)[-k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append((self._memory_ids[idx], float(similarities[idx])))

        return results

    @property
    def size(self) -> int:
        return len(self._vectors)


# ============================================================================
# Main Memory System
# ============================================================================

class MemorySystem:
    """
    Brian's complete memory system integrating all memory types.

    Provides:
    - Store: Save new memories from experiences
    - Recall: Retrieve relevant memories by semantic similarity
    - Consolidate: Move working memory to episodic, episodic to semantic
    - Forget: Decay unimportant or old memories
    - Persist: Save/load memory to disk for cross-session continuity
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}

        # Memory stores
        self._memories: Dict[str, Memory] = {}
        self._episodic_events: Dict[str, EpisodicEvent] = {}
        self._semantic_facts: Dict[str, SemanticFact] = {}
        self._skill_memories: Dict[str, SkillMemory] = {}

        # Working memory
        self.working_memory = WorkingMemory(
            capacity=config.get("working_memory_capacity", 7),
            decay_time=config.get("working_memory_decay", 30.0),
        )

        # Vector index for similarity search
        self._index = MemoryIndex(dimension=config.get("embedding_dim", 2048))

        # Capacity limits
        self._max_episodic = config.get("max_episodic_memories", 10000)
        self._max_semantic = config.get("max_semantic_memories", 50000)
        self._max_skill = config.get("max_skill_memories", 1000)

        # Persistence
        self._storage_path = Path(config.get("storage_path", "brian_memory"))

        # Stats
        self._total_stored = 0
        self._total_recalled = 0
        self._total_forgotten = 0
        self._consolidation_count = 0

        logger.info(f"MemorySystem initialized | "
                    f"working={self.working_memory.capacity} | "
                    f"episodic_max={self._max_episodic} | "
                    f"semantic_max={self._max_semantic}")

    # === Store ===

    def store_episodic(self, event: EpisodicEvent, semantic_vector: np.ndarray,
                       importance: MemoryImportance = MemoryImportance.MEDIUM,
                       tags: Optional[List[str]] = None) -> str:
        """Store a new episodic memory (a specific event that happened)."""
        memory_id = self._generate_id("ep")
        now = time.time()

        # Determine emotional valence from outcome
        valence = 0.0
        if event.outcome == "success":
            valence = 0.5 + importance.value * 0.1
        elif event.outcome == "failure":
            valence = -0.3 - importance.value * 0.1

        memory = Memory(
            memory_id=memory_id,
            memory_type=MemoryType.EPISODIC,
            content={
                "event_type": event.event_type,
                "location": event.location.tolist() if event.location is not None else None,
                "participants": event.participants,
                "objects": event.objects,
                "outcome": event.outcome,
                "duration_s": event.duration_s,
            },
            semantic_vector=semantic_vector,
            importance=importance,
            timestamp_created=now,
            timestamp_accessed=now,
            emotional_valence=valence,
            tags=tags or [event.event_type],
            decay_rate=0.001 * (5 - importance.value),  # important = slower decay
        )

        self._memories[memory_id] = memory
        self._episodic_events[memory_id] = event
        self._index.add(memory_id, semantic_vector)
        self._total_stored += 1

        # Enforce capacity
        self._enforce_capacity(MemoryType.EPISODIC, self._max_episodic)

        logger.debug(f"Stored episodic memory {memory_id}: {event.event_type} "
                     f"({event.outcome})")
        return memory_id

    def store_semantic(self, fact: SemanticFact, semantic_vector: np.ndarray,
                       tags: Optional[List[str]] = None) -> str:
        """Store a semantic fact (general knowledge)."""
        # Check for existing similar facts
        similar = self._index.search(semantic_vector, k=3)
        for mid, sim in similar:
            if sim > 0.95 and mid in self._semantic_facts:
                # Very similar fact exists, reinforce instead of duplicating
                existing = self._memories[mid]
                existing.strength = min(existing.strength + 0.1, 1.0)
                existing.confidence = min(existing.confidence + 0.05, 1.0)
                existing.access_count += 1
                existing.timestamp_accessed = time.time()
                logger.debug(f"Reinforced existing fact {mid} (strength={existing.strength:.2f})")
                return mid

        memory_id = self._generate_id("sem")
        now = time.time()

        memory = Memory(
            memory_id=memory_id,
            memory_type=MemoryType.SEMANTIC,
            content={
                "subject": fact.subject,
                "predicate": fact.predicate,
                "object": fact.obj,
                "source": fact.source,
            },
            semantic_vector=semantic_vector,
            importance=MemoryImportance.MEDIUM,
            timestamp_created=now,
            timestamp_accessed=now,
            confidence=fact.confidence,
            tags=tags or [fact.subject, fact.predicate],
            decay_rate=0.0001,  # semantic memories decay very slowly
        )

        self._memories[memory_id] = memory
        self._semantic_facts[memory_id] = fact
        self._index.add(memory_id, semantic_vector)
        self._total_stored += 1

        self._enforce_capacity(MemoryType.SEMANTIC, self._max_semantic)
        return memory_id

    def store_skill(self, skill: SkillMemory, semantic_vector: np.ndarray) -> str:
        """Store a motor skill memory."""
        # Check if skill already exists
        for mid, existing_skill in self._skill_memories.items():
            if existing_skill.skill_name == skill.skill_name:
                # Update existing skill
                existing_skill.success_rate = (
                    0.9 * existing_skill.success_rate + 0.1 * skill.success_rate)
                existing_skill.execution_count += skill.execution_count
                existing_skill.last_executed = time.time()
                if skill.success_rate > existing_skill.success_rate:
                    existing_skill.best_parameters = skill.best_parameters
                    existing_skill.joint_trajectory = skill.joint_trajectory
                self._memories[mid].timestamp_accessed = time.time()
                self._memories[mid].access_count += 1
                logger.debug(f"Updated skill memory: {skill.skill_name}")
                return mid

        memory_id = self._generate_id("sk")
        now = time.time()

        memory = Memory(
            memory_id=memory_id,
            memory_type=MemoryType.SKILL,
            content={
                "skill_name": skill.skill_name,
                "success_rate": skill.success_rate,
                "execution_count": skill.execution_count,
                "best_parameters": skill.best_parameters,
            },
            semantic_vector=semantic_vector,
            importance=MemoryImportance.HIGH,
            timestamp_created=now,
            timestamp_accessed=now,
            tags=[skill.skill_name, "motor_skill"],
            decay_rate=0.0,  # skills don't decay (use it or lose it handled separately)
        )

        self._memories[memory_id] = memory
        self._skill_memories[memory_id] = skill
        self._index.add(memory_id, semantic_vector)
        self._total_stored += 1

        self._enforce_capacity(MemoryType.SKILL, self._max_skill)
        return memory_id

    # === Recall ===

    def recall(self, query_vector: np.ndarray, k: int = 5,
               memory_type: Optional[MemoryType] = None,
               min_importance: MemoryImportance = MemoryImportance.TRIVIAL,
               tags: Optional[List[str]] = None) -> List[RecallResult]:
        """
        Recall memories most relevant to the query.

        Relevance = similarity * importance_weight * recency * strength
        """
        self._total_recalled += 1
        candidates = self._index.search(query_vector, k=k * 3)  # over-fetch for filtering
        now = time.time()

        results = []
        for memory_id, similarity in candidates:
            if memory_id not in self._memories:
                continue

            memory = self._memories[memory_id]

            # Filter by type
            if memory_type is not None and memory.memory_type != memory_type:
                continue

            # Filter by importance
            if memory.importance.value < min_importance.value:
                continue

            # Filter by tags
            if tags and not any(t in memory.tags for t in tags):
                continue

            # Compute relevance score
            time_since = now - memory.timestamp_accessed
            recency = math.exp(-time_since / (3600 * 24))  # 24h half-life
            importance_weight = (memory.importance.value + 1) / 5.0
            relevance = similarity * importance_weight * recency * memory.strength

            # Update access stats
            memory.timestamp_accessed = now
            memory.access_count += 1
            # Strengthen recalled memories
            memory.strength = min(memory.strength + 0.02, 1.0)

            results.append(RecallResult(
                memory=memory,
                similarity=similarity,
                relevance=relevance,
                time_since_access=time_since,
            ))

        # Sort by relevance
        results.sort(key=lambda r: r.relevance, reverse=True)
        return results[:k]

    def recall_skill(self, skill_name: str) -> Optional[SkillMemory]:
        """Directly recall a skill by name."""
        for mid, skill in self._skill_memories.items():
            if skill.skill_name == skill_name:
                self._memories[mid].timestamp_accessed = time.time()
                self._memories[mid].access_count += 1
                return skill
        return None

    def recall_recent_episodes(self, n: int = 10) -> List[Tuple[Memory, EpisodicEvent]]:
        """Recall the most recent episodic memories."""
        episodes = [
            (self._memories[mid], event)
            for mid, event in self._episodic_events.items()
            if mid in self._memories
        ]
        episodes.sort(key=lambda x: x[0].timestamp_created, reverse=True)
        return episodes[:n]

    # === Consolidation ===

    def consolidate(self, encoder) -> int:
        """
        Consolidate memories: extract patterns from episodic memories
        and create semantic facts. Similar to sleep-based memory consolidation.

        Returns number of new semantic facts created.
        """
        self._consolidation_count += 1
        new_facts = 0

        # Find clusters of similar episodic memories
        episodes = list(self._episodic_events.items())
        if len(episodes) < 3:
            return 0

        # Group by event type
        type_groups: Dict[str, List[str]] = {}
        for mid, event in episodes:
            if event.event_type not in type_groups:
                type_groups[event.event_type] = []
            type_groups[event.event_type].append(mid)

        for event_type, memory_ids in type_groups.items():
            if len(memory_ids) < 3:
                continue

            # Analyze outcomes for this event type
            outcomes = []
            for mid in memory_ids:
                if mid in self._episodic_events:
                    outcomes.append(self._episodic_events[mid].outcome)

            success_rate = outcomes.count("success") / len(outcomes) if outcomes else 0

            # Create semantic fact about success rates
            fact_text = f"{event_type} has {success_rate:.0%} success rate"
            vector = encoder.quantum_encode(fact_text)

            fact = SemanticFact(
                subject=event_type,
                predicate="has_success_rate",
                obj=f"{success_rate:.2f}",
                confidence=min(len(memory_ids) / 10.0, 1.0),
                source="consolidation",
            )
            self.store_semantic(fact, vector, tags=[event_type, "consolidated"])
            new_facts += 1

            # Extract common objects/context
            all_objects = []
            for mid in memory_ids:
                if mid in self._episodic_events:
                    all_objects.extend(self._episodic_events[mid].objects)

            if all_objects:
                from collections import Counter
                common = Counter(all_objects).most_common(3)
                for obj_name, count in common:
                    if count >= 2:
                        fact_text = f"{event_type} often involves {obj_name}"
                        vector = encoder.quantum_encode(fact_text)
                        fact = SemanticFact(
                            subject=event_type,
                            predicate="often_involves",
                            obj=obj_name,
                            confidence=count / len(memory_ids),
                            source="consolidation",
                        )
                        self.store_semantic(fact, vector)
                        new_facts += 1

        if new_facts > 0:
            logger.info(f"Memory consolidation: created {new_facts} semantic facts "
                        f"from {len(episodes)} episodes")

        return new_facts

    # === Decay & Forgetting ===

    def decay(self, dt: float = 1.0) -> int:
        """
        Apply memory decay. Memories that fall below a threshold are forgotten.

        Args:
            dt: Time elapsed since last decay call (seconds).

        Returns:
            Number of memories forgotten.
        """
        forgotten = 0
        to_remove = []

        for memory_id, memory in self._memories.items():
            # Apply decay
            memory.strength -= memory.decay_rate * dt

            # Boost from importance and recent access
            time_since_access = time.time() - memory.timestamp_accessed
            if time_since_access < 3600:  # accessed in last hour
                memory.strength += 0.001 * dt

            # Forget weak memories
            if memory.strength <= 0.0:
                to_remove.append(memory_id)

        for memory_id in to_remove:
            self._forget(memory_id)
            forgotten += 1

        self._total_forgotten += forgotten
        return forgotten

    def _forget(self, memory_id: str) -> None:
        """Remove a memory completely."""
        self._index.remove(memory_id)
        self._memories.pop(memory_id, None)
        self._episodic_events.pop(memory_id, None)
        self._semantic_facts.pop(memory_id, None)
        self._skill_memories.pop(memory_id, None)

    def _enforce_capacity(self, memory_type: MemoryType, max_count: int) -> None:
        """Remove weakest memories if over capacity."""
        type_memories = [
            (mid, m) for mid, m in self._memories.items()
            if m.memory_type == memory_type
        ]
        if len(type_memories) <= max_count:
            return

        # Sort by strength (weakest first)
        type_memories.sort(key=lambda x: x[1].strength)
        n_remove = len(type_memories) - max_count
        for mid, _ in type_memories[:n_remove]:
            self._forget(mid)

    # === Persistence ===

    def save(self, path: Optional[str] = None) -> None:
        """Save all memories to disk."""
        save_path = Path(path) if path else self._storage_path
        save_path.mkdir(parents=True, exist_ok=True)

        # Save memories
        memories_data = {}
        for mid, memory in self._memories.items():
            memories_data[mid] = {
                "memory_type": memory.memory_type.name,
                "content": memory.content,
                "importance": memory.importance.name,
                "timestamp_created": memory.timestamp_created,
                "timestamp_accessed": memory.timestamp_accessed,
                "access_count": memory.access_count,
                "emotional_valence": memory.emotional_valence,
                "confidence": memory.confidence,
                "tags": memory.tags,
                "decay_rate": memory.decay_rate,
                "strength": memory.strength,
            }

        with open(save_path / "memories.json", "w") as f:
            json.dump(memories_data, f, indent=2)

        # Save vectors separately (binary for efficiency)
        vectors = {}
        for mid, memory in self._memories.items():
            vectors[mid] = memory.semantic_vector
        np.savez_compressed(save_path / "vectors.npz", **vectors)

        # Save skill memories
        skills_data = {}
        for mid, skill in self._skill_memories.items():
            skills_data[mid] = {
                "skill_name": skill.skill_name,
                "success_rate": skill.success_rate,
                "execution_count": skill.execution_count,
                "best_parameters": skill.best_parameters,
                "avg_duration": skill.avg_duration,
                "last_executed": skill.last_executed,
            }
        with open(save_path / "skills.json", "w") as f:
            json.dump(skills_data, f, indent=2)

        logger.info(f"Saved {len(self._memories)} memories to {save_path}")

    def load(self, path: Optional[str] = None) -> int:
        """Load memories from disk. Returns number of memories loaded."""
        load_path = Path(path) if path else self._storage_path
        if not load_path.exists():
            logger.info(f"No saved memories at {load_path}")
            return 0

        count = 0

        # Load vectors
        vectors = {}
        vec_path = load_path / "vectors.npz"
        if vec_path.exists():
            data = np.load(vec_path)
            vectors = {key: data[key] for key in data.files}

        # Load memories
        mem_path = load_path / "memories.json"
        if mem_path.exists():
            with open(mem_path, "r") as f:
                memories_data = json.load(f)

            for mid, mdata in memories_data.items():
                vector = vectors.get(mid, np.zeros(2048))
                memory = Memory(
                    memory_id=mid,
                    memory_type=MemoryType[mdata["memory_type"]],
                    content=mdata["content"],
                    semantic_vector=vector,
                    importance=MemoryImportance[mdata["importance"]],
                    timestamp_created=mdata["timestamp_created"],
                    timestamp_accessed=mdata["timestamp_accessed"],
                    access_count=mdata["access_count"],
                    emotional_valence=mdata.get("emotional_valence", 0),
                    confidence=mdata.get("confidence", 1.0),
                    tags=mdata.get("tags", []),
                    decay_rate=mdata.get("decay_rate", 0.01),
                    strength=mdata.get("strength", 1.0),
                )
                self._memories[mid] = memory
                self._index.add(mid, vector)
                count += 1

        # Load skills
        skills_path = load_path / "skills.json"
        if skills_path.exists():
            with open(skills_path, "r") as f:
                skills_data = json.load(f)
            for mid, sdata in skills_data.items():
                self._skill_memories[mid] = SkillMemory(
                    skill_name=sdata["skill_name"],
                    success_rate=sdata["success_rate"],
                    execution_count=sdata["execution_count"],
                    best_parameters=sdata.get("best_parameters", {}),
                    avg_duration=sdata.get("avg_duration", 0),
                    last_executed=sdata.get("last_executed", 0),
                )

        logger.info(f"Loaded {count} memories from {load_path}")
        return count

    # === Utilities ===

    def _generate_id(self, prefix: str) -> str:
        """Generate a unique memory ID."""
        raw = f"{prefix}:{time.time()}:{self._total_stored}"
        return f"{prefix}_{hashlib.md5(raw.encode()).hexdigest()[:12]}"

    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        type_counts = {}
        for memory in self._memories.values():
            name = memory.memory_type.name
            type_counts[name] = type_counts.get(name, 0) + 1

        avg_strength = (
            float(np.mean([m.strength for m in self._memories.values()]))
            if self._memories else 0.0
        )

        return {
            "total_memories": len(self._memories),
            "type_counts": type_counts,
            "working_memory_items": self.working_memory.size,
            "episodic_events": len(self._episodic_events),
            "semantic_facts": len(self._semantic_facts),
            "skill_memories": len(self._skill_memories),
            "index_size": self._index.size,
            "avg_strength": avg_strength,
            "total_stored": self._total_stored,
            "total_recalled": self._total_recalled,
            "total_forgotten": self._total_forgotten,
            "consolidations": self._consolidation_count,
        }
