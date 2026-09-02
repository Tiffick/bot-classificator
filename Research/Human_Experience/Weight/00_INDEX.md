# HUMAN EXPERIENCE — WEIGHT

## INDEX / RESEARCH MAP

### STATUS

Research layer + approved semantic Discovery architecture.

Documents `01–11` describe the research model of human experience,
Discovery mechanisms, and boundaries.

Documents `12–17` define the approved semantic architecture of the
Discovery thinking process, memory, and logical data model.

This is not a prompt and not a deterministic dialogue script.
The concrete technical runtime schema has not yet been defined.

### PURPOSE

This folder describes:

- what may be important to understand about a person who comes with a weight-related concern;
- how that understanding may develop through conversation;
- how to distinguish user material, user interpretation, system proposals,
  support, rejection, correction, and uncertainty;
- how accumulated understanding survives across turns;
- how a new reply is reconciled with the system's previous active action;
- when further Discovery stops adding value;
- where the current AI consultant ends and a human consultant begins.

The central object is not “weight” itself and not maximum profile completeness.

The central object is the working picture of this concrete person:
their experience, changes, personal meaning, interpretations, relations,
previous attempts / barriers, and desired change, while preserving
provenance and uncertainty.

---

# I. CORE MODEL

Useful Discovery may develop around semantic objects such as:

**STATED PROBLEM**  
What the person currently says is wrong or wants to change.

**LIFE CHANGE**  
What has actually become different in the person's lived experience.

**BEFORE → NOW**  
Comparison across time that can reveal change without automatically proving causality.

**HYPOTHESES FOR RECOGNITION**  
Context-sensitive possibilities the person may recognize, reject, or correct.
They are not facts.

**ADAPTATION / RATIONALIZATION**  
A possible process by which changed life gradually becomes experienced as normal.
It is investigated only as a hypothesis.

**PERSONAL MEANING**  
Why a particular change matters to this particular person.

**DESIRED CHANGE / RETURN**  
What the person would like to change or regain, where relevant.

**PREVIOUS ATTEMPT / BARRIER**  
What has already been tried and what meaningful constraint should not simply be repeated.

Several elements may form a connected Meaningful Change Picture.
It does not need to be long, linear, dramatic, or complete.

---

# II. DISCOVERY IS NOT A FIXED SEQUENCE

These semantic objects are NOT mandatory dialogue stages.

The user may reveal several of them in one message.
The system should not ask for information that is already known.
It may move forward, backward, or skip an area entirely.

Research examples describe semantic possibilities.

They must not become:

- keyword triggers;
- phrase libraries;
- fixed questionnaires;
- deterministic routing.

The LLM handles meaning and context.

Deterministic code manages state, contracts, validation, boundaries,
and process, but should not enumerate human language.

---

# III. EPISTEMIC MODEL

The system must preserve provenance and current status of material.

Conceptually distinguish:

- USER PROVIDED;
- USER INTERPRETATION;
- SYSTEM PROPOSED;
- OTHER PERSON;
- SUPPORTED;
- REJECTED;
- PARTIALLY SUPPORTED;
- UNCERTAIN;
- CORRECTED.

A system proposal does not become a user fact merely because
the system formulated or repeated it.

A causal interpretation made by the user remains a user interpretation,
not objective causality.

Ambiguity must not be forced into `SUPPORTED` or `REJECTED`.

---

# IV. COGNITIVE CYCLE

Discovery uses one Cognitive Cycle repeated on every turn.

Input:

USER MESSAGE  
+ HUMAN MODEL  
+ ACTIVE CONVERSATION STATE  
+ HUMAN EXPERIENCE

Dialogue History is available as conversation context.

Working sequence:

1. PERCEPTION
2. INTEGRATION + STATE RECONCILIATION
3. FORMULATION
4. DECISION + VALUE / SUFFICIENCY
5. IMPACT
6. EMOTIONAL ADAPTATION
7. RESPONSE + BOUNDARY VALIDATION
8. PERSIST

Stage names do not imply mandatory separate Engines.

---

# V. DISCOVERY MEMORY

Three user-specific entities persist across turns:

**HUMAN MODEL**  
What we understand about this concrete person.

**ACTIVE CONVERSATION STATE**  
Which meanings from the system's last turn the user may now be responding to.

**DIALOGUE HISTORY**  
What was literally said.

**HUMAN EXPERIENCE** exists separately as professional knowledge:
what may happen to people generally and where it may sometimes be useful to look.

Human Experience is not memory about this concrete user
and cannot independently create Human Model facts.

---

# VI. HUMAN MODEL

Human Model is not a questionnaire and not a set of mandatory slots.

The basic logical unit of the future model is a `MODEL ITEM`.

It must be able to express:

- identity;
- content;
- kind;
- provenance;
- status.

Meaningful connections between elements are represented separately as `RELATIONS`.

Human Model must be able to preserve:

- user material;
- user interpretations;
- supported, rejected, and uncertain system proposals;
- corrections and supersession;
- Before → Now;
- meaningful relations between elements.

---

# VII. ACTIVE CONVERSATION STATE

ACS is a semantic bookmark between adjacent turns, not conversation memory.

Minimal contract:

**ACTIVE CONTENT**  
one or more addressable semantic items

+

**RESPONSE TARGETS**  
one or more references to active items / relations

This allows one user reply to independently support, reject, correct,
or leave uncertain different parts of the system's previous response.

ACS is not a backlog of every unresolved question or hypothesis.

---

# VIII. DISCOVERY PROGRESS / SUFFICIENCY

Discovery Progress is not a questionnaire such as:

`problem = true`  
`motivation = true`  
`pain = true`  
`limitations = true`  
`discovery_complete = true`

Progress is expressed by the quality of the Human Model
and the current Formulation.

Decision evaluates Sufficiency dynamically:

**“Will another exploratory move add meaningful understanding?”**

If the answer would merely add another interesting fact,
continuing Discovery may have less value than moving forward.

Discovery completion is not a mandatory separate persistent boolean.

---

# IX. REFLECTION AND TRANSITION

Reflection returns the working picture to the user,
built primarily from their own material.

It may be supported, partially supported, corrected, rejected,
or left uncertain.

Reflection does not become truth merely because the system formulated it.

Transition is an offer of the next step, not automatic handoff.

Meaningful user consent is required except when the user explicitly
requests handoff or asks a question outside the current AI product boundary.

---

# X. NON-NEGOTIABLE BOUNDARIES

The system must not:

- turn hypotheses into facts;
- treat any “yes” as strong confirmation;
- invent unavailable tone or intonation;
- infer objective causality from a user-perceived A/B relation;
- treat every change as a loss;
- treat adaptation as self-deception;
- announce the user's “real motivation”;
- maximize pain;
- use fear of future losses as the primary motivational mechanism;
- optimize hypotheses for obtaining “yes”;
- convert Human Experience into phrase matching;
- ask questions whose answers are already known;
- investigate every possible life domain;
- equate Human Model completeness with Discovery quality;
- continue Discovery by inertia;
- drift into full nutrition / program design;
- mix user material with system proposals;
- make Reflection stronger than the source material;
- hand off without meaningful consent;
- replace understanding with lead conversion as the true goal.

---

# XI. RESEARCH FILE MAP

### `01_STATED_PROBLEM.txt`
The user's stated concern as an entry point into exploration.

### `02_LIFE_CHANGES.txt`
Concrete changes in lived experience supported by the user.

### `03_BEFORE_NOW.txt`
Past-versus-present comparison without automatic causal inference.

### `04_ADAPTATIONS_RATIONALIZATIONS.txt`
Possible normalization of changed life; always a working hypothesis.

### `05_HYPOTHESES_FOR_RECOGNITION.txt`
Context-sensitive possibilities for recognition, rejection, or correction.

### `06_PERSONAL_MEANING.txt`
The personal significance of a supported change.

### `07_CONFIRMED_LOSS_CHAINS.txt`
Connected chains of changes and meanings. `Loss Chain` remains a working term.

### `08_PREVIOUS_ATTEMPTS_BARRIERS.txt`
Previous attempts and barriers within the current product boundary.

### `09_REFLECTION_AWARENESS.txt`
Reflection, verification of understanding, and possible Awareness.

### `10_TRANSITION_TO_HUMAN_CONSULTANT.txt`
Conditions and meaning of transition to a human consultant.

### `11_BOUNDARIES.txt`
Epistemic, conversational, ethical, product, and architectural boundaries.

### `12_Cognitive_Cycle.txt`
Approved semantic model of one repeating Cognitive Cycle:
Perception → Integration → Formulation → Decision → Impact →
Emotional Adaptation → Response → Persist.

### `13_Active_Conversation_State.txt`
Semantic ACS contract. Current minimum:
addressable Active Content + one or more Response Targets.

### `14_Human_Model.txt`
Semantic contract for accumulated working understanding of the concrete person.

### `15_Discovery_Memory.txt`
Responsibility split between Human Model, ACS, Dialogue History,
and Human Experience; dynamic Sufficiency instead of slot-based progress.

### `16_Discovery_Data_Model.txt`
Approved semantic requirements for the future technical Data Model:
Model Items, Relations, provenance, status, corrections, uncertainty,
addressable ACS, and Dialogue History.

### `17_Discovery_Data_Model_Validation.txt`
Adversarial validation of the logical Data Model.
Confirmed that no new persistent memory entity is required.
The ACS refinement discovered by validation is already incorporated into `13` and `16`.

### `TEMP_CORE_MOTIVATIONS.txt`
Historical exploratory material.
Do not treat it as the current architecture specification.

---

# XII. WHAT IS ALREADY SETTLED

At the semantic architecture level:

- there is one repeating Cognitive Cycle;
- Human Model, ACS, Dialogue History, and Human Experience have distinct responsibilities;
- persistent Discovery Memory = Human Model + ACS + Dialogue History;
- Human Experience exists separately;
- Discovery Progress is not a set of mandatory slots;
- Sufficiency is evaluated dynamically inside Decision;
- Human Model must preserve provenance, status, relations,
  rejected / corrected / uncertain material;
- ACS contains addressable Active Content and one or more Response Targets;
- Dialogue History is not duplicated in Human Model;
- the logical Data Model passed adversarial validation;
- the logical model is sufficient to proceed to a concrete technical schema.

---

# XIII. OPEN QUESTIONS / NEXT STAGE

Not yet defined:

- the concrete technical runtime schema: dataclass / Pydantic / JSON;
- concrete fields, types, identifiers, and serialization;
- exact technical representation of `MODEL ITEM`, `RELATION`,
  `ACTIVE CONTENT`, and `RESPONSE TARGETS`;
- retention / cleanup policy for old rejected and superseded elements;
- exact mapping of approved semantic responsibilities
  onto the existing Python Engines;
- how much Human Experience belongs in runtime context
  versus distilled knowledge artifacts;
- exact payload for handoff to a human consultant;
- generalization from Weight to other domains;
- objective quality criteria in Conversation Lab;
- which test results should force revision of current architectural hypotheses;
- whether `Confirmed Loss Chain` is an appropriate production term.

The next stage is to design the concrete technical Data Model
from the approved requirements in `12–17`, then map it
onto the existing project implementation.

---

# ONE-SENTENCE SUMMARY

**Understand the concrete person, preserve the provenance and uncertainty
of that understanding, dynamically choose the next useful move, stop Discovery
when further questions stop adding value, and offer a human consultant as
the next meaningful step rather than the hidden purpose of the conversation.**
