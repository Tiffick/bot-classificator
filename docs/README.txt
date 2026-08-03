===============================================================================
AI CONSULTANT PROJECT
OFFICIAL DOCUMENTATION
===============================================================================

Version: 3.0

This directory contains the official architecture and development
documentation of the AI Consultant project.

The documentation is organized from philosophy
to implementation.

Each document builds upon the previous ones.

The documentation is the single source of truth for the project.

Historical documents are stored separately in:

    /architecture_history

===============================================================================
DOCUMENT STRUCTURE
===============================================================================

00_PROJECT_MANIFESTO.txt

Why does this project exist?

-------------------------------------------------------------------------------

01_AI_CONSULTANT_VISION.txt

What kind of consultant are we building?

-------------------------------------------------------------------------------

02_AI_CONSULTANT_MIND.txt

How does the consultant think?

-------------------------------------------------------------------------------

03_AI_CONSULTANT_ARCHITECTURE.txt

Overall architecture of the AI Consultant.

Responsibilities of every Engine.

-------------------------------------------------------------------------------

04_CONSULTATIVE_LOOP.txt

Complete processing cycle
of a single user message.

-------------------------------------------------------------------------------

05.0_DATA_CONTRACTS.txt

Overview of all internal Data Contracts.

-------------------------------------------------------------------------------

05.1_SEMANTIC_CONTEXT.txt

SemanticContext specification.

-------------------------------------------------------------------------------

05.2_HUMAN_MODEL.txt

HumanModel specification.

-------------------------------------------------------------------------------

05.3_REASONING_CONTEXT.txt

ReasoningContext specification.

-------------------------------------------------------------------------------

05.4_DECISION_CONTEXT.txt

DecisionContext specification.

-------------------------------------------------------------------------------

05.5_IMPACT_CONTEXT.txt

ImpactContext specification.

-------------------------------------------------------------------------------

05.6_EMOTIONAL_CONTEXT.txt

EmotionalContext specification.

-------------------------------------------------------------------------------

06_ENGINE_ARCHITECTURE.txt

General principles shared by every Engine.

-------------------------------------------------------------------------------

07_SEMANTIC_ENGINE.txt

Semantic Engine specification.

-------------------------------------------------------------------------------

08_HUMAN_MODEL_ENGINE.txt

Human Model Engine specification.

-------------------------------------------------------------------------------

09_REASONING_ENGINE.txt

Reasoning Engine specification.

-------------------------------------------------------------------------------

10_DECISION_ENGINE.txt

Decision Engine specification.

-------------------------------------------------------------------------------

11_IMPACT_ENGINE.txt

Impact Engine specification.

-------------------------------------------------------------------------------

12_EMOTIONAL_ENGINE.txt

Emotional Engine specification.

-------------------------------------------------------------------------------

13_RESPONSE_ENGINE.txt

Response Engine specification.

-------------------------------------------------------------------------------

14_DEVELOPMENT_PROTOCOL.txt

Project development protocol.

Implementation order.

Development rules.

===============================================================================
READING ORDER
===============================================================================

PROJECT MANIFESTO

↓

AI CONSULTANT VISION

↓

AI CONSULTANT MIND

↓

AI CONSULTANT ARCHITECTURE

↓

CONSULTATIVE LOOP

↓

DATA CONTRACTS

↓

ENGINE ARCHITECTURE

↓

ENGINE DOCUMENTS

↓

DEVELOPMENT PROTOCOL

===============================================================================
PROJECT PRINCIPLES
===============================================================================

1.

Architecture defines the system.

-------------------------------------------------------------------------------

2.

Documentation is the single source of truth.

-------------------------------------------------------------------------------

3.

Code implements documentation.

-------------------------------------------------------------------------------

4.

Every Engine has exactly one responsibility.

-------------------------------------------------------------------------------

5.

Every Engine produces exactly one immutable Context.

-------------------------------------------------------------------------------

6.

Engine implementations are developed independently.

-------------------------------------------------------------------------------

7.

Each completed Engine must:

• satisfy its documentation;

• pass tests;

• preserve architectural boundaries.

===============================================================================
CURRENT DEVELOPMENT STAGE
===============================================================================

Architecture Phase

Completed.

Documentation Phase

Completed.

Implementation Phase

In progress.

Current task:

Implementation of Semantic Engine.

===============================================================================