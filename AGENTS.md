# AGENTS.md

## 1. Think Before Coding

State assumptions explicitly. If unclear, ask. Present alternatives; don't pick silently. Prefer simpler approaches.

## 2. Simplicity First

Minimum code that solves the problem. No speculative features, no single-use abstractions, no unrequested config. If 200 lines could be 50, rewrite.

## 3. Surgical Changes

Touch only what you must. Don't improve adjacent code or refactor unrelated things. Match existing style. Remove only the dead code your changes create.

## 4. Goal-Driven Execution

Define success criteria up front. Turn tasks into verifiable goals: add validation → test invalid inputs; fix bug → reproduce then pass; refactor → tests pass before and after. For multi-step work, list steps with verify checks.

Every changed line should trace directly to the request.