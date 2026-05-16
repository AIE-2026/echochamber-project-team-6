# Ethics and limitations

EchoChamber is a teaching and research prototype for simulating discursive responses with AI agents.

The goal is educational: to understand RAG, role prompts, LangGraph workflows, multi-agent interaction, and the risks of generated political discourse.

EchoChamber must not be presented as a system that measures real public opinion, predicts political behavior, or represents real social groups.

---

# 1. What the agents are

The agents in this project are simulated discursive roles.

They are not real people.

They are not voters.

They are not representatives of parties, communities, demographic groups, or social categories.

Each agent is created from:

- a YAML role prompt;
- retrieved fragments from a corpus;
- an LLM-generated response;
- workflow rules defined in the application.

Correct wording:

> “The anti-system agent generated a simulated anti-system framing.”

Incorrect wording:

> “Anti-system voters think this.”

---

# 2. What the outputs mean

Generated responses are not factual claims.

They are synthetic comments produced by a model under a role constraint.

Even when RAG is used, retrieved context is only supporting material.

Retrieved context does not automatically make the generated answer true.

All outputs must be interpreted critically by students, researchers, or instructors.

---

# 3. Main risks

| Risk | What it means | Minimum control |
|---|---|---|
| Anthropomorphism | Users may treat agents as real people | Always label them as simulated agents |
| False factuality | Generated text may sound like verified information | Separate retrieved context from generated response |
| Corpus bias | The corpus may contain biased or unbalanced content | Document corpus source and limitations |
| Amplification | Multi-agent threads may intensify conflict | Limit number of turns and review outputs |
| Misrepresentation | A role may be confused with a real social group | Use “constructed discursive position” |
| Privacy | Public comments may contain personal data | Avoid unnecessary personal data |
| Political misuse | Outputs could be reused as persuasion material | Use only for education and analysis |

---

# 4. Data protection rules

Use only public or classroom-approved data.

Do not commit API keys, `.env` files, private documents, or personal data to GitHub.

Avoid storing unnecessary identifiers such as usernames, names, or links to personal profiles.

Do not build profiles of real users.

Do not infer sensitive attributes from comments.

---

# 5. Responsible use

EchoChamber may be used to:

- test role prompts;
- compare discursive framings;
- inspect RAG behavior;
- analyze multi-agent escalation;
- support classroom discussion about AI and discourse.

EchoChamber must not be used to:

- generate political persuasion material;
- imitate real citizens or groups;
- publish synthetic comments as public opinion;
- target individuals or communities;
- replace empirical research.

---

# 6. Human oversight

Human interpretation is required.

Students must inspect:

- retrieved context relevance;
- role consistency;
- unsupported claims;
- escalation risks;
- misleading interpretations.

No generated output should be used publicly without review.

---

# 7. Transparency in the application

The application should clearly show:

- which agent generated the response;
- the input prompt;
- the provider/model used;
- whether RAG was used;
- the number of retrieved fragments;
- that the output is synthetic.

Recommended disclaimer:

> EchoChamber generates simulated political-discourse responses. The agents are fictional analytical constructs, not real people and not representatives of real social groups.

---

# 8. How to present results

Recommended wording:

- “The agent generated a simulated framing.”
- “The thread shows possible escalation.”
- “The result requires human interpretation.”

Avoid:

- “This group believes...”
- “The public thinks...”
- “The model proves...”

---

# 9. Minimal logging for reproducibility

Document:

- input text;
- agent slug;
- provider/model;
- retrieval parameter k;
- number of turns;
- routing strategy;
- observed issues.

Example:

```text
input: CCR a decis anularea alegerilor.
agents: anti_suveranist, conspirationist
provider: gemini
k: 3
turns: 4
router: round-robin
observed problem: repetition in later turns