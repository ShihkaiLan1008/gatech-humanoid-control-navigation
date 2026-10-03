# Sky Lan — robotics research retrospective

**Private research archive · team-supported, AI-assisted work · simulation studies**

My interest began at a 2024 Georgia Tech / CIC Asia Future Tech Forum event, where I saw robotics work from professors, university students and high-school students like me. What surprised me was the possibility of using engineering to contribute to the world while still being a student. That experience drew me toward mechanical engineering and robotics.

Watching changes in robot inputs turn into movement made the subject personal: to me, seeing a robot learn to walk felt like a parent watching a child take its first steps. My long-term interest is adaptive robot walking with less dependence on repeated training, transferable movement knowledge and cleaner, more efficient use of energy.

This portfolio brings together preliminary independent G1 locomotion exploration and extensions of the team's tactile robotics work. The central case study asks a concrete question: **can an existing touch-based humanoid navigation framework work on map layouts beyond those originally tested?** The studies below are steps toward those interests; they do not yet demonstrate training-free locomotion, universal transfer or clean-energy gains.

## The project in one minute

Think of a robot walking around obstacles with a map, but an imperfect estimate of where it is. A planned route is only the first step. As it moves, its estimate can drift. Touching a wall provides information that can help correct that estimate. The system must decide both how to move and when it has enough evidence to say it has arrived.

The team supplied the navigation framework and trained walking controller. This extension added three map layouts, checked their geometry, ran the existing system, and collected quantitative results and full replay videos. **Every completed trial reached the physical goal region without a recorded fall, but only one of the four met the system's formal completion criterion.** The others remained below its 95% belief-based arrival threshold. This is the key finding: physical arrival and confident arrival are different engineering goals.

These are single-seed simulation observations. They do not establish a general success rate, a reliable product or real-robot performance.

## What the team built, and what this extension added

| Team foundation | Work in this extension |
|---|---|
| Existing maps and map-loading framework | Three new layouts: offset opening, alternating dividers, central island |
| Global planner and uncertainty-aware route execution | Geometry checks and experiments using shared settings |
| Particle-filter localization and contact-sensing strategy | Analysis of estimated versus true position, contacts and arrival confidence |
| Trained G1 locomotion controller and simulator integration | Four completed physical-simulation records, videos, figures and reproducible reports |

Zhenyu Wu, Huaze Liu and the team retain authorship of their framework and models. Codex assisted with code inspection, map/script implementation, experiment operation, analysis and writing. The portfolio identifies that assistance rather than presenting the framework or all implementation as independently authored by Sky. [Full contribution statement](CONTRIBUTIONS.md).

## Three research threads

| Study | What it investigates | Main evidence and limits |
|---|---|---|
| [Independent G1 rough terrain](projects/independent-g1-rough-terrain/README.md) | How a learned locomotion policy and a custom command can support exploration and return toward an environment origin | Code, saved model_3000, configuration and training log; preliminary exploration, no guaranteed fall prevention or energy-efficiency claim |
| [Tactile controller and foot tracking](projects/tactile-controller/RETROSPECTIVE.md) | Setup, local reward/training experiments and measurement of foot-reference tracking | Reports and screening; local tuned model had higher mean reference error than the reference model, so this is not an improvement claim |
| [New-map tactile navigation](docs/ADMISSIONS_RESEARCH_OVERVIEW.md) | Whether the existing navigation system executes on different geometries and meets its arrival criterion | Three new maps plus an original-map pilot; four recordings, one formal success, one seed per map |

## Why this work matters

Robotics needs both action and evidence: a robot must move, estimate where it is, and know when its estimate is trustworthy enough for the task. Contact-based localization could complement other sensing where visual information is difficult to use. This study does not test darkness, smoke, assistive deployment or search-and-rescue; those are potential application directions rather than demonstrated capabilities.

The work provides a reproducible way to examine an existing system on new layouts and expose a failure mode that a success-only demo would miss. The current research artifact can support map evaluation, replay-based diagnosis and follow-up experiments on uncertainty and sensing.

## Where the research can go next

1. Repeat all layouts across seeds with a common time budget, and retain every outcome.
2. Vary one geometric factor at a time, such as opening width, to identify reproducible limits.
3. Inspect final belief distributions and contact observations to understand uncertain arrival before changing the planner or threshold.
4. Compare sensing choices or additional final probes in a controlled experiment.
5. After repeatable simulation evidence, investigate transfer with the team on real hardware and its appropriate operating procedures.

My personal motivation, the experience that drew me into robotics, and a concrete path toward my longer-term goals are in [Why I care and where I want to go](docs/MY_RESEARCH_DIRECTION.md). This narrative is based on my account of the forum and my response to the robotics work; technical claims remain tied to the saved evidence.

## Read at your preferred depth

- **Admissions overview:** [problem, contribution, meaning and future direction](docs/ADMISSIONS_RESEARCH_OVERVIEW.md).
- **Technical depth:** [architecture, metric definitions and open research questions](docs/TECHNICAL_DEPTH.md).
- **Evidence:** [full report and results](projects/tactile-navigation/outputs/Sky%20Navigation%20Experiment%20Report.md), [trajectory figure](projects/tactile-navigation/outputs/sky_navigation_trajectories.png), [video index](docs/VIDEOS.md).
- **Application writing:** [report/conclusion instructions](projects/tactile-navigation/outputs/College%20Admission%20Research%20Writing%20Guide.md).
- **Trust and provenance:** [contributions](CONTRIBUTIONS.md), [source hashes](SOURCE_MANIFEST.json), [upstream revisions](PROVENANCE.json), [confidentiality](CONFIDENTIALITY.md).

## Archive and retrieval

This personal repository preserves source snapshots, maps, models, research datasets, logs, reports and full execution recordings. Original checkouts remain intact. Generated compiler caches, credentials and virtual environments are excluded. Large artifacts use Git LFS:

```bash
git clone https://github.com/azureowl372/ShihKai-Lan-Research-Project-with-Georgia-Tech-LIDAR-LAB..git research-retrospective
cd research-retrospective
git lfs pull
```

This archive is not one standalone runtime. WSL/controller links and Isaac Lab dependencies are documented by project; some historical scripts retain original local paths. Keep it private. Admissions readers cannot access a private link unless you explicitly grant access or provide an approved excerpt.
