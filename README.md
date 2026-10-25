# RLander: Autonomous Planetary Lander with Gemma 4 Mission Lab

RLander is an end-to-end guidance and landing simulation system for spacecraft descending onto Jupiter's icy moon, Europa. It simulates realistic rocket physics—including thrust vectoring, fuel depletion, and sensor noise—and evaluates a trained Reinforcement Learning (PPO) neural policy against a classical Proportional-Derivative (PD) flight control baseline.

Integrated directly into the system is the **Gemma 4 Mission Test Lab**, which uses Google's **Gemma 4** model via the **Gemini API** as an autonomous flight test engineer to design stress tests, discover controller failure limits, and interpret flight telemetry.

---

### 1. What the Project Does

The objective of RLander is to safely touch down an uncrewed lander on Europa. The lander starts at a high altitude with randomized initial velocities, altitudes, and fuel reserves. To land successfully, it must fire a single throttleable, gimbaled rocket engine to decelerate smoothly.

A landing is judged successful only if it meets strict physical criteria upon ground contact:
- Vertical touchdown speed must be under 1.0 m/s (`|vy| < 1.0 m/s`).
- Horizontal drift speed must be under 0.5 m/s (`|vx| < 0.5 m/s`).
- The vehicle must reach the ground before exhausting propellant or exceeding the mission time limit.

---

### 2. Technologies and Tools Used

- **Physics and Simulation Engine:** Built with Python 3.13 and JAX for fast, hardware-accelerated numerical computing and vectorized physics. It uses a 4th-order Runge-Kutta (RK4) numerical integrator to model continuous planetary dynamics.
- **Reinforcement Learning and Neural Networks:** Developed using Flax for the actor-critic neural network architecture, Optax for gradient optimization, Orbax for safe checkpoint serialization, and Gymnasium for reinforcement learning environment interfaces.
- **Large Language Model (LLM) Integration:** Powered by Google Gemma 4 (`gemma-4-26b-a4b-it`) accessed through the Gemini API. It uses strict JSON schema validation to enforce structured responses without hallucinations.
- **Interactive Dashboard:** Built using Streamlit for a multi-page web application, Plotly for interactive telemetry graphs and trajectory plots, and an HTML5 Canvas for real-time 2D flight animations.
- **Data Processing and Statistics:** Uses Pandas and PyArrow for flight log storage and telemetry analysis, combined with Wilson Score 95% Confidence Intervals for rigorous statistical auditing across 2,000 test seeds.
- **Development and Quality Assurance:** Managed via uv for fast package installation and lockfile integrity, Ruff for code linting, Pytest for unit and integration testing, and GitHub Actions for automated continuous integration.

---

### 3. Gemma 4 Mission Test Lab

In this project, Gemma 4 does not act as a generic chatbot or modify the neural network's weights. Instead, it serves as an **autonomous, hypothesis-driven test engineer**:

- **Natural Language to Scientific Plan:** An engineer enters an everyday question, such as *"How does increasing sideways drift affect landing safety?"* Gemma 4 formulates a formal hypothesis and selects exactly one physical parameter to vary across three probe scenarios around a nominal reference.
- **Strict Physical Criteria and Safety Guardrails:** Gemma must output structured JSON conforming to exact simulation limits (altitude between 100 m and 3,000 m; horizontal speed between -15 m/s and +15 m/s; descent speed between -60 m/s and -1 m/s). If an operator requests something outside the simulation's scope—such as wind or terrain craters—Gemma rejects it explicitly rather than making up answers.
- **Executing the Experiment:** The simulation automatically runs all probe scenarios across matching random seeds for both the learned PPO policy and the classical PD controller.
- **Automated Engineering Diagnosis:** After flights complete, Gemma reviews the actual flight telemetry (touchdown speeds, propellant usage, success/crash labels) and generates a structured summary explaining what failed, what succeeded, and why.
- **Real-World Discovery:** In the bundled demonstration (`app/gemma_demo.json`), Gemma tested horizontal drift at -1 m/s, 5 m/s, and 12 m/s. The test showed that while the classical PD controller landed all flights, the PPO policy crashed at higher drift speeds (outside its training distribution)—identifying a clear operational limit through automated testing.
- **Zero API Key Requirement for Demos:** The app bundles a pre-recorded real Gemma test flight set in `app/gemma_demo.json` that functions immediately without requiring an API key. For live queries, supply `GEMINI_API_KEY`.

---

### 4. How the Model Runs During Simulation

During a live flight simulation, the system executes the following cycle at 10 Hz:

1. **Integrity Check:** The app verifies the SHA-256 hash of the frozen PPO weights against a committed manifest to ensure the model has not been altered.
2. **State Measurement:** The simulator reads the lander's altitude, horizontal position, vertical velocity, horizontal velocity, and remaining propellant mass.
3. **Sensor Noise Injection:** Zero-mean Gaussian noise is added to the raw state to emulate real-world sensor inaccuracies (altimeter and radar error).
4. **Neural Forward Pass:** The noisy observation vector is passed into the frozen PPO actor network implemented in Flax and JAX.
5. **Continuous Action Decision:** The neural network outputs two continuous numbers:
   - Engine Throttle (0.0 to 1.0), determining thrust force.
   - Gimbal Angle (-θ to +θ), tilting the engine to steer horizontally and correct drift.
6. **Physics Update (RK4 Integration):** The simulation computes the lander's acceleration, accounting for Europa's surface gravity (1.315 m/s²) and engine thrust. It calculates the propellant burned using the Tsiolkovsky rocket equation and updates positions and velocities across 4 numerical sub-steps.
7. **Termination:** The loop repeats until the lander touches down on the surface or runs out of fuel, logging the full flight telemetry for immediate replay and graphing.

---

### 5. Benchmark Results & Verification

Training is complete and locked. The release model uses **seed 15 at +80,000 transitions**, selected using 917/1,000 validation landings before the fresh test. Benchmarking was performed across 2,000 identical unseen test seeds (90000–91999) under full-height starts, sensor noise, and physical success rules on Kaggle GPU:

- **Selected V7 PPO Controller:**
  - Landed: 1,809
  - Crashed: 184
  - Timeout: 7
  - Success Rate: **90.45%** (Wilson 95% Confidence Interval: 89.08%–91.66%)
- **Parent PPO Controller:**
  - Landed: 1,810
  - Crashed: 186
  - Timeout: 4
  - Success Rate: 90.50% (Wilson 95% Confidence Interval: 89.14%–91.71%)
- **Classical PD Baseline:**
  - Landed: 1,960
  - Crashed: 40
  - Timeout: 0
  - Success Rate: **98.00%** (Wilson 95% Confidence Interval: 97.29%–98.53%)

The selected PPO controller achieved the target observed score of at least 90% (7.55 percentage points below the PD baseline). Detailed diagnostic logs and timeout analyses are preserved in [PPO Recovery V7 Result](docs/PPO_RECOVERY_V7_RESULT.md).

---

### 6. Run Locally — Python 3.13 Only

From the repository root, with Python 3.13 and `uv` installed:

```sh
uv sync --locked --python 3.13
uv run --locked --python 3.13 python -m streamlit run app/Home.py
```

The multi-page dashboard includes:
- **Mission Overview:** Release results, project architecture, and representative flight examples.
- **Gemma Mission Test Lab:** Natural-language flight experiment designer powered by Gemma 4 with automated telemetry interpretation.
- **Flight Replay:** Telemetry inspection and 2D canvas playback across recorded test flights.
- **Compare Controllers:** Side-by-side performance analysis between PPO, Parent PPO, and PD controllers.
- **Training & Evidence:** Training curves, validation snapshots, and checkpoint hash registry.
- **Live Simulation:** Real-time episode generation, interactive playback controls, and custom initial state inputs.

---

### 7. Live Inference & Custom Starts

Actual live PPO inference requires JAX and the model files in `outputs/final_release/selected/` (`sha256.json`, `manifest.json`, and the `checkpoint/` directory). The loader validates hashes before initializing the policy.

- **Lightweight Mode:** With dependencies from `app/requirements.txt`, recorded replays and evidence run without JAX. An optional Python physics approximation is available with a persistent UI disclaimer.
- **Custom Starts:** Users can enter exact initial physical conditions (horizontal position, altitude, horizontal speed `vx`, and vertical descent speed `vy`). The saved PPO model computes actions in real time directly from these inputs. Custom starts require actual inference to prevent ungrounded approximations.

---

### 8. Verification & Testing

Run linting and the test suite:

```sh
uv run --locked --python 3.13 ruff check .
uv run --locked --python 3.13 python -m pytest -q
```

All continuous integration checks run automatically on GitHub Actions using Python 3.13 with locked dependencies.

---

### 9. Evidence and Documentation Links

- [Release notes](docs/EUROPA_RELEASE.md): Packaged artifacts, verification protocol, and release procedures.
- [Release summary](app/release_data/summary.json): Quantitative metrics, controller selection criteria, and replay hashes.
- [Checkpoint registry](app/release_data/checkpoint_registry.json): Registry of validation-selected run winners.
- [V7 analysis](docs/PPO_RECOVERY_V7_RESULT.md): Test breakdown and timeout root-cause analysis.
