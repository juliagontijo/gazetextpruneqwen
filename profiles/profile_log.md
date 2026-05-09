# VLM Profiling Log

## Run 1
**Date:** 2026-04-28
**Model:** Qwen/Qwen2-VL-2B-Instruct
**Device:** MPS | torch 2.11.0
**Frames:** 8
**Input shape:** input_ids (1, 2977) | pixel_values (11664, 1176)

### Component Timing
| Component | Time (ms) | % Total | Notes |
|---|---|---|---|
| vision_encoder | 18733.1 | 20.5% | mem Δ +39.0 MB |
| └─ token_merger | 80.4 | 0.4% of enc | mem Δ +9.1 MB |
| prefill | 5826.3 | 6.4% | mem Δ +137.4 MB |
| decode | 66720.3 | 73.1% | 132 steps, avg 505.5 ms/tok |
| **TOTAL** | **91279.8** | 100% | |

### End-to-end
| Stage | Time (ms) |
|---|---|
| input_preprocessing | 277.7 |
| end_to_end | 93300.7 |

### Model Output (truncated)
> In the video, the person is in a room with a TV displaying a screen with various game options. The person is reaching for a game console, specifically a Nintendo Switch, from a shelf. They are holding a game case in their hands, which they are likely preparing to open and play. The person's hands ar...

---

## Run 2
**Date:** 2026-04-28
**Model:** Qwen/Qwen2-VL-2B-Instruct
**Device:** MPS | torch 2.11.0
**Frames:** 8
**Input shape:** input_ids (1, 2977) | pixel_values (11664, 1176)
**Input logs:**

Sample  file:      ego4d/b70b1c34-46b6-49af-bddb-505dc7738f96/123_1205.mp4
        qa_type:   causal
        question:  Why did I pick a memory card while organizing the gaming setup, given the observed changes in my attention?
        options:   A: I was distracted by the bright light from the television and accidentally grabbed a memory card.
B: I was checking if any specific memory card was missing from the container.
C: I was preparing to insert the memory card into the console for a new game session.
D: I was trying to hide the memory cards from others who might use them.
E: I was simply moving items around without any particular purpose.
        answer:    C


### Component Timing
| Component | Time (ms) | % Total | Notes |
|---|---|---|---|
| vision_encoder | 18856.0 | 25.3% | mem Δ +39.0 MB |
| └─ token_merger | 87.5 | 0.5% of enc | mem Δ +9.1 MB |
| prefill | 5358.1 | 7.2% | mem Δ +210.8 MB |
| decode | 50341.0 | 67.5% | 132 steps, avg 381.4 ms/tok |
| **TOTAL** | **74555.2** | 100% | |

### End-to-end
| Stage | Time (ms) |
|---|---|
| input_preprocessing | 261.0 |
| end_to_end | 76643.2 |

### Model Output
> In the video, the person is in a room with a TV displaying a screen with various game options. The person is reaching for a game console, specifically a Nintendo Switch, from a shelf. They are holding a game case in their hands, which they are likely preparing to open and play. The person's hands are actively engaged in the task of handling the game case and the console, indicating they are about to engage in a gaming session. The person is looking at the TV screen, which shows a selection of games, suggesting they are considering which game to play next. The overall activity appears to be related to gaming, specifically playing a Nintendo Switch game....

---

## Run 3
**Date:** 2026-04-28
**Model:** Qwen/Qwen2-VL-2B-Instruct
**Device:** MPS | torch 2.11.0
**Frames:** 4
**Input shape:** input_ids (1, 1519) | pixel_values (5832, 1176)
**Input logs:**

Sample  file:      ego4d/b70b1c34-46b6-49af-bddb-505dc7738f96/123_1205.mp4
        qa_type:   causal
        question:  Why did I pick a memory card while organizing the gaming setup, given the observed changes in my attention?
        options:   A: I was distracted by the bright light from the television and accidentally grabbed a memory card.
B: I was checking if any specific memory card was missing from the container.
C: I was preparing to insert the memory card into the console for a new game session.
D: I was trying to hide the memory cards from others who might use them.
E: I was simply moving items around without any particular purpose.
        answer:    C


### Component Timing
| Component | Time (ms) | % Total | Notes |
|---|---|---|---|
| vision_encoder | 9199.5 | 8.0% | mem Δ +29.9 MB |
| └─ token_merger | 40.3 | 0.4% of enc | mem Δ +14.9 MB |
| prefill | 2334.4 | 2.0% | mem Δ +48.6 MB |
| decode | 103958.5 | 90.0% | 160 steps, avg 649.7 ms/tok |
| **TOTAL** | **115492.3** | 100% | |

### End-to-end
| Stage | Time (ms) |
|---|---|
| input_preprocessing | 215.9 |
| end_to_end | 118004.9 |

### Model Output
> In the video, the person is sitting in front of a television screen. The screen displays a video game interface with various game options and a news feed. The person is holding a remote control in their right hand and appears to be interacting with the game interface. The person's left hand is resting on a surface, possibly a table or a shelf.  The person is looking at the television screen, likely focusing on the game interface or the news feed. The person seems to be engaged in some form of entertainment or leisure activity, possibly playing a video game or browsing the news.  The objects visible in the video include the television screen, the remote control, and the person's hands. The person is likely in a living room or a similar setting, given the presence of the television and the casual posture....

---

## Run 4
**Date:** 2026-04-28
**Model:** Qwen/Qwen2-VL-2B-Instruct
**Device:** MPS | torch 2.11.0
**Frames:** 4
**Input shape:** input_ids (1, 1519) | pixel_values (5832, 1176)
**Input logs:**

Sample  file:      ego4d/b70b1c34-46b6-49af-bddb-505dc7738f96/123_1205.mp4
        qa_type:   causal
        question:  Why did I pick a memory card while organizing the gaming setup, given the observed changes in my attention?
        options:   A: I was distracted by the bright light from the television and accidentally grabbed a memory card.
B: I was checking if any specific memory card was missing from the container.
C: I was preparing to insert the memory card into the console for a new game session.
D: I was trying to hide the memory cards from others who might use them.
E: I was simply moving items around without any particular purpose.
        answer:    C


### Component Timing
| Component | Time (ms) | % Total | Notes |
|---|---|---|---|
| vision_encoder | 9234.0 | 42.6% | mem Δ +29.9 MB |
| └─ token_merger | 40.5 | 0.4% of enc | mem Δ +14.9 MB |
| prefill | 2181.2 | 10.1% | mem Δ +58.5 MB |
| decode | 10276.8 | 47.4% | 160 steps, avg 64.2 ms/tok |
| **TOTAL** | **21691.9** | 100% | |

### End-to-end
| Stage | Time (ms) |
|---|---|
| input_preprocessing | 828.3 |
| end_to_end | 23199.2 |

### Model Output
> In the video, the person is sitting in front of a television screen. The screen displays a video game interface with various game options and a news feed. The person is holding a remote control in their right hand and appears to be interacting with the game interface. The person's left hand is resting on a surface, possibly a table or a shelf.  The person is looking at the television screen, likely focusing on the game interface or the news feed. The person seems to be engaged in some form of entertainment or leisure activity, possibly playing a video game or browsing the news.  The objects visible in the video include the television screen, the remote control, and the person's hands. The person is likely in a living room or a similar setting, given the presence of the television and the casual posture....

---

## Run 5
**Date:** 2026-04-28
**Model:** Qwen/Qwen2-VL-2B-Instruct
**Device:** MPS | torch 2.11.0
**Frames:** 6
**Input shape:** input_ids (1, 2248) | pixel_values (8748, 1176)
**Input logs:**

Sample  file:      ego4d/b70b1c34-46b6-49af-bddb-505dc7738f96/123_1205.mp4
        qa_type:   causal
        question:  Why did I pick a memory card while organizing the gaming setup, given the observed changes in my attention?
        options:   A: I was distracted by the bright light from the television and accidentally grabbed a memory card.
B: I was checking if any specific memory card was missing from the container.
C: I was preparing to insert the memory card into the console for a new game session.
D: I was trying to hide the memory cards from others who might use them.
E: I was simply moving items around without any particular purpose.
        answer:    C


### Component Timing
| Component | Time (ms) | % Total | Notes |
|---|---|---|---|
| vision_encoder | 14024.6 | 50.0% | mem Δ +29.3 MB |
| └─ token_merger | 60.6 | 0.4% of enc | mem Δ +6.9 MB |
| prefill | 3464.0 | 12.3% | mem Δ +228.6 MB |
| decode | 10584.6 | 37.7% | 142 steps, avg 74.5 ms/tok |
| **TOTAL** | **28073.2** | 100% | |

### End-to-end
| Stage | Time (ms) |
|---|---|
| input_preprocessing | 287.6 |
| end_to_end | 29404.1 |

### Model Output
> In the video, the person is in a car, and the camera is mounted on the dashboard. The screen shows a TV with a video game interface. The person is reaching for a remote control, which they are holding in their right hand. They are looking at the TV screen, which displays a video game interface with various options and a title screen. The person appears to be preparing to play the game. The car's interior is visible, including a shelf with various items, including a bottle of water and a remote control. The person's hand is also visible, and they are wearing a watch on their left wrist. The overall scene suggests that the person is about to engage in a video game session....

---

## Run 6
**Date:** 2026-04-28
**Model:** Qwen/Qwen2-VL-2B-Instruct
**Device:** MPS | torch 2.11.0
**Frames:** 1
**Input shape:** input_ids (1, 790) | pixel_values (2916, 1176)
**Input logs:**

Sample  file:      ego4d/b70b1c34-46b6-49af-bddb-505dc7738f96/123_1205.mp4
        qa_type:   causal
        question:  Why did I pick a memory card while organizing the gaming setup, given the observed changes in my attention?
        options:   A: I was distracted by the bright light from the television and accidentally grabbed a memory card.
B: I was checking if any specific memory card was missing from the container.
C: I was preparing to insert the memory card into the console for a new game session.
D: I was trying to hide the memory cards from others who might use them.
E: I was simply moving items around without any particular purpose.
        answer:    C


### Component Timing
| Component | Time (ms) | % Total | Notes |
|---|---|---|---|
| vision_encoder | 5248.3 | 37.0% | mem Δ +14.9 MB |
| └─ token_merger | 21.7 | 0.4% of enc | mem Δ +7.5 MB |
| prefill | 1094.0 | 7.7% | mem Δ +30.1 MB |
| decode | 7847.2 | 55.3% | 150 steps, avg 52.3 ms/tok |
| **TOTAL** | **14189.5** | 100% | |

### End-to-end
| Stage | Time (ms) |
|---|---|
| input_preprocessing | 975.7 |
| end_to_end | 15628.4 |

### Model Output
> In the video, a person is sitting in front of a television screen. The screen displays a blue interface with various icons and text. The person appears to be engaged in a video game or watching a video, as indicated by the game interface and the presence of a game title on the screen. The person's hands are not visible in the frame, but they seem to be resting on the table or a surface in front of the television.  The person is looking at the television screen, likely focusing on the game or video being played. The activity appears to be leisurely, possibly watching a video or playing a game that requires minimal interaction with the screen. The person's attention is directed towards the screen, suggesting they are enjoying the content being displayed....

---

## Run 7
**Date:** 2026-04-29
**Model:** Qwen/Qwen2-VL-2B-Instruct
**Device:** MPS | torch 2.11.0
**Frames:** 8
**Input shape:** input_ids (1, 283) | pixel_values (888, 1176)
**Input logs:**

Sample  file:      ego4d/b70b1c34-46b6-49af-bddb-505dc7738f96/123_1205.mp4
        qa_type:   causal
        question:  Why did I pick a memory card while organizing the gaming setup, given the observed changes in my attention?
        options:   A: I was distracted by the bright light from the television and accidentally grabbed a memory card.
B: I was checking if any specific memory card was missing from the container.
C: I was preparing to insert the memory card into the console for a new game session.
D: I was trying to hide the memory cards from others who might use them.
E: I was simply moving items around without any particular purpose.
        answer:    C


### Component Timing
| Component | Time (ms) | % Total | Notes |
|---|---|---|---|
| vision_encoder | 18276.4 | 52.9% | mem Δ +39.0 MB |
| └─ token_merger | 82.5 | 0.5% of enc | mem Δ +9.1 MB |
| prefill | 4747.7 | 13.7% | mem Δ +309.7 MB |
| decode | 11518.7 | 33.3% | 132 steps, avg 87.3 ms/tok |
| **TOTAL** | **34542.8** | 100% | |

### End-to-end
| Stage | Time (ms) |
|---|---|
| input_preprocessing | 919.3 |
| end_to_end | 35955.5 |

### Model Output
> Describe in detail what is happening in this egocentric video. What objects are visible? What is the person doing with their hands? Where are they looking and why? What activity are they performing? assistant In the video, the person is in a room with a TV displaying a screen with various game options. The person is reaching for a game console, specifically a Nintendo Switch, from a shelf. They are holding a game case in their hands, which they are likely preparing to open and play. The person's hands are actively engaged in the task of handling the game case and the console, indicating they are about to engage in a gaming session. The person is looking at the TV screen, which shows a selection of games, suggesting they are considering which game to play next. The overall activity appears to be related to gaming, specifically playing a Nintendo Switch game....

---

## Run 8
**Date:** 2026-04-29
**Model:** Qwen/Qwen2-VL-2B-Instruct
**Device:** MPS | torch 2.11.0
**Frames:** 8
**Input shape:** input_ids (1, 2977) | pixel_values (11664, 1176)
**Input logs:**

Sample  file:      ego4d/b70b1c34-46b6-49af-bddb-505dc7738f96/123_1205.mp4
        qa_type:   causal
        question:  Why did I pick a memory card while organizing the gaming setup, given the observed changes in my attention?
        options:   A: I was distracted by the bright light from the television and accidentally grabbed a memory card.
B: I was checking if any specific memory card was missing from the container.
C: I was preparing to insert the memory card into the console for a new game session.
D: I was trying to hide the memory cards from others who might use them.
E: I was simply moving items around without any particular purpose.
        answer:    C


### Component Timing
| Component | Time (ms) | % Total | Notes |
|---|---|---|---|
| vision_encoder | 18767.4 | 24.9% | mem Δ +39.0 MB |
| └─ token_merger | 80.4 | 0.4% of enc | mem Δ +9.1 MB |
| prefill | 5536.0 | 7.3% | mem Δ +210.8 MB |
| decode | 51087.1 | 67.8% | 132 steps, avg 387.0 ms/tok |
| **TOTAL** | **75390.5** | 100% | |

### End-to-end
| Stage | Time (ms) |
|---|---|
| input_preprocessing | 258.6 |
| end_to_end | 77402.6 |

### Model Output
> In the video, the person is in a room with a TV displaying a screen with various game options. The person is reaching for a game console, specifically a Nintendo Switch, from a shelf. They are holding a game case in their hands, which they are likely preparing to open and play. The person's hands are actively engaged in the task of handling the game case and the console, indicating they are about to engage in a gaming session. The person is looking at the TV screen, which shows a selection of games, suggesting they are considering which game to play next. The overall activity appears to be related to gaming, specifically playing a Nintendo Switch game....

---

## Run 9
**Date:** 2026-04-29
**Model:** Qwen/Qwen2-VL-2B-Instruct
**Device:** MPS | torch 2.11.0
**Frames:** 8
**Input shape:** input_ids (1, 2977) | pixel_values (11664, 1176)
**Input logs:**

Sample  file:      ego4d/b70b1c34-46b6-49af-bddb-505dc7738f96/123_1205.mp4
        qa_type:   causal
        question:  Why did I pick a memory card while organizing the gaming setup, given the observed changes in my attention?
        options:   A: I was distracted by the bright light from the television and accidentally grabbed a memory card.
B: I was checking if any specific memory card was missing from the container.
C: I was preparing to insert the memory card into the console for a new game session.
D: I was trying to hide the memory cards from others who might use them.
E: I was simply moving items around without any particular purpose.
        answer:    C


### Component Timing
| Component | Time (ms) | % Total | Notes |
|---|---|---|---|
| vision_encoder | 18995.7 | 54.0% | mem Δ +39.0 MB |
| └─ token_merger | 82.2 | 0.4% of enc | mem Δ +9.1 MB |
| prefill | 4725.6 | 13.4% | mem Δ +309.7 MB |
| decode | 11429.0 | 32.5% | 132 steps, avg 86.6 ms/tok |
| **TOTAL** | **35150.3** | 100% | |

### End-to-end
| Stage | Time (ms) |
|---|---|
| input_preprocessing | 263.9 |
| end_to_end | 36416.5 |

### Model Output
> In the video, the person is in a room with a TV displaying a screen with various game options. The person is reaching for a game console, specifically a Nintendo Switch, from a shelf. They are holding a game case in their hands, which they are likely preparing to open and play. The person's hands are actively engaged in the task of handling the game case and the console, indicating they are about to engage in a gaming session. The person is looking at the TV screen, which shows a selection of games, suggesting they are considering which game to play next. The overall activity appears to be related to gaming, specifically playing a Nintendo Switch game....

---

## Run 10
**Date:** 2026-04-29
**Model:** Qwen/Qwen2-VL-2B-Instruct
**Device:** MPS | torch 2.11.0
**Frames:** 8
**Input shape:** input_ids (1, 2977) | pixel_values (11664, 1176)
**Input logs:**

Sample  file:      ego4d/b70b1c34-46b6-49af-bddb-505dc7738f96/123_1205.mp4
        qa_type:   causal
        question:  Why did I pick a memory card while organizing the gaming setup, given the observed changes in my attention?
        options:   A: I was distracted by the bright light from the television and accidentally grabbed a memory card.
B: I was checking if any specific memory card was missing from the container.
C: I was preparing to insert the memory card into the console for a new game session.
D: I was trying to hide the memory cards from others who might use them.
E: I was simply moving items around without any particular purpose.
        answer:    C


### Component Timing
| Component | Time (ms) | % Total | Notes |
|---|---|---|---|
| vision_encoder | 18922.5 | 18.1% | mem Δ +39.0 MB |
| └─ token_merger | 81.1 | 0.4% of enc | mem Δ +9.2 MB |
| prefill | 5972.1 | 5.7% | mem Δ +146.5 MB |
| decode | 79698.8 | 76.2% | 132 steps, avg 603.8 ms/tok |
| **TOTAL** | **104593.4** | 100% | |

### End-to-end
| Stage | Time (ms) |
|---|---|
| input_preprocessing | 600.2 |
| end_to_end | 106663.4 |

### Model Output
> In the video, the person is in a room with a TV displaying a screen with various game options. The person is reaching for a game console, specifically a Nintendo Switch, from a shelf. They are holding a game case in their hands, which they are likely preparing to open and play. The person's hands are actively engaged in the task of handling the game case and the console, indicating they are about to engage in a gaming session. The person is looking at the TV screen, which shows a selection of games, suggesting they are considering which game to play next. The overall activity appears to be related to gaming, specifically playing a Nintendo Switch game....

---

## Run 11
**Date:** 2026-04-29
**Model:** Qwen/Qwen2-VL-2B-Instruct
**Device:** MPS | torch 2.11.0
**Frames:** 4
**Input shape:** input_ids (1, 790) | pixel_values n/a
**Input logs:**

Sample  file:      ego4d/b70b1c34-46b6-49af-bddb-505dc7738f96/123_1205.mp4
        qa_type:   causal
        question:  Why did I pick a memory card while organizing the gaming setup, given the observed changes in my attention?
        options:   A: I was distracted by the bright light from the television and accidentally grabbed a memory card.
B: I was checking if any specific memory card was missing from the container.
C: I was preparing to insert the memory card into the console for a new game session.
D: I was trying to hide the memory cards from others who might use them.
E: I was simply moving items around without any particular purpose.
        answer:    C


### Component Timing
| Component | Time (ms) | % Total | Notes |
|---|---|---|---|
| vision_encoder | 12150.1 | 47.5% | mem Δ +19.6 MB |
| └─ token_merger | 54.6 | 0.4% of enc | mem Δ +4.7 MB |
| prefill | 1451.0 | 5.7% | mem Δ +27.3 MB |
| decode | 11969.8 | 46.8% | 143 steps, avg 83.7 ms/tok |
| **TOTAL** | **25571.0** | 100% | |

### End-to-end
| Stage | Time (ms) |
|---|---|
| input_preprocessing | 268.3 |
| probe_prefill | 17071.8 |
| end_to_end | 26772.0 |

### Model Output
> In the video, the person is interacting with a television screen. The screen displays a news feed with various articles and images, including a prominent article titled "What's New Week of" and a section labeled "Xenoblade Chronicles." The person is holding a remote control in their right hand, which is positioned near the television. The person's left hand is not visible in the frame.  The person appears to be looking at the television screen, possibly to read the news or select an article. The remote control is likely being used to navigate through the news feed or to access a specific article. The person's focus is on the screen, indicating they are engaged in watching the news or checking the latest updates....

---

## Run 12
**Date:** 2026-04-29
**Model:** Qwen/Qwen2-VL-2B-Instruct
**Device:** MPS | torch 2.11.0
**Frames:** 8
**Input shape:** input_ids (1, 1519) | pixel_values n/a
**Input logs:**

Sample  file:      ego4d/b70b1c34-46b6-49af-bddb-505dc7738f96/123_1205.mp4
        qa_type:   causal
        question:  Why did I pick a memory card while organizing the gaming setup, given the observed changes in my attention?
        options:   A: I was distracted by the bright light from the television and accidentally grabbed a memory card.
B: I was checking if any specific memory card was missing from the container.
C: I was preparing to insert the memory card into the console for a new game session.
D: I was trying to hide the memory cards from others who might use them.
E: I was simply moving items around without any particular purpose.
        answer:    C


### Component Timing
| Component | Time (ms) | % Total | Notes |
|---|---|---|---|
| vision_encoder | 42222.8 | 65.7% | mem Δ +39.0 MB |
| └─ token_merger | 105.0 | 0.2% of enc | mem Δ +9.1 MB |
| prefill | 7530.4 | 11.7% | mem Δ +52.7 MB |
| decode | 14541.0 | 22.6% | 128 steps, avg 113.6 ms/tok |
| **TOTAL** | **64294.2** | 100% | |

### End-to-end
| Stage | Time (ms) |
|---|---|
| input_preprocessing | 274.9 |
| probe_prefill | 31718.8 |
| end_to_end | 65614.4 |

### Model Output
> In the video, the person is in a car, and the screen shows a TV with a video playing. The person is reaching for a game console, specifically a Nintendo Switch, from a shelf. The person is holding a game case, which is likely the console itself. The person is looking at the TV screen, which is displaying a video titled "What's New Week of 11/3." The person appears to be preparing to play a game.               ...

---

## Run 13
**Date:** 2026-04-29
**Model:** Qwen/Qwen2-VL-2B-Instruct
**Device:** MPS | torch 2.11.0
**Frames:** 4
**Input shape:** input_ids (1, 790) | pixel_values n/a
**Input logs:**

Sample  file:      ego4d/b70b1c34-46b6-49af-bddb-505dc7738f96/123_1205.mp4
        qa_type:   causal
        question:  Why did I pick a memory card while organizing the gaming setup, given the observed changes in my attention?
        options:   A: I was distracted by the bright light from the television and accidentally grabbed a memory card.
B: I was checking if any specific memory card was missing from the container.
C: I was preparing to insert the memory card into the console for a new game session.
D: I was trying to hide the memory cards from others who might use them.
E: I was simply moving items around without any particular purpose.
        answer:    C


### Component Timing
| Component | Time (ms) | % Total | Notes |
|---|---|---|---|
| vision_encoder | 12279.2 | 47.2% | mem Δ +19.6 MB |
| └─ token_merger | 179.4 | 1.5% of enc | mem Δ +4.7 MB |
| prefill | 1698.0 | 6.5% | mem Δ +27.3 MB |
| decode | 12030.3 | 46.3% | 143 steps, avg 84.1 ms/tok |
| **TOTAL** | **26007.5** | 100% | |

### End-to-end
| Stage | Time (ms) |
|---|---|
| input_preprocessing | 230.3 |
| probe_prefill | 15465.1 |
| end_to_end | 27264.9 |

### Model Output
> In the video, the person is interacting with a television screen. The screen displays a news feed with various articles and images, including a prominent article titled "What's New Week of" and a section labeled "Xenoblade Chronicles." The person is holding a remote control in their right hand, which is positioned near the television. The person's left hand is not visible in the frame.  The person appears to be looking at the television screen, possibly to read the news or select an article. The remote control is likely being used to navigate through the news feed or to access a specific article. The person's focus is on the screen, indicating they are engaged in watching the news or checking the latest updates....

---

## Run 14
**Date:** 2026-04-29
**Model:** Qwen/Qwen2-VL-2B-Instruct
**Device:** MPS | torch 2.11.0
**Frames:** 4
**Input shape:** input_ids (1, 764) | pixel_values n/a
**Input logs:**

Sample  file:      ego4d/b70b1c34-46b6-49af-bddb-505dc7738f96/123_1205.mp4
        qa_type:   causal
        question:  Why did I pick a memory card while organizing the gaming setup, given the observed changes in my attention?
        options:   A: I was distracted by the bright light from the television and accidentally grabbed a memory card.
B: I was checking if any specific memory card was missing from the container.
C: I was preparing to insert the memory card into the console for a new game session.
D: I was trying to hide the memory cards from others who might use them.
E: I was simply moving items around without any particular purpose.
        answer:    C


### Component Timing
| Component | Time (ms) | % Total | Notes |
|---|---|---|---|
| vision_encoder | 9673.9 | 49.5% | mem Δ +19.5 MB |
| └─ token_merger | 42.8 | 0.4% of enc | mem Δ +4.6 MB |
| prefill | 1070.5 | 5.5% | mem Δ +26.5 MB |
| decode | 8788.2 | 45.0% | 134 steps, avg 65.6 ms/tok |
| **TOTAL** | **19532.5** | 100% | |

### End-to-end
| Stage | Time (ms) |
|---|---|
| input_preprocessing | 241.1 |
| probe_prefill | 12448.9 |
| end_to_end | 20649.8 |

### Model Output
> The person in the video is holding a remote control and appears to be interacting with a television screen. The remote control is being used to navigate the TV interface, possibly to change channels or adjust settings. The person's hands are positioned in a way that suggests they are actively engaged in this task. What is the person doing with their hands?   The person in the video is holding a remote control and appears to be interacting with a television screen. The remote control is being used to navigate the TV interface, possibly to change channels or adjust settings. The person's hands are positioned in a way that suggests they are actively engaged in this task....

---

## Run 15
**Date:** 2026-04-29
**Model:** Qwen/Qwen2-VL-2B-Instruct
**Device:** MPS | torch 2.11.0
**Frames:** 4
**Input shape:** input_ids (1, 1493) | pixel_values (5832, 1176)
**Input logs:**

Sample  file:      ego4d/b70b1c34-46b6-49af-bddb-505dc7738f96/123_1205.mp4
        qa_type:   causal
        question:  Why did I pick a memory card while organizing the gaming setup, given the observed changes in my attention?
        options:   A: I was distracted by the bright light from the television and accidentally grabbed a memory card.
B: I was checking if any specific memory card was missing from the container.
C: I was preparing to insert the memory card into the console for a new game session.
D: I was trying to hide the memory cards from others who might use them.
E: I was simply moving items around without any particular purpose.
        answer:    C


### Component Timing
| Component | Time (ms) | % Total | Notes |
|---|---|---|---|
| vision_encoder | 9122.1 | 45.5% | mem Δ +29.9 MB |
| └─ token_merger | 40.1 | 0.4% of enc | mem Δ +14.9 MB |
| prefill | 2354.4 | 11.7% | mem Δ +57.7 MB |
| decode | 8569.8 | 42.8% | 131 steps, avg 65.4 ms/tok |
| **TOTAL** | **20046.3** | 100% | |

### End-to-end
| Stage | Time (ms) |
|---|---|
| input_preprocessing | 237.9 |
| probe_prefill | 0.0 |
| end_to_end | 21383.2 |

### Model Output
> The person in the video is interacting with a television screen. They are holding a remote control and appear to be selecting or adjusting something on the screen. The person's hands are visible, and they are likely controlling the TV or a device connected to it. The background shows a room with a television and some other electronic devices. The person seems to be engaged in some form of media or entertainment activity. The remote control is being used to navigate through the available options or settings on the screen. The overall scene suggests a casual, everyday activity involving using a television or a similar device. The person's hands are in motion, indicating active engagement with the screen....

---

## Run 16
**Date:** 2026-04-29
**Model:** Qwen/Qwen2-VL-2B-Instruct
**Device:** MPS | torch 2.11.0
**Frames:** 4
**Input shape:** input_ids (1, 1493) | pixel_values (5832, 1176)
**Input logs:**

Sample  file:      ego4d/b70b1c34-46b6-49af-bddb-505dc7738f96/123_1205.mp4
        qa_type:   causal
        question:  Why did I pick a memory card while organizing the gaming setup, given the observed changes in my attention?
        options:   A: I was distracted by the bright light from the television and accidentally grabbed a memory card.
B: I was checking if any specific memory card was missing from the container.
C: I was preparing to insert the memory card into the console for a new game session.
D: I was trying to hide the memory cards from others who might use them.
E: I was simply moving items around without any particular purpose.
        answer:    C


### Component Timing
| Component | Time (ms) | % Total | Notes |
|---|---|---|---|
| vision_encoder | 8882.6 | 38.9% | mem Δ +29.9 MB |
| └─ token_merger | 40.3 | 0.5% of enc | mem Δ +14.9 MB |
| prefill | 2165.4 | 9.5% | mem Δ +57.7 MB |
| decode | 11773.2 | 51.6% | 150 steps, avg 78.5 ms/tok |
| **TOTAL** | **22821.2** | 100% | |

### End-to-end
| Stage | Time (ms) |
|---|---|
| input_preprocessing | 234.7 |
| probe_prefill | 0.0 |
| end_to_end | 24290.0 |

### Model Output
> The person in the video is interacting with a television screen. They are holding a remote control and appear to be selecting or adjusting something on the screen. The remote control is positioned near the bottom of the screen, and the person's hands are moving towards it. The background shows a room with a television and some other electronic devices. The person seems to be engaged in some form of media or entertainment activity, possibly using the remote control to control the television or another device. The overall scene suggests a casual, everyday setting. The person's hands are in motion, indicating they are actively involved in the task at hand. The television screen is displaying a variety of content, which could be a news feed, a video, or some other form of media....

---

## Run 17
**Date:** 2026-04-29
**Model:** Qwen/Qwen2-VL-2B-Instruct
**Device:** MPS | torch 2.11.0
**Frames:** 4
**Input shape:** input_ids (1, 764) | pixel_values n/a
**Input logs:**

Sample  file:      ego4d/b70b1c34-46b6-49af-bddb-505dc7738f96/123_1205.mp4
        qa_type:   causal
        question:  Why did I pick a memory card while organizing the gaming setup, given the observed changes in my attention?
        options:   A: I was distracted by the bright light from the television and accidentally grabbed a memory card.
B: I was checking if any specific memory card was missing from the container.
C: I was preparing to insert the memory card into the console for a new game session.
D: I was trying to hide the memory cards from others who might use them.
E: I was simply moving items around without any particular purpose.
        answer:    C


### Component Timing
| Component | Time (ms) | % Total | Notes |
|---|---|---|---|
| vision_encoder | 9548.1 | 49.5% | mem Δ +19.5 MB |
| └─ token_merger | 40.3 | 0.4% of enc | mem Δ +4.6 MB |
| prefill | 1096.4 | 5.7% | mem Δ +26.5 MB |
| decode | 8637.0 | 44.8% | 134 steps, avg 64.5 ms/tok |
| **TOTAL** | **19281.5** | 100% | |

### End-to-end
| Stage | Time (ms) |
|---|---|
| input_preprocessing | 191.9 |
| probe_prefill | 11705.3 |
| end_to_end | 20322.2 |

### Model Output
> The person in the video is holding a remote control and appears to be interacting with a television screen. The remote control is being used to navigate the TV interface, possibly to change channels or adjust settings. The person's hands are positioned in a way that suggests they are actively engaged in this task. What is the person doing with their hands?   The person in the video is holding a remote control and appears to be interacting with a television screen. The remote control is being used to navigate the TV interface, possibly to change channels or adjust settings. The person's hands are positioned in a way that suggests they are actively engaged in this task....

---
