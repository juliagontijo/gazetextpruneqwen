# Forked Qwen2-VL Profiling Log

## Run 1
**Date:** 2026-04-30
**Model:** Qwen/Qwen2-VL-2B-Instruct
**Forked model file:** /Users/juliagontijolopes/Desktop/LLMBuilder/Sara/modeling_qwen2_vl.py
**Device:** MPS | torch 2.11.0
**Frames:** 4
**Prune text:** False
**Prune layers:** [27]
**Prune ratio:** 0.5
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


### Timings
| Stage | Time (ms) | Mem Δ (MB) |
|---|---:|---:|
| model_load | 5954.1 | +2733.2 |
| input_preprocessing | 189.2 | +27.5 |
| vision_encoder | 8882.7 | +13.7 |
| decoder_only | 132238.7 | +0.0 |
| full_run | 141594.3 | +0.1 |

### Outputs
**Decoder-only output**
> The person in the video is interacting with a television screen. They are holding a remote control and appear to be selecting or adjusting something on the screen. The remote control is positioned near the bottom of the screen, and the person's hands are moving towards it. The background shows a room with a television and some other electronic devices. The person seems to be engaged in some form of media or entertainment activity, possibly using the remote control to control the television or another device. The overall scene suggests a casual, everyday setting. The person's hands are in motion, indicating they are actively involved in the task at hand. The television screen is displaying a variety of content, which could be a news feed, a video, or some other form of media.

**Full-run output**
> The person in the video is interacting with a television screen. They are holding a remote control and appear to be selecting or adjusting something on the screen. The remote control is positioned near the bottom of the screen, and the person's hands are moving towards it. The background shows a room with a television and some other electronic devices. The person seems to be engaged in some form of media or entertainment activity, possibly using the remote control to control the television or another device. The overall scene suggests a casual, everyday setting. The person's hands are in motion, indicating they are actively involved in the task at hand. The television screen is displaying a variety of content, which could be a news feed, a video, or some other form of media.

---

## Run 2
**Date:** 2026-04-30
**Model:** Qwen/Qwen2-VL-2B-Instruct
**Forked model file:** /Users/juliagontijolopes/Desktop/LLMBuilder/Sara/modeling_qwen2_vl.py
**Device:** MPS | torch 2.11.0
**Frames:** 4
**Prune text:** True
**Prune layers:** [10]
**Prune ratio:** 0.5
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


### Timings
| Stage | Time (ms) | Mem Δ (MB) |
|---|---:|---:|
| model_load | 6401.5 | +2639.7 |
| input_preprocessing | 181.8 | +27.5 |
| vision_encoder | 8903.1 | +13.7 |
| decoder_only | 104981.5 | +0.0 |
| full_run | 127411.6 | +0.1 |

### Outputs
**Decoder-only output**
> The person in the video is interacting with a television screen and a remote control. They are holding the remote control in their right hand and appear to be pressing a button on the remote. The television screen shows a news article or a video, and the person seems to be either selecting or adjusting something on the screen. The person's left hand is not visible in the frame, but it is likely holding a device or object that is not visible in the video. The overall scene suggests that the person is engaged in some form of media consumption or interaction with the television. The remote control is likely being used to control the television or another device.

**Full-run output**
> The person in the video is interacting with a television screen and a remote control. They are holding the remote control in their right hand and appear to be pressing a button on the remote. The television screen shows a news article or a video, and the person seems to be either selecting or adjusting something on the screen. The person's left hand is not visible in the frame, but it is likely holding a device or object that is not visible in the video. The overall scene suggests that the person is engaged in some form of media consumption or interaction with the television. The remote control is likely being used to control the television or another device.

---

## Run 3
**Date:** 2026-04-30
**Model:** Qwen/Qwen2-VL-2B-Instruct
**Forked model file:** /Users/juliagontijolopes/Desktop/LLMBuilder/Sara/modeling_qwen2_vl.py
**Device:** MPS | torch 2.11.0
**Frames:** 4
**Prune text:** False
**Prune layers:** [10]
**Prune ratio:** 0.5
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


### Timings
| Stage | Time (ms) | Mem Δ (MB) |
|---|---:|---:|
| model_load | 5856.1 | +2733.2 |
| input_preprocessing | 245.9 | +27.5 |
| vision_encoder | 9149.9 | +13.7 |
| decoder_only | 147953.0 | +0.0 |
| full_run | 161851.3 | +0.1 |

### Outputs
**Decoder-only output**
> The person in the video is holding a remote control and a phone. They are likely interacting with the remote control, possibly to navigate or select an option on the TV screen. The phone is being held in a way that suggests it might be used for additional tasks or information, such as checking the time, browsing the internet, or accessing a different application. The person's hands are in motion, indicating they are actively engaged in these tasks. The background shows a room with a TV and some furniture, suggesting a home or office setting. The person's focus appears to be on the remote control and phone, indicating they are likely using them to control the TV or access additional information.

**Full-run output**
> The person in the video is holding a remote control and a phone. They are likely interacting with the remote control, possibly to navigate or select an option on the TV screen. The phone is being held in a way that suggests it might be used for additional tasks or information, such as checking the time, browsing the internet, or accessing a different application. The person's hands are in motion, indicating they are actively engaged in these tasks. The background shows a room with a TV and some furniture, suggesting a home or office setting. The person's focus appears to be on the remote control and phone, indicating they are likely using them to control the TV or access additional information.

---

## Run 4
**Date:** 2026-04-30
**Model:** Qwen/Qwen2-VL-2B-Instruct
**Forked model file:** /Users/juliagontijolopes/Desktop/LLMBuilder/Sara/modeling_qwen2_vl.py
**Device:** MPS | torch 2.11.0
**Frames:** 4
**Prune text:** True
**Prune layers:** [10]
**Prune ratio:** 0.5
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


### Timings
| Stage | Time (ms) | Mem Δ (MB) |
|---|---:|---:|
| model_load | 8428.3 | +4418.0 |
| input_preprocessing | 283.7 | +27.5 |
| vision_encoder | 9811.5 | +13.7 |
| decoder_only | 12615.0 | +0.1 |
| full_run | 33588.0 | +0.4 |

### Outputs
**Decoder-only output**
> The person in the video is interacting with a television screen. They are holding a remote control and appear to be selecting or adjusting something on the screen. The person's hands are visible, and they seem to be focused on the screen. The background shows a room with a TV and some furniture. The person's wrist is also visible, suggesting they might be wearing a watch or bracelet. The overall scene suggests they are using the remote control to control the TV or another device. The person's actions indicate they are engaged in some form of media or entertainment activity. The TV screen shows a blue background with some text and images, but the specific content is not clear from the image alone.

**Full-run output**
> The person in the video is interacting with a television screen. They are holding a remote control and appear to be selecting or adjusting something on the screen. The person's hands are visible, and they seem to be focused on the screen. The background shows a room with a TV and some furniture. The person's wrist is also visible, suggesting they might be wearing a watch or bracelet. The overall scene suggests they are using the remote control to control the TV or another device. The person's actions indicate they are engaged in some form of media or entertainment activity. The TV screen shows a blue background with some text and images, but the specific content is not clear from the image alone.

---

## Run 5
**Date:** 2026-04-30
**Model:** Qwen/Qwen2-VL-2B-Instruct
**Forked model file:** /Users/juliagontijolopes/Desktop/LLMBuilder/Sara/modeling_qwen2_vl.py
**Device:** MPS | torch 2.11.0
**Frames:** 4
**Prune text:** True
**Prune layers:** [10]
**Prune ratio:** 0.5
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


### Timings
| Stage | Time (ms) | Mem Δ (MB) |
|---|---:|---:|
| model_load | 6257.6 | +4418.0 |
| input_preprocessing | 250.8 | +27.5 |
| vision_encoder | 8957.1 | +13.7 |
| decoder_only | 13852.0 | +0.2 |
| full_run | 37594.7 | +0.4 |

### Outputs
**Decoder-only output**
> The person in the video is interacting with a television screen. They are holding a remote control and appear to be adjusting or selecting something on the screen. The person's hands are visible, and they seem to be focused on the screen. The background shows a room with a television and some furniture. The person's wrist is also visible, and they are wearing a watch. The overall scene suggests that the person is engaged in some form of media or entertainment activity, possibly watching a show or movie. The remote control in their hand indicates that they might be controlling the device or making adjustments to the screen. The setting appears to be indoors, likely in a living room or similar space.

**Full-run output**
> The person in the video is interacting with a television screen. They are holding a remote control and appear to be adjusting or selecting something on the screen. The person's hands are visible, and they seem to be focused on the screen. The background shows a room with a television and some furniture. The person's wrist is also visible, and they are wearing a watch. The overall scene suggests that the person is engaged in some form of media or entertainment activity, possibly watching a show or movie. The remote control in their hand indicates that they might be controlling the device or making adjustments to the screen. The setting appears to be indoors, likely in a living room or similar space.

---

## Run 6
**Date:** 2026-04-30
**Model:** Qwen/Qwen2-VL-2B-Instruct
**Forked model file:** /Users/juliagontijolopes/Desktop/LLMBuilder/Sara/modeling_qwen2_vl.py
**Device:** MPS | torch 2.11.0
**Frames:** 4
**Prune text:** False
**Prune layers:** [27]
**Prune ratio:** 0.5
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


### Timings
| Stage | Time (ms) | Mem Δ (MB) |
|---|---:|---:|
| model_load | 8309.1 | +4418.0 |
| input_preprocessing | 288.4 | +27.5 |
| vision_encoder | 9494.8 | +13.7 |
| decoder_only | 15785.9 | +0.4 |
| full_run | 50250.7 | +0.8 |

### Outputs
**Decoder-only output**
> The person in the video is interacting with a television screen. They are holding a remote control and appear to be selecting or adjusting something on the screen. The remote control is positioned near the bottom of the screen, and the person's hands are moving towards it. The background shows a room with a television and some other electronic devices. The person seems to be engaged in some form of media or entertainment activity, possibly using the remote control to control the television or another device. The overall scene suggests a casual, everyday setting. The person's hands are in motion, indicating they are actively involved in the task at hand. The television screen is displaying a variety of content, which could be a news feed, a video, or some other form of media.

**Full-run output**
> The person in the video is interacting with a television screen. They are holding a remote control and appear to be selecting or adjusting something on the screen. The remote control is positioned near the bottom of the screen, and the person's hands are moving towards it. The background shows a room with a television and some other electronic devices. The person seems to be engaged in some form of media or entertainment activity, possibly using the remote control to control the television or another device. The overall scene suggests a casual, everyday setting. The person's hands are in motion, indicating they are actively involved in the task at hand. The television screen is displaying a variety of content, which could be a news feed, a video, or some other form of media.

---

## Run 7
**Date:** 2026-04-30
**Model:** Qwen/Qwen2-VL-2B-Instruct
**Forked model file:** /Users/juliagontijolopes/Desktop/LLMBuilder/Sara/modeling_qwen2_vl.py
**Device:** MPS | torch 2.11.0
**Frames:** 4
**Prune text:** False
**Prune layers:** [10]
**Prune ratio:** 0.5
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


### Timings
| Stage | Time (ms) | Mem Δ (MB) |
|---|---:|---:|
| model_load | 8399.6 | +4418.0 |
| input_preprocessing | 274.1 | +27.5 |
| vision_encoder | 9317.4 | +13.7 |
| decoder_only | 12355.1 | +0.4 |
| full_run | 34213.8 | +0.4 |

### Outputs
**Decoder-only output**
> The person in the video is holding a remote control and a phone. They are likely interacting with the remote control, possibly to navigate or select an option on the TV screen. The phone is being held in a way that suggests it might be used for additional tasks or information, such as checking the time, browsing the internet, or accessing a different application. The person's hands are in motion, indicating they are actively engaged in these tasks. The background shows a room with a TV and some furniture, suggesting a home or office setting. The person's focus appears to be on the remote control and phone, indicating they are likely using them to control the TV or access additional information.

**Full-run output**
> The person in the video is holding a remote control and a phone. They are likely interacting with the remote control, possibly to navigate or select an option on the TV screen. The phone is being held in a way that suggests it might be used for additional tasks or information, such as checking the time, browsing the internet, or accessing a different application. The person's hands are in motion, indicating they are actively engaged in these tasks. The background shows a room with a TV and some furniture, suggesting a home or office setting. The person's focus appears to be on the remote control and phone, indicating they are likely using them to control the TV or access additional information.

---

## Run 8
**Date:** 2026-04-30
**Model:** Qwen/Qwen2-VL-2B-Instruct
**Forked model file:** /Users/juliagontijolopes/Desktop/LLMBuilder/Sara/modeling_qwen2_vl.py
**Device:** MPS | torch 2.11.0
**Frames:** 4
**Prune text:** True
**Prune layers:** [10]
**Prune ratio:** 0.5
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


### Timings
| Stage | Time (ms) | Mem Δ (MB) |
|---|---:|---:|
| model_load | 5641.7 | +4418.0 |
| input_preprocessing | 241.9 | +27.5 |
| vision_encoder | 9041.3 | +13.7 |
| decoder_only | 12393.1 | +0.1 |
| full_run | 34005.5 | +0.4 |

### Outputs
**Decoder-only output**
> The person in the video is interacting with a television screen. They are holding a remote control and appear to be selecting or adjusting something on the screen. The person's hands are visible, and they seem to be focused on the screen. The background shows a room with a TV and some furniture. The person's wrist is also visible, suggesting they might be wearing a watch or bracelet. The overall scene suggests they are using the remote control to control the TV or another device. The person's actions indicate they are engaged in some form of media or entertainment activity. The TV screen shows a blue background with some text and images, but the specific content is not clear from the image alone.

**Full-run output**
> The person in the video is interacting with a television screen. They are holding a remote control and appear to be selecting or adjusting something on the screen. The person's hands are visible, and they seem to be focused on the screen. The background shows a room with a TV and some furniture. The person's wrist is also visible, suggesting they might be wearing a watch or bracelet. The overall scene suggests they are using the remote control to control the TV or another device. The person's actions indicate they are engaged in some form of media or entertainment activity. The TV screen shows a blue background with some text and images, but the specific content is not clear from the image alone.

---

## Run 9
**Date:** 2026-04-30
**Model:** Qwen/Qwen2-VL-2B-Instruct
**Forked model file:** /Users/juliagontijolopes/Desktop/LLMBuilder/Sara/modeling_qwen2_vl.py
**Device:** MPS | torch 2.11.0
**Frames:** 4
**Prune text:** True
**Prune layers:** [10]
**Prune ratio:** 0.5
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


### Timings
| Stage | Time (ms) | Mem Δ (MB) |
|---|---:|---:|
| model_load | 5914.2 | +4418.0 |
| input_preprocessing | 244.1 | +27.5 |
| vision_encoder | 9121.0 | +13.7 |
| decoder_only | 12762.0 | +0.2 |
| full_run | 36667.9 | +0.4 |

### Outputs
**Decoder-only output**
> The person in the video is interacting with a television screen. They are holding a remote control and appear to be adjusting or selecting something on the screen. The person's hands are visible, and they seem to be focused on the screen. The background shows a room with a television and some furniture. The person's wrist is also visible, and they are wearing a watch. The overall scene suggests that the person is engaged in some form of media or entertainment activity, possibly watching a show or movie. The remote control in their hand indicates that they might be controlling the device or making adjustments to the screen. The setting appears to be indoors, likely in a living room or similar space.

**Full-run output**
> The person in the video is interacting with a television screen. They are holding a remote control and appear to be adjusting or selecting something on the screen. The person's hands are visible, and they seem to be focused on the screen. The background shows a room with a television and some furniture. The person's wrist is also visible, and they are wearing a watch. The overall scene suggests that the person is engaged in some form of media or entertainment activity, possibly watching a show or movie. The remote control in their hand indicates that they might be controlling the device or making adjustments to the screen. The setting appears to be indoors, likely in a living room or similar space.

---

## Run 10
**Date:** 2026-04-30
**Model:** Qwen/Qwen2-VL-2B-Instruct
**Forked model file:** /Users/juliagontijolopes/Desktop/LLMBuilder/Sara/modeling_qwen2_vl.py
**Device:** MPS | torch 2.11.0
**Frames:** 4
**Prune text:** False
**Prune gaze:** True
**Prune layers:** [10]
**Prune ratio:** 0.5
**Prune alpha:** 0.5
**Predicted answer:** C ✓ (correct: C)
**Visualisation:** `/Users/juliagontijolopes/Desktop/LLMBuilder/Sara/viz/run10_g_l10_r0.5.png`
**Input shape:** input_ids (1, 1629) | pixel_values (5832, 1176)
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


### Timings
| Stage | Time (ms) | Mem Δ (MB) |
|---|---:|---:|
| model_load | 8301.8 | +4418.0 |
| input_preprocessing | 271.2 | +27.5 |
| vision_encoder | 8850.0 | +13.7 |
| decoder_only | 19911.6 | +0.4 |
| full_run | 47605.2 | +0.5 |

### Outputs
**Decoder-only output**
> To determine why you picked a memory card while organizing the gaming setup, let's analyze the observed changes in your attention:  1. You were initially looking at the TV screen. 2. You then moved your attention to the TV screen again. 3. You picked up a memory card. 4. You placed the memory card on the TV screen.  Given these observations, the most likely reason for picking up the memory card is:  C: I was preparing to insert the memory card into the console for a new game session.  Answer: C. X (where X is the letter of the correct option).   This is because the action of picking up the memory card aligns with the context of preparing to use the console for a new game session. The other options do not fit the observed changes in your attention. The bright light from the TV screen and the accidental action of grabbing a memory card do not directly relate to organizing the gaming setup. The memory card was not hidden, and the action of moving items around was not the primary focus.

**Full-run output**
> To determine why you picked a memory card while organizing the gaming setup, let's analyze the observed changes in your attention:  1. You were initially looking at the TV screen. 2. You then moved your attention to the TV screen again. 3. You picked up a memory card. 4. You placed the memory card on the TV screen.  Given these observations, the most likely reason for picking up the memory card is:  C: I was preparing to insert the memory card into the console for a new game session.  Answer: C. X (where X is the letter of the correct option).   This is because the action of picking up the memory card aligns with the context of preparing to use the console for a new game session. The other options do not fit the observed changes in your attention. The bright light from the TV screen and the accidental action of grabbing a memory card do not directly relate to organizing the gaming setup. The memory card was not hidden, and the action of moving items around was not the primary focus.

---

## Run 11
**Date:** 2026-04-30
**Model:** Qwen/Qwen2-VL-2B-Instruct
**Forked model file:** /Users/juliagontijolopes/Desktop/LLMBuilder/Sara/modeling_qwen2_vl.py
**Device:** MPS | torch 2.11.0
**Frames:** 4
**Prune text:** True
**Prune gaze:** False
**Prune layers:** [10]
**Prune ratio:** 0.5
**Prune alpha:** 0.5
**Predicted answer:** C ✓ (correct: C)
**Visualisation:** `/Users/juliagontijolopes/Desktop/LLMBuilder/Sara/viz/run11_t_l10_r0.5.png`
**Input shape:** input_ids (1, 1629) | pixel_values (5832, 1176)
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


### Timings
| Stage | Time (ms) | Mem Δ (MB) |
|---|---:|---:|
| model_load | 6713.6 | +4418.0 |
| input_preprocessing | 260.1 | +27.5 |
| vision_encoder | 9367.9 | +13.7 |
| decoder_only | 18202.7 | +0.4 |
| full_run | 44490.0 | +0.5 |

### Outputs
**Decoder-only output**
> To determine why you picked a memory card while organizing the gaming setup, let's analyze the observed changes in your attention:  1. You were initially watching a video on the TV. 2. You then picked up a memory card. 3. You placed the memory card on the TV stand. 4. You then organized the gaming setup.  Given these observations, the most likely reason for picking up the memory card is:  C: I was preparing to insert the memory card into the console for a new game session.  This is the most logical explanation for the action you took, as it aligns with the context of organizing the gaming setup and preparing for a new game session. Therefore, the correct answer is:  C X. The letter 'X' represents the correct option. Answer: C X. The letter 'X' represents the correct option. Answer: C X. The letter 'X' represents the correct option. Answer: C X. The letter 'X' represents the correct option. Answer: C

**Full-run output**
> To determine why you picked a memory card while organizing the gaming setup, let's analyze the observed changes in your attention:  1. You were initially watching a video on the TV. 2. You then picked up a memory card. 3. You placed the memory card on the TV stand. 4. You then organized the gaming setup.  Given these observations, the most likely reason for picking up the memory card is:  C: I was preparing to insert the memory card into the console for a new game session.  This is the most logical explanation for the action you took, as it aligns with the context of organizing the gaming setup and preparing for a new game session. Therefore, the correct answer is:  C X. The letter 'X' represents the correct option. Answer: C X. The letter 'X' represents the correct option. Answer: C X. The letter 'X' represents the correct option. Answer: C X. The letter 'X' represents the correct option. Answer: C

---

## Run 12
**Date:** 2026-04-30
**Model:** Qwen/Qwen2-VL-2B-Instruct
**Forked model file:** /Users/juliagontijolopes/Desktop/LLMBuilder/Sara/modeling_qwen2_vl.py
**Device:** MPS | torch 2.11.0
**Frames:** 4
**Prune text:** True
**Prune gaze:** True
**Prune layers:** [10]
**Prune ratio:** 0.5
**Prune alpha:** 0.5
**Predicted answer:** C ✓ (correct: C)
**Visualisation:** `/Users/juliagontijolopes/Desktop/LLMBuilder/Sara/viz/run12_tg_l10_r0.5.png`
**Input shape:** input_ids (1, 1629) | pixel_values (5832, 1176)
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


### Timings
| Stage | Time (ms) | Mem Δ (MB) |
|---|---:|---:|
| model_load | 6806.5 | +4418.0 |
| input_preprocessing | 255.8 | +27.5 |
| vision_encoder | 8944.6 | +13.7 |
| decoder_only | 20513.0 | +0.1 |
| full_run | 46732.6 | +0.4 |

### Outputs
**Decoder-only output**
> To determine why you picked a memory card while organizing the gaming setup, let's analyze the observed changes in your attention:  1. You were initially looking at the TV screen. 2. You then moved to the TV stand. 3. You picked up a memory card. 4. You placed the memory card on the TV stand.  Given these observations, the most likely reason for picking up the memory card is:  C: I was preparing to insert the memory card into the console for a new game session.  Therefore, the correct answer is:  C X. Answer: C X. The memory card was picked up to prepare for a new game session. This is the most logical explanation given the observed changes in your attention. The other options do not fit the context of organizing the gaming setup. The TV screen and the TV stand were not the focus of your attention, and there was no indication of distraction or a need to hide the memory cards. The memory card was picked up to facilitate a new game session, which aligns with the observed actions.

**Full-run output**
> To determine why you picked a memory card while organizing the gaming setup, let's analyze the observed changes in your attention:  1. You were initially looking at the TV screen. 2. You then moved to the TV stand. 3. You picked up a memory card. 4. You placed the memory card on the TV stand.  Given these observations, the most likely reason for picking up the memory card is:  C: I was preparing to insert the memory card into the console for a new game session.  Therefore, the correct answer is:  C X. Answer: C X. The memory card was picked up to prepare for a new game session. This is the most logical explanation given the observed changes in your attention. The other options do not fit the context of organizing the gaming setup. The TV screen and the TV stand were not the focus of your attention, and there was no indication of distraction or a need to hide the memory cards. The memory card was picked up to facilitate a new game session, which aligns with the observed actions.

---
