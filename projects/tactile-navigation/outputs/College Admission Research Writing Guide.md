# 大學申請研究結論文章：寫作指南

依據 2026 年 10 月 3 日完成的觸覺導航模擬先導研究。這份指南提供研究報告的結構與可引用的結論；不替你編造個人經歷。這是早期寫作指南；現行的[個人角色說明](../../../docs/HOW_I_DIRECTED_AI.md)已補上模型訓練、團隊討論、週報與導航評估中的數據工作。提交時仍須依各校指定格式、字數及 AI 使用要求調整。

## 先決定文章用途

準備兩份不同用途的文字：

- **研究摘要／補充報告**：約 800–1,200 英文字、3–5 頁，含一張地圖／軌跡圖、一張結果表、參考來源及貢獻說明。適合 GitHub 專案或學校允許提交的研究補充材料。
- **個人申請文章中的研究段落**：依題目挑一個具體觀察，描述你如何理解問題、做決定、改變想法。不要把技術報告整篇搬進個人文章。

先閱讀 [Sky Navigation Experiment Report.md](Sky%20Navigation%20Experiment%20Report.md)，觀看影片，再用自己的話解釋結果。只描述你確實理解、審核或執行過的工作。

## 建議題目與研究問題

**英文題目：Evaluating Tactile Humanoid Navigation on New Simulated Maps**

**研究問題：**在不針對各張地圖重新訓練或調整導航設定的情況下，團隊既有的觸覺導航框架能否在不同障礙配置中執行路徑，並符合自己的完成判定？

這個問題與 Zhenyu 的任務一致：先跑通既有地圖、提供 UI 或影片，再設計新地圖探索通用性與穩健性。研究已完成單一 seed 的初步實驗；尚未證明跨 seed 的穩健性。

## 建議章節：照這個順序寫

### 1. Abstract：100–150 字

寫四件事：問題、三種新地圖的設計、相同框架／模型下的模擬方法、最重要的發現與限制。摘要最後一句應指出「物理到達與系統信心判定不同」，避免只寫全部成功。

### 2. Background and Role：100–150 字

解釋觸覺接觸如何協助機器人定位；介紹團隊已提供框架、規劃器、粒子濾波定位、感測策略和訓練好的 G1 控制器。再說明本次工作範圍：新地圖、實驗執行與結果分析。

不要把團隊的演算法或訓練模型寫成你發明的。署名與共同作者需反映實際參與；參考作者姓名和機構用團隊確認的版本。

### 3. Map Design and Method：150–250 字

每張新地圖都要回答「為什麼這樣設計」：

| 地圖 | 設計目的 | 預期能觀察什麼 |
|---|---|---|
| Offset gate | 偏移開口，需要繞路 | 間接路徑與障礙避讓 |
| Staggered baffles | 交錯隔板，需要多次轉向 | 長路徑與轉向中的定位表現 |
| Island choice | 中央實心障礙，可從兩侧繞過 | 路徑選擇與座標轉換 |

方法交代：Windows 電腦上的 WSL Ubuntu、MuJoCo 物理模擬、團隊既有 footstep 控制器、相同導航參數、每張地圖 seed 0，以及保存的狀態、感測和終止紀錄。影片由已執行的機器人狀態重播產生。

模型使用 35500 checkpoint，因下載內容缺少 README 的預設 34000；不可稱為團隊預設實驗的完全重現。原有 room2 的時間上限是 240 秒，新地圖 900 秒，均在上限前結束，但仍需揭露。

### 4. Results：150–200 字，加一張表

直接使用已核對的結果，不重新挑選好看的數字：

| 地圖 | 最終目標距離 | 定位 RMSE | 到達信心 | 正式成功 |
|---|---:|---:|---:|---|
| Original room2 | 7.31 cm | 3.48 cm | 93.35% | 否：arrival_uncertain |
| Island choice | 6.41 cm | 6.57 cm | 92.41% | 否：arrival_uncertain |
| Offset gate | 4.07 cm | 6.64 cm | 91.89% | 否：arrival_uncertain |
| Staggered baffles | 4.16 cm | 6.83 cm | 96.04% | 是：route_complete |

四次都進入各自設定的目標區域，沒有紀錄到跌倒。正式成功還要求框架回報完成；到達判定要求至少 95% 粒子濾波權重同時落在目標範圍及末端朝向界限內。因此前三次物理到達，但仍被判為信心不足。

圖使用 `sky_navigation_trajectories.png`，圖說註明「執行過的物理模擬、seed 0、真實與估計路徑、單位公尺」。提供實際影片與原始紀錄連結。不要把規劃線或這裡的模擬稱為實體機器人測試。

### 5. Discussion and Limitations：150–250 字

主張：本次實驗顯示框架在所測新地圖上能執行到達動作，但成功判定揭露了末端信心不足。這支持進一步研究導航與到達判定的關係。

不能推論：路徑較複雜就一定較差。交錯隔板的正式成功不能證明它比較容易或系統比較穩健；每張只有一次執行，也沒有消融實驗能分離地圖、接觸策略和定位不確定性的影響。95% 是系統內部到達門檻，不是統計信賴區間；也不是「真實世界有 95% 成功率」。

列出限制：單一 seed、模擬而非實體、checkpoint 替代、不同時間上限，以及一次 Windows 系統中斷。中斷紀錄另存，重啟後同 seed 的 offset gate 完成，兩者不可重複計算或當成兩次獨立樣本。電腦當機原因尚未定位到具體驅動程式。

下一步提出可以執行的實驗：相同設定下先測多個 seed，再單獨改變開口寬度或障礙位置；保留失敗、逾時和信心不足，分別比較物理到達、正式成功、定位誤差和接觸次數。先檢查末端粒子分布和接觸紀錄，再決定是否修改框架。

### 6. Conclusion：100–150 字

依序回答：做了什麼、觀察到什麼、證據支持到哪裡、下一步是什麼。下方英文是可核對的技術結論草稿；先用自己的語氣改寫。

> This pilot study evaluated an existing tactile humanoid navigation framework on one original map and three newly authored simulation maps. All four completed trials ended inside the configured goal region without a recorded fall. However, only the staggered-baffle trial satisfied the framework's completion criterion. The other three trials were classified as arrival uncertain because their final belief-based arrival probabilities remained below the 95% threshold. These results demonstrate execution on the tested layouts while highlighting a distinction between physical arrival and sufficiently confident arrival. Because each map was tested with only one seed and an alternative team checkpoint, the findings do not establish general robustness or exactly reproduce the team's default experiments. Future work should repeat trials across seeds and systematically vary map geometry while preserving common settings and reporting all outcomes.

### 7. Contributions, Acknowledgements, and References

分清三種貢獻：

- **團隊**：導航演算法、定位與感測流程、原有地圖、訓練模型、模擬整合。
- **本次擴充**：三張地圖、地圖註冊、幾何驗證、執行紀錄、影片、量化整理與分析。
- **本人工作與 AI 協助**：Sky 主導獲指派的新地圖評估流程，參與團隊討論、數據檢視與報告；Codex 對地圖實作、程式檢查、實驗操作、計算及文字整理提供大量協助。具體歸屬以[現行貢獻說明](../../../CONTRIBUTIONS.md)為準，不把團隊既有演算法或 AI 實作稱為自己獨立完成。

引用團隊 repository 的確切 commit、採用的模型、研究背景來源和你實際閱讀過的文獻。來源基礎為 [HuazeLiu/tactile-navigation](https://github.com/HuazeLiu/tactile-navigation)，branch `sky/evaluation`、base commit `f6ba1df1ed45f6f482eadfe2f84b2182692abd60`；完整資料來源與 SHA-256 保存在 `sky_results.json`。私人 repository 的連結不一定能讓審查者讀取，GitHub 展示應使用你已有授權範圍內的材料。

## 你自己的反思：先回答這五題

1. 看影片前，你認為什麼算「到達成功」？看完信心判定後想法如何改變？
2. 哪張地圖的路徑最讓你意外？指出影片時間或軌跡位置，再解釋你看到什麼。
3. 你如何向不懂機器人的人說明「位置誤差」與「信心不足」不同？
4. 你實際修改、驗證或選擇了什麼？保留一個能指出檔案或紀錄的具體例子。
5. 若只能再做一個實驗，你會改哪一個變因？為什麼？

用回答組成個人文章的材料。不要編造挫折、指導教授對話或「我親手寫了全部程式」；寫出你真的學會的思考方法，比誇大成果更有說服力。

## 完稿操作順序

1. 看完一個完整新地圖影片，再比較四張軌跡。
2. 在紙上說明 95% 門檻、定位 RMSE、物理到達和正式成功的差異。
3. 依上方章節寫初稿；把資料表和圖放入 Results。
4. 加上真實的個人貢獻、AI 使用說明、團隊致謝與引用。
5. 請 Zhenyu 檢查框架描述、團隊署名及公開內容；這是建議的研究審閱步驟。
6. 逐句核對數字、單位和主張，確認每個結果都能連回保存的紀錄。
7. 依各校提交規則輸出 PDF，並在 GitHub README 放摘要、圖、影片與完整報告入口。

## 本地資料入口

- [完整研究報告](Sky%20Navigation%20Experiment%20Report.md)
- [軌跡比較圖](sky_navigation_trajectories.png)
- [指標與來源](sky_results.json)
- [資料驗證說明](Validation%20Report.md)
- [GitHub 展示構想](GitHub%20Navigation%20Portfolio%20Plan.md)
- [原有 room2 影片](../out/sky_evaluation_20261003/original_room2/footstep/room2/seed_0/room2_footstep_pf_global_s0.mp4)
- [中央障礙影片](../out/sky_evaluation_20261003/new_maps_physics/footstep/sky_island_choice/seed_0/sky_island_choice_footstep_pf_global_s0.mp4)
- [偏移開口影片](../out/sky_evaluation_20261003/new_maps_physics_restart/footstep/sky_offset_gate/seed_0/sky_offset_gate_footstep_pf_global_s0.mp4)
- [交錯障礙影片](../out/sky_evaluation_20261003/new_maps_physics_restart/footstep/sky_staggered_baffles/seed_0/sky_staggered_baffles_footstep_pf_global_s0.mp4)

你現在已有研究證據與技術結論；申請文章仍需要你自己的理解和反思。沒有在本次工作中向 GitHub 發布任何材料。
