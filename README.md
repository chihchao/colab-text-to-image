# colab-cli-texttoimage-skill

用 Google 官方 Colab CLI（`google-colab-cli`）在 Colab GPU 上文字生圖，可以選 **SDXL-Turbo**（T4，快速又便宜）或 **Z-Image-Turbo**（L4，高品質，能畫中英文字），把圖片下載回本機，可以選擇自動去背（輸出透明 PNG），並自動驗證結果。可以當作 Claude Code skill 使用，也能讓其他 AI agent 或程式直接呼叫。

## 檔案
| 檔案 | 用途 |
|---|---|
| `SKILL.md` | Claude Code skill 的定義：完整流程（檢查 → 登入 → 生成 → 驗證 → 關機） |
| `scripts/t2i.py` | 主程式（只用 Python 標準函式庫），子指令有 `doctor` / `login` / `generate` / `stop` |
| `scripts/generate.py` | 在 Colab kernel 裡執行：生圖（SDXL-Turbo / Z-Image-Turbo）、去背（BiRefNet）、計算像素統計和 CLIP 吻合分數 |
| `scripts/t2i.sh` | 舊介面的相容腳本：`t2i.sh -p "..."` 等同 `generate`，`t2i.sh --stop-only` 等同 `stop` |

## 需求
- macOS 或 Linux（Colab CLI 不支援 Windows）
- `python3`（3.8 以上），本機不需要另外安裝 Python 套件
- 一個可以使用 Colab 的 Google 帳號；免費方案通常可以用 T4，但可用時數有限制
- Z-Image-Turbo 需要 L4，要有 Colab compute units（付費方案，或 Google AI Pro / Ultra 附的額度）

## 第一次使用
```bash
python3 scripts/t2i.py doctor
```
- 沒有 `colab` CLI 時會自動安裝，依序嘗試 uv → pipx → pip --user
- 會自動處理 macOS 上 python.org 版 Python 缺少憑證的問題
- 新裝的 `colab` 如果不在 PATH 上，會提示該把哪個資料夾加進 PATH（t2i.py 本身不受影響）
- 加上 `--no-install` 可以關閉自動安裝
- 如果還沒登入，會列出登入步驟。照著做，或直接執行：
```bash
python3 scripts/t2i.py login
```
打開它顯示的 Google 網址，登入並按「允許」，再把授權碼貼回終端機。

## 生圖
```bash
python3 scripts/t2i.py generate -p "two orange cats sleeping on a blue sofa, photo" -n 2
```
參數：`-m` 模型（`sdxl-turbo` 或 `z-image-turbo`，見下表）、`-n` 張數（1–8）、`-W/-H` 尺寸（會調整成 8 的倍數，預設值和範圍依模型而定）、`--steps` 步數（預設值依模型而定）、`-s` session 名稱（預設 `t2i` 或 `t2i-z`）、`--seed`、`--negative`、`-o` 輸出資料夾、`--gpu`（預設依模型而定）、`--stop`（下載完就關機）、`--min-score`（CLIP 分數低於這個值就判定失敗）、`--no-score`（不計算 CLIP 分數）、`--remove-bg`（另外輸出去背的透明 PNG）、`--no-bg-hint`（去背時不自動在 prompt 後面加上背景描述）。

### 選擇模型
| `-m` | GPU | 預設尺寸 | 尺寸範圍 | 預設步數 | 每張速度（session 已開著時） | 適合 |
|---|---|---|---|---|---|---|
| `sdxl-turbo`（預設） | T4 | 512 | 256–1024 | 1（最多 8） | 約 2 秒 | 草稿、快速多試幾張、省 compute units |
| `z-image-turbo` | L4 | 1024 | 512–1280 | 9（最多 16） | 約 21 秒 | 正式素材、複雜畫面、需要精準照 prompt 畫、**圖裡要有中英文字** |

```bash
python3 scripts/t2i.py generate -m z-image-turbo -p 'a cozy coffee shop, a wooden sign that reads "早安咖啡", warm morning light, photo'
```
- 兩個模型各用一個 session（`t2i` 和 `t2i-z`），切換模型不用重新載入；`stop` 沒指定 `-s` 時兩個都會關
- Z-Image-Turbo（阿里巴巴通義，6B，Apache 2.0，不用登入 Hugging Face）中文和英文 prompt 都能用；要在圖裡畫字，把文字放在引號裡
- Z-Image-Turbo 不支援 `--negative`，不想要的東西改用正面描述
- L4 的顯存（24 GB）放不下全部模型，所以文字編碼器平常放在 CPU，只在處理 prompt 時才搬上 GPU。這樣每張多花約 5 秒，但能同時使用去背和 CLIP 檢查
- L4 暫時開不了（Service Unavailable）時會自動重試 3 次；連到 kernel 的連線中斷時也會自動重試一次
- 實測 L4 每小時約 1.94 compute units

### 去背
```bash
python3 scripts/t2i.py generate -p "a cute cartoon knight game character, full body, centered" --remove-bg
```
- 除了原圖，還會另外輸出 `<檔名>_nobg.png`，背景是透明的。適合遊戲角色、貼圖、圖示這類要疊在其他畫面上的素材
- 使用 [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet)（MIT 授權，可以商用）。模型版本固定在某一次 commit，因為它會從 Hugging Face 下載並執行模型自帶的程式碼
- prompt 裡沒有提到背景時，會自動加上 `isolated on a plain white background`，讓去背更乾淨。不想自動加的話，用 `--no-bg-hint` 關閉
- 第一次使用會多花約 10 秒載入模型，之後每張只多約 1 秒
- 驗證項目：去背檔有成功下載、有透明通道、透明區域佔 5%–97%（超出這個範圍代表背景沒去掉，或主體被一起去掉了），以及圖片邊框上的不透明比例（超過 10% 會警告主體可能被畫面切到）

### 速度
- 新 session 第一次生圖：SDXL-Turbo 約 2 分鐘；Z-Image-Turbo 約 4 分鐘（開 VM，加上下載並載入約 20 GB 的模型）
- 同一個 session 之後的呼叫：SDXL-Turbo 每次約 9 秒（生圖本身每張 1–3 秒，其餘是連線和下載的時間）；Z-Image-Turbo 每張約 21 秒，整個指令約 27 秒
- 圖片預設存到執行指令時所在資料夾的 `./outputs/`

### 自動驗證項目
每張圖都會檢查：檔案是否下載成功、是否為有效的 PNG、尺寸是否和要求相同、是否為空白或全黑，以及 CLIP 計算的 prompt 吻合分數。實測時，吻合的圖約 30–37 分，同一張圖比對不相關的 prompt 只有約 12 分。低於 22 分會發出警告。AI agent 另外會依照 SKILL.md 的規定，親自看過每張圖再回報結果。

## 給 AI agent / 程式呼叫
加上 `--json` 後，stdout 只會輸出一個 JSON 物件，進度訊息都輸出到 stderr，執行過程中也不會停下來等使用者輸入。

| 結束碼 | 意義 |
|---|---|
| 0 | 成功 |
| 2 | 參數錯誤 |
| 3 | 需要登入（JSON 內附 `login_steps`） |
| 4 | 無法安裝 colab CLI |
| 5 | 拿不到 GPU（配額不足） |
| 6 | 生圖或下載失敗 |
| 7 | 驗證沒通過 |
| 8 | 不支援的作業系統 |

```bash
python3 ~/.claude/skills/colab-text-to-image/scripts/t2i.py generate -p "a lighthouse at dusk" --json --stop
```
JSON 的欄位格式請見 SKILL.md。

## 用完記得關機（閒置也會消耗 compute units）
```bash
python3 scripts/t2i.py stop
```

## 安裝成 Claude Code skill
```bash
git clone https://github.com/chihchao/colab-text-to-image.git ~/.claude/skills/colab-text-to-image
```
裝好後執行一次 `python3 ~/.claude/skills/colab-text-to-image/scripts/t2i.py doctor`，完成 CLI 安裝和 Google 登入。

如果已經在其他地方 clone 過，也可以用連結的方式安裝：
```bash
ln -s "$PWD" ~/.claude/skills/colab-text-to-image
```

## 授權
本專案以 [MIT License](LICENSE) 授權。

使用的模型和服務另有各自的授權與條款：SDXL-Turbo 採用 Stability AI 的 [SDXL-Turbo 授權](https://huggingface.co/stabilityai/sdxl-turbo)（商業用途需要另外確認）；[Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) 和 [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) 分別採用 Apache 2.0 和 MIT 授權；Colab 則適用 Google Colab 的服務條款。
