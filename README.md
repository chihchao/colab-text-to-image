# colab-cli-texttoimage-skill

用 Google 官方 Colab CLI（`google-colab-cli`）在 Colab 的 T4 GPU 上跑 **SDXL-Turbo** 文字生圖，把圖片下載回本機，可以選擇自動去背（輸出透明 PNG），並自動驗證結果。可以當作 Claude Code skill 使用，也能讓其他 AI agent 或程式直接呼叫。

## 檔案
| 檔案 | 用途 |
|---|---|
| `SKILL.md` | Claude Code skill 的定義：完整流程（檢查 → 登入 → 生成 → 驗證 → 關機） |
| `scripts/t2i.py` | 主程式（只用 Python 標準函式庫），子指令有 `doctor` / `login` / `generate` / `stop` |
| `scripts/generate.py` | 在 Colab kernel 裡執行：生圖、去背（BiRefNet）、計算像素統計和 CLIP 吻合分數 |
| `scripts/t2i.sh` | 舊介面的相容腳本：`t2i.sh -p "..."` 等同 `generate`，`t2i.sh --stop-only` 等同 `stop` |

## 需求
- macOS 或 Linux（Colab CLI 不支援 Windows）
- `python3`（3.8 以上），本機不需要另外安裝 Python 套件
- 一個可以使用 Colab 的 Google 帳號；免費方案通常可以用 T4，但可用時數有限制

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
參數：`-n` 張數（1–8）、`-W/-H` 尺寸（預設 512，可設 256–1024，會調整成 8 的倍數；T4 上建議不超過 768）、`--steps` 步數（預設 1，建議 1–4，最多 8）、`-s` session 名稱（預設 `t2i`）、`--seed`、`--negative`、`-o` 輸出資料夾、`--gpu`（預設 T4）、`--stop`（下載完就關機）、`--min-score`（CLIP 分數低於這個值就判定失敗）、`--no-score`（不計算 CLIP 分數）、`--remove-bg`（另外輸出去背的透明 PNG）、`--no-bg-hint`（去背時不自動在 prompt 後面加上背景描述）。

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
- 新 session 第一次生圖：約 2 分鐘（開 T4 VM 約 1 分鐘，加上下載並載入模型）
- 同一個 session 之後的呼叫：每次約 9 秒，其中生圖本身每張 1–3 秒，其餘是連線和下載的時間
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

使用的模型和服務另有各自的授權與條款：SDXL-Turbo 採用 Stability AI 的 [SDXL-Turbo 授權](https://huggingface.co/stabilityai/sdxl-turbo)（商業用途需要另外確認），Colab 則適用 Google Colab 的服務條款。
