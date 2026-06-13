# llama.cpp OpenAI互換プロキシ

FastAPIで動く、`llama.cpp` の `llama-server` 管理つきOpenAI互換プロキシです。Web UIからGGUFモデルを複数起動し、OpenAI API互換クライアントから `http://localhost:8000/v1/chat/completions` と `http://localhost:8000/v1/embeddings` を利用できます。

## セットアップ

```bash
cp .env.example .env
uv sync
```

必要に応じて `.env` を編集します。
`MODEL_DIR` と、少なくとも1つの `LLAMA_BIN_DIR_VULKAN`、`LLAMA_BIN_DIR_ROCM`、
または互換設定の `LLAMA_BIN_DIR` が必要です。未設定または空の場合、サーバーはエラーを表示して起動を中止します。

```env
LLAMA_BIN_DIR_VULKAN=/home/user/llama.cpp/build-vulkan/bin
LLAMA_BIN_DIR_ROCM=/home/user/llama.cpp/build-hip/bin
DEFAULT_LLAMA_BACKEND=vulkan
MODEL_DIR=/home/user/models
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8080
PROXY_HOST=0.0.0.0
PROXY_PORT=8000
DEFAULT_MAX_TOKENS=512
GRAMMAR_DEFAULT_MAX_TOKENS=256
MODEL_LOAD_TIMEOUT_SECONDS=60
PROXY_API_KEY=
```

`PROXY_API_KEY` を設定した場合は、Web UIの「Proxy API Key」に同じ値を入力してください。APIクライアントも `Authorization: Bearer <key>` が必要になります。

`max_tokens` がリクエストにない場合、proxyが `DEFAULT_MAX_TOKENS` を自動で追加します。`grammar` または `grammar_file` がある場合は `GRAMMAR_DEFAULT_MAX_TOKENS` を使います。

## 起動

```bash
uv run python server.py
```

ブラウザで以下を開き、モデルと起動設定を選んで `Start` を押します。複数モデルを起動すると、`BACKEND_PORT` を開始ポートとして `8080`, `8081`, `8082`... のようにモデルごとの `llama-server` が別ポートで起動します。

`Backend` では設定済みのVulkan版またはROCm版をモデルごとに選択できます。同じモデルは一度に1プロセスだけ起動します。実行中モデルのBackendを変更する場合は `Restart` を押してください。ROCm版では、mmapされたモデルからHIPメモリへのコピーによるロード遅延を避けるため、proxyが `--direct-io` を自動的に追加します。Vulkan版には追加しません。

```text
http://localhost:8000/
```

JSONやgrammar制約つきの出力を `message.content` に返したい場合は、Web UIで以下を選ぶのがおすすめです。

```text
Reasoning: off
Reasoning Format: none
```

Qwen系などのthinking対応モデルでは、reasoningが有効だと生成結果が `message.reasoning_content` に分離され、`message.content` が空になることがあります。

同じLAN内の別端末から使う場合は、UbuntuマシンのLAN IPを使います。

```text
http://<ubuntu-lan-ip>:8000/
```

backendの `llama-server` は `127.0.0.1` の各backendポートにのみ起動し、LAN公開はproxyの `0.0.0.0:8000` 側で行います。

Web UIから起動した `llama-server` のログは、`uv run python server.py` を実行しているターミナルに表示されます。

`/v1/models` の `id` がAPIで指定するモデル名です。既知の未ロードモデルを `model` に指定した場合、proxyが用途に合うモードでロードしてから転送します。`model` が未指定、`"local"`、または存在しない値の場合は、最後に起動した同用途のモデルへ転送します。

モデル用途はGGUF先頭のarchitecture、pooling、embedding次元数から自動判定します。Web UIの `Mode` と `Pooling` でモデルごとに上書きできます。embeddingモデルは `llama-server --embeddings` で起動されるため、同じGGUFをchat用とembedding用に同時ロードすることはありません。実行中モデルの用途を変更した場合は `Restart` してください。

## モデル起動設定

Web UIでモデルを選択すると、`Start` または `Restart` 時に以下の値を設定できます。数値欄を空欄にした場合、そのオプションはproxyから渡さず、使用中の `llama-server` とGGUFモデルの既定値に任せます。入力欄の薄い文字は例または代表的な既定値であり、入力済みの値ではありません。

| UI項目 | 指定できる値 | 説明 | `llama-server` オプション |
| --- | --- | --- | --- |
| `Backend` | `vulkan`, `rocm` | 使用する `llama-server` ビルドを選択します。設定はモデルごとに保存されます。 | 実行ファイルを切替 |
| `Use MMProj` | on / off | モデルと同じディレクトリから検出した `mmproj*.gguf` を使います。画像や音声などのマルチモーダル入力に必要です。対応するMMProjがない場合は選択できません。 | `--mmproj` |
| `Mode` | `auto`, `chat`, `embeddings` | モデルの用途です。`auto` はGGUFのarchitecture、pooling、embedding次元数から判定します。`embeddings` ではembedding専用モードで起動します。 | `--embeddings` |
| `Pooling` | `auto`, `mean`, `cls`, `last` | embeddingベクトルの集約方法です。`auto` はGGUFの設定を使います。chatモードでは使用しません。embeddingモデルに利用可能なpooling情報がない場合は明示指定が必要です。 | `--pooling` |
| `Context` | 整数または空欄 | promptを保持できるコンテキストサイズをtoken数で指定します。空欄ではGGUFから読み込まれる値を使います。値を大きくすると長い入力を扱えますが、KV cacheのメモリ使用量も増えます。 | `--ctx-size` |
| `GPU Layers` | `auto`, `all`, `custom` | VRAMへ配置するモデル層数です。`auto` はbackendに任せ、`all` は可能な限り全層、`custom` は指定した非負整数の層をGPUへ配置します。VRAMを超える指定ではロードに失敗する場合があります。 | `--n-gpu-layers` |
| `Threads` | 整数または空欄 | 生成に使用するCPU thread数です。空欄ではbackendの自動設定を使います。CPU推論やGPUへ配置されない処理の性能に影響します。 | `--threads` |
| `Batch` | 整数または空欄 | prompt処理で一度に扱える論理batchの最大token数です。大きいほどprompt処理が速くなる場合がありますが、メモリ使用量が増えます。 | `--batch-size` |
| `UBatch` | 整数または空欄 | 実際の計算単位となる物理batchの最大token数です。小さくするとピークメモリを抑えられますが、prompt処理が遅くなる場合があります。 | `--ubatch-size` |
| `Parallel` | 整数または空欄 | 同時処理に使うserver slot数です。空欄ではbackendが自動決定します。値を増やすと同時リクエストを処理しやすくなりますが、Contextとメモリをslot間で使用します。 | `--parallel` |
| `Flash Attention` | `auto`, `on`, `off` | Flash Attentionの使用方法です。`auto` はbackendとデバイスの対応状況に任せます。対応環境では速度向上やメモリ削減が期待できます。 | `--flash-attn` |
| `Reasoning` | `off`, `auto`, `on` | chat templateのreasoning/thinking機能を無効化、自動判定、または有効化します。embeddingモードでは通常使用しません。 | `--reasoning` |
| `Reasoning Format` | `none`, `auto`, `deepseek`, `deepseek-legacy` | thinking部分をレスポンスから抽出する形式を指定します。`none` は抽出せず `message.content` に残し、`auto` はtemplateから判定します。`deepseek` 系は対応するthought tagを解析します。 | `--reasoning-format` |

### Contextとembeddingの注意点

`Context` はchatモデルだけでなくembeddingモデルにも適用されます。実際の1リクエストあたりの上限は、指定したContext、モデルの学習時Context、`Parallel` によるslot構成などからbackendが決定します。起動ログの `n_ctx`、`n_ctx_seq`、`new slot, n_ctx = ...` が実際に割り当てられた値です。

embeddingではpooling方式やモデル構造により、入力全体を1回のUBatchで処理する必要があります。この場合はContextが十分でも `UBatch` が入力token数より小さいと処理できません。長文をembeddingする場合は、`Context`、`Batch`、`UBatch` を想定する最大入力token数以上に設定してください。値を大きくするとメモリ消費も増えるため、ロードログと実際の入力長を見ながら調整します。

設定はモデルごとに `.llm-server/model-settings.json` へ保存されます。APIリクエストで既知の未ロードモデルが自動ロードされる場合も、そのモデルの保存済み設定が使われます。実行中のモデルに変更を反映するには `Restart` を押してください。

## OpenAI互換API

通常の非streamingリクエスト:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local",
    "messages": [
      {"role": "user", "content": "短い俳句を書いて"}
    ],
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_tokens": 128,
    "stream": false
  }'
```

streamingリクエスト:

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local",
    "messages": [
      {"role": "user", "content": "ローカルLLMの利点を3つ"}
    ],
    "stream": true
  }'
```

embeddingリクエスト:

```bash
curl http://localhost:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Abiray/harrier-oss-v1-27b-GGUF/harrier-27b-Q4_K_M.gguf",
    "input": ["検索対象の文章", "もう1つの文章"],
    "encoding_format": "float"
  }'
```

OpenAI Pythonクライアント:

```python
from openai import OpenAI

client = OpenAI(api_key="dummy", base_url="http://localhost:8000/v1")
result = client.embeddings.create(
    model="Abiray/harrier-oss-v1-27b-GGUF/harrier-27b-Q4_K_M.gguf",
    input=["検索対象の文章", "もう1つの文章"],
    encoding_format="float",
)
print(len(result.data[0].embedding))
```

`input` は文字列、文字列配列、token ID配列を指定できます。`encoding_format` は `float` または `base64` です。`dimensions` とstreamingは現在のbackendでは対応していないため、proxyが `400 unsupported_parameter` を返します。

embeddingモデルを `/v1/chat/completions` に指定した場合は `400 model_not_chat_capable`、chatモデルを `/v1/embeddings` に指定した場合は `400 model_not_embedding_capable` になります。

`grammar` を直接指定:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local",
    "messages": [
      {"role": "user", "content": "JSONで thought と answer を返して"}
    ],
    "grammar": "root ::= \"ok\"",
    "stream": false
  }'
```

`grammar_file` を指定:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local",
    "messages": [
      {"role": "user", "content": "thought と answer を持つJSONで返して"}
    ],
    "grammar_file": "structured_cot_short.gbnf",
    "max_tokens": 256,
    "stream": false
  }'
```

`grammar_file` は `./grammars/` 配下のみ許可されます。絶対パス、`../`、存在しないファイルは400エラーになります。

`grammar` / `grammar_file` 利用時は、文法上まだ続けられる出力をモデルが繰り返す場合があります。このproxyは `max_tokens` 未指定時に `GRAMMAR_DEFAULT_MAX_TOKENS` を自動設定しますが、用途が短いJSONに限定される場合はリクエスト側で `128` から `256` 程度を明示するのもおすすめです。

## `/v1/chat/completions` リクエスト項目

このproxyは、リクエストJSONをほぼそのままbackendの `llama-server` へ転送します。proxy側で独自に処理するのは `grammar_file` と `max_tokens` の自動補完です。

`max_tokens` が未指定の場合、`grammar` または `grammar_file` があれば `GRAMMAR_DEFAULT_MAX_TOKENS`、なければ `DEFAULT_MAX_TOKENS` が自動で入ります。

### 基本項目

| 項目 | 説明 | 例 |
| --- | --- | --- |
| `model` | `/v1/models` の `id` です。既知の未ロードモデルなら自動ロードされます。未指定、`"local"`、存在しない値は最後に起動したモデルへフォールバックします。 | `"model": "Qwen/Qwen3.gguf"` |
| `messages` | 必須。会話履歴です。各要素に `role` と `content` を指定します。 | `"messages": [{"role": "user", "content": "こんにちは"}]` |
| `stream` | `true` ならServer-Sent Eventsで逐次返します。 | `"stream": true` |
| `stream_options` | streaming時の追加オプションです。`include_usage` を使うと最後にusage情報を含めます。 | `"stream_options": {"include_usage": true}` |
| `max_tokens` | 最大生成トークン数です。未指定時はproxyが自動設定します。 | `"max_tokens": 256` |
| `max_completion_tokens` | `max_tokens` 相当のOpenAI互換名です。 | `"max_completion_tokens": 256` |
| `n_predict` | llama.cpp名の最大生成トークン数です。指定すると `max_tokens` より優先されます。 | `"n_predict": 256` |
| `n` | 生成候補数です。通常は `1` で使います。 | `"n": 1` |
| `stop` | 生成を止める文字列です。文字列または配列を指定できます。 | `"stop": ["\nUser:", "</s>"]` |
| `seed` | 乱数シードです。再現性を上げたい場合に指定します。 | `"seed": 1234` |

基本例:

```json
{
  "model": "local",
  "messages": [
    {"role": "system", "content": "簡潔に答えてください。"},
    {"role": "user", "content": "ローカルLLMの利点を3つ"}
  ],
  "temperature": 0.7,
  "top_p": 0.95,
  "max_tokens": 128,
  "stream": false
}
```

### `messages` の形式

| 項目 | 説明 | 例 |
| --- | --- | --- |
| `role` | 発話者です。主に `system`、`user`、`assistant`、`tool` を使います。 | `"role": "user"` |
| `content` | メッセージ本文です。文字列、`null`、またはcontent part配列を指定できます。 | `"content": "俳句を書いて"` |
| `tool_calls` | assistant側のツール呼び出し履歴です。tool callingを使う場合に指定します。 | `"tool_calls": [...]` |
| `reasoning_content` | reasoning対応モデルの思考履歴を保持したい場合に使います。テンプレート側の対応が必要です。 | `"reasoning_content": "..."` |

content part配列の例:

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "この画像を説明して"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
  ]
}
```

音声入力を使う場合:

```json
{
  "role": "user",
  "content": [
    {
      "type": "input_audio",
      "input_audio": {
        "data": "...base64...",
        "format": "wav"
      }
    }
  ]
}
```

画像や音声は、backend起動時に対応モデルや `mmproj` などが必要です。

### サンプリング項目

| 項目 | 説明 | 例 |
| --- | --- | --- |
| `temperature` | 出力のランダムさです。低いほど決定的、高いほど多様になります。 | `"temperature": 0.7` |
| `top_p` | nucleus samplingのしきい値です。候補確率の累積がこの値までのtokenから選びます。 | `"top_p": 0.95` |
| `top_k` | 上位K個のtokenだけを候補にします。`0` で無効です。 | `"top_k": 40` |
| `min_p` | 最高確率tokenに対する相対しきい値です。低確率tokenを落とします。 | `"min_p": 0.05` |
| `top_n_sigma` | 平均から指定sigma以内のtokenを使うサンプリングです。負値で無効です。 | `"top_n_sigma": -1` |
| `typical_p` | locally typical samplingです。`1.0` でほぼ無効です。 | `"typical_p": 1.0` |
| `xtc_probability` | XTC samplingを使う確率です。`0.0` で無効です。 | `"xtc_probability": 0.0` |
| `xtc_threshold` | XTC samplingのしきい値です。 | `"xtc_threshold": 0.1` |
| `dynatemp_range` | dynamic temperatureの変動幅です。`0.0` で無効です。 | `"dynatemp_range": 0.0` |
| `dynatemp_exponent` | dynamic temperatureの変化カーブです。 | `"dynatemp_exponent": 1.0` |
| `repeat_last_n` | 繰り返しペナルティを見る直近token数です。`-1` でcontext全体です。 | `"repeat_last_n": 64` |
| `repeat_penalty` | 繰り返しtokenへのペナルティです。`1.0` で無効です。 | `"repeat_penalty": 1.1` |
| `frequency_penalty` | 出現頻度に応じたペナルティです。 | `"frequency_penalty": 0.0` |
| `presence_penalty` | 既出tokenへのペナルティです。 | `"presence_penalty": 0.0` |
| `dry_multiplier` | DRY repetition penaltyの強さです。`0.0` で無効です。 | `"dry_multiplier": 0.8` |
| `dry_base` | DRY penaltyのベース値です。 | `"dry_base": 1.75` |
| `dry_allowed_length` | DRYで許容する繰り返し長です。 | `"dry_allowed_length": 2` |
| `dry_penalty_last_n` | DRY penaltyを見る直近token数です。 | `"dry_penalty_last_n": -1` |
| `dry_sequence_breakers` | DRYの繰り返し判定を切る文字列です。 | `"dry_sequence_breakers": ["\n", ":", "\""]` |
| `mirostat` | Mirostat samplingです。`0` 無効、`1` Mirostat、`2` Mirostat 2.0です。 | `"mirostat": 0` |
| `mirostat_tau` | Mirostatの目標entropyです。 | `"mirostat_tau": 5.0` |
| `mirostat_eta` | Mirostatの学習率です。 | `"mirostat_eta": 0.1` |
| `adaptive_target` | adaptive-pの目標確率です。負値で無効です。 | `"adaptive_target": -1` |
| `adaptive_decay` | adaptive-pの減衰率です。 | `"adaptive_decay": 0.9` |
| `samplers` | samplerの順序を指定します。配列または短縮文字列を指定できます。 | `"samplers": ["top_k", "top_p", "temperature"]` |
| `min_keep` | sampler後に最低限残す候補token数です。 | `"min_keep": 1` |
| `ignore_eos` | EOSを無視して生成を続けます。 | `"ignore_eos": false` |
| `logit_bias` | token IDや文字列にbiasをかけます。falseは生成禁止扱いです。 | `"logit_bias": {"</think>": -5}` |

creative寄りの例:

```json
{
  "messages": [{"role": "user", "content": "短い物語を書いて"}],
  "temperature": 0.9,
  "top_p": 0.95,
  "top_k": 50,
  "repeat_penalty": 1.05,
  "max_tokens": 300
}
```

安定寄りの例:

```json
{
  "messages": [{"role": "user", "content": "要点だけ箇条書きで説明して"}],
  "temperature": 0.2,
  "top_p": 0.8,
  "top_k": 20,
  "max_tokens": 160
}
```

### logprobs

| 項目 | 説明 | 例 |
| --- | --- | --- |
| `logprobs` | OpenAI互換のlogprobs要求です。`true` にするとtoken確率情報を要求します。 | `"logprobs": true` |
| `top_logprobs` | `logprobs: true` のとき、上位何件を返すかです。 | `"top_logprobs": 5` |
| `n_probs` | llama.cpp名の確率情報件数です。 | `"n_probs": 5` |
| `post_sampling_probs` | sampling後の確率情報を返すための llama.cpp 独自項目です。 | `"post_sampling_probs": true` |

```json
{
  "messages": [{"role": "user", "content": "yes か no で答えて"}],
  "max_tokens": 4,
  "logprobs": true,
  "top_logprobs": 5
}
```

### Grammar / JSON制約

| 項目 | 説明 | 例 |
| --- | --- | --- |
| `grammar` | GBNF文字列を直接指定して出力を制約します。 | `"grammar": "root ::= \"ok\""` |
| `grammar_file` | proxy独自項目です。`./grammars/` 配下のGBNFファイル名を指定します。backendへは `grammar` として転送されます。 | `"grammar_file": "structured_cot_short.gbnf"` |
| `json_schema` | JSON Schemaを指定します。llama.cpp側でGBNFへ変換されます。 | `"json_schema": {"type": "object"}` |
| `response_format` | OpenAI風の出力形式指定です。`text`、`json_object`、`json_schema` を使います。 | `"response_format": {"type": "json_object"}` |
| `grammar_lazy` | grammarをlazy適用します。通常は不要です。使う場合はtrigger指定も必要です。 | `"grammar_lazy": true` |
| `grammar_triggers` | lazy grammarを開始するtriggerです。tool callingや高度な制約向けです。 | `"grammar_triggers": [...]` |
| `preserved_tokens` | grammar triggerなどで保持したい特殊token文字列です。 | `"preserved_tokens": ["<tool_call>"]` |

`grammar` と `json_schema` は同時に使えません。`grammar_file` は絶対パス、`../`、存在しないファイルを拒否します。

JSON objectを強制する例:

```json
{
  "messages": [{"role": "user", "content": "name と age をJSONで返して"}],
  "response_format": {"type": "json_object"},
  "max_tokens": 128
}
```

schemaつきJSONの例:

```json
{
  "messages": [{"role": "user", "content": "架空の人物を1人作って"}],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "schema": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "age": {"type": "integer"}
        },
        "required": ["name", "age"]
      }
    }
  },
  "max_tokens": 128
}
```

GBNFファイル指定の例:

```json
{
  "messages": [{"role": "user", "content": "thought と answer を持つJSONで返して"}],
  "grammar_file": "structured_cot_short.gbnf",
  "max_tokens": 256
}
```

### Reasoning / thinking

| 項目 | 説明 | 例 |
| --- | --- | --- |
| `reasoning_format` | reasoningの返し方です。`none` は未解析で `message.content` に残し、`deepseek` は `message.reasoning_content` に分離し、`deepseek-legacy` はcontentにもtagを残します。 | `"reasoning_format": "none"` |
| `thinking_budget_tokens` | thinking対応テンプレートで思考に使うtoken予算です。 | `"thinking_budget_tokens": 128` |
| `chat_template_kwargs` | chat templateへ渡す追加引数です。thinkingのON/OFFなどに使います。 | `"chat_template_kwargs": {"enable_thinking": false}` |
| `add_generation_prompt` | chat template末尾にassistant生成開始promptを追加するかです。通常は `true` です。 | `"add_generation_prompt": true` |
| `generation_prompt` | parser向けに、templateがprefillした生成開始文字列を明示します。通常は自動値で足ります。 | `"generation_prompt": "assistant:"` |

thinkingをcontentへ出したい例:

```json
{
  "messages": [{"role": "user", "content": "短く考えて答えて"}],
  "reasoning_format": "none",
  "thinking_budget_tokens": 64,
  "max_tokens": 256
}
```

thinkingを切りたい例:

```json
{
  "messages": [{"role": "user", "content": "結論だけ答えて"}],
  "chat_template_kwargs": {"enable_thinking": false},
  "reasoning_format": "none",
  "max_tokens": 128
}
```

### Tool calling

| 項目 | 説明 | 例 |
| --- | --- | --- |
| `tools` | OpenAI互換のfunction tool定義です。backendのJinja template対応が必要です。 | `"tools": [{"type": "function", "function": {...}}]` |
| `tool_choice` | tool使用方針です。`auto`、`none`、または特定function指定を使います。 | `"tool_choice": "auto"` |
| `parallel_tool_calls` | 複数tool callを許可します。templateが対応している場合のみ有効です。 | `"parallel_tool_calls": true` |
| `parse_tool_calls` | 生成結果からtool callをparseするかです。通常はtools利用時に自動設定されます。 | `"parse_tool_calls": true` |

```json
{
  "messages": [{"role": "user", "content": "東京の天気を調べて"}],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "指定地点の天気を返す",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          },
          "required": ["location"]
        }
      }
    }
  ],
  "tool_choice": "auto"
}
```

### llama.cpp 独自・上級項目

| 項目 | 説明 | 例 |
| --- | --- | --- |
| `cache_prompt` | prompt cacheを使うかです。backend起動設定の既定値を上書きします。 | `"cache_prompt": true` |
| `return_tokens` | token ID列などをレスポンスに含めます。デバッグ向けです。 | `"return_tokens": true` |
| `return_progress` | prompt処理の進捗情報を返します。 | `"return_progress": true` |
| `timings_per_token` | tokenごとのtiming情報を返します。 | `"timings_per_token": true` |
| `n_keep` | context shift時などに先頭から保持するtoken数です。 | `"n_keep": 0` |
| `n_discard` | context shift時などに破棄するtoken数です。 | `"n_discard": 0` |
| `n_cmpl` | llama.cpp名の生成候補数です。OpenAI互換の `n` と似ています。 | `"n_cmpl": 1` |
| `n_cache_reuse` | prompt cache再利用を試みる最小chunkサイズです。 | `"n_cache_reuse": 0` |
| `t_max_predict_ms` | 生成に使う最大時間をミリ秒で指定します。 | `"t_max_predict_ms": 10000` |
| `response_fields` | レスポンスに含めるフィールドを絞るための項目です。 | `"response_fields": ["content", "timings"]` |
| `backend_sampling` | experimentalなbackend samplingを有効化します。 | `"backend_sampling": false` |
| `lora` | 読み込み済みLoRAの適用をリクエスト単位で指定します。 | `"lora": [{"id": 0, "scale": 0.8}]` |
| `speculative.n_min` | speculative decodingの最小draft token数です。 | `"speculative.n_min": 0` |
| `speculative.n_max` | speculative decodingの最大draft token数です。 | `"speculative.n_max": 16` |
| `speculative.p_min` | speculative decodingで採用する最小確率です。 | `"speculative.p_min": 0.75` |
| `speculative.type` | speculative decoding方式です。 | `"speculative.type": "draft"` |
| `speculative.ngram_size_n` | n-gram speculative decoding用のn値です。 | `"speculative.ngram_size_n": 3` |
| `speculative.ngram_size_m` | n-gram speculative decoding用のm値です。 | `"speculative.ngram_size_m": 5` |
| `speculative.ngram_m_hits` | n-gram speculative decodingのhit数しきい値です。 | `"speculative.ngram_m_hits": 1` |

debug寄りの例:

```json
{
  "messages": [{"role": "user", "content": "hello"}],
  "max_tokens": 32,
  "cache_prompt": true,
  "return_tokens": true,
  "timings_per_token": true
}
```

## APIキーありの例

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Authorization: Bearer your-secret-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "hello"}],
    "stream": false
  }'
```

## 補足

- `/api/models` は `MODEL_DIR` 配下の `.gguf` を再帰検索します。LM Studioのシンボリックリンク先も探索対象です。
- GGUFメタデータはテンソル本体を読まず、ファイルのパス・サイズ・更新時刻をキーにプロセス内でキャッシュします。
- 分割GGUFは `00001-of-xxxxx.gguf` の先頭shardだけをモデルとして列挙し、そのメタデータを使います。
- `mmproj*.gguf` はモデル一覧から除外し、同じディレクトリの対応モデルに関連付けます。
- proxyが終了すると、このproxyから起動した `llama-server` も停止します。
