from flask import Flask, render_template, request
from openai import OpenAI
import pandas as pd
import json
import difflib
from dotenv import load_dotenv
import os

# Flask アプリ
app = Flask(__name__)

# OpenAI API
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# TRIZパラメータ一覧
TRIZ_PARAMETERS = [
    "移動物体の重量","静止物体の重量","移動物体の長さ/角度","静止物体の長さ/角度",
    "移動物体の面積","静止物体の面積","移動物体の体積","静止物体の体積","形状","物質の量",
    "情報の量","移動物体の動作時間","静止物体の動作時間","速度[スピード]","力/トルク",
    "移動物体の使用エネルギー","静止物体の使用エネルギー","パワー","応力/圧力","強度",
    "(物体の構の)安定性","温度","照明強度","機能の効率","物質の損失","時間の損失","エネルギーの損失",
    "情報の損失","騒音[ノイズ]","有害なものの放出","システムが作り出すその他の有益な効果",
    "適応性/汎用性","両立性/接続性","操作の容易性","信頼性/ロバスト性[頑健性]",
    "修理可能性","セキュリティ","安全性/脆弱性","美しさ/見かけ",
    "システムが作り出すその他の有害な効果","製造性","製造精度/一貫性",
    "自動化","生産性","システムの複雑さ","制御の複雑さ",
    "検出/測定の能力","測定の精度"
]

# 発明原理辞書
TRIZ_PRINCIPLES = {
    1:"分割",2:"取り出し",3:"局所品質",4:"非対称",5:"統合",6:"万能性",
    7:"入れ子",8:"バランス調整",9:"予め反作用",10:"予め作用",11:"予備",
    12:"等ポテンシャル",13:"逆の操作",14:"曲面",15:"動作の柔軟性",
    16:"部分的/過剰な作用",17:"他次元",18:"機械的振動",19:"周期的作用",
    20:"有益な作用の連続",21:"高速実行",22:"有害作用の逆利用",23:"フィードバック",
    24:"仲介",25:"自己修復",26:"コピー",27:"廉価短命",28:"機械システムの代替",
    29:"流体膜・柔軟殻",30:"薄膜・強い殻",31:"多孔質材料",32:"色の変更",
    33:"均質性",34:"廃棄と再生",35:"パラメータ変化",36:"相転移",37:"熱膨張",
    38:"強い酸化剤",39:"不活性環境",40:"複合材料"
}

# TRIZマトリックス読み込み
matrix_df = pd.read_excel("matrix.xlsx", index_col=0)

def infer_parameters(requirement_text):
    """GPTで改善・劣化パラメータを推定"""
    prompt = f"""
あなたはTRIZの専門家です。
次の技術的要求を読んで、それが改善するパラメータと劣化するパラメータを推定してください。

必ず次のリストから2つ選び、JSON形式で出力してください。
リスト: {TRIZ_PARAMETERS}

出力例:
{{
  "改善": "騒音[ノイズ]",
  "劣化": "機能の効率"
}}

技術的要求: {requirement_text}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={ "type": "json_object" }
    )
    result = json.loads(response.choices[0].message.content)
    improved = result.get("改善")
    worsened = result.get("劣化")
    if improved not in TRIZ_PARAMETERS:
        improved = difflib.get_close_matches(improved, TRIZ_PARAMETERS, n=1)[0]
    if worsened not in TRIZ_PARAMETERS:
        worsened = difflib.get_close_matches(worsened, TRIZ_PARAMETERS, n=1)[0]
    return improved, worsened

def get_triz_principles(improved, worsened, requirement_text):
    """TRIZマトリックスから発明原理を取得し、GPTで事例を生成"""
    try:
        cell_value = matrix_df.at[improved, worsened]
        if pd.isna(cell_value):
            return []
        principle_nums = [int(p.strip()) for p in str(cell_value).split(",")]
        results = []
        for num in principle_nums:
            principle_name = TRIZ_PRINCIPLES.get(num, "不明")
            example_prompt = f"""
技術的要求: {requirement_text}
該当するTRIZ発明原理: {num}: {principle_name}
この技術的要求と関連する具体的な事例を1つ挙げてください。
"""
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":example_prompt}],
                temperature=0.7
            )
            example = response.choices[0].message.content.strip()
            results.append((num, principle_name, example))
        return results
    except KeyError:
        return []

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        requirement = request.form["requirement"]
        improved, worsened = infer_parameters(requirement)
        principles = get_triz_principles(improved, worsened, requirement)
        return render_template("index.html", 
                               requirement=requirement,
                               improved=improved,
                               worsened=worsened,
                               principles=principles)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

