import json
import os
import multiprocessing

from predictor import Predictor

data_path = "data/input"  # The directory of the input data
output_path = "data/output"  # The directory of the output data


def format_result(result):
    rex = {"accusation": [], "articles": [], "imprisonment": -3}

    res_acc = []
    for x in result["accusation"]:
        if not (x is None):
            res_acc.append(int(x))
    rex["accusation"] = res_acc

    if not (result["imprisonment"] is None):
        rex["imprisonment"] = int(result["imprisonment"])
    else:
        rex["imprisonment"] = -3

    res_art = []
    for x in result["articles"]:
        if not (x is None):
            res_art.append(int(x))
    rex["articles"] = res_art

    return rex


if __name__ == "__main__":
    user = Predictor()
    cnt = 0


    def get_batch():
        v = user.batch_size
        if not (type(v) is int) or v <= 0:
            raise NotImplementedError

        return v


    def solve(fact):
        result = user.predict(fact)
        
        # 确保结果是列表格式
        if not isinstance(result, list):
            result = [result]
        
        # 格式化每个结果
        formatted_results = []
        for r in result:
            formatted_results.append(format_result(r))
        
        return formatted_results


    for file_name in os.listdir(data_path):
        inf = open(os.path.join(data_path, file_name), "r", encoding="utf-8")
        ouf = open(os.path.join(output_path, file_name), "w", encoding="utf-8")

        fact = []

        for line in inf:
            fact.append(json.loads(line)["fact"])
            if len(fact) == get_batch():
                results = solve(fact)
                cnt += len(results)
                # 每行输出一个JSON对象
                for result in results:
                    ouf.write(json.dumps(result, ensure_ascii=False) + "\n")
                fact = []

        if len(fact) != 0:
            results = solve(fact)
            cnt += len(results)
            # 每行输出一个JSON对象
            for result in results:
                ouf.write(json.dumps(result, ensure_ascii=False) + "\n")
            fact = []

        ouf.close()
