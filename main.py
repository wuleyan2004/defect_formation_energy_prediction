from fastapi import FastAPI, UploadFile, File
from predictor import FormationEnergyPredictor  # 导入你第一阶段封装的类
import os
from ase.io import read

app = FastAPI()

# 1. 在程序启动时就加载模型（避免重复加载）
# 使用基于当前文件的绝对路径，适配当前项目结构
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "models", "formation_energy_model.pth")
feature_path = os.path.join(base_dir, "atom_features.pth")

predictor = FormationEnergyPredictor(model_path=model_path, feature_path=feature_path)

@app.get("/")
def read_root():
    return {"message": "二维材料缺陷预测平台后端已启动"}

@app.post("/predict")
async def predict_formation_energy(file: UploadFile = File(...)):
    # 2. 接收并保存上传的文件
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        # 3. 使用 ase 读取晶体结构文件，并调用写好的预测器
        structure = read(temp_path)
        result = predictor.predict(structure) 
        
        # 4. 返回结果
        return {
            "filename": file.filename,
            "formation_energy": float(result),
            "status": "success"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        # 5. 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)