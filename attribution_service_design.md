# **归因服务 (Attribution Service) \- 详细设计与实现**

本文档基于 EmSHAP 论文，为归因服务（Attribution Service）提供详细的设计与实现方案。该服务是整个能效调度系统的智能核心，负责将底层的硬件指标转化为可解释的能耗归因。

## **1\. 核心设计决策：训练与推理分离**

尽管整个后端系统主要使用Go构建，但在机器学习领域，Python拥有无与伦比的生态系统（PyTorch, TensorFlow, Scikit-learn等），是模型研究和训练的**事实标准**。直接在Go中实现复杂的模型训练（如论文中的GRU和MLP网络、反向传播、动态掩码等）不仅工作量巨大，而且缺乏现成的、经过验证的库。

因此，我们采用业界成熟的\*\*训练与推理分离（Training/Inference Decoupling）\*\*架构：

* **模型训练（Training）**: 使用Python和PyTorch/TensorFlow实现一个独立的训练服务。该服务负责消费Kafka中的原始指标数据，完成EmSHAP模型的完整训练流程，并将最终训练好的模型导出为**ONNX (Open Neural Network Exchange)** 格式。
* **模型推理（Inference）**: 在现有的Go语言Attribution Service中，我们**只负责加载和执行**这个ONNX模型。Go语言有成熟的ONNX Runtime库，可以高效地进行模型推理，完全满足在线服务对高性能、低延迟的要求。

这种架构兼顾了研究/训练的灵活性和在线服务的性能与稳定性。

```mermaid
%%{init: { 
        "theme": "base", 
        "themeVariables": { 
            "clusterBkg": "#f8f9fa", 
            "clusterBorder": "#e1e4e8" 
        } 
    }}%%
graph TD
    %% 样式定义
    classDef baseNode fill:#ffffff,stroke:#68ACD4,stroke-width:1.5px,rx:8,ry:8,font-family:Arial,font-size:12px,color:#333,font-weight:bold
    classDef accentNode fill:#1D50A2,stroke:#163E7A,stroke-width:1.5px,rx:8,ry:8,font-family:Arial,font-size:12px,color:#ffffff,font-weight:bold
    classDef storageNode fill:#E7F3F5,stroke:#108A95,stroke-width:1.5px,rx:8,ry:8,font-family:Arial,font-size:12px,color:#333,font-weight:bold
    classDef subgraphStyle rx:10,ry:10,padding:25px
    classDef subgraphTitle fill:none,stroke:none,font-weight:bold,font-size:15px,color:#1D50A2

    subgraph sg1 [离线训练平台 Python]
        direction LR
        class sg1 subgraphStyle
        A[<fa:fa-download> Kafka Consumer]:::baseNode
        B[<fa:fa-filter> 数据清洗与特征工程]:::baseNode
        C{<fa:fa-cogs> EmSHAP 模型训练}:::accentNode
        D[<fa:fa-file-export> 导出为 ONNX 模型]:::baseNode
        E[<fa:fa-archive> 模型仓库 如 S3/MinIO]:::storageNode

        A --> B;
        B --> C;
        C -- "应用动态掩码机制" --> C;
        C --> D;
        D --> E;
    end
    class sg1 subgraphTitle

    %% 两个子图之间添加垂直空间
    sg1 -->|模型流转| sg2

    subgraph sg2 [在线归因服务 Go]
        direction LR
        class sg2 subgraphStyle
        F[<fa:fa-power-off> Attribution Service 启动]:::baseNode
        G{<fa:fa-download> 从模型仓库加载 ONNX 模型}:::baseNode
        H[<fa:fa-brain> ONNX Runtime 内存中]:::accentNode
        I[<fa:fa-server> gRPC Server: GetAttribution]:::baseNode
        J[<fa:fa-calculator> 推理逻辑]:::baseNode
        K[<fa:fa-file-import> 返回归因结果]:::baseNode

        F --> G;
        G --> H;
        I -- "收到请求" --> J;
        J -- "调用模型" --> H;
        H -- "返回推理结果" --> J;
        J -- "计算Shapley值" --> K;
        I --> K;
    end
    class sg2 subgraphTitle

    %% 数据流向样式
    linkStyle default stroke:#1D50A2,stroke-width:2px
    linkStyle 2 stroke:#FF5733,stroke-width:2px  
    linkStyle 7 stroke:#FF5733,stroke-width:2px  
    linkStyle 8 stroke:#FF5733,stroke-width:2px  
    linkStyle 9 stroke:#68ACD4,stroke-dasharray:5,5,stroke-width:1.5px 

    %% 节点分类
    class A,B,D,F,G,I,J,K baseNode;
    class C,H accentNode;
    class E storageNode;

    

## **2\. EmSHAP 模型实现 (internal/attribution/ebm)**

这个Go包将封装所有与EmSHAP模型推理相关的逻辑。

**相关代码:**

* internal/attribution/ebm/model.go: 定义模型结构，加载ONNX文件，并执行推理。
* internal/attribution/ebm/shapley.go: 实现基于模型输出的Shapley值计算逻辑。

### **2.1 模型结构 (model.go)**

我们将定义一个Model结构体，它持有一个ONNX Runtime的会话实例。

// internal/attribution/ebm/model.go

package ebm

import ( "github.com/ortgo/onnxruntime" // 示例ONNX Runtime库 "xai\_energy\_scheduler/pkg/models")

// EmSHAPModel 封装了ONNX模型和推理逻辑type EmSHAPModel struct { session \*onnxruntime.Session // 可以包含模型的元数据，如输入/输出张量的名称和形状 inputName string outputNames \[\]string}

// NewEmSHAPModel 从指定的路径加载ONNX模型文件并初始化一个模型实例func NewEmSHAPModel(modelPath string) (\*EmSHAPModel, error) { // ... 加载ONNX文件并创建session的逻辑 ... // ... 获取模型的输入输出信息 ... return \&EmSHAPModel{session: session /\* ... \*/}, nil}

// PredictEnergyContribution 使用加载的ONNX模型进行单次推理// 输入是一个特征向量，输出是论文中定义的能量函数 gθ(x) 和提议分布 q(x) 的参数func (m \*EmSHAPModel) PredictEnergyContribution(vector \*models.FeatureVector) (energy float32, proposalParams map\[string\]interface{}, err error) { // 1\. 将 FeatureVector 转换为 ONNX Runtime 所需的张量 (Tensor) inputTensor, err := createInputTensor(vector) if err \!= nil { return 0, nil, err } // 2\. 执行模型推理 inputs := map\[string\]onnxruntime.Tensor{m.inputName: inputTensor} outputs, err := m.session.Run(inputs) if err \!= nil { return 0, nil, err } // 3\. 解析模型的输出张量，提取能量值和提议分布的参数（均值、方差等） energy, proposalParams \= parseOutputTensors(outputs) return energy, proposalParams, nil

}

// ... 其他辅助函数 ...

### **2.2 Shapley 值计算 (shapley.go)**

这是实现论文核心算法的关键部分。它将调用PredictEnergyContribution来估算不同特征子集的贡献函数v(S)，最终算出每个特征的Shapley值。

```mermaid
%%{init: { 
        "theme": "base", 
        "themeVariables": { 
            "actorBkg": "#f8f9fa", 
            "actorBorder": "#e1e4e8", 
            "messageArrowColor": "#1D50A2", 
            "lifeLineColor": "#68ACD4", 
            "lifeLineTextColor": "#333" 
        } 
    }}%%
sequenceDiagram
    title Shapley值计算流程

    participant GetAttribution as <fa:fa-server> gRPC Handler
    participant ShapleyCalc as <fa:fa-calculator> CalculateShapleyValue
    participant EmSHAPModel as <fa:fa-brain> EBM模型
    participant MonteCarlo as <fa:fa-random> 蒙特卡洛采样

    GetAttribution->>ShapleyCalc: CalculateShapleyValue(baseVector)
    Note right of ShapleyCalc: 循环遍历所有特征子集 S

    loop 对每个特征 i
        ShapleyCalc->>ShapleyCalc: 生成包含 i 的子集 S ∪ {i}
        ShapleyCalc->>ShapleyCalc: 生成不含 i 的子集 S

        ShapleyCalc->>MonteCarlo: 估算 v(S ∪ {i})
        MonteCarlo->>EmSHAPModel: PredictEnergyContribution(maskedVector)
        EmSHAPModel-->>MonteCarlo: 返回能量和提议分布
        MonteCarlo-->>ShapleyCalc: 返回 v(S ∪ {i}) 的估算值

        ShapleyCalc->>MonteCarlo: 估算 v(S)
        MonteCarlo->>EmSHAPModel: PredictEnergyContribution(maskedVector)
        EmSHAPModel-->>MonteCarlo: 返回能量和提议分布
        MonteCarlo-->>ShapleyCalc: 返回 v(S) 的估算值

        ShapleyCalc->>ShapleyCalc: 累加边际贡献 v(S ∪ {i}) - v(S)
    end

    ShapleyCalc->>ShapleyCalc: 根据公式(1)加权平均，得到phi_i
    ShapleyCalc-->>GetAttribution: 返回所有特征的Shapley值

    

这个过程的核心是估算贡献函数 $v(S) = E[f(x) | x_S = x_S^t]$ 。根据论文的公式(6)，这通过蒙特卡洛采样实现：

1. 对于给定的特征子集$S$，我们保持$x_S$的特征值不变。
2. 对于S之外的特征$x_S̄$，我们使用EmSHAPModel预测出的提议分布$q(x\_S̄ | x\_S)$进行多次采样，生成K个样本 $x\_S̄^(k)$。
3. 对于每个样本，我们组合成完整的特征向量$(x\_S, x\_S̄^(k))$，并用一个预先训练好的**功耗预测模型** $f(x)$（这可以是另一个简单的MLP模型，与EmSHAP一同导出）来预测其功耗。
4. 最终$v(S)$的估算值是这K个功耗预测值的平均。

**2.3 模型训练与部署生命周期**

为了让归因服务能够持续演进和优化，我们需要建立一个标准化的模型训练与部署（MLOps）流程。

#### **2.3.1 数据准备与特征工程 (Python)**

* **数据源**: 训练脚本将作为Kafka消费者，订阅raw\_metrics主题，持续获取由Data Agent采集的FeatureVector数据流。
* **数据清洗**: 对原始数据进行预处理，包括处理缺失值、去除异常点等。
* **特征缩放**: 对数值型特征（如PMU计数器）进行标准化（Standardization）或归一化（Normalization），使其落入相似的数值范围，这对于神经网络模型的稳定训练至关重要。
* **数据存储**: 预处理后的数据将被存储在专门用于机器学习的数据湖或特征存储（Feature Store）中，以供后续模型训练和分析使用。

#### **2.3.2 模型训练 (Python/PyTorch)**

我们将训练两个独立的模型：

1. **EmSHAP 模型**:
  * **架构**: 严格遵循论文中的设计，由一个**能量网络**（带Skip Connection的MLP）和一个**GRU网络**（用于生成提议分布）组成。
  * **训练核心**: 关键在于实现论文中的**动态掩码机制 (Dynamic Masking Scheme)**。在训练的每个周期（Epoch），我们会动态调整特征的掩码率（例如从0.2线性增加到0.8），这使得模型能够学习到各种不同特征组合下的条件依赖关系，极大地提升了模型的泛化能力。
  * **损失函数**: 使用最大似然估计（MLE）作为损失函数，如论文公式(14)所示，目标是让模型学习到的概率分布p(x)尽可能地逼近真实的数据分布。
2. **功耗预测模型 f(x)**:
  * **架构**: 这是一个相对简单的监督学习模型，可以使用一个标准的多层感知机（MLP）或者梯度提升树（如XGBoost）来实现。
  * **目标**: 学习从一个**完整**的FeatureVector到瞬时功耗（PowerConsumption）的映射关系。这个模型在Shapley值计算的蒙特卡洛采样环节中扮演着“价值函数”的角色。

#### **2.3.3 模型部署与 CI/CD**

* **模型导出**: 两个模型训练完成后，都将被导出为通用的ONNX格式。这使得模型本身与训练框架（PyTorch）解耦，便于在不同语言环境（Go）中部署。
* **模型版本控制与存储**: 每个导出的ONNX模型都会被赋予一个唯一的版本号（例如，使用Git Commit Hash或时间戳），并存储在对象存储服务（如AWS S3, MinIO）中，形成一个模型仓库。
* **自动化部署流水线 (CI/CD)**:
  1. **触发**: 当训练代码有新的提交，或手动触发时，CI/CD流水线启动。
  2. **执行**: 流水线自动执行数据准备和模型训练脚本。
  3. **评估**: 训练完成后，使用预留的测试数据集评估新模型的性能。只有当新模型的关键指标（如损失、准确率）优于当前生产环境中的模型时，才继续下一步。
  4. **发布**: 将通过评估的ONNX模型推送到模型仓库。
  5. **上线**: 自动触发Attribution Service的滚动更新（Rolling Update），新的服务实例在启动时会从模型仓库拉取并加载最新版本的模型，实现无缝切换。

## **3\. 服务实现 (internal/attribution/server.go)**

server.go中的gRPC服务实现将变得非常清晰。

// internal/attribution/server.go

package attribution

import ( // ... imports ... "xai\_energy\_scheduler/internal/attribution/ebm")

type Server struct { // ... 其他字段 ... ebmModel \*ebm.EmSHAPModel}

// NewServer 创建一个新的归因服务实例func NewServer() (\*Server, error) { // 从配置中获取模型路径 modelPath := "/models/emshap\_model.onnx" // 路径应可配置 model, err := ebm.NewEmSHAPModel(modelPath) if err \!= nil { return nil, fmt.Errorf("failed to load ebm model: %w", err) } return \&Server{ebmModel: model}, nil

}

// GetAttribution 是 gRPC 接口的实现func (s \*Server) GetAttribution(ctx context.Context, req \*pb.AttributionRequest) (\*pb.AttributionResponse, error) { featureVector := req.GetFeatureVector() // 调用Shapley值计算逻辑 // shapley.CalculateShapleyValue 会在内部多次调用 s.ebmModel.PredictEnergyContribution attributionShares, err := ebm.CalculateShapleyValue(s.ebmModel, featureVector) if err \!= nil { // ... 错误处理 ... return nil, err } // 构建并返回gRPC响应 response := \&pb.AttributionResponse{ AttributionShares: attributionShares, } return response, nil

}

## **4\. 总结与后续步骤**

通过上述设计，我们成功地将一篇前沿的科研论文思想，转化为了一个工程上健壮、职责清晰的微服务实现方案。

**下一步行动计划:**

1. **搭建Python训练环境**: 建立一个独立的Python项目，用于实现EmSHAP模型的训练脚本。
2. **定义ONNX模型接口**: 明确Python训练脚本输出的ONNX模型的输入和输出张量格式，确保Go端的推理代码可以正确解析。
3. **实现Go端推理逻辑**: 按照本文档的设计，完成internal/attribution/ebm包中的代码实现。
4. **集成与测试**: 部署训练服务生成第一个版本的模型，并让Go服务加载它，进行端到端的集成测试。

这个设计为实现系统的“智能大脑”提供了清晰的路线图。