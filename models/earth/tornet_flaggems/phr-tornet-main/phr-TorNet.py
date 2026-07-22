#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import os
import sys

# 避免在TorNet源码目录生成不必要的__pycache__
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

# 必须在第一次import keras之前设置
os.environ["KERAS_BACKEND"] = "torch"

# 当前环境的固定路径
REPO_ROOT = Path(
    "/root/private_data/phr/phr-tornet-main"
).resolve()

DATA_ROOT = Path(
    "/root/private_data/phr/phr-TorNet-data"
).resolve()

MODEL_PATH = (
    DATA_ROOT
    / "tornado_detector_baseline.keras"
)

GEMS_LOG_PATH = (
    REPO_ROOT
    / "gems_debug.log"
)

# 当前只测试2013年
TEST_YEAR = 2013
BATCH_SIZE = 1

# 第一次快速测试时保持False
RUN_FULL_2013_TEST = True

# 检查TorNet源码
assert REPO_ROOT.is_dir(), (
    f"源码目录不存在：{REPO_ROOT}"
)

assert (REPO_ROOT / "README.md").is_file(), (
    "源码目录中缺少README.md"
)

assert (REPO_ROOT / "tornet").is_dir(), (
    "源码目录中缺少tornet包"
)

assert (
    REPO_ROOT
    / "scripts"
    / "tornado_detection"
).is_dir(), (
    "源码目录中缺少官方测试脚本目录"
)

# 检查2013数据
assert DATA_ROOT.is_dir(), (
    f"数据目录不存在：{DATA_ROOT}"
)

assert (
    DATA_ROOT / "catalog.csv"
).is_file(), (
    "数据目录中缺少catalog.csv"
)

assert (
    DATA_ROOT / "train"
).is_dir(), (
    "数据目录中缺少train目录"
)

assert (
    DATA_ROOT / "test"
).is_dir(), (
    "数据目录中缺少test目录"
)

# 检查手动上传的预训练模型
assert MODEL_PATH.is_file(), (
    f"预训练模型不存在：{MODEL_PATH}"
)

# 设置TorNet数据根目录
os.environ["TORNET_ROOT"] = str(DATA_ROOT)

# 切换到TorNet源码目录
os.chdir(REPO_ROOT)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )

# 新Kernel中不能已有FlagGems启用标志
if globals().get(
    "_FLAGGEMS_ENABLED",
    False,
):
    raise RuntimeError(
        "当前Kernel中已经存在FlagGems启用标志。"
        "请重启Kernel后重新执行。"
    )

print("TorNet源码目录：", REPO_ROOT)
print("TorNet数据目录：", DATA_ROOT)
print("预训练模型：", MODEL_PATH)
print("TORNET_ROOT：", os.environ["TORNET_ROOT"])
print("KERAS_BACKEND：", os.environ["KERAS_BACKEND"])
print("FlagGems日志：", GEMS_LOG_PATH)
print("完整2013测试：", RUN_FULL_2013_TEST)


# In[2]:


import platform
import numpy as np
import torch
import triton
import flag_gems
import keras

from importlib.metadata import (
    version,
    PackageNotFoundError,
)


def package_version(name):
    try:
        return version(name)

    except PackageNotFoundError:
        return "unknown"


def synchronize_device():
    """
    等待加速设备上的异步任务执行完成。
    """

    try:
        torch.cuda.synchronize()

    except Exception:
        pass


def to_numpy(value):
    """
    将PyTorch或Keras输出复制到CPU NumPy。
    """

    if isinstance(value, torch.Tensor):
        return (
            value
            .detach()
            .float()
            .cpu()
            .numpy()
            .copy()
        )

    return np.asarray(
        keras.ops.convert_to_numpy(value)
    ).copy()


print("运行环境")
print("=" * 72)

print(
    "Python版本：",
    platform.python_version(),
)

print(
    "PyTorch版本：",
    torch.__version__,
)

print(
    "Triton版本：",
    getattr(
        triton,
        "__version__",
        "unknown",
    ),
)

print(
    "FlagGems版本：",
    getattr(
        flag_gems,
        "__version__",
        package_version("flag-gems"),
    ),
)

print(
    "Keras版本：",
    keras.__version__,
)

print(
    "Keras后端：",
    keras.config.backend(),
)

print(
    "FlagGems设备：",
    flag_gems.device,
)

print(
    "FlagGems厂商：",
    getattr(
        flag_gems,
        "vendor_name",
        "unknown",
    ),
)

print(
    "HIP版本：",
    getattr(
        torch.version,
        "hip",
        None,
    ),
)

print(
    "CUDA接口可用：",
    torch.cuda.is_available(),
)

print(
    "设备数量：",
    torch.cuda.device_count(),
)

if torch.cuda.is_available():
    for index in range(
        torch.cuda.device_count()
    ):
        print(
            f"设备{index}：",
            torch.cuda.get_device_name(index),
        )

assert keras.config.backend() == "torch", (
    "Keras后端不是torch。"
    "请重启Kernel并从单元格1重新执行。"
)

assert torch.cuda.is_available(), (
    "PyTorch没有识别到加速设备。"
)

assert not globals().get(
    "_FLAGGEMS_ENABLED",
    False,
), (
    "当前Kernel已经启用过FlagGems。"
)

print("\n环境检查：通过")
print("当前尚未启用FlagGems。")


# In[3]:


from tornet.data.constants import (
    ALL_VARIABLES,
)

from tornet.data.preprocess import (
    get_shape,
)

from tornet.data.keras.loader import (
    KerasDataLoader,
)

from tornet.models.keras.cnn_baseline import (
    build_model,
)


sample_weights_config = {
    "wN": 1.0,
    "w0": 1.0,
    "w1": 1.0,
    "w2": 2.0,
    "wW": 0.5,
}

select_keys = list(
    ALL_VARIABLES
) + [
    "range_folded_mask",
    "coordinates",
]


train_loader = KerasDataLoader(
    data_root=str(DATA_ROOT),
    data_type="train",
    years=[TEST_YEAR],
    batch_size=BATCH_SIZE,
    weights=sample_weights_config,
    include_az=False,
    random_state=2026,
    select_keys=select_keys,
    tilt_last=True,
    workers=1,
    use_multiprocessing=False,
    max_queue_size=2,
)


test_loader = KerasDataLoader(
    data_root=str(DATA_ROOT),
    data_type="test",
    years=[TEST_YEAR],
    batch_size=BATCH_SIZE,
    weights=sample_weights_config,
    include_az=False,
    random_state=2026,
    select_keys=select_keys,
    tilt_last=True,
    workers=1,
    use_multiprocessing=False,
    max_queue_size=2,
)


print(
    "TorNet输入变量：",
    ALL_VARIABLES,
)

print(
    "训练批次数：",
    len(train_loader),
)

print(
    "测试批次数：",
    len(test_loader),
)

assert len(train_loader) > 0, (
    "2013训练加载器为空"
)

assert len(test_loader) > 0, (
    "2013测试加载器为空"
)

print(
    "\n2013年数据加载器创建：通过"
)


# In[4]:


# 读取第一个训练批次
train_batch = train_loader[0]

assert isinstance(
    train_batch,
    (tuple, list),
), (
    "训练加载器返回格式异常"
)

assert len(train_batch) == 3, (
    "训练批次应返回x、y和sample_weight"
)

x_train, y_train, weight_train = (
    train_batch
)


# 读取第一个测试批次
test_batch = test_loader[0]

assert isinstance(
    test_batch,
    (tuple, list),
), (
    "测试加载器返回格式异常"
)


if len(test_batch) == 3:
    x_test, y_test, weight_test = (
        test_batch
    )

elif len(test_batch) == 2:
    x_test, y_test = test_batch

    weight_test = np.ones_like(
        np.asarray(y_test),
        dtype=np.float32,
    )

else:
    raise RuntimeError(
        "测试加载器返回了"
        f"{len(test_batch)}个对象，"
        "无法识别。"
    )


# 标签和样本权重统一转成float32
y_train = np.asarray(
    y_train,
    dtype=np.float32,
)

weight_train = np.asarray(
    weight_train,
    dtype=np.float32,
)

y_test = np.asarray(
    y_test,
    dtype=np.float32,
)

weight_test = np.asarray(
    weight_test,
    dtype=np.float32,
)


required_keys = set(
    ALL_VARIABLES
) | {
    "range_folded_mask",
    "coordinates",
}


missing_train_keys = (
    required_keys
    - set(x_train.keys())
)

missing_test_keys = (
    required_keys
    - set(x_test.keys())
)


assert not missing_train_keys, (
    "训练输入缺少："
    f"{sorted(missing_train_keys)}"
)

assert not missing_test_keys, (
    "测试输入缺少："
    f"{sorted(missing_test_keys)}"
)

assert np.isfinite(y_train).all(), (
    "训练标签含NaN或Inf"
)

assert np.isfinite(
    weight_train
).all(), (
    "训练权重含NaN或Inf"
)

assert np.isfinite(y_test).all(), (
    "测试标签含NaN或Inf"
)

assert np.isfinite(
    weight_test
).all(), (
    "测试权重含NaN或Inf"
)


print("训练批次")
print("=" * 72)

for key, value in x_train.items():
    array = np.asarray(value)

    print(
        f"{key:25s}"
        f"shape={array.shape}, "
        f"dtype={array.dtype}, "
        f"finite={np.isfinite(array).all()}"
    )


print(
    "\n训练标签：",
    y_train.reshape(-1),
)

print(
    "训练权重：",
    weight_train.reshape(-1),
)


print("\n测试批次")
print("=" * 72)

for key, value in x_test.items():
    array = np.asarray(value)

    print(
        f"{key:25s}"
        f"shape={array.shape}, "
        f"dtype={array.dtype}"
    )


print(
    "\n测试标签：",
    y_test.reshape(-1),
)

print(
    "测试权重：",
    weight_test.reshape(-1),
)

print(
    "\n真实数据批次读取：通过"
)

print(
    "雷达输入中存在NaN可能是正常现象，"
    "模型内部会处理。"
)


# In[5]:


inference_model = (
    keras.saving.load_model(
        str(MODEL_PATH),
        compile=False,
    )
)

print("官方预训练模型加载成功")

print(
    "模型参数量：",
    inference_model.count_params(),
)


# ==================================================
# 检查模型输出
# Keras模型即使只有一个输出，也可能用list保存
# ==================================================

model_outputs = list(
    inference_model.outputs
)

print(
    "模型输出数量：",
    len(model_outputs),
)

for index, output_tensor in enumerate(
    model_outputs
):
    print(
        f"输出{index}名称：",
        output_tensor.name,
    )

    print(
        f"输出{index}形状：",
        tuple(output_tensor.shape),
    )


# ==================================================
# 定义统一提取单个模型输出的函数
# 后面的原生推理和FlagGems推理都会使用
# ==================================================

def unwrap_single_output(value):
    """
    将只包含一个元素的list、tuple或dict模型输出，
    转换为其中的单个张量。
    """

    if isinstance(value, dict):
        if len(value) != 1:
            raise RuntimeError(
                "模型返回多个命名输出："
                f"{list(value.keys())}"
            )

        value = next(
            iter(value.values())
        )

    if isinstance(
        value,
        (list, tuple),
    ):
        if len(value) != 1:
            raise RuntimeError(
                "模型返回多个输出，"
                f"输出数量为{len(value)}。"
                "当前流程只支持单输出分类模型。"
            )

        value = value[0]

    return value


# ==================================================
# 检查模型输入
# ==================================================

if isinstance(
    inference_model.input,
    dict,
):
    model_input_names = list(
        inference_model.input.keys()
    )

else:
    model_input_names = [
        tensor.name.split(":")[0]

        for tensor
        in inference_model.inputs
    ]


print(
    "模型输入数量：",
    len(model_input_names),
)

print(
    "模型输入名称：",
    model_input_names,
)


missing_model_inputs = (
    set(model_input_names)
    - set(x_test.keys())
)

assert not missing_model_inputs, (
    "测试批次缺少模型输入："
    + ", ".join(
        sorted(missing_model_inputs)
    )
)


print(
    "测试批次字段：",
    list(x_test.keys()),
)

print(
    "\n模型与数据输入匹配：通过"
)


# In[6]:


import time

assert not globals().get(
    "_FLAGGEMS_ENABLED",
    False,
), (
    "本单元必须在FlagGems启用前运行"
)


keras.utils.set_random_seed(2026)
torch.manual_seed(2026)


# ==================================================
# 原生PyTorch模型预热
# ==================================================

native_warmup = inference_model(
    x_test,
    training=False,
)

synchronize_device()


# ==================================================
# 原生PyTorch正式推理
# ==================================================

native_start = time.perf_counter()

native_prediction_raw = (
    inference_model(
        x_test,
        training=False,
    )
)

synchronize_device()

native_elapsed = (
    time.perf_counter()
    - native_start
)


# 模型输出可能是单元素list
native_prediction = (
    unwrap_single_output(
        native_prediction_raw
    )
)

native_prediction_np = to_numpy(
    native_prediction
)


assert np.isfinite(
    native_prediction_np
).all(), (
    "原生PyTorch模型输出含NaN或Inf"
)


print(
    "原生PyTorch预训练模型推理：通过"
)

print(
    "输出形状：",
    native_prediction_np.shape,
)

print(
    "输出值：",
    native_prediction_np.reshape(-1),
)

print(
    "推理时间：",
    native_elapsed,
    "秒",
)


# ==================================================
# 原生PyTorch矩阵乘基准
# ==================================================

torch.manual_seed(2026)

mm_a = torch.randn(
    (256, 256),
    dtype=torch.float32,
    device=flag_gems.device,
)

mm_b = torch.randn(
    (256, 256),
    dtype=torch.float32,
    device=flag_gems.device,
)


native_mm = torch.mm(
    mm_a,
    mm_b,
)

synchronize_device()

native_mm_np = to_numpy(
    native_mm
)


print(
    "\n原生PyTorch矩阵乘基准：完成"
)


# In[7]:


if globals().get(
    "_FLAGGEMS_ENABLED",
    False,
):
    raise RuntimeError(
        "当前Kernel已经启用过FlagGems。"
        "请重启Kernel并从单元格1开始。"
    )


UNUSED_OPS = [
    # TorNet模型中暂时回退的归一化算子
    "batch_norm",
    "batch_norm_backward",

    # Hygon后端已确认有问题的比较辅助算子
    "isclose",
    "all",

    # 当前测试发现：FlagGems conv2d反向Triton编译失败
    "conv2d",

    # 当前日志提示缺少Autograd注册
    # 为避免梯度静默错误，一并回退
    "square",
    "divide",
]


# 删除旧日志，避免与本次测试混在一起
if GEMS_LOG_PATH.exists():
    GEMS_LOG_PATH.unlink()


# 整个Notebook唯一一次启用FlagGems
flag_gems.enable(
    unused=UNUSED_OPS,
    record=True,
    path=str(GEMS_LOG_PATH),
    once=True,
)


# 只有enable成功后才设置标志
_FLAGGEMS_ENABLED = True


print("FlagGems已启用")

print(
    "回退到原生PyTorch的算子：",
    UNUSED_OPS,
)

print(
    "日志路径：",
    GEMS_LOG_PATH,
)


# ==================================================
# 第一部分：FlagGems矩阵乘
# ==================================================

gems_mm = torch.mm(
    mm_a,
    mm_b,
)

synchronize_device()

gems_mm_np = to_numpy(
    gems_mm
)


mm_abs_error = np.abs(
    gems_mm_np
    - native_mm_np
)

mm_max_abs_error = float(
    np.max(mm_abs_error)
)

mm_mean_abs_error = float(
    np.mean(mm_abs_error)
)


# 在CPU NumPy中比较
# 避免调用FlagGems isclose
np.testing.assert_allclose(
    gems_mm_np,
    native_mm_np,
    rtol=1e-3,
    atol=1e-4,
    equal_nan=False,
)


print(
    "\nFlagGems矩阵乘：通过"
)

print(
    "最大绝对误差：",
    mm_max_abs_error,
)

print(
    "平均绝对误差：",
    mm_mean_abs_error,
)


# ==================================================
# 第二部分：FlagGems官方模型推理
# ==================================================

# 第一次FlagGems模型运行可能包含Triton编译
gems_warmup = inference_model(
    x_test,
    training=False,
)

synchronize_device()


# 第二次运行记录结果
gems_start = time.perf_counter()

gems_prediction_raw = (
    inference_model(
        x_test,
        training=False,
    )
)

synchronize_device()

gems_elapsed = (
    time.perf_counter()
    - gems_start
)

gems_prediction = (
    unwrap_single_output(
        gems_prediction_raw
    )
)

gems_prediction_np = to_numpy(
    gems_prediction
)

assert np.isfinite(
    gems_prediction_np
).all(), (
    "FlagGems模型输出含NaN或Inf"
)


# ==================================================
# 第三部分：比较原生与FlagGems模型输出
# ==================================================

model_abs_error = np.abs(
    gems_prediction_np
    - native_prediction_np
)

model_max_abs_error = float(
    np.max(model_abs_error)
)

model_mean_abs_error = float(
    np.mean(model_abs_error)
)


np.testing.assert_allclose(
    gems_prediction_np,
    native_prediction_np,
    rtol=1e-3,
    atol=1e-4,
    equal_nan=False,
)


print(
    "\nFlagGems官方预训练模型推理：通过"
)

print(
    "原生PyTorch输出：",
    native_prediction_np.reshape(-1),
)

print(
    "FlagGems输出：",
    gems_prediction_np.reshape(-1),
)

print(
    "最大绝对误差：",
    model_max_abs_error,
)

print(
    "平均绝对误差：",
    model_mean_abs_error,
)

print(
    "FlagGems推理时间：",
    gems_elapsed,
    "秒",
)

print(
    "\n原生PyTorch与FlagGems前向精度比较：通过"
)


# In[8]:


assert globals().get(
    "_FLAGGEMS_ENABLED",
    False,
), (
    "必须先执行FlagGems启用单元"
)


keras.utils.set_random_seed(2027)
torch.manual_seed(2027)


input_shape = (
    None,
    None,
    get_shape(x_train)[-1],
)

coordinate_shape = (
    None,
    None,
    x_train[
        "coordinates"
    ].shape[-1],
)


print(
    "雷达输入形状参数：",
    input_shape,
)

print(
    "坐标输入形状参数：",
    coordinate_shape,
)


training_model = build_model(
    shape=input_shape,
    c_shape=coordinate_shape,
    input_variables=ALL_VARIABLES,
    start_filters=48,
    l2_reg=1e-5,
    head="maxpool",
)


training_model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-4,
    ),

    loss=keras.losses.BinaryCrossentropy(
        from_logits=True,
    ),
)


print(
    "训练模型参数量：",
    training_model.count_params(),
)


# ==================================================
# 先执行一次前向，使模型权重完成初始化
# ==================================================

initial_output_raw = training_model(
    x_train,
    training=False,
)

initial_output = unwrap_single_output(
    initial_output_raw
)

synchronize_device()


# 保存第一个可训练参数的训练前副本
assert training_model.trainable_weights, (
    "训练模型没有可训练参数"
)

first_weight_before = to_numpy(
    training_model.trainable_weights[0].value
)


# ==================================================
# 执行一次训练
# ==================================================

train_start = time.perf_counter()


train_result = (
    training_model.train_on_batch(
        x_train,
        y_train,
        sample_weight=weight_train,
        return_dict=True,
    )
)


synchronize_device()

train_elapsed = (
    time.perf_counter()
    - train_start
)


train_metrics = {}


print("\n单批次训练结果")
print("=" * 72)


for (
    metric_name,
    metric_value,
) in train_result.items():

    value = float(
        np.asarray(metric_value)
    )

    train_metrics[
        metric_name
    ] = value

    print(
        f"{metric_name}：{value}"
    )

    assert np.isfinite(value), (
        f"{metric_name}出现NaN或Inf"
    )


# ==================================================
# 检查参数是否发生更新
# ==================================================

first_weight_after = to_numpy(
    training_model.trainable_weights[0].value
)


weight_change = np.abs(
    first_weight_after
    - first_weight_before
)

max_weight_change = float(
    np.max(weight_change)
)

mean_weight_change = float(
    np.mean(weight_change)
)

changed_element_count = int(
    np.count_nonzero(weight_change)
)


print(
    "\n首个可训练参数名称：",
    training_model.trainable_weights[0].name,
)

print(
    "参数最大变化：",
    max_weight_change,
)

print(
    "参数平均变化：",
    mean_weight_change,
)

print(
    "发生变化的元素数量：",
    changed_element_count,
)


assert np.isfinite(
    max_weight_change
), (
    "参数变化中含NaN或Inf"
)

assert changed_element_count > 0, (
    "训练后参数没有发生变化，"
    "不能确认优化器完成了参数更新。"
)


print(
    "\n单批次训练时间：",
    train_elapsed,
    "秒",
)

print(
    "\n前向传播、损失计算、"
    "反向传播和Adam参数更新：通过"
)


# In[9]:


from tqdm.auto import tqdm
import numpy as np
import time


# 无论是否执行完整测试，
# 都先定义这个变量，保证最终摘要单元可以正常运行
full_test_metrics = None


if RUN_FULL_2013_TEST is False:

    # ==================================================
    # 快速验证模式
    # ==================================================

    print(
        "RUN_FULL_2013_TEST=False"
    )

    print(
        "当前只进行快速验证，"
        "跳过完整2013测试集。"
    )

    print(
        "前面的单批次推理、精度比较、"
        "反向传播和日志测试结果保持有效。"
    )


else:

    # ==================================================
    # 完整2013测试模式
    # ==================================================

    print(
        "RUN_FULL_2013_TEST=True"
    )

    print(
        "开始运行完整2013测试集。"
    )


    assert globals().get(
        "_FLAGGEMS_ENABLED",
        False,
    ), (
        "当前Kernel还没有启用FlagGems。"
        "请从前面的单元格开始顺序运行。"
    )


    assert len(test_loader) > 0, (
        "2013测试加载器为空。"
    )


    all_test_logits = []
    all_test_labels = []


    print(
        "测试批次数：",
        len(test_loader),
    )

    print(
        "批大小：",
        BATCH_SIZE,
    )

    print(
        "当前回退算子：",
        UNUSED_OPS,
    )


    full_test_start = (
        time.perf_counter()
    )


    # ==================================================
    # 遍历完整2013测试集
    # ==================================================

    for batch_index in tqdm(
        range(len(test_loader)),
        desc="TorNet 2013完整测试",
    ):

        batch = test_loader[
            batch_index
        ]


        # 兼容加载器返回：
        # (x, y, sample_weight)
        # 或(x, y)
        if len(batch) == 3:

            (
                batch_x,
                batch_y,
                _,
            ) = batch

        elif len(batch) == 2:

            (
                batch_x,
                batch_y,
            ) = batch

        else:

            raise RuntimeError(
                f"第{batch_index}批返回了"
                f"{len(batch)}个对象，"
                "无法识别。"
            )


        # 官方模型可能返回只包含一个张量的list
        batch_prediction_raw = (
            inference_model(
                batch_x,
                training=False,
            )
        )


        # 从list中提取真正的输出张量
        batch_prediction = (
            unwrap_single_output(
                batch_prediction_raw
            )
        )


        # 转成CPU NumPy
        prediction_np = to_numpy(
            batch_prediction
        ).reshape(-1)


        label_np = np.asarray(
            batch_y,
            dtype=np.float32,
        ).reshape(-1)


        assert np.isfinite(
            prediction_np
        ).all(), (
            f"第{batch_index}批预测值"
            "出现NaN或Inf。"
        )


        assert np.isfinite(
            label_np
        ).all(), (
            f"第{batch_index}批标签"
            "出现NaN或Inf。"
        )


        assert (
            len(prediction_np)
            == len(label_np)
        ), (
            f"第{batch_index}批预测数量"
            "与标签数量不一致。"
        )


        all_test_logits.append(
            prediction_np
        )

        all_test_labels.append(
            label_np
        )


    synchronize_device()


    full_test_elapsed = (
        time.perf_counter()
        - full_test_start
    )


    # ==================================================
    # 合并全部批次
    # ==================================================

    all_test_logits = np.concatenate(
        all_test_logits
    )

    all_test_labels = np.concatenate(
        all_test_labels
    )


    assert (
        len(all_test_logits)
        == len(all_test_labels)
    )


    # 标签转为0和1
    binary_labels = (
        all_test_labels >= 0.5
    ).astype(np.int64)


    # 模型输出是logit
    # logit >= 0等价于概率 >= 0.5
    predicted_labels = (
        all_test_logits >= 0.0
    ).astype(np.int64)


    # 防止计算sigmoid时数值溢出
    clipped_logits = np.clip(
        all_test_logits,
        -60.0,
        60.0,
    )


    test_probabilities = (
        1.0
        / (
            1.0
            + np.exp(
                -clipped_logits
            )
        )
    )


    # ==================================================
    # 混淆矩阵
    # ==================================================

    tp = int(
        np.sum(
            (predicted_labels == 1)
            & (binary_labels == 1)
        )
    )


    tn = int(
        np.sum(
            (predicted_labels == 0)
            & (binary_labels == 0)
        )
    )


    fp = int(
        np.sum(
            (predicted_labels == 1)
            & (binary_labels == 0)
        )
    )


    fn = int(
        np.sum(
            (predicted_labels == 0)
            & (binary_labels == 1)
        )
    )


    accuracy_2013 = float(
        np.mean(
            predicted_labels
            == binary_labels
        )
    )


    precision_2013 = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else float("nan")
    )


    recall_2013 = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else float("nan")
    )


    f1_2013 = (
        2.0
        * precision_2013
        * recall_2013
        / (
            precision_2013
            + recall_2013
        )
        if (
            np.isfinite(
                precision_2013
            )
            and np.isfinite(
                recall_2013
            )
            and (
                precision_2013
                + recall_2013
            ) > 0
        )
        else float("nan")
    )


    # ==================================================
    # ROC AUC
    # ==================================================

    def binary_roc_auc(
        y_true,
        scores,
    ):

        y_true = np.asarray(
            y_true,
            dtype=np.int64,
        )

        scores = np.asarray(
            scores,
            dtype=np.float64,
        )


        positive_mask = (
            y_true == 1
        )


        n_positive = int(
            np.sum(
                positive_mask
            )
        )


        n_negative = int(
            len(y_true)
            - n_positive
        )


        if (
            n_positive == 0
            or n_negative == 0
        ):
            return float("nan")


        order = np.argsort(
            scores,
            kind="mergesort",
        )


        sorted_scores = scores[
            order
        ]


        ranks = np.empty(
            len(scores),
            dtype=np.float64,
        )


        index = 0


        while index < len(scores):

            next_index = (
                index + 1
            )


            while (
                next_index < len(scores)
                and sorted_scores[
                    next_index
                ]
                == sorted_scores[
                    index
                ]
            ):
                next_index += 1


            average_rank = (
                (index + 1)
                + next_index
            ) / 2.0


            ranks[
                order[
                    index:next_index
                ]
            ] = average_rank


            index = next_index


        positive_rank_sum = float(
            np.sum(
                ranks[
                    positive_mask
                ]
            )
        )


        auc = (
            positive_rank_sum
            - (
                n_positive
                * (
                    n_positive + 1
                )
                / 2.0
            )
        ) / (
            n_positive
            * n_negative
        )


        return float(auc)


    # ==================================================
    # Average Precision
    # ==================================================

    def binary_average_precision(
        y_true,
        scores,
    ):

        y_true = np.asarray(
            y_true,
            dtype=np.int64,
        )

        scores = np.asarray(
            scores,
            dtype=np.float64,
        )


        positive_count = int(
            np.sum(
                y_true == 1
            )
        )


        if positive_count == 0:
            return float("nan")


        order = np.argsort(
            -scores,
            kind="mergesort",
        )


        sorted_labels = y_true[
            order
        ]


        cumulative_positive = (
            np.cumsum(
                sorted_labels == 1
            )
        )


        precision_at_rank = (
            cumulative_positive
            / np.arange(
                1,
                len(sorted_labels) + 1,
            )
        )


        return float(
            np.sum(
                precision_at_rank
                * (
                    sorted_labels == 1
                )
            )
            / positive_count
        )


    roc_auc_2013 = binary_roc_auc(
        binary_labels,
        all_test_logits,
    )


    average_precision_2013 = (
        binary_average_precision(
            binary_labels,
            all_test_logits,
        )
    )


    # ==================================================
    # 保存结果
    # ==================================================

    full_test_metrics = {
        "测试批次数":
            int(len(test_loader)),

        "样本数量":
            int(len(binary_labels)),

        "正例数量":
            int(
                np.sum(
                    binary_labels == 1
                )
            ),

        "负例数量":
            int(
                np.sum(
                    binary_labels == 0
                )
            ),

        "Accuracy":
            accuracy_2013,

        "Precision":
            precision_2013,

        "Recall":
            recall_2013,

        "F1":
            f1_2013,

        "ROC AUC":
            roc_auc_2013,

        "Average Precision":
            average_precision_2013,

        "TP":
            tp,

        "TN":
            tn,

        "FP":
            fp,

        "FN":
            fn,

        "最小预测概率":
            float(
                np.min(
                    test_probabilities
                )
            ),

        "最大预测概率":
            float(
                np.max(
                    test_probabilities
                )
            ),

        "测试耗时秒":
            full_test_elapsed,
    }


    print(
        "\nTorNet 2013完整测试结果"
    )

    print("=" * 72)


    for (
        metric_name,
        metric_value,
    ) in full_test_metrics.items():

        print(
            f"{metric_name}："
            f"{metric_value}"
        )


    print(
        "\n2013年完整测试集推理：通过"
    )


# In[10]:


import time

# 等待日志写入完成
time.sleep(1)


assert GEMS_LOG_PATH.is_file(), (
    "没有生成FlagGems日志："
    f"{GEMS_LOG_PATH}"
)


log_text = (
    GEMS_LOG_PATH.read_text(
        encoding="utf-8",
        errors="replace",
    )
)


log_lines = [
    line
    for line in log_text.splitlines()
    if line.strip()
]


print(
    "日志路径：",
    GEMS_LOG_PATH,
)

print(
    "日志大小：",
    GEMS_LOG_PATH.stat().st_size,
    "bytes",
)

print(
    "非空日志行数：",
    len(log_lines),
)


assert log_lines, (
    "FlagGems日志为空"
)


print("\n前100条日志")
print("=" * 72)

for line in log_lines[:100]:
    print(line)


operator_keywords = [
    "conv",
    "mm",
    "matmul",
    "relu",
    "max_pool",
    "maxpool",
    "add",
    "mul",
    "div",
    "sum",
]


matched_lines = [
    line

    for line in log_lines

    if any(
        keyword in line.lower()

        for keyword
        in operator_keywords
    )
]


print(
    "\n匹配到的主要算子日志数量：",
    len(matched_lines),
)


for line in matched_lines[:100]:
    print(line)


if not matched_lines:
    print(
        "\n日志非空，但关键词没有匹配到"
        "常见算子。请人工查看上面的日志，"
        "不要直接判断测试失败。"
    )


# In[11]:


final_summary = {
    "测试模型":
        "TorNet CNN baseline",

    "模型权重":
        str(MODEL_PATH),

    "数据范围":
        "TorNet v1.1 2013年",

    "Keras后端":
        keras.config.backend(),

    "Python版本":
        platform.python_version(),

    "PyTorch版本":
        torch.__version__,

    "Triton版本":
        getattr(
            triton,
            "__version__",
            "unknown",
        ),

    "FlagGems版本":
        getattr(
            flag_gems,
            "__version__",
            package_version(
                "flag-gems"
            ),
        ),

    "FlagGems厂商":
        getattr(
            flag_gems,
            "vendor_name",
            "unknown",
        ),

    "设备":
        torch.cuda.get_device_name(0),

    "回退算子":
        UNUSED_OPS,

    "矩阵乘最大绝对误差":
        mm_max_abs_error,

    "矩阵乘平均绝对误差":
        mm_mean_abs_error,

    "模型输出最大绝对误差":
        model_max_abs_error,

    "模型输出平均绝对误差":
        model_mean_abs_error,

    "单批次训练结果":
        train_metrics,

    "完整2013测试结果":
        full_test_metrics,

    "FlagGems日志":
        str(GEMS_LOG_PATH),

    "FlagGems日志行数":
        len(log_lines),
}


print(
    "TorNet 2013 FlagGems适配测试摘要"
)

print("=" * 72)


for key, value in final_summary.items():
    print(
        f"{key}：{value}"
    )


print("\n结论：")

print(
    "本次仅使用TorNet v1.1的2013年数据，"
    "验证Keras 3 PyTorch后端上的TorNet CNN"
    "在FlagGems环境中的真实数据加载、"
    "前向推理、精度一致性和单批次"
    "反向传播兼容性。"
)

print(
    "该结果不代表2013—2022完整数据集上的"
    "官方benchmark复现结果。"
)


# In[12]:


# 若想在快速验证或完整测试模式间进行切换，请您在第一个单元格手动更改RUN_FULL_2013_TEST的Bool值，False开启快速验证，True开启完整测试
# 更换完后需要在菜单栏内选择Kernel→Restart Kernel，并且从首个单元格顺序运行到最后

