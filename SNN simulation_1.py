import torch
import matplotlib.pyplot as plt

#LIT 뉴런 파라미터
beta=0.5 #leak 정도
threshold = 1.0 #spike 임계값

#초기 막전위
mem = torch.zeros(1)

#input spike
inputs = torch.tensor([0.3, 0.5, 0.8, 0.2, 0.9, 0.1, 0.0, 0.4, 0.7, 0.6, 0.3, 0.8, 0.5, 0.2, 0.9], dtype=torch.float)

mem_before_reset_history = []  # 리셋 전 막전위

mem_history = []
spk_history = []

for t in range(len(inputs)):
    #1. leak + intergrate
    mem = beta * mem + inputs[t]

    #1.5. 리셋 전 값 저장
    mem_before_reset_history.append(mem.item())

    #2. fire
    spk = (mem >= threshold).float()

    #3. reset
    mem = mem * (1-spk)

    mem_history.append(mem.item())
    spk_history.append(spk.item())

# 시각화
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (10, 6))

ax1.plot(range(len(mem_history)), mem_history, 'b-o', label='리셋 후')
ax1.plot(range(len(mem_before_reset_history)), mem_before_reset_history, 'g--o', label='리셋 전')
ax1.legend()

ax2.bar(range(len(spk_history)), spk_history, color = 'orange')
ax2.set_ylabel('spike')
ax2.set_xlabel('time step')

plt.tight_layout()
plt.show()

print ("스파이크 발생 시점:",[i for i, s in enumerate(spk_history) if s == 1.0])