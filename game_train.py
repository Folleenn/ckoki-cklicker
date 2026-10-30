import math
import time
from pathlib import Path
import torch
from game_env import CookieClickerEnv
from game_brain import DQNAgent

MODEL_PATH = Path("komorebi_brain_old.pth")
TOTAL_STEPS = 20000

def prepare_state(raw_state):
    cookies, cps, can_buy_1, can_buy_2 = raw_state

    return [
        math.log1p(max(0, cookies)) / 20,
        math.log1p(max(0, cps)) / 10,
        float(can_buy_1),
        float(can_buy_2),
    ]
def main():
    env = CookieClickerEnv(headless=False)
    print("bimbim")
    env.close_ad()
    agent = DQNAgent(state_size=4, action_size=3)

    try:
        if MODEL_PATH.exists():
            weights = torch.load(
                MODEL_PATH,
                map_location="cpu",
                weights_only=True,
            )
            agent.brain.load_state_dict(weights)
            print("Загрузили предыдущие веса Komorebi!")

        print("Начинаем обучение Komorebi!")
        print("Действия: 0 — клик, 1 — первый товар, 2 — второй товар.")
        print(f"Количество шагов: {TOTAL_STEPS}")

        raw_state = env.get_state()
        state = prepare_state(raw_state)
        old_cps = raw_state[1]

        for step in range(TOTAL_STEPS):
            env.close_ad()
            products = env.get_products()
            available = sorted(
                [p for p in products if p["can_buy"] and p["price"] > 0],
                key=lambda p: p["price"]
            )
            action = agent.choose_action(state)
            if action == 0 or not available:
                env.click_cookie(15)
            elif action == 1:
                if not env.wait_for_ad_warning():
                    time.sleep(2)
                    continue
                if not env.buy_product(available[0]["index"]):
                    env.click_cookie(10)
            elif action == 2:
                target = available[1] if len(available) >= 2 else available[0]

                if not env.buy_product(target["index"]):
                    env.click_cookie(10)


            time.sleep(0.01)

            raw_next_state = env.get_state()
            next_state = prepare_state(raw_next_state)
            new_cps = raw_next_state[1]

            reward = (new_cps - old_cps) / max(1, old_cps + 1)
            reward += 0.01 * (next_state[0] - state[0])

            agent.remember(
                state, action, reward, next_state, False
            )

            loss = agent.train_step(batch_size=8)

            print(
                f"Шаг {step + 1}/{TOTAL_STEPS} | "
                f"Действие: {action} | "
                f"Печенья: {raw_next_state[0]:.1f} | "
                f"Доход/сек: {new_cps:.1f} | "
                f"Награда: {reward:.4f} | "
                f"Ошибка: {loss} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Состояние: {state}"
            )

            state = next_state
            old_cps = new_cps

        torch.save(agent.brain.state_dict(), MODEL_PATH)

        print("\nОбучение завершено.")
        print("Случайность выбора:", round(agent.epsilon, 3))
        print(f"Модель сохранена: {MODEL_PATH}")

    finally:
        env.close()

if __name__ == "__main__":
    main()
