import time

from playwright.sync_api import sync_playwright
from sympy.concrete import products

GAME_URL = "https://yandex.ru/games/app/cookie-clicker-old-238578"

class CookieClickerEnv:
    def __init__(self, headless=False):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=headless)
        self.page = self.browser.new_page()
        self.page.goto( GAME_URL, wait_until="domcontentloaded" )
        self.page.wait_for_timeout(5000)
        self.frame = next(
        frame for frame in self.page.frames
        if "app-238578" in frame.url
        )

    def get_state(self):
        import re

        text = self.observe()

        def read_number(pattern):
            match = re.search(pattern, text, re.IGNORECASE)

            if not match:
                return 0.0

            number = match.group(1).replace(",", "").replace(" ", "")

            try:
                return float(number)
            except ValueError:
                return 0.0

        cookies = read_number(r"печенья:\s*([\d., ]+)")
        cps = read_number(r"в секунду:\s*([\d., ]+)")

        products = self.frame.locator("#store [id^='product']")
        first_price = 0.0
        can_buy = 0.0

        if products.count() > 0:
            item = products.first

            if item.is_visible():
                classes = item.get_attribute("class") or ""
                can_buy = float("enabled" in classes.split())

                price_text = item.locator(".price").inner_text() \
                    if item.locator(".price").count() > 0 else ""

                price_match = re.search(r"[\d., ]+", price_text)

                if price_match:
                    try:
                        first_price = float(
                            price_match.group().replace(",", "").replace(" ", "")
                        )
                    except ValueError:
                        pass

        return [cookies, cps, first_price, can_buy]

    def observe(self):
        return self.frame.locator("body").inner_text()

    def click_cookie(self, count=1):
        cookie = self.frame.locator("#bigCookie")
        for _ in range(count):
            try:
                cookie.click(timeout=1000)
            except Exception as e:
                print(f"Не удалось нажать на печенье: {e}")
                return False

        return True

    def is_cookie_blocked(self):
        cookie = self.frame.locator("#bigCookie")

        try:
            box = cookie.bounding_box()
            if box is None:
                return True

            return not cookie.is_visible() or not cookie.is_enabled()
        except Exception:
            return True

    def buy_upgrade(self, index=0):
        products = self.frame.locator("#store [id^='product']")

        if index < 0 or index >= products.count():
            return False

        item = products.nth(index)

        if not item.is_visible():
            return False

        item.click(timeout=1000)
        return True

    def get_products(self):
        products = self.frame.locator("#store [id^='product']")
        result = []
        for i in range(products.count()):
            item = products.nth(i)

            if not item.is_visible():
                continue

            price_element = item.locator(".price").first
            price_text = (
                price_element.inner_text()
                if price_element.count() > 0
                else ""
            )

            price = self.parse_cookie_number(price_text)
            classes = (item.get_attribute("class") or "").split()

            result.append({
                "index": i,
                "price": price,
                "can_buy": "enabled" in classes,
            })

        return result

    def parse_cookie_number(self, text):
        import re

        text = text.strip().lower().replace("\xa0", " ")
        text = text.replace(",", ".")

        match = re.search(r"([\d.]+)\s*(тысяч[а-я]*|миллион[а-я]*|миллиард[а-я]*)?", text)

        if not match:
            return 0.0

        value = float(match.group(1))
        suffix = match.group(2) or ""

        if suffix.startswith("тысяч"):
            value *= 1_000
        elif suffix.startswith("миллион"):
            value *= 1_000_000
        elif suffix.startswith("миллиард"):
            value *= 1_000_000_000

        return value

    def buy_product(self, index):
        products = self.frame.locator("#store [id^='product']")

        if index < 0 or index >= products.count():
            return False

        item = products.nth(index)

        if not item.is_visible():
            return False

        classes = (item.get_attribute("class") or "").split()

        if "enabled" not in classes:
            return False

        try:
            item.click(timeout=1000)
            return True

        except Exception:
            # Реклама могла перекрыть магазин
            self.close_ad()
            time.sleep(0.5)

            try:
                # Проверяем, что товар всё ещё доступен
                if not item.is_visible():
                    return False

                classes = (item.get_attribute("class") or "").split()

                if "enabled" not in classes:
                    return False

                item.click(timeout=2000)
                return True

            except Exception as e:
                print(f"Не удалось купить товар: {e}")
                return False

    def close(self):
        self.browser.close()
        self.playwright.stop()

    def wait_for_ad_warning(self, timeout=30000):
        warning = self.frame.locator("#yandexAdWarning")
        try:
            warning.wait_for(state="hidden", timeout=timeout)
            return True
        except Exception:
            print("Реклама или предупреждение всё ещё перекрывает игру.")
            return False


    def close_ad_by_image(self):
        import cv2
        import numpy as np
        template = cv2.imread("cross_template.png")

        if template is None:
            print("ОШИБКА: шаблон не найден!")
            return False

        try:
            screenshot = self.frame.locator("body").screenshot()
            screen = cv2.imdecode(
                np.frombuffer(screenshot, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )

            if (template.shape[0] > screen.shape[0]
                    or template.shape[1] > screen.shape[1]):
                print("Шаблон крестика больше игрового окна.")
                return False

            result = cv2.matchTemplate(
                screen, template, cv2.TM_CCOEFF_NORMED
            )

            _, confidence, _, location = cv2.minMaxLoc(result)
            print(f"Совпадение крестика: {confidence:.3f}")

            if confidence < 0.80:
                print("Крестик не найден уверенно.")
                return False

            h, w = template.shape[:2]
            x = location[0] + w // 2
            y = location[1] + h // 2

            self.frame.mouse.click(x, y)
            print(f"Нажатие внутри iframe: ({x}, {y})")
            return True

        except Exception as e:
            print(f"Не удалось закрыть рекламу: {e}")
            return False

    def close_ad(self):
        buttons = self.page.locator('button.close-button_type_adv-fullscreen[aria-label="Закрыть"]')
        try:
            count = buttons.count()
            print(f"Найдено кнопок закрытия: {count}")

            if count == 0:
                return False

            button = buttons.first
            print("Видна:", button.is_visible())
            print("Доступна:", button.is_enabled())

            if not button.is_visible():
                return False

            button.click(timeout=3000)
            print("Нажатие выполнено.")
            return True

        except Exception as e:
            print(f"Ошибка закрытия рекламы: {e}")
            return False

    def debug_ad_frames(self):
        print("\n--- Проверка рекламы ---")

        try:
            print(
                "Рекламный слой:",
                self.page.locator(".aab-adv").evaluate_all("""
                                                           els => els.map(el => ({
                                                               className: el.className,
                                                               visible: !!(el.offsetWidth || el.offsetHeight),
                                                               text: (el.innerText || "").slice(0, 150),
                                                               buttons: [...el.querySelectorAll(
                                                                   'button, [role="button"], [aria-label], [title]'
                                                               )].map(b => ({
                                                                   tag: b.tagName,
                                                                   text: (b.innerText || "").trim(),
                                                                   aria: b.getAttribute("aria-label"),
                                                                   title: b.getAttribute("title"),
                                                                   className: String(b.className).slice(0, 100)
                                                               })).slice(0, 30)
                                                           }))
                                                           """)
            )
        except Exception as e:
            print("Ошибка проверки рекламного слоя:", e)
        try:
            info = self.page.evaluate("""() => ({
                overlays: [...document.querySelectorAll(
                    '.aab-adv, #yandexAdWarning, [class*="ad"], [id*="ad"]'
                )].filter(el => {
                    const r = el.getBoundingClientRect();
                    return r.width > 0 && r.height > 0;
                }).map(el => {
                    const r = el.getBoundingClientRect();
                    return {
                        tag: el.tagName,
                        id: el.id,
                        className: String(el.className).slice(0, 120),
                        text: (el.innerText || '').slice(0, 80),
                        x: Math.round(r.x),
                        y: Math.round(r.y),
                        width: Math.round(r.width),
                        height: Math.round(r.height)
                    };
                }).slice(0, 30)
            })""")
            print("\nОсновная страница:", info)
        except Exception as e:
            print("Ошибка основной страницы:", e)

        # Проверяем игровые и рекламные фреймы
        for i, frame in enumerate(self.page.frames):
            try:
                info = frame.evaluate("""() => ({
                    url: location.href,
                    title: document.title,
                    buttons: [...document.querySelectorAll(
                        'button, [aria-label], [title], [class*="close"], [id*="close"]'
                    )].filter(el => {
                        const r = el.getBoundingClientRect();
                        return r.width > 0 && r.height > 0;
                    }).map(el => ({
                        tag: el.tagName,
                        id: el.id,
                        className: String(el.className).slice(0, 100),
                        text: (el.innerText || '').slice(0, 60),
                        aria: el.getAttribute('aria-label'),
                        title: el.getAttribute('title')
                    })).slice(0, 30)
                })""")
                print(f"\nФрейм {i}: {info}")
            except Exception as e:
                print(f"\nФрейм {i}: ошибка проверки: {e}")

    def get_state(self):
        import re

        text = self.observe()

        def read_number(pattern):
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                return 0.0

            try:
                return float(match.group(1).replace(",", "."))
            except ValueError:
                return 0.0

        cookies = read_number(r"печенья:\s*([\d.,]+)")
        cps = read_number(r"в секунду:\s*([\d.,]+)")

        products = self.frame.locator("#store [id^='product']")
        prices = []
        can_buy = []

        for i in range(min(products.count(), 2)):
            item = products.nth(i)
            classes = (item.get_attribute("class") or "").split()

            price_text = ""
            price_element = item.locator(".price").first

            if price_element.count() > 0:
                price_text = price_element.inner_text()

            match = re.search(r"[\d., ]+", price_text)
            price = 0.0

            if match:
                try:
                    price = float(
                        match.group().replace(",", "").replace(" ", "")
                    )
                except ValueError:
                    pass
            prices.append(price)
            can_buy.append(
                float(item.is_visible() and "enabled" in classes)
            )
        while len(prices) < 2:
            prices.append(0.0)
            can_buy.append(0.0)

        self.last_prices = prices

        return [cookies, cps, can_buy[0], can_buy[1]]

    def debug_first_product(self):
        item = self.frame.locator("#store [id^='product']").first

        print("ID:", item.get_attribute("id"))
        print("HTML:")
        print(item.evaluate("(element) => element.outerHTML"))

    def observe(self):
        return self.frame.locator("body").inner_text()

    def get_store_text(self):
        return self.frame.locator("#store").inner_text()

    def inspect_store(self):
        store = self.frame.locator("#store")
        print("Магазин найден:", store.count())
        print("Текст магазина:")
        print(store.inner_text()[:1500])

        print("\nЭлементы магазина:")
        items = store.locator("[id^='product']")
        for i in range(min(items.count(), 10)):
            item = items.nth(i)
            print(item.get_attribute("id"), item.inner_text())

    def buy_upgrade(self, index=0):
        item = self.frame.locator("#store [id^='product']").nth(index)

        if item.count() == 0:
            print("Улучшение не найдено")
            return False

        if not item.is_visible():
            print("Улучшение не видно")
            return False

        item.click()
        print("Нажали на улучшение:", item.get_attribute("id"))
        return True

    def close(self):
        self.browser.close()
        self.playwright.stop()

if __name__ == "__main__":
    env = CookieClickerEnv(headless=False)
    try:
        print("Состояние игры:")
        print(env.observe()[:500])
        env.inspect_store()
        env.click_cookie(0)
        env.buy_upgrade(0)
        env.debug_first_product()

        print("\nПосле пяти кликов:")
        print(env.observe())
        print("\n=====Store=====")
        print(env.inspect_store())
        print("Cостояние для нейросити:", env.get_state())

        input("\nНажми Enter для завершения...")
    finally:
        env.close()