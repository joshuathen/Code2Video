from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section4Scene(TeachingScene):
    def construct(self):
        title = "The Brute Force Sisyphus"
        lines = [
            "Brute force means trying every single combination.",
            "Even billions of supercomputers would take trillions of years.",
            "Time itself is too short to crack this code."
        ]
        self.setup_layout(title, lines)

        # Colors
        TURBO_BOT_COLOR = "#808080"
        HIGHLIGHT_COLOR = "#FFFF00"
        COUNTER_COLOR = "#00FF00"
        BAR_COLOR = "#FF0000"

        # Asset Paths
        BOT_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/bot.svg"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        # Turbo Bot representation using SVG
        turbo_bot = SVGMobject(BOT_ASSET, color=TURBO_BOT_COLOR)
        # Using a fixed scale for the main bot
        turbo_bot.scale(0.3)
        self.place_at_grid(turbo_bot, "B3")
        bot_label = Text("Turbo Bot", font_size=14, color=TURBO_BOT_COLOR)
        bot_label.next_to(turbo_bot, UP, buff=0.1)

        # Counter
        counter_val = ValueTracker(0)
        counter_text = Text("Keys Checked:", font_size=14, color=WHITE)
        counter_num = DecimalNumber(0, num_decimal_places=0, group_with_commas=True, font_size=18, color=COUNTER_COLOR)
        counter_num.add_updater(lambda d: d.set_value(counter_val.get_value()))
        
        counter_group = VGroup(counter_text, counter_num).arrange(RIGHT, buff=0.1)
        # Fix for Issue 40: Move counter_group to C6
        self.place_at_grid(counter_group, "C6", scale_factor=0.8)

        self.play(FadeIn(turbo_bot), FadeIn(bot_label), FadeIn(counter_group))
        self.play(counter_val.animate.set_value(10**12), run_time=2, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)

        # Fill the screen with tiny bots using SVG
        bots_grid = VGroup()
        bot_template = SVGMobject(BOT_ASSET, color=TURBO_BOT_COLOR).scale(0.15)
        for r in ["A", "B", "C", "D"]:
            for c in ["1", "2", "3", "4", "5", "6"]:
                if r == "B" and c == "3": continue # Skip the main bot area center but keep the grid logic
                tiny_bot = bot_template.copy()
                tiny_bot.move_to(self.grid[f"{r}{c}"])
                bots_grid.add(tiny_bot)
        
        # Label for "4 Billion Bots"
        bots_count_label = Text("4,000,000,000 Supercomputers", font_size=20, color=TURBO_BOT_COLOR)
        # Fix for Issue 41: Position bots_count_label in area A2 to A5
        self.place_in_area(bots_count_label, 'A2', 'A5', scale_factor=0.8)

        self.play(
            FadeOut(turbo_bot), 
            FadeOut(bot_label),
            FadeOut(counter_group),
            FadeIn(bots_grid),
            Write(bots_count_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)

        # Progress bar
        bar_bg = Rectangle(height=0.4, width=4.0, color=WHITE)
        # Fix for Issue 42: Position bar_bg in area E2 to E6
        self.place_in_area(bar_bg, 'E2', 'E6', scale_factor=1.0)
        
        # 0% progress fill (tiny thin line)
        bar_fill = Rectangle(height=0.3, width=0.01, color=BAR_COLOR, fill_opacity=1.0)
        bar_fill.align_to(bar_bg, LEFT).shift(RIGHT * 0.05)
        
        bar_label = Text("Estimated Time: Trillions of Universe Ages", font_size=18, color=BAR_COLOR)
        bar_label.next_to(bar_bg, DOWN, buff=0.2)
        
        progress_text = Text("0.0000000001% Complete", font_size=14, color=WHITE)
        progress_text.next_to(bar_bg, UP, buff=0.1)

        self.play(
            Create(bar_bg),
            FadeIn(bar_fill),
            Write(bar_label),
            Write(progress_text)
        )
        self.wait(2)
