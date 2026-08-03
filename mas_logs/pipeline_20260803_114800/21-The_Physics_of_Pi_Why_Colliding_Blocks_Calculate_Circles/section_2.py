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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Power of 100", [
            "Mass ratios of 100 reveal a surprising pattern.",
            "Ratio 1 to 1 produces 3 collisions.",
            "Ratio 1 to 100 produces 31 collisions.",
            "Ratio 1 to 10,000 produces 314 collisions.",
            "These blocks are calculating the digits of Pi."
        ])

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        ratio_1 = Text("Ratio 1:1", font_size=24, color=WHITE)
        self.place_in_area(ratio_1, "B1", "B2")
        
        # Simple representation of blocks
        block_m = Square(side_length=0.4, color=BLUE, fill_opacity=0.8)
        block_M = Square(side_length=0.4, color=RED, fill_opacity=0.8)
        self.place_at_grid(block_m, "B4")
        self.place_at_grid(block_M, "B5")
        
        count_1 = Integer(0, color=YELLOW).scale(0.8)
        self.place_at_grid(count_1, "B3")
        
        self.play(FadeIn(ratio_1), FadeIn(block_m), FadeIn(block_M), FadeIn(count_1))
        
        # Animate 3 collisions (simplified visual)
        for i in range(1, 4):
            self.play(
                block_M.animate.shift(LEFT * 0.2),
                block_m.animate.shift(LEFT * 0.1),
                run_time=0.15
            )
            count_1.set_value(i)
            self.play(
                block_M.animate.shift(RIGHT * 0.2),
                block_m.animate.shift(RIGHT * 0.1),
                run_time=0.15
            )
        
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        ratio_100 = Text("Ratio 1:100", font_size=24, color=WHITE)
        self.place_in_area(ratio_100, "C1", "C2")
        
        count_100 = Integer(0, color=YELLOW).scale(0.8)
        self.place_at_grid(count_100, "C3")
        
        self.play(FadeIn(ratio_100), FadeIn(count_100))
        
        # ValueTracker for counter
        vt_100 = ValueTracker(0)
        count_100.add_updater(lambda d: d.set_value(int(vt_100.get_value())))
        
        self.play(vt_100.animate.set_value(31), run_time=1.5, rate_func=linear)
        count_100.clear_updaters()
        count_100.set_value(31)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        ratio_10000 = Text("Ratio 1:10,000", font_size=24, color=WHITE)
        self.place_in_area(ratio_10000, "D1", "D2")
        
        count_10000 = Integer(0, color=YELLOW).scale(0.8)
        self.place_at_grid(count_10000, "D3")
        
        self.play(FadeIn(ratio_10000), FadeIn(count_10000))
        
        vt_10000 = ValueTracker(0)
        count_10000.add_updater(lambda d: d.set_value(int(vt_10000.get_value())))
        
        self.play(vt_10000.animate.set_value(314), run_time=2, rate_func=linear)
        count_10000.clear_updaters()
        count_10000.set_value(314)
        
        # Highlight sequence and compare to Pi
        pi_approx = MathTex("\\pi \\approx 3.14159...", color=WHITE, font_size=30)
        self.place_in_area(pi_approx, "E1", "E3")
        
        self.play(
            count_1.animate.set_color(YELLOW),
            count_100.animate.set_color(YELLOW),
            count_10000.animate.set_color(YELLOW),
            FadeIn(pi_approx)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        pi_symbol = MathTex("\\pi", color=PINK)
        self.place_in_area(pi_symbol, "E5", "F6", scale_factor=0.8)
        
        self.play(FadeIn(pi_symbol))
        
        # Pulse animation
        self.play(pi_symbol.animate.scale(1.2), run_time=0.4, rate_func=there_and_back)
        self.play(pi_symbol.animate.scale(1.2), run_time=0.4, rate_func=there_and_back)
        
        self.wait(2)
