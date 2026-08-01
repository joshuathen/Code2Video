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
        # Updated lines from ScriptWriter via Issue 53
        lines = [
            "Squaring the coefficients reveals the real-world odds.",
            "This shifting chameleon represents our uncertain quantum state.",
            "Probabilities often start balanced at fifty-fifty.",
            "But these chances can vary based on the coefficients.",
            "Measuring forces one outcome with one hundred percent certainty."
        ]
        self.setup_layout("The Math of Probability (Born's Rule)", lines)

        # Colors
        COLOR_ALPHA = "#00FFFF"  # Cyan
        COLOR_BETA = "#FF00FF"   # Magenta
        COLOR_GREEN = "#00FF00"
        COLOR_GOLD = "#FFD700"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_ALPHA))
        
        # Equation: |α|² + |β|² = 1
        eq_alpha = Text("|α|²", color=COLOR_ALPHA, font_size=28)
        eq_plus = Text(" + ", color=WHITE, font_size=28)
        eq_beta = Text("|β|²", color=COLOR_BETA, font_size=28)
        eq_equals = Text(" = 1", color=WHITE, font_size=28)
        equation = VGroup(eq_alpha, eq_plus, eq_beta, eq_equals).arrange(RIGHT, buff=0.1)
        self.place_in_area(equation, "A2", "A5", scale_factor=1.0)

        label_a = Text("Prob(0)", font_size=20, color=COLOR_ALPHA)
        label_b = Text("Prob(1)", font_size=20, color=COLOR_BETA)
        self.place_at_grid(label_a, "B2", scale_factor=0.8)
        self.place_at_grid(label_b, "B4", scale_factor=0.8)

        self.play(FadeIn(equation), FadeIn(label_a), FadeIn(label_b))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_BETA))
        
        # Chameleon asset integration (Issue 35, Issue 45)
        chameleon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/ch.svg")
        self.place_at_grid(chameleon, "C2", scale_factor=1.1)
        
        self.play(FadeIn(chameleon))
        
        # Rapidly shifting color
        for _ in range(3):
            self.play(chameleon.animate.set_color(COLOR_GREEN), run_time=0.25)
            self.play(chameleon.animate.set_color(COLOR_GOLD), run_time=0.25)
        
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_GREEN))

        # Bar Chart (Issue 46 alignment)
        bar_width = 0.5
        max_height = 1.5
        
        green_bar = Rectangle(width=bar_width, height=max_height * 0.5, color=COLOR_GREEN, fill_opacity=0.8)
        gold_bar = Rectangle(width=bar_width, height=max_height * 0.5, color=COLOR_GOLD, fill_opacity=0.8)
        
        # Grid positions
        self.place_at_grid(green_bar, "F2")
        self.place_at_grid(gold_bar, "F4")
        # Align to bottom of cell
        green_bar.align_to(self.grid["F2"], DOWN)
        gold_bar.align_to(self.grid["F4"], DOWN)

        label_green = Text("Green", font_size=18, color=COLOR_GREEN)
        label_gold = Text("Gold", font_size=18, color=COLOR_GOLD)
        self.place_at_grid(label_green, "E2", scale_factor=0.8)
        self.place_at_grid(label_gold, "E4", scale_factor=0.8)

        self.play(
            GrowFromEdge(green_bar, DOWN), 
            GrowFromEdge(gold_bar, DOWN), 
            FadeIn(label_green), 
            FadeIn(label_gold)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(COLOR_GOLD))
        
        # Change heights to 70/30
        h_70 = max_height * 0.7
        h_30 = max_height * 0.3
        
        self.play(
            green_bar.animate.stretch_to_fit_height(h_70).align_to(self.grid["F2"], DOWN),
            gold_bar.animate.stretch_to_fit_height(h_30).align_to(self.grid["F4"], DOWN),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(WHITE))
        
        # MEASURE! (Issue 44 fix)
        measure_flash = Text("MEASURE!", color=RED, weight=BOLD)
        self.place_at_grid(measure_flash, "C4", scale_factor=0.8)
        
        self.play(FadeIn(measure_flash))
        self.play(Flash(measure_flash, color=RED, line_length=0.4))
        self.play(FadeOut(measure_flash))

        # Collapse to 100% Gold
        self.play(
            chameleon.animate.set_color(COLOR_GOLD),
            gold_bar.animate.stretch_to_fit_height(max_height).align_to(self.grid["F4"], DOWN),
            green_bar.animate.stretch_to_fit_height(0.01).align_to(self.grid["F2"], DOWN),
            FadeOut(green_bar),
            run_time=1
        )
        self.wait(2)
