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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup the stage
        title = "Conclusion: The Universal Cycle"
        lines = [
            "Derivatives track speed; integrals track distance.",
            "They form a perfect mathematical cycle.",
            "Together, they unlock the secrets of change."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color("#00FFFF"))

        # Speedometer Icon [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/speedometer.svg] (#00FFFF)
        speedometer = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/speedometer.svg", color="#00FFFF")
        self.place_in_area(speedometer, "B1", "C2", scale_factor=1.0)

        # Odometer Icon [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/od.svg] (#FFD700)
        odometer = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/od.svg", color="#FFD700")
        self.place_in_area(odometer, "B5", "C6", scale_factor=1.0)

        self.play(FadeIn(speedometer), FadeIn(odometer))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition lecture highlight
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFD700")
        )

        # Labels for the cycle (Corrected positions per Issue 41 and 42)
        deriv_label = Text("Derivative", font_size=24, color="#00FFFF")
        integ_label = Text("Integral", font_size=24, color="#FFD700")
        self.place_in_area(deriv_label, "E1", "E2", scale_factor=0.8)
        self.place_in_area(integ_label, "E5", "E6", scale_factor=0.8)

        # Circular loop arrows (#FFFFFF)
        # Top arrow: Speedometer side to Odometer side
        arrow_top = CurvedArrow(
            self.grid["B2"] + 0.5 * RIGHT + 0.3 * UP,
            self.grid["B5"] + 0.5 * LEFT + 0.3 * UP,
            angle=-TAU/8,
            color=WHITE
        )
        # Bottom arrow: Odometer side to Speedometer side
        arrow_bottom = CurvedArrow(
            self.grid["C5"] + 0.5 * LEFT + 0.3 * DOWN,
            self.grid["C2"] + 0.5 * RIGHT + 0.3 * DOWN,
            angle=-TAU/8,
            color=WHITE
        )

        self.play(
            Write(deriv_label),
            Write(integ_label),
            Create(arrow_top),
            Create(arrow_bottom)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Final highlight
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )

        # Pulsing effect: Labels and arrows pulse in sync to summarize the relationship
        pulse_group = VGroup(deriv_label, integ_label, arrow_top, arrow_bottom)
        
        for _ in range(2):
            self.play(
                pulse_group.animate.scale(1.1),
                rate_func=there_and_back,
                run_time=0.8
            )
            self.wait(0.1)

        self.wait(2)
