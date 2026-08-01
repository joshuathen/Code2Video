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
        title = "The Mathematical Mechanics"
        lines = [
            "For the theorem to work, sample size matters.",
            "The distribution's center marks the true population mean.",
            "We calculate the spread using the standard error.",
            "Increasing the sample size makes our estimate narrower.",
            "Larger samples lead to much higher precision."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF6347"))
        n_text = Text("n ≥ 30", font_size=48, weight=BOLD, color="#FF6347")
        # Issue 27 Fix: Positioned at A3-A5 for better balance.
        self.place_in_area(n_text, "A3", "A5", scale_factor=0.8)
        self.play(FadeIn(n_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#ADD8E6"))
        
        # Visual anchor for the distribution centered on column 4
        vis_center = self.grid["D4"] 
        sigma_tracker = ValueTracker(1.0)
        
        # Bell curve defined by sigma tracker
        bell_curve = always_redraw(lambda: 
            ParametricFunction(
                lambda t: np.array([t, (1.2/sigma_tracker.get_value()) * np.exp(-0.5 * (t/(sigma_tracker.get_value()*0.8))**2), 0]),
                t_range=[-3, 3],
                color=BLUE_A
            ).move_to(vis_center).shift(DOWN * 1.5)
        )
        
        # Horizontal axis centered on Col 4
        baseline = Line(
            self.grid["F1"] + LEFT*0.5, 
            self.grid["F6"] + RIGHT*0.5, 
            color=GRAY_C
        ).shift(UP * 0.2)
        baseline.move_to(np.array([3.5, self.grid["F4"][1] + 0.2, 0]))

        # Vertical mean line (mu)
        mu_line = DashedLine(
            start=baseline.get_center(),
            end=baseline.get_center() + UP * 3.5,
            color="#ADD8E6"
        )
        
        # Issue 28 Fix: mu_label at B4, centered with the peak.
        mu_label = Text("μ", color="#ADD8E6", font_size=36)
        self.place_at_grid(mu_label, "B4", scale_factor=1.0)
        mu_label.shift(UP * 0.3)

        self.add(baseline)
        self.play(Create(bell_curve), Create(mu_line), Write(mu_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#F0E68C"))
        se_formula = Text("Standard Error = σ / √n", color="#F0E68C", font_size=24)
        # Issue 29 Fix: se_formula in area D5-F6 to avoid overlaps.
        self.place_in_area(se_formula, "D5", "F6", scale_factor=0.8)
        self.play(Write(se_formula))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFD700")) 
        # Visualization: decreasing sigma (narrower curve)
        self.play(sigma_tracker.animate.set_value(0.4), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFFFFF"))
        
        # Flash animation for precision
        flash_point = vis_center + DOWN * 0.5
        flash_rect = Rectangle(width=0.4, height=1.0, color=WHITE, fill_opacity=0.5).move_to(flash_point)
        self.play(Flash(flash_rect, color=WHITE, line_length=0.4, num_lines=12))
        
        self.wait(2)
