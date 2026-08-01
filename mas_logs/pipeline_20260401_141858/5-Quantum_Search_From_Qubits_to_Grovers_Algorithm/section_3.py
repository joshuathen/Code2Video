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

class Section3Scene(TeachingScene):
    def construct(self):
        # Title and Lecture Lines
        title = "State Vectors and Probability Amplitudes"
        lines = [
            "Vector projections onto the axes are called amplitudes.",
            "Squaring an amplitude reveals the probability of that state.",
            "Measurement collapses the vector into a single result."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Highlight Line 1
        self.play(self.lecture[0].animate.set_color("#FFFF00"))

        # Setup Coordinate System (Axes)
        axes = Axes(
            x_range=[0, 1.2, 1],
            y_range=[0, 1.2, 1],
            x_length=3,
            y_length=3,
            axis_config={"include_tip": True, "color": WHITE},
            tips=True
        )
        # Resolved Issue #40: Relocate axes
        self.place_in_area(axes, "B3", "E5")
        
        # Labels for axes - Using Text to avoid LaTeX dependency
        label_0 = Text("|0⟩", font_size=24).next_to(axes.c2p(1.1, 0), DOWN)
        label_1 = Text("|1⟩", font_size=24).next_to(axes.c2p(0, 1.1), LEFT)
        
        # Resolved Issue #31: Integrate Asset (vector.svg)
        # Load the asset
        state_vec = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/vector.svg")
        state_vec.set_color("#FFFF00")
        
        # Scale to match unit length of axes and rotate to 45 degrees
        unit_len = np.linalg.norm(axes.c2p(1,0) - axes.c2p(0,0))
        state_vec.scale_to_fit_height(unit_len)
        state_vec.move_to(axes.c2p(0,0), aligned_edge=DOWN) # Aligning based on vertical SVG orientation
        state_vec.rotate(45 * DEGREES, about_point=axes.c2p(0,0))
        
        # Tip of vector is at 45 degrees (cos 45, sin 45)
        tip_pos = axes.c2p(np.cos(PI/4), np.sin(PI/4))
        
        # Projections
        proj_x = DashedLine(tip_pos, axes.c2p(np.cos(PI/4), 0), color=WHITE)
        proj_y = DashedLine(tip_pos, axes.c2p(0, np.sin(PI/4)), color=WHITE)
        
        # Amplitude labels α and β - Using Text
        alpha_label = Text("α", color=WHITE, font_size=30)
        beta_label = Text("β", color=WHITE, font_size=30)
        
        # Resolved Issue #40: Use grid for alpha/beta labels
        self.place_at_grid(alpha_label, "E4", scale_factor=0.8) 
        self.place_at_grid(beta_label, "C3", scale_factor=0.8)

        self.play(Create(axes), Write(label_0), Write(label_1))
        self.play(FadeIn(state_vec))
        self.play(Create(proj_x), Create(proj_y))
        self.play(Write(alpha_label), Write(beta_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight Line 2
        self.play(self.lecture[1].animate.set_color("#00FF00"))

        # Probability calculations - Using Text
        prob_alpha = Text("|α|² = 0.5", color="#00FF00", font_size=32)
        prob_beta = Text("|β|² = 0.5", color="#00FF00", font_size=32)
        
        # Resolved Issue #41: Use grid for probability labels
        self.place_at_grid(prob_alpha, "B6", scale_factor=0.9)
        self.place_at_grid(prob_beta, "C6", scale_factor=0.9)

        self.play(Write(prob_alpha), Write(prob_beta))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight Line 3
        self.play(self.lecture[2].animate.set_color("#58ACFA"))

        # Measurement collapse: Snap to |0>
        # Define the target collapsed vector (pointing horizontal along x-axis)
        collapsed_vec = state_vec.copy()
        collapsed_vec.generate_target()
        collapsed_vec.target.rotate(-45 * DEGREES, about_point=axes.c2p(0,0))
        
        self.play(
            FadeOut(proj_x),
            FadeOut(proj_y),
            FadeOut(alpha_label),
            FadeOut(beta_label),
            MoveToTarget(collapsed_vec),
            run_time=1.5
        )
        
        # Final label for collapse result
        result_text = Text("Result: |0⟩", font_size=24, color="#FFFF00")
        # Resolved Issue #42: Relocate result_text to E6
        self.place_at_grid(result_text, "E6", scale_factor=0.9)
        self.play(Write(result_text))
        
        self.wait(2)
