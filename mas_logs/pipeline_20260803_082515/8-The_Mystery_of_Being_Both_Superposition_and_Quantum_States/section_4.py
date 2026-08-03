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
        self.setup_layout("The Math of Superposition", [
            "- Mathematical superposition combines these states using Greek letters.",
            "- Alpha and Beta represent the vector's horizontal and vertical shadows.",
            "- These 'amplitudes' define the probability of each outcome.",
            "- The squared sum of these shadows must equal one.",
            "- This ensures the total probability always remains one hundred percent."
        ])

        # === Animation for Lecture Line 1 ===
        # Display the superposition equation: "|ψ⟩ = α|0⟩ + β|1⟩" (#FFFFFF).
        self.play(self.lecture[0].animate.set_color(YELLOW))
        eqn = MathTex(r"|\psi\rangle = \alpha|0\rangle + \beta|1\rangle", color=WHITE)
        self.place_in_area(eqn, 'A3', 'B6', scale_factor=1.2) # Fixed position per Issue 42
        self.play(Write(eqn))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show a vector at a 45-degree angle on the coordinate plane.
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        
        axes = Axes(
            x_range=[0, 1.2, 0.5],
            y_range=[0, 1.2, 0.5],
            x_length=3,
            y_length=3,
            axis_config={"include_tip": True}
        )
        self.place_in_area(axes, 'C3', 'E6', scale_factor=1.0) # Fixed position per Issue 41
        
        # Vector at 45 degrees, length 1
        # sin(45) = cos(45) = 1/sqrt(2) approx 0.707
        val_0707 = 0.707
        vec_end = axes.c2p(val_0707, val_0707)
        vec_start = axes.c2p(0, 0)
        vector = Arrow(vec_start, vec_end, buff=0, color=WHITE)
        
        self.play(Create(axes), GrowArrow(vector))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Draw projections (dotted lines) from the vector tip to both axes.
        # Label projections α (#00FF00) on X-axis and β (#FF0000) on Y-axis.
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        
        proj_x = DashedLine(vec_end, axes.c2p(val_0707, 0), color=GRAY)
        proj_y = DashedLine(vec_end, axes.c2p(0, val_0707), color=GRAY)
        
        label_alpha = MathTex(r"\alpha", color="#00FF00")
        label_beta = MathTex(r"\beta", color="#FF0000")
        
        # Position labels within 1 grid unit of their corresponding objects
        label_alpha.next_to(axes.c2p(val_0707/2, 0), DOWN, buff=0.1)
        label_beta.next_to(axes.c2p(0, val_0707/2), LEFT, buff=0.1)
        
        # Highlight projection segments on axes
        segment_x = Line(axes.c2p(0, 0), axes.c2p(val_0707, 0), color="#00FF00", stroke_width=4)
        segment_y = Line(axes.c2p(0, 0), axes.c2p(0, val_0707), color="#FF0000", stroke_width=4)
        
        self.play(Create(proj_x), Create(proj_y))
        self.play(Create(segment_x), Create(segment_y), Write(label_alpha), Write(label_beta))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Display and highlight "α² + β² = 1" (#FFFFFF) next to the vector.
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(YELLOW))
        
        sum_eqn = MathTex(r"\alpha^2 + \beta^2 = 1", color=WHITE)
        self.place_in_area(sum_eqn, 'F3', 'F6', scale_factor=1.2) # Fixed position per Issue 43
        
        self.play(Write(sum_eqn))
        self.play(Indicate(sum_eqn))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This ensures the total probability always remains one hundred percent.
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(YELLOW))
        
        prob_text = Text("Total Probability = 100%", font_size=20, color=WHITE)
        # Position slightly below the sum_eqn
        prob_text.next_to(sum_eqn, DOWN, buff=0.2)
        
        self.play(Write(prob_text))
        self.wait(2)
        
        # Final cleanup or highlight
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
