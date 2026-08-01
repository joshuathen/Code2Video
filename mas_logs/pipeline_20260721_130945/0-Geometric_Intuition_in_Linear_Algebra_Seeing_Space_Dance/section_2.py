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
        # Fetch data from storyboard
        title = "The Foundation: Basis Vectors"
        lines = [
            "Standard basis vectors i-hat and j-hat define our grid.",
            "i-hat is one step right; j-hat is one step up.",
            "Every vector is a combination of these basic hops."
        ]
        
        self.setup_layout(title, lines)
        
        # Colors
        I_HAT_COLOR = "#FF0000"
        J_HAT_COLOR = "#00FF00"
        VECTOR_SUM_COLOR = "#FFFF00"
        HIGHLIGHT_COLOR = "#FFFF00"
        
        # Coordinate system
        # Ensure we don't call unsupported methods like hide_tip (L002)
        plane = NumberPlane(
            x_range=[-1, 5, 1],
            y_range=[-1, 4, 1],
            x_length=5,
            y_length=4,
            background_line_style={
                "stroke_color": BLUE_D,
                "stroke_width": 1,
                "stroke_opacity": 0.5
            }
        )
        # Using place_in_area for the plane (L003)
        # Position plane in a central area on the right
        self.place_in_area(plane, "B2", "E6", scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        # Standard basis vectors i-hat and j-hat define our grid.
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        
        # i-hat and j-hat vectors
        # Using Arrow for vectors; use plane.c2p for mapping (L013)
        i_hat = Arrow(plane.c2p(0, 0), plane.c2p(1, 0), buff=0, color=I_HAT_COLOR, stroke_width=4)
        j_hat = Arrow(plane.c2p(0, 0), plane.c2p(0, 1), buff=0, color=J_HAT_COLOR, stroke_width=4)
        
        self.play(
            Create(plane),
            GrowArrow(i_hat),
            GrowArrow(j_hat),
            run_time=2
        )
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # i-hat is one step right; j-hat is one step up.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Labels: i-hat and j-hat
        # Fix for Issue 28 and 29: Use place_at_grid (L003) to prevent overlap with axes (L015)
        i_label = MathTex(r"\hat{i}", color=I_HAT_COLOR)
        j_label = MathTex(r"\hat{j}", color=J_HAT_COLOR)
        
        # Applying specific grid coordinates suggested by VideoCritic
        self.place_at_grid(i_label, 'F3', scale_factor=0.6) 
        self.place_at_grid(j_label, 'D1', scale_factor=0.6) 
        
        self.play(
            Write(i_label),
            Write(j_label)
        )
        
        # Use Indicate for emphasis (L004)
        self.play(Indicate(i_hat), Indicate(i_label))
        self.play(Indicate(j_hat), Indicate(j_label))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        # Every vector is a combination of these basic hops.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Show combination for vector [3, 2]
        # Fade out labels to prevent clutter (L015)
        self.play(FadeOut(i_label), FadeOut(j_label))
        
        # Create hops: 3 i-hats end-to-end
        i_hops = VGroup(*[
            Arrow(plane.c2p(k, 0), plane.c2p(k+1, 0), buff=0, color=I_HAT_COLOR, stroke_width=4)
            for k in range(3)
        ])
        
        # Create hops: 2 j-hats stacked
        j_hops = VGroup(*[
            Arrow(plane.c2p(3, k), plane.c2p(3, k+1), buff=0, color=J_HAT_COLOR, stroke_width=4)
            for k in range(2)
        ])
        
        # Final vector [3,2]
        result_vec = Arrow(plane.c2p(0, 0), plane.c2p(3, 2), buff=0, color=VECTOR_SUM_COLOR, stroke_width=6)
        # Fix for Issue 30: Use place_at_grid for result_label to avoid grid intersection overlap
        result_label = MathTex(r"\begin{bmatrix} 3 \\ 2 \end{bmatrix}", color=VECTOR_SUM_COLOR)
        self.place_at_grid(result_label, 'B5', scale_factor=0.7)
        
        # Sequential growth of hops to demonstrate linear combination
        # Transform the initial i_hat into the first hop and continue
        self.play(ReplacementTransform(i_hat, i_hops[0]), run_time=0.5)
        for i in range(1, 3):
            self.play(GrowArrow(i_hops[i]), run_time=0.5)
        
        # Transform the initial j_hat into the first hop (at its new position)
        self.play(ReplacementTransform(j_hat, j_hops[0]), run_time=0.5)
        for i in range(1, 2):
            self.play(GrowArrow(j_hops[i]), run_time=0.5)
            
        self.wait(0.5)
        
        # Show the resultant vector
        self.play(
            GrowArrow(result_vec),
            Write(result_label),
            i_hops.animate.set_stroke(opacity=0.4),
            j_hops.animate.set_stroke(opacity=0.4)
        )
        self.wait(2)
        
        # Final cleanup: Reset lecture line color
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
