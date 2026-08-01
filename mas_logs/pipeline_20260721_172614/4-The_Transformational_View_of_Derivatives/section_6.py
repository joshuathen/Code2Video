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
        self.setup_layout(
            "Conclusion: Why This Matters",
            [
                "This perspective is vital for multivariable calculus.",
                "It treats derivatives as matrices transforming space.",
                "Think of derivatives as local growth settings for functions."
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        
        # Display a white 2D grid that warps and stretches locally.
        # Instead of a heavy NumberPlane, we'll use a VGroup of lines.
        grid_lines = VGroup()
        for i in range(7):
            # Vertical lines
            v_line = Line(start=[0.5 + i*0.83, 2.2, 0], end=[0.5 + i*0.83, -2.8, 0], stroke_width=2, color=WHITE)
            grid_lines.add(v_line)
            # Horizontal lines
            h_line = Line(start=[0.5, 2.2 - i*0.83, 0], end=[5.5, 2.2 - i*0.83, 0], stroke_width=2, color=WHITE)
            grid_lines.add(h_line)
        
        def warp_func(p):
            # Apply a local warp centered at grid center (3.0, -0.3, 0)
            center = np.array([3.0, -0.3, 0])
            dist = np.linalg.norm(p - center)
            if dist < 2:
                # Use np.cos and np.sin for warping effect
                return p + 0.5 * np.array([np.sin(p[1]*2), np.cos(p[0]*2), 0]) * (2-dist)/2
            return p

        self.play(Create(grid_lines))
        self.wait(1)
        
        # Apply transformation
        self.play(grid_lines.animate.apply_function(warp_func), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color(YELLOW)

        # "Derivatives as matrices transforming space"
        # Displaying matrix. Issue 35 fix: scale_factor=1.5
        matrix = MathTex(
            r"\begin{bmatrix} a & b \\ c & d \end{bmatrix}",
            color=YELLOW, font_size=36
        )
        self.place_at_grid(matrix, "C3", scale_factor=1.5)
        
        self.play(Write(matrix))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color(YELLOW)

        # Fade in a yellow slider icon labeled 'Local Growth Setting'.
        # Issue 36 fix: move to area D1-E6, scale 0.8
        slider_line = Line(LEFT, RIGHT, color=YELLOW).scale(1.5)
        slider_knob = Dot(color=YELLOW)
        slider_label = Text("Local Growth Setting", font_size=20, color=YELLOW)
        
        slider_group = VGroup(slider_line, slider_knob, slider_label).arrange(DOWN, buff=0.2)
        self.place_in_area(slider_group, "D1", "E6", scale_factor=0.8)
        
        self.play(FadeIn(slider_group))
        
        # Animate knob moving
        self.play(slider_knob.animate.shift(RIGHT * 0.5), run_time=1)
        self.play(slider_knob.animate.shift(LEFT * 1.0), run_time=1)
        self.play(slider_knob.animate.shift(RIGHT * 0.5), run_time=1)
        self.wait(1)

        # Final transition
        # Fade out all elements and show 'Derivatives = Local Scaling' in white.
        # Issue 34 fix: scale_factor=0.6 and area B2-E6
        self.play(
            FadeOut(grid_lines),
            FadeOut(matrix),
            FadeOut(slider_group)
        )
        
        final_text = Text("Derivatives = Local Scaling", font_size=36, color=WHITE)
        self.place_in_area(final_text, "B2", "E6", scale_factor=0.6)
        
        self.play(Write(final_text))
        self.wait(3)
