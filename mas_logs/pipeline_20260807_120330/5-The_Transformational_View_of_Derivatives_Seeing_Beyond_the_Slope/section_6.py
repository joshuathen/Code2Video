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
        # Initialize layout with title and lecture lines
        self.setup_layout("Conclusion: Preparing for Higher Dimensions", [
            "The derivative is a local expansion or contraction factor.",
            "This view extends to higher dimensions via the Jacobian.",
            "Higher-order derivatives describe complex transformations of areas and volumes."
        ])
        
        # Initial visual setup: 1D parallel lines representing mapping from previous context
        line_input = Line(self.grid["B2"], self.grid["B6"], color="#555555")
        line_output = Line(self.grid["D2"], self.grid["D6"], color="#555555")
        self.add(line_input, line_output)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Display 'Derivative = Local Scaling'
        scaling_text = Text("Derivative = Local Scaling", font_size=24, color="#FFFFFF")
        # Apply B002/B021: Use area for horizontally extended text starting at Column 2
        self.place_in_area(scaling_text, "A2", "A6", scale_factor=0.8)
        
        self.play(
            FadeIn(scaling_text, shift=UP),
            line_input.animate.set_stroke(opacity=0.3),
            line_output.animate.set_stroke(opacity=0.3)
        )
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Morph the number lines into a 2D coordinate grid in #333333
        grid_2d = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=4,
            y_length=4,
            background_line_style={"stroke_color": "#333333", "stroke_width": 2, "stroke_opacity": 0.6},
            axis_config={"stroke_color": "#333333", "stroke_width": 2}
        )
        # Apply B021 and positioning constraints: place grid in the center-right visual area
        self.place_in_area(grid_2d, "B2", "F6")
        
        self.play(
            FadeOut(scaling_text),
            ReplacementTransform(VGroup(line_input, line_output), grid_2d),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Asset integration (Issue 26): Use square.svg
        square_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg"
        square = SVGMobject(square_path).set_color("#00FFFF")
        # Center it relative to the visual area
        self.place_at_grid(square, "D4", scale_factor=1.0)
        
        # Asset integration (Issue 26): Use paralle.svg
        paralle_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/paralle.svg"
        parallelogram = SVGMobject(paralle_path).set_color("#FF00FF")
        self.place_at_grid(parallelogram, "D4", scale_factor=1.0)
        
        # Jacobian label (Issue 37): Move to C5 for proximity to D4
        jacobian_label = Text("Jacobian", font_size=24, color="#FFFFFF")
        self.place_at_grid(jacobian_label, "C5", scale_factor=0.8) 

        self.play(FadeIn(square))
        self.wait(1)
        
        self.play(
            ReplacementTransform(square, parallelogram),
            Write(jacobian_label),
            run_time=2
        )
        self.wait(3)
        
        # Final cleanup
        self.lecture[2].set_color(WHITE)
        self.wait(2)
