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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup title and lecture lines
        self.setup_layout(
            "The Critical Case: Determinant Zero", 
            [
                'A zero determinant means space collapses into a line.', 
                'Since area is lost, we cannot reverse this transformation.', 
                'This is why matrices with zero determinant lack inverses.'
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Highlight Line 1
        self.play(self.lecture[0].animate.set_color(YELLOW), run_time=0.5)
        
        # Create a local grid for transformation
        grid_lines = VGroup()
        for x in np.linspace(-1, 1, 5):
            grid_lines.add(Line([x, -1, 0], [x, 1, 0], color=BLUE_B, stroke_width=1, stroke_opacity=0.6))
        for y in np.linspace(-1, 1, 5):
            grid_lines.add(Line([-1, y, 0], [1, y, 0], color=BLUE_B, stroke_width=1, stroke_opacity=0.6))
            
        unit_square = Square(side_length=0.5, fill_opacity=0.4, fill_color=YELLOW, stroke_width=2, stroke_color=YELLOW_A)
        unit_square.move_to([0.25, 0.25, 0])
        
        basis_i = Arrow(start=[0,0,0], end=[0.5,0,0], buff=0, color=GREEN, stroke_width=4)
        basis_j = Arrow(start=[0,0,0], end=[0,0.5,0], buff=0, color=RED, stroke_width=4)
        
        # Group everything and place it in the center area of the right side
        animation_group = VGroup(grid_lines, unit_square, basis_i, basis_j)
        self.place_in_area(animation_group, "A1", "F6", scale_factor=1.2)
        
        self.add(animation_group)
        self.wait(1)

        # Apply transformation: [[1, 2], [2, 4]]
        center = animation_group.get_center()
        
        def transform_func(point):
            # Point is absolute scene point. Calculate relative to local center.
            rel_p = point - center
            x, y = rel_p[0], rel_p[1]
            # Apply matrix [[1, 2], [2, 4]]
            # We scale the matrix effect slightly so it stays within the panel bounds if needed,
            # but let's stick to the math:
            new_x = 1*x + 2*y
            new_y = 2*x + 4*y
            return np.array([new_x, new_y, 0]) + center

        self.play(
            animation_group.animate.apply_function(transform_func),
            run_time=3
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Update lecture highlight
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(GREEN),
            run_time=0.5
        )
        
        # Highlight the collapsed line in red (#FF0000)
        # Find the extent of the line
        start_pt = transform_func(np.array([-1, -1, 0]) + center - center)
        end_pt = transform_func(np.array([1, 1, 0]) + center - center)
        
        collapsed_line_highlight = Line(
            start_pt, 
            end_pt, 
            color="#FF0000", 
            stroke_width=6
        )
        
        # Show area shrinking to line
        area_label = Text("Area = 0", font_size=24, color=WHITE)
        self.place_at_grid(area_label, "D1")

        self.play(
            Create(collapsed_line_highlight),
            Write(area_label),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Update lecture highlight
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(TEAL),
            run_time=0.5
        )
        
        # Display 'det = 0' and 'No Inverse' in white next to the collapsed grid
        det_text = Text("det = 0", font_size=24, color=WHITE)
        inv_text = Text("No Inverse", font_size=24, color=WHITE)
        
        # Position labels near the collapse
        self.place_at_grid(det_text, "B5")
        self.place_at_grid(inv_text, "C5")
        
        self.play(
            Write(det_text),
            Write(inv_text),
            run_time=1.5
        )
        self.wait(2)
