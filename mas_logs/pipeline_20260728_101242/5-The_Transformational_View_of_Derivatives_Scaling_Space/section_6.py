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
        lecture_lines = [
            "Derivatives measure the change in local spatial density.",
            "This transformational view extends to higher dimensions.",
            "Visualize calculus as the geometry of moving space."
        ]
        self.setup_layout("Conclusion: The Dynamic Viewpoint", lecture_lines)
        
        # Colors
        highlight_color_dense = "#0000FF" # Blue
        highlight_color_sparse = "#FF0000" # Red
        concluding_text_color = "#FFFFFF"
        string_color = GRAY

        # === Animation for Lecture Line 1 ===
        # Derivatives measure the change in local spatial density.
        self.lecture[0].set_color(YELLOW)
        
        # Create Input and Output lines using the grid
        # Horizontal lines
        input_line = Line(self.grid["B2"], self.grid["B6"], color=WHITE)
        output_line = Line(self.grid["E2"], self.grid["E6"], color=WHITE)
        
        input_label = Text("Input", font_size=18)
        self.place_at_grid(input_label, "B2", scale_factor=0.8) # Fixed Issue 36
        
        output_label = Text("Output", font_size=18)
        self.place_at_grid(output_label, "E2", scale_factor=0.8) # Fixed Issue 37
        
        self.add(input_line, output_line, input_label, output_label)
        
        # Create mapping strings
        # Create dense points in the middle and sparse at ends on the input line
        input_points_list = []
        
        # Sparse side 1 (0.1 to 0.3)
        for i in range(10):
            alpha = 0.1 + 0.2 * (i / 9)
            input_points_list.append(input_line.point_from_proportion(alpha))
            
        # Dense middle (0.4 to 0.6)
        for i in range(40):
            alpha = 0.4 + 0.2 * (i / 39)
            input_points_list.append(input_line.point_from_proportion(alpha))
            
        # Sparse side 2 (0.7 to 0.9)
        for i in range(10):
            alpha = 0.7 + 0.2 * (i / 9)
            input_points_list.append(input_line.point_from_proportion(alpha))
            
        input_points_list.sort(key=lambda p: p[0])
            
        # Output points (regularly spaced)
        num_points = len(input_points_list)
        output_points = [output_line.point_from_proportion(0.1 + 0.8 * (i/(num_points-1))) for i in range(num_points)]
        
        mapping_strings = VGroup()
        for i in range(num_points):
            line = Line(input_points_list[i], output_points[i], stroke_width=0.5, color=string_color, stroke_opacity=0.4)
            mapping_strings.add(line)
            
        self.play(Create(mapping_strings), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # This transformational view extends to higher dimensions.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Create a colored input line using segments to show density gradient
        num_segments = 60
        colored_input_line = VGroup(*[
            Line(
                input_line.point_from_proportion(i/num_segments),
                input_line.point_from_proportion((i+1)/num_segments),
                stroke_width=6
            ) for i in range(num_segments)
        ])
        
        # Red (sparse) at ends, Blue (dense) in middle
        colored_input_line.set_color_gradient([highlight_color_sparse, highlight_color_dense, highlight_color_sparse])
        
        self.play(
            Transform(input_line, colored_input_line),
            mapping_strings.animate.set_color(WHITE).set_stroke(opacity=0.2),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Visualize calculus as the geometry of moving space.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Display concluding text
        concluding_text = Text("Derivative =\nLocal Spatial Density Change", 
                               font_size=28, 
                               color=concluding_text_color,
                               t2c={"Derivative": YELLOW})
        
        # Clear specific elements
        self.play(
            FadeOut(input_line),
            FadeOut(output_line),
            FadeOut(mapping_strings),
            FadeOut(input_label),
            FadeOut(output_label),
            run_time=1
        )
        
        # Position concluding text
        self.place_in_area(concluding_text, "C1", "D6", scale_factor=1.0) # Fixed Issue 35
        
        self.play(Write(concluding_text), run_time=2)
        self.wait(3)

        # Final state
        self.lecture[2].set_color(WHITE)
        self.wait(1)
