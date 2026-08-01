from manim import *
import numpy as np

# Fix for KeyError: 'iπ' caused by curly braces in the file path.
# Manim's config system uses .format() on directory paths, which crashes 
# if the path contains braces. We override the input_file config to bypass this.
config.input_file = "section_3.py"

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
        # Colors for lines
        color1 = WHITE
        color2 = "#FFFF00" # Yellow for scaling
        color3 = "#00CCFF" # Light blue for imaginary/rotation

        self.setup_layout(
            "The Meaning of 'e': Continuous Growth", 
            [
                'Constant e describes continuous growth in any direction.', 
                'Real exponents scale growth along a straight line.', 
                'Imaginary exponents push this growth in a new direction.'
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(color1)
        
        # Define origin for the growth (D2)
        origin = self.grid["D2"]
        # Horizontal path from D2 to D5
        start_point = self.grid["D2"]
        end_point_real = self.grid["D5"]
        
        path_real = Line(start_point, end_point_real, color=WHITE, stroke_width=2)
        moving_dot = Dot(color=color1)
        moving_dot.move_to(start_point)
        
        # Replaced MathTex with Text to avoid FileNotFoundError: 'latex'
        label_e = Text("e^x", font_size=32, color=color1)
        # Resolved Issue 25: Moved from C2 to C3 to avoid clutter
        self.place_at_grid(label_e, "C3")
        
        self.play(FadeIn(path_real), FadeIn(moving_dot), FadeIn(label_e))
        self.play(moving_dot.animate.move_to(self.grid["D4"]), run_time=2, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(color2)
        
        # Show scaling: stretch the path further or highlight the growth
        scale_arrow = Arrow(self.grid["D4"], self.grid["D6"], color=color2, buff=0.1)
        label_scale = Text("Scaling", font_size=20, color=color2)
        # Resolved Issue 26: Moved from E5 to D5 for better visual association
        self.place_at_grid(label_scale, "D5")
        
        self.play(Create(scale_arrow), Write(label_scale))
        self.play(moving_dot.animate.move_to(self.grid["D5"]), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(color3)
        
        # Point is at D5. We apply a 'push' perpendicular.
        # Radius is length between D2 and D5 (3 units)
        radius = 3.0 
        
        # Perpendicular arrow 'i' at current dot position
        push_arrow = Arrow(self.grid["D5"], self.grid["B5"], color=color3, buff=0.1)
        # Replaced MathTex with Text to avoid FileNotFoundError: 'latex'
        label_i = Text("i", font_size=36, color=color3)
        # Resolved Issue 27: Moved from B6 to B3 for better positioning
        self.place_at_grid(label_i, "B3")
        
        # Arc path: Starting at 0 degrees relative to D2, going to 90 degrees
        arc_path = Arc(radius=radius, start_angle=0, angle=PI/2, arc_center=origin, color=color3)
        
        self.play(
            FadeOut(scale_arrow),
            FadeOut(label_scale),
            Create(push_arrow),
            Write(label_i)
        )
        
        # As the dot moves along the arc, the push arrow stays perpendicular
        def update_arrow(obj):
            curr_pos = moving_dot.get_center()
            vec = curr_pos - origin
            # Rotate vector 90 degrees (tangent)
            perp_vec = np.array([-vec[1], vec[0], 0])
            if np.linalg.norm(perp_vec) > 0:
                perp_vec = perp_vec / np.linalg.norm(perp_vec) * 0.8
            obj.put_start_and_end_on(curr_pos, curr_pos + perp_vec)

        push_arrow.add_updater(update_arrow)
        
        # Label 'i' follows the arrow end
        def update_label_i(obj):
            obj.move_to(push_arrow.get_end() + RIGHT*0.2 + UP*0.2)
        
        label_i.add_updater(update_label_i)

        self.play(
            MoveAlongPath(moving_dot, arc_path),
            Create(arc_path),
            run_time=3,
            rate_func=linear
        )
        
        push_arrow.remove_updater(update_arrow)
        label_i.remove_updater(update_label_i)
        
        self.wait(2)
