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
        # Setup layout with title and lecture lines
        self.setup_layout("The Visual Vector: Magnitude and Direction", [
            "- Visually, a vector is an arrow.",
            "- Magnitude is the arrow's physical length.",
            "- Direction is the angle of the path."
        ])
        
        # Colors
        COLOR_ARROW = "#00FFFF"
        COLOR_GRID = "#444444"
        COLOR_ORIGIN = "#FFFFFF"
        COLOR_MAGNITUDE = "#FFD700"
        COLOR_DIRECTION = "#FF4500"

        # === Animation for Lecture Line 1 ===
        # "Visually, a vector is an arrow."
        
        # Draw the background grid manually using grid positions
        grid_group = VGroup()
        for r in ["A", "B", "C", "D", "E", "F"]:
            grid_group.add(Line(self.grid[f"{r}1"], self.grid[f"{r}6"], color=COLOR_GRID, stroke_width=1, stroke_opacity=0.5))
        for c in ["1", "2", "3", "4", "5", "6"]:
            grid_group.add(Line(self.grid[f"A{c}"], self.grid[f"F{c}"], color=COLOR_GRID, stroke_width=1, stroke_opacity=0.5))
        
        origin_dot = Dot(color=COLOR_ORIGIN, radius=0.08)
        self.place_at_grid(origin_dot, 'E2')
        
        # Vector v from origin E2 to A5 (3 units right, 4 units up)
        start_pos = self.grid['E2']
        end_pos = self.grid['A5']
        vector_v = Arrow(start=start_pos, end=end_pos, color=COLOR_ARROW, buff=0, stroke_width=6)
        
        label_v = MathTex("\\vec{v}", color=COLOR_ARROW)
        # Resolved Issue 32: Move label_v to A4
        self.place_at_grid(label_v, 'A4', scale_factor=0.8)

        self.play(self.lecture[0].animate.set_color(COLOR_ARROW))
        self.play(Create(grid_group), FadeIn(origin_dot))
        self.play(GrowArrow(vector_v), Write(label_v))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Magnitude is the arrow's physical length."
        
        # Define offset points for the brace to represent magnitude
        # Vector direction: (3, 4). Normal: (-4, 3). Normalized offset: (-0.4, 0.3)
        normal_offset = np.array([-0.4, 0.3, 0])
        brace_start = start_pos + normal_offset
        brace_end = end_pos + normal_offset
        
        magnitude_brace = BraceBetweenPoints(brace_end, brace_start, color=COLOR_MAGNITUDE)
        label_magnitude = Text("Magnitude", color=COLOR_MAGNITUDE, font_size=24)
        # Resolved Issue 31: Move label_magnitude to C1
        self.place_at_grid(label_magnitude, 'C1', scale_factor=0.8)

        self.play(self.lecture[1].animate.set_color(COLOR_MAGNITUDE))
        self.play(Create(magnitude_brace), FadeIn(label_magnitude))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Direction is the angle of the path."
        
        # Arc representing the angle at the origin
        angle_val = np.arctan2(4, 3)
        direction_arc = Arc(
            radius=0.7, 
            start_angle=0, 
            angle=angle_val, 
            arc_center=start_pos, 
            color=COLOR_DIRECTION
        )
        
        label_direction = Text("Direction", color=COLOR_DIRECTION, font_size=24)
        # Resolved Issue 30: Move label_direction to F3
        self.place_at_grid(label_direction, 'F3', scale_factor=0.8)

        self.play(self.lecture[2].animate.set_color(COLOR_DIRECTION))
        self.play(Create(direction_arc), FadeIn(label_direction))
        self.wait(2)
