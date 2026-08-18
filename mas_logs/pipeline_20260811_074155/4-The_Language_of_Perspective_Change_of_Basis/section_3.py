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
        # Setup layout
        self.setup_layout(
            "The 'New' Ruler: Defining Basis B", 
            [
                "Basis B introduces a new set of coordinate axes.",
                "Vectors u and v point in non-standard directions.",
                "This creates a skewed grid over the standard system."
            ]
        )
        
        # Define base colors
        STANDARD_GREY = "#555555"
        VECTOR_U_COLOR = "#FFA500"  # Bright orange
        VECTOR_V_COLOR = "#800080"  # Bright purple
        HIGHLIGHT_COLOR = YELLOW

        # Create Standard Grid
        # Range adjusted to keep the visual contained
        std_grid = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            background_line_style={
                "stroke_color": STANDARD_GREY,
                "stroke_width": 1.5,
                "stroke_opacity": 0.5
            },
            axis_config={"stroke_color": STANDARD_GREY, "stroke_width": 2}
        )

        # Place standard grid in the visual area as requested in Issue #28 and #29
        # Moving to B2-F5 area with a smaller scale factor to prevent overlap with lecture notes
        self.place_in_area(std_grid, 'B2', 'F5', scale_factor=0.7)
        origin_pos = std_grid.get_origin()

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        self.play(FadeIn(std_grid))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Define vectors u and v relative to std_grid coordinates
        # u = (1, 1), v = (-1, 1)
        u_vec = Arrow(
            start=origin_pos, 
            end=std_grid.c2p(1, 1), 
            buff=0, 
            color=VECTOR_U_COLOR,
            stroke_width=6
        )
        v_vec = Arrow(
            start=origin_pos, 
            end=std_grid.c2p(-1, 1), 
            buff=0, 
            color=VECTOR_V_COLOR,
            stroke_width=6
        )
        
        # Labels for vectors - positioned within 1 grid unit of their corresponding objects
        u_label = Text("u", slant=ITALIC, color=VECTOR_U_COLOR, font_size=24).next_to(u_vec.get_end(), RIGHT, buff=0.1)
        v_label = Text("v", slant=ITALIC, color=VECTOR_V_COLOR, font_size=24).next_to(v_vec.get_end(), LEFT, buff=0.1)

        # Update lecture line colors
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(VECTOR_U_COLOR)
        )
        
        self.play(Create(u_vec), Write(u_label))
        self.play(Create(v_vec), Write(v_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Create skewed grid lines (parallels to u and v)
        # Based on Basis B coordinates: x = c1*u + c2*v
        # Since u=(1,1) and v=(-1,1), grid lines are c1=const or c2=const
        skewed_lines_u = VGroup() # Lines parallel to u
        skewed_lines_v = VGroup() # Lines parallel to v
        
        # Grid range from -3 to 3 for basis coordinates
        for i in range(-3, 4):
            # Line where c1 = i (parallel to v)
            # L(c2) = i*u + c2*v. Ends at c2 = +/- 3
            start_v = std_grid.c2p(i - 3, i + 3)
            end_v = std_grid.c2p(i + 3, i - 3)
            line_v = Line(start_v, end_v, color=VECTOR_V_COLOR, stroke_width=1, stroke_opacity=0.4)
            skewed_lines_v.add(line_v)
            
            # Line where c2 = i (parallel to u)
            # L(c1) = c1*u + i*v. Ends at c1 = +/- 3
            start_u = std_grid.c2p(-3 - i, -3 + i)
            end_u = std_grid.c2p(3 - i, 3 + i)
            line_u = Line(start_u, end_u, color=VECTOR_U_COLOR, stroke_width=1, stroke_opacity=0.4)
            skewed_lines_u.add(line_u)

        skewed_grid = VGroup(skewed_lines_u, skewed_lines_v)
        
        # Origin point for flashing
        origin_dot = Dot(origin_pos, color=WHITE, radius=0.06)

        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        self.play(Create(skewed_grid))
        self.play(Flash(origin_dot, color=WHITE, line_length=0.2, flash_radius=0.15, num_lines=10))
        self.add(origin_dot)
        self.wait(2)

        # Reset final line color
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
