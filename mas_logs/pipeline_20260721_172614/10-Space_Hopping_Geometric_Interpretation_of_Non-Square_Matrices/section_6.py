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
        title = "Summary & Application"
        lecture_lines = [
            "Non-square matrices act as bridges between different dimensions.",
            "Cameras project 3D worlds onto 2D digital sensors.",
            "Linear algebra powers these inter-dimensional transformations."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors
        COLOR_BRIDGE = BLUE_C
        COLOR_MOUNTAIN = "#00FF00"
        COLOR_SCREEN = "#FFFFFF"
        COLOR_LENS = GRAY_B
        HIGHLIGHT_COLOR = YELLOW

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        # Create Grids
        grid_3d = NumberPlane(
            x_range=[0, 3, 1], y_range=[0, 3, 1], 
            background_line_style={"stroke_color": BLUE_E, "stroke_width": 1}
        )
        grid_2d = NumberPlane(
            x_range=[0, 2, 1], y_range=[0, 2, 1], 
            background_line_style={"stroke_color": BLUE_E, "stroke_width": 1}
        )
        
        self.place_in_area(grid_3d, "A1", "B3", scale_factor=0.3)
        self.place_in_area(grid_2d, "A5", "B6", scale_factor=0.3)
        
        # Matrix Bridge Icon
        bridge_rect = Rectangle(width=1.2, height=0.8, color=COLOR_BRIDGE)
        bridge_text = MathTex(r"A_{2 \times 3}", color=COLOR_BRIDGE, font_size=24)
        bridge_group = VGroup(bridge_rect, bridge_text)
        self.place_at_grid(bridge_group, "A4", scale_factor=0.8)
        
        arrow1 = Arrow(grid_3d.get_right(), bridge_rect.get_left(), buff=0.1, color=COLOR_BRIDGE)
        arrow2 = Arrow(bridge_rect.get_right(), grid_2d.get_left(), buff=0.1, color=COLOR_BRIDGE)

        self.play(
            Create(grid_3d),
            Create(grid_2d),
            Create(bridge_group),
            Create(arrow1),
            Create(arrow2),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)

        # Mountains (3D Representation in 2D)
        mountain_path = VGroup(
            Polygon([0, 0, 0], [1, 2, 0], [2, 0, 0], color=COLOR_MOUNTAIN, fill_opacity=0.3),
            Polygon([0.5, 0, 0], [1.5, 1.5, 0], [2.5, 0, 0], color=COLOR_MOUNTAIN, fill_opacity=0.3)
        )
        # Resolved Issue 33: Move to C1-D2
        self.place_in_area(mountain_path, "C1", "D2", scale_factor=0.6)
        
        # Lens
        lens = Circle(radius=0.5, color=COLOR_LENS, fill_opacity=0.2)
        lens_text = Text("LENS", font_size=16, color=COLOR_LENS)
        lens_group = VGroup(lens, lens_text)
        # Resolved Issue 34: Move to C4
        self.place_at_grid(lens_group, "C4", scale_factor=0.8)
        
        # Rays
        rays = VGroup(*[
            Line(mountain_path.get_center() + i*0.2*UP, lens.get_center(), color=YELLOW, stroke_width=1, stroke_opacity=0.5)
            for i in range(-2, 3)
        ])

        self.play(
            Create(mountain_path),
            Create(lens_group),
            run_time=1.5
        )
        self.play(Create(rays), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)

        # Digital Screen (Sensor)
        screen = Rectangle(width=2.5, height=1.8, color=COLOR_SCREEN)
        screen_label = Text("SENSOR (2D)", font_size=14, color=COLOR_SCREEN).next_to(screen, DOWN, buff=0.1)
        screen_group = VGroup(screen, screen_label)
        # Resolved Issue 35: Move to C5-D6
        self.place_in_area(screen_group, "C5", "D6", scale_factor=0.6)
        
        # Flattened mountains on screen
        flattened_mountains = mountain_path.copy().set_color(COLOR_SCREEN).set_fill(COLOR_SCREEN, opacity=0.5)
        flattened_mountains.scale(0.5)
        flattened_mountains.move_to(screen.get_center())

        self.play(
            Create(screen_group),
            run_time=1
        )
        self.play(
            Transform(rays.copy(), flattened_mountains),
            run_time=2
        )
        self.wait(2)
