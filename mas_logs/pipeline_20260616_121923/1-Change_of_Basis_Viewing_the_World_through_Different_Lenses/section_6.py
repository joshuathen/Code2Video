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
        # Initial Setup
        title = "Practical Application: Why Bother?"
        lines = [
            'Some problems are simpler in a custom basis.',
            "An eagle's flight is easy along its path.",
            'Aligning the basis simplifies complex diagonal motion.'
        ]
        self.setup_layout(title, lines)

        # Colors
        VEC_COLOR = YELLOW
        EAGLE_COLOR = WHITE
        AXIS_COLOR = GREY_B

        # === Animation for Lecture Line 1 ===
        # Line 1: Highlighted
        self.lecture[0].set_color(VEC_COLOR)

        # Coordinate Plane
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=5,
            y_length=5,
            axis_config={"color": AXIS_COLOR, "include_tip": True},
            background_line_style={"stroke_opacity": 0.3}
        )
        
        # Velocity parameters (2, 1.5) -> magnitude 2.5, angle ~36.87 degrees
        vx, vy = 2.0, 1.5
        mag = np.sqrt(vx**2 + vy**2)
        angle = np.arctan2(vy, vx)
        
        origin_pt = plane.c2p(0, 0, 0)
        end_pt = plane.c2p(vx, vy, 0)
        
        # Velocity Vector
        velocity_vec = Arrow(origin_pt, end_pt, buff=0, color=VEC_COLOR, stroke_width=4)
        
        # Eagle Icon (SVG Asset Integration - Issue 40)
        # Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/eagle.svg
        eagle_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/eagle.svg")
        eagle_icon.set_color(EAGLE_COLOR).scale(0.2)
        eagle_icon.move_to(end_pt).rotate(angle)
        
        eagle_text = Text("Eagle", font_size=18, color=EAGLE_COLOR).next_to(eagle_icon, UR, buff=0.1)
        
        # Standard basis labels
        coord_label = Text(
            "v = [2.0, 1.5]", 
            font_size=32, color=VEC_COLOR
        )
        # Issue 52: Move coord_label to B4 and scale to 0.7 to avoid overlap
        self.place_at_grid(coord_label, "B4", scale_factor=0.7)
        
        # Group the world to rotate together
        world = VGroup(plane, velocity_vec, eagle_icon, eagle_text)
        # Issue 53: Scale world to 0.8 to prevent obstruction of lecture notes
        self.place_in_area(world, "A2", "F6", scale_factor=0.8)
        
        self.play(
            FadeIn(world),
            Write(coord_label),
            run_time=1.5
        )
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Transition highlights
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(VEC_COLOR)
        
        # Rotate world by negative the eagle's angle to align x-axis with flight path
        rotation_angle = -angle
        
        self.play(
            Rotate(world, angle=rotation_angle, about_point=plane.c2p(0,0,0)),
            # Counter-rotate text to keep it readable
            Rotate(eagle_text, angle=-rotation_angle, about_point=eagle_text.get_center()),
            run_time=2.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition highlights
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(VEC_COLOR)
        
        # New basis coordinates
        new_coord_label = Text(
            "v' = [2.5, 0]", 
            font_size=32, color=VEC_COLOR
        )
        new_coord_label.move_to(coord_label.get_center())
        
        # Simple highlight of the aligned axis
        path_highlight = Line(
            plane.c2p(0,0,0), 
            plane.c2p(2.5, 0, 0), 
            color=VEC_COLOR, 
            stroke_width=6, 
            stroke_opacity=0.5
        )
        # Add highlight to world so it stays in the rotated frame
        world.add(path_highlight)

        self.play(
            Transform(coord_label, new_coord_label),
            FadeIn(path_highlight),
            run_time=1.5
        )
        self.wait(3)
