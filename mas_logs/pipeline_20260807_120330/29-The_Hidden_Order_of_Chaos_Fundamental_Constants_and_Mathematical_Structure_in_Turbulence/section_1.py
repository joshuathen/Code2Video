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

class Section1Scene(TeachingScene):
    def construct(self):
        # Setup the scene layout
        title = "The Mystery of the Mess: Defining Turbulence"
        lines = [
            "Turbulence is a great mystery in physics.",
            "Chaos seems random, but hidden structures exist.",
            "Smooth flows break into complex, turbulent swirls."
        ]
        self.setup_layout(title, lines)
        
        # Colors based on storyboard
        COLOR_LAMINAR = "#0000FF"
        COLOR_ROCK = "#808080"
        COLOR_TURBULENT = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Fade in smooth stream of horizontal blue lines representing laminar flow.
        # Resolving Issue 26: Moving flow lines to start at column 4 and using place_in_area.
        laminar_lines = VGroup()
        for row in ["B", "C", "D", "E"]:
            # Construct lines relative to local space first
            line = Line(LEFT * 1.5, RIGHT * 1.5, color=COLOR_LAMINAR, stroke_width=4)
            laminar_lines.add(line)
        laminar_lines.arrange(DOWN, buff=1.0)
        
        # Position the group in the area B4-E6
        self.place_in_area(laminar_lines, "B4", "E6", scale_factor=0.9)
        
        self.lecture[0].set_color(BLUE)
        self.play(Create(laminar_lines), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Introduce a gray rock (#808080) in the path of the stream.
        # Resolving Issue 25: Reducing rock scale to 0.6.
        rock = Dot(radius=0.35, color=COLOR_ROCK)
        self.place_at_grid(rock, "D4", scale_factor=0.6)
        
        self.lecture[1].set_color(BLUE)
        self.play(FadeIn(rock, scale=0.5))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transform the smooth lines into a chaotic web of white swirls (#FFFFFF) past the rock.
        # Swirls are designed to match the new laminar line positions.
        
        swirls = VGroup()
        
        # Path B (Top) - starts from Column 4 position
        path_b = CubicBezier(
            self.grid["B4"], 
            self.grid["B5"] + UP * 0.3, 
            self.grid["B5"] + DOWN * 0.5, 
            self.grid["B6"] + UP * 0.2
        )
        
        # Path C (Upper Middle)
        path_c = VMobject()
        path_c.set_points_as_corners([
            self.grid["C4"],
            self.grid["C4"] + UP * 0.5,
            self.grid["C5"] + DOWN * 0.8 + RIGHT * 0.2,
            self.grid["C5"] + UP * 0.4 + RIGHT * 0.5,
            self.grid["C6"]
        ]).make_smooth()
        
        # Path D (Interacts with Rock at D4)
        path_d = VMobject()
        path_d.set_points_as_corners([
            self.grid["D4"] + LEFT * 0.5,
            self.grid["D4"] + UP * 0.4 + RIGHT * 0.2,
            self.grid["D5"] + DOWN * 0.6,
            self.grid["D6"] + UP * 0.3
        ]).make_smooth()
        
        # Path E (Bottom)
        path_e = CubicBezier(
            self.grid["E4"],
            self.grid["E5"] + DOWN * 0.2,
            self.grid["E5"] + UP * 0.4,
            self.grid["E6"] + DOWN * 0.1
        )
        
        swirls.add(path_b, path_c, path_d, path_e)
        swirls.set_color(COLOR_TURBULENT)
        # Apply the same scale/positioning logic to the swirls group for Transform consistency
        # although here we defined them using grid points directly, which is more precise for the effect.
        # We'll rely on the grid points starting from Col 4.

        self.lecture[2].set_color(BLUE)
        self.play(
            Transform(laminar_lines, swirls),
            rock.animate.scale(1.2),
            run_time=3
        )
        self.wait(3)
