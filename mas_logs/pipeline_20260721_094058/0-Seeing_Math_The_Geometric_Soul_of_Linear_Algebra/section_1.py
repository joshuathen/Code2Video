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
        # LECTURE DATA
        title = "Vectors: Movement, Not Just Numbers"
        lines = [
            "A vector is more than just a list of numbers.",
            "Imagine it as an arrow pointing in a direction.",
            "This arrow represents a specific movement through space.",
            "Its length, or magnitude, shows the distance moved.",
            "Watch the arrow stretch as its values increase."
        ]
        
        self.setup_layout(title, lines)
        
        # Colors
        COLOR_GREEN = "#00FF00"
        COLOR_YELLOW = "#FFFF00"
        COLOR_BLUE = "#0000FF"
        COLOR_GRAY = "#A9A9A9"
        COLOR_ORANGE = "#FFA500"
        COLOR_WHITE = "#FFFFFF"

        # Assets
        DRONE_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/drone.svg"

        # Initialize Coordinate Grid
        plane = NumberPlane(
            x_range=[0, 5, 1],
            y_range=[0, 4, 1],
            background_line_style={"stroke_color": COLOR_WHITE, "stroke_width": 1, "stroke_opacity": 0.3},
            axis_config={"stroke_color": COLOR_WHITE}
        )
        # Scale to fit the area nicely
        self.place_in_area(plane, "A1", "F6", scale_factor=0.8)
        origin = plane.c2p(0, 0)

        # === Animation for Lecture Line 1 ===
        # A vector is more than just a list of numbers.
        self.lecture[0].set_color(COLOR_GREEN)
        self.play(Create(plane))
        
        vector_1 = Arrow(
            start=origin,
            end=plane.c2p(2, 1),
            buff=0,
            color=COLOR_GREEN,
            stroke_width=6
        )
        label_vector = Text("Vector", font_size=22, color=COLOR_GREEN)
        self.place_at_grid(label_vector, 'C5', scale_factor=0.6)
        
        self.play(GrowArrow(vector_1), Write(label_vector))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Imagine it as an arrow pointing in a direction.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_YELLOW)
        
        # Flash the arrow's length (magnitude) and tip (direction)
        tip_circle = Circle(radius=0.1, color=COLOR_YELLOW).move_to(vector_1.get_end())
        self.play(
            Indicate(vector_1, color=COLOR_YELLOW),
            Flash(vector_1.get_end(), color=COLOR_YELLOW, line_length=0.2),
            Create(tip_circle)
        )
        self.play(FadeOut(tip_circle))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # This arrow represents a specific movement through space.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_BLUE)
        
        # Clear previous vector
        self.play(FadeOut(vector_1), FadeOut(label_vector))
        
        # Drone movement [Asset: drone.svg]
        drone = SVGMobject(DRONE_ASSET)
        drone.set_color(COLOR_BLUE)
        drone.scale(0.3)
        drone.move_to(origin)
        
        path = DashedLine(origin, plane.c2p(3, 2), color=COLOR_GRAY)
        
        self.play(Create(path))
        self.play(drone.animate.move_to(plane.c2p(3, 2)), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Its length, or magnitude, shows the distance moved.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_WHITE)
        
        # Bold white vector arrow (0,0) to (3,2)
        vector_2 = Arrow(
            start=origin,
            end=plane.c2p(3, 2),
            buff=0,
            color=COLOR_WHITE,
            stroke_width=8
        )
        label_coords = Text("[3, 2]", font_size=22, color=COLOR_WHITE)
        self.place_at_grid(label_coords, 'C5', scale_factor=0.6)
        
        self.play(FadeOut(drone), FadeOut(path))
        self.play(GrowArrow(vector_2), Write(label_coords))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Watch the arrow stretch as its values increase.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_ORANGE)
        
        # Scale the [3, 2] vector by 1.5x. Change color to orange.
        # Use value tracker or updater if needed, but animate with scale works well here.
        self.play(
            vector_2.animate.scale(1.5, about_point=origin).set_color(COLOR_ORANGE),
            label_coords.animate.set_color(COLOR_ORANGE),
            run_time=2
        )
        # Update label position to B6 as per Refinement issue
        self.play(
            label_coords.animate.move_to(self.grid['B6']).scale(1.0), # scale factor was already 0.6
            run_time=1
        )
        self.wait(2)
