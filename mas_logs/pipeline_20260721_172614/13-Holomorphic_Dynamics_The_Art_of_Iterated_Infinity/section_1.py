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
        self.setup_layout("The Playground: The Complex Plane", [
            "Welcome to the playground of holomorphic dynamics.",
            "We begin with the infinite complex plane.",
            "Every point is defined by real and imaginary parts.",
            "This plane is a dynamic map for movement.",
            "Watch as points shift across this mathematical space."
        ])
        
        # Colors for lines
        c1 = "#ADD8E6" # Light Blue
        c2 = "#FFFFFF" # White
        c3 = "#FFFFFF" # White
        c4 = "#FFFF00" # Yellow
        c5 = "#FFFF00" # Yellow

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(c1))
        title_obj = Text("Holomorphic Dynamics", color=c1, font_size=36)
        self.place_in_area(title_obj, "B2", "E5")
        self.play(Write(title_obj))
        self.wait(1)
        self.play(FadeOut(title_obj))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(c2))
        grid = NumberPlane(
            x_range=[-2.5, 2.5, 1],
            y_range=[-2.5, 2.5, 1],
            x_length=5,
            y_length=5,
            background_line_style={"stroke_color": WHITE, "stroke_opacity": 0.3},
            axis_config={"stroke_color": WHITE}
        )
        self.place_in_area(grid, "A1", "F6")
        self.play(Create(grid))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(c3))
        real_label = Text("Real", color="#ADD8E6", font_size=20)
        imag_label = Text("Imaginary", color="#FFB6C1", font_size=20)
        
        # Manually positioned relative to axes ends
        # Fixes for Issue 22, 23, 24
        self.place_at_grid(real_label, "C6", scale_factor=0.6)
        self.place_in_area(imag_label, "A3", "A4", scale_factor=0.6)
        
        self.play(Write(real_label), Write(imag_label))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(c4))
        zoe_start_coords = [1, 1]
        zoe = Dot(point=grid.c2p(*zoe_start_coords), color=YELLOW)
        zoe_label = Text("Zoe", color=YELLOW, font_size=18)
        zoe_label.next_to(zoe, UR, buff=0.1)
        
        self.play(FadeIn(zoe), Write(zoe_label))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(c5))
        zoe_end_coords = [2, -1]
        
        # persistent tracker for movement
        move_tracker = ValueTracker(0)
        
        # update positions
        zoe.add_updater(lambda m: m.move_to(
            interpolate(grid.c2p(*zoe_start_coords), grid.c2p(*zoe_end_coords), move_tracker.get_value())
        ))
        zoe_label.add_updater(lambda m: m.next_to(zoe, UR, buff=0.1))
        
        self.play(move_tracker.animate.set_value(1), run_time=2)
        self.wait(2)
        
        # cleanup
        zoe.clear_updaters()
        zoe_label.clear_updaters()
