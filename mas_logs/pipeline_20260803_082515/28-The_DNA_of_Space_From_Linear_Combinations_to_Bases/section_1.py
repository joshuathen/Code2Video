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
        # Fetching data from storyboard
        title_text = "Prerequisites: Vectors as Directions"
        lecture_lines = [
            "Think of vectors as movement instructions, not points.",
            "Scaling a vector stretches or shrinks its length.",
            "Adding vectors follows a simple tip-to-tail path."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors matching the theme and animation elements
        COLOR_V1 = "#00FF00"  # Green for North
        COLOR_V2 = "#0000FF"  # Blue for East
        COLOR_SCALE = "#FFFF00"  # Yellow for Scaling
        COLOR_ADD = "#FF00FF"  # Magenta for Addition

        # Setup Coordinate System (NumberPlane) on the right side area (A1 to F6)
        plane = NumberPlane(
            x_range=[-1, 4, 1],
            y_range=[-1, 4, 1],
            x_length=4.5,
            y_length=4.5,
            background_line_style={"stroke_opacity": 0.3}
        )
        self.place_in_area(plane, 'A1', 'F6')
        self.add(plane)

        # === Animation for Lecture Line 1 ===
        # "Think of vectors as movement instructions, not points."
        # No specific color change for Line 1 in instructions, but usually we highlight current line.
        # I will keep it White for now or highlight it if implied. Let's stick to white.
        
        v1 = Arrow(plane.c2p(0, 0), plane.c2p(0, 1), buff=0, color=COLOR_V1)
        north_label = Text("North", font_size=24, color=COLOR_V1)
        # Fix for Issue 19: place_at_grid 'D1'
        self.place_at_grid(north_label, 'D1', scale_factor=0.6)

        v2 = Arrow(plane.c2p(0, 0), plane.c2p(1, 0), buff=0, color=COLOR_V2)
        east_label = Text("East", font_size=24, color=COLOR_V2)
        # Fix for Issue 20: place_at_grid 'F3'
        self.place_at_grid(east_label, 'F3', scale_factor=0.6)

        self.play(
            GrowArrow(v1),
            Write(north_label),
            run_time=1
        )
        self.play(
            GrowArrow(v2),
            Write(east_label),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Scaling a vector stretches or shrinks its length."
        self.play(self.lecture[1].animate.set_color(COLOR_SCALE))
        
        v1_scaled = Arrow(plane.c2p(0, 0), plane.c2p(0, 2), buff=0, color=COLOR_SCALE)
        scaled_north_label = Text("2 * North", font_size=24, color=COLOR_SCALE)
        # Fix for Issue 19: place_at_grid 'C1'
        self.place_at_grid(scaled_north_label, 'C1', scale_factor=0.6)

        self.play(
            Transform(v1.copy(), v1_scaled),
            Write(scaled_north_label),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Adding vectors follows a simple tip-to-tail path."
        self.play(self.lecture[2].animate.set_color(COLOR_ADD))

        # Show addition by moving v2 to tip of v1
        # Target position for v2 is from (0,1) to (1,1)
        v2_shifted = Arrow(plane.c2p(0, 1), plane.c2p(1, 1), buff=0, color=COLOR_V2)
        
        v_sum = Arrow(plane.c2p(0, 0), plane.c2p(1, 1), buff=0, color=COLOR_ADD)
        sum_label = Text("North + East", font_size=24, color=COLOR_ADD)
        # Fix for Issue 21: place_in_area 'D4' to 'D5'
        self.place_in_area(sum_label, 'D4', 'D5', scale_factor=0.5)

        self.play(
            Transform(v2, v2_shifted),
            east_label.animate.set_opacity(0), # Hide old label
            run_time=1.5
        )
        self.play(
            GrowArrow(v_sum),
            Write(sum_label),
            run_time=1
        )
        self.wait(2)
