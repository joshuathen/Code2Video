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
        # Title and lecture lines for Section 1
        title_text = "Vectors as Physical Movements"
        lecture_lines = [
            "Start with a standard Cartesian coordinate system.",
            "Think of a vector as a movement through space.",
            "An arrow from the origin shows direction and distance."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        WHITE_COLOR = "#FFFFFF"
        BLUE_COLOR = "#0000FF"

        # === Animation for Lecture Line 1 ===
        # Step 1: Show a faint white (#FFFFFF) coordinate grid.
        # Shifted origin to D3 to avoid crowding (Issue 28).
        grid_plane = NumberPlane(
            x_range=[-2, 4, 1],
            y_range=[-2, 3, 1],
            background_line_style={
                "stroke_color": WHITE_COLOR,
                "stroke_width": 1,
                "stroke_opacity": 0.2
            }
        )
        self.place_at_grid(grid_plane, "D3")
        
        self.play(
            self.lecture[0].animate.set_color(WHITE_COLOR),
            Create(grid_plane)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Step 2: Think of a vector as a movement through space.
        # Robot starts at origin D3 (Issue 29).
        robot = Circle(radius=0.15, color=WHITE_COLOR, fill_opacity=1)
        self.place_at_grid(robot, "D3")
        
        self.play(
            self.lecture[1].animate.set_color(WHITE_COLOR),
            FadeIn(robot)
        )
        
        # Animate robot moving 3 units right, then 2 units up.
        # Origin at D3.
        # Right 3: D3 -> D6.
        # Up 2: D6 -> B6.
        self.play(robot.animate.move_to(self.grid["D6"]), run_time=1)
        self.play(robot.animate.move_to(self.grid["B6"]), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Step 3: An arrow from the origin shows direction and distance.
        # Draw bold blue (#0000FF) arrow from origin (D3) to end (B6).
        vector_arrow = Arrow(
            start=self.grid["D3"],
            end=self.grid["B6"],
            buff=0,
            color=BLUE_COLOR,
            stroke_width=6
        )
        
        # Label the arrow '[3, 2]' in blue (#0000FF).
        # Shifted label to A6 to avoid overlap with robot (Issue 30).
        label = Text("[3, 2]", color=BLUE_COLOR)
        self.place_at_grid(label, "A6", scale_factor=0.6)
        
        self.play(
            self.lecture[2].animate.set_color(BLUE_COLOR),
            GrowArrow(vector_arrow)
        )
        self.play(Write(label))
        self.wait(2)
