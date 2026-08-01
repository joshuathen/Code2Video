from manim import *
import numpy as np

# Fix potential KeyError in Manim's config by cleaning the input file path of braces.
if hasattr(config, "input_file") and config.input_file:
    config.input_file = str(config.input_file).replace("{", "").replace("}", "")

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

class Section1Scene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        """
        Sets up the educational layout with a title and bullet points.
        """
        # Set background
        self.camera.background_color = "#000000"
        
        # Add Title
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP, buff=0.5)
        self.add(self.title)

        # Left-side lecture content (bullets)
        lecture_texts = [Text("- " + line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT, buff=0.4).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.5)
        self.add(self.lecture)

        # Define animation grid (6x6 grid on right side) for positioning elements
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                # Offset x to the right half of the screen
                x = 1.5 + j * 0.8
                y = 2.0 - i * 0.8
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def construct(self):
        # Initialize the layout with content relevant to Euler's Identity
        self.setup_layout(
            "The Visual Harmony of Euler's Identity",
            [
                "Defining e, i, and pi",
                "The complex plane rotation",
                "Angle of pi radians (180 deg)",
                "The result: -1 on the real axis",
                "Mathematical elegance"
            ]
        )
        
        # Visualize the calculated grid points
        grid_dots = VGroup(*[
            Dot(point=pos, radius=0.05, color=BLUE_B) 
            for pos in self.grid.values()
        ])
        
        # Add labels to the grid points for reference
        grid_labels = VGroup(*[
            Text(key, font_size=10, color=GRAY).next_to(self.grid[key], DOWN, buff=0.1)
            for key in self.grid.keys()
        ])
        
        # Main formula display using Text to avoid 'latex' dependency issues
        formula = Text("e^{iπ} = -1", font_size=48, color=YELLOW)
        formula.move_to(self.grid["C3"])

        # Animations
        self.play(
            Create(grid_dots), 
            Write(grid_labels), 
            run_time=1.5
        )
        self.wait(0.5)
        
        self.play(
            Write(formula),
            self.lecture.animate.set_color_by_gradient(WHITE, BLUE_A),
            run_time=2
        )
        
        self.play(
            Indicate(formula, scale_factor=1.2),
            formula.animate.set_color(WHITE),
            run_time=1
        )
        
        self.wait(2)
