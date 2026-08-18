from manim import *

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
        self.setup_layout("The Dimensional Paradox", ["Can a 1D line cover a 2D area?", "Intuition suggests infinite folding is required.", "We challenge the boundary between dimensions."])
        
        # Define elements
        # Using placeholder icon for the requested asset paths
        icon_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg"
        icon = SVGMobject(icon_path)
        
        square = Square(side_length=2.0, color="#FFFFFF")
        boundary = Square(side_length=2.0, color="#FF00FF")
        line_1d = Line(start=LEFT*1, end=RIGHT*1, color="#00FFFF")
        paradox_text = Text("Line Filling Square", color="#FFFF00", font_size=24)
        density_label = Text("Infinite Density", color="#FF0000", font_size=20)

        # Apply constraints and fixes from VideoCritic issues
        self.place_in_area(square, 'B3', 'C4', scale_factor=0.9)
        self.place_in_area(boundary, 'B3', 'C4', scale_factor=0.9)
        self.place_at_grid(line_1d, 'D4', scale_factor=0.8)
        self.place_at_grid(paradox_text, 'E2', scale_factor=0.8)
        self.place_at_grid(density_label, 'E5', scale_factor=0.7)
        self.place_at_grid(icon, 'A5', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(Create(square))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(Create(boundary))
        self.play(Create(line_1d))
        self.lecture[1].set_color("#FF00FF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(Write(paradox_text), Write(density_label))
        self.lecture[2].set_color("#00FFFF")
        self.wait(2)
