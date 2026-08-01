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
        # Script and Setup based on prompt requirements
        title = "Welcome to the Rubber-Sheet World"
        lines = [
            'Welcome to Topology, the study of rubber-sheet geometry.',
            'Imagine a shape like this simple square outline.',
            'It can smoothly morph into a circle without tearing.'
        ]
        self.setup_layout(title, lines)

        # Colors from prompt
        COLOR_TEXT = "#56B4E9"
        COLOR_SHAPE = "#0072B2"

        # === Animation for Lecture Line 1 ===
        # Fade in the text 'TOPOLOGY' in a light blue color (#56B4E9)
        # Position adjusted per Issue 31
        topo_text = Text("TOPOLOGY", color=COLOR_TEXT)
        self.place_at_grid(topo_text, 'B4', scale_factor=1.2)
        
        self.play(
            FadeIn(topo_text),
            self.lecture[0].animate.set_color(COLOR_TEXT),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transform 'TOPOLOGY' into square outline (#0072B2) on sheet icon asset
        # Position adjusted per Issue 32
        square = Square(side_length=3.0, color=COLOR_SHAPE)
        self.place_in_area(square, 'C3', 'E5', scale_factor=0.8)
        
        # Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/sheet.svg
        sheet_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/sheet.svg")
        self.place_in_area(sheet_icon, 'C3', 'E5', scale_factor=0.8)
        sheet_icon.set_opacity(0.3) # Subtle backdrop
        
        self.play(
            ReplacementTransform(topo_text, square),
            FadeIn(sheet_icon),
            self.lecture[1].animate.set_color(COLOR_SHAPE),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Morph square outline into circle outline (#0072B2) with rubber icon asset
        # Position adjusted per Issue 33
        circle = Circle(radius=1.5, color=COLOR_SHAPE)
        self.place_in_area(circle, 'C3', 'E5', scale_factor=0.8)
        
        # Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/rubber.svg
        rubber_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/rubber.svg")
        self.place_in_area(rubber_icon, 'C3', 'E5', scale_factor=0.8)
        rubber_icon.set_opacity(0.3) # Subtle backdrop
        
        self.play(
            ReplacementTransform(square, circle),
            ReplacementTransform(sheet_icon, rubber_icon),
            self.lecture[2].animate.set_color(COLOR_SHAPE),
            run_time=2
        )
        self.wait(2)
