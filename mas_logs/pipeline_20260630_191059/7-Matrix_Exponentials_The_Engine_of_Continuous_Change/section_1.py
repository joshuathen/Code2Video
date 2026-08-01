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
        # Setup layout with title and lecture lines
        title_text = "The Hook: The Growth of a Digital Colony"
        lecture_lines = [
            "Meet our digital colony: Cyber-Rabbits and Digital-Foxes.",
            "A single population grows proportionally to its size.",
            "But these two populations are coupled together.",
            "Matrix A describes how their populations interact.",
            "This linear system models their continuous evolution."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        RABBIT_COLOR = "#00FF00"
        FOX_COLOR = "#FF8C00"
        MATRIX_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Meet our digital colony: Cyber-Rabbits and Digital-Foxes.
        self.play(self.lecture[0].animate.set_color(RABBIT_COLOR))
        
        # Asset for Rabbit
        # [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/rabbit.svg]
        try:
            rabbit_icon = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/rabbit.svg", color=RABBIT_COLOR).scale(0.4)
        except:
            # Fallback if asset is missing
            rabbit_icon = Square(color=RABBIT_COLOR, fill_opacity=0.6).scale(0.4)
            
        rabbit_label = Text("Rabbit", font_size=16, color=RABBIT_COLOR)
        rabbit_group = VGroup(rabbit_icon, rabbit_label).arrange(DOWN, buff=0.1)
        self.place_at_grid(rabbit_group, "B2")

        # Fox icon (Using Circle as an alternative icon since no asset was provided)
        fox_icon = Circle(color=FOX_COLOR, fill_opacity=0.6).scale(0.4)
        fox_label = Text("Fox", font_size=16, color=FOX_COLOR)
        fox_group = VGroup(fox_icon, fox_label).arrange(DOWN, buff=0.1)
        self.place_at_grid(fox_group, "B5")

        self.play(FadeIn(rabbit_group), FadeIn(fox_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A single population grows proportionally to its size.
        self.play(self.lecture[1].animate.set_color(MATRIX_COLOR))
        
        scalar_eq = Text("dx/dt = ax", color=WHITE)
        self.place_in_area(scalar_eq, "C2", "C5", scale_factor=0.9)
        
        self.play(Write(scalar_eq))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # But these two populations are coupled together.
        self.play(self.lecture[2].animate.set_color(MATRIX_COLOR))
        
        vector_x = Text(
            "x = [r, f]", 
            color=WHITE, 
            t2c={"r": RABBIT_COLOR, "f": FOX_COLOR}
        )
        # Fix from Issue 29: scale_factor=0.8 to avoid crowding
        self.place_at_grid(vector_x, "D2", scale_factor=0.8)

        self.play(Write(vector_x))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Matrix A describes how their populations interact.
        self.play(self.lecture[3].animate.set_color(MATRIX_COLOR))
        
        matrix_a = Text(
            "A = [[a, b], [c, d]]", 
            color=MATRIX_COLOR
        )
        # Fix from Issue 27: Move to 'D6' and scale to 0.8 to avoid horizontal overlap
        self.place_at_grid(matrix_a, "D6", scale_factor=0.8)
        
        self.play(Write(matrix_a))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This linear system models their continuous evolution.
        self.play(self.lecture[4].animate.set_color(MATRIX_COLOR))

        system_eq = Text(
            "dx/dt = Ax", 
            color=WHITE,
            t2c={"A": MATRIX_COLOR}
        )
        # Fix from Issue 28: Scale factor reduced to 1.0 for better visual balance
        self.place_in_area(system_eq, "E2", "F5", scale_factor=1.0)

        self.play(Write(system_eq))
        self.wait(2)
