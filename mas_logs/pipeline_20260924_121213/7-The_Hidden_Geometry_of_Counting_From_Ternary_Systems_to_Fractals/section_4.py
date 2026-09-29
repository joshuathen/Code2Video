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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Ternary numbers are our fractal GPS.",
            "They locate vertices within the structure.",
            "The graph represents the total state-space.",
            "Logic and geometry unify perfectly here.",
            "This concludes our hidden geometry exploration."
        ]
        self.setup_layout("Synthesis: The Deep Connection", lecture_lines)
        
        # Sierpinski Triangle (simplified)
        sierpinski = Polygon(UP * 2, LEFT * 2 + DOWN * 2, RIGHT * 2 + DOWN * 2, color=BLUE_D)
        self.place_in_area(sierpinski, 'B4', 'D5', scale_factor=0.5)

        # GPS Icon
        gps_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/gps.svg")
        self.place_at_grid(gps_icon, 'B3', scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#8ECAE6"))
        gps_label = Text("Ternary: (1, 0, 2)", color="#8ECAE6", font_size=24)
        self.place_at_grid(gps_label, 'B3', scale_factor=0.8)
        self.play(FadeIn(gps_label), FadeIn(gps_icon))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFB703"))
        target_dot = Dot(color="#FFB703").move_to(sierpinski.get_center() + RIGHT * 0.5 + UP * 0.2)
        self.play(Create(target_dot))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FB8500"))
        state_space_box = Rectangle(color="#FB8500", width=2, height=2)
        self.place_in_area(state_space_box, 'E4', 'F6', scale_factor=0.7)
        self.play(Create(state_space_box))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#219EBC"))
        self.play(
            sierpinski.animate.set_color("#219EBC"),
            state_space_box.animate.set_color("#219EBC")
        )

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFFFFF"))
        self.wait(2)
