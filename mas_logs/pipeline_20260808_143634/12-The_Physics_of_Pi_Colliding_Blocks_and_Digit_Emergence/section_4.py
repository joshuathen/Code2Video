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
        lecture_lines = ["Mass ratios dictate the wedge angle size.", "As mass grows, reflections trace an arc.", "The arc length corresponds to value Pi."]
        self.setup_layout("The Limit of Mass Ratios", lecture_lines)
        
        # Assets
        wedge = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wedge.svg")
        
        # === Animation for Lecture Line 1 ===
        # Position wedge and color it #FFFFFF
        self.place_in_area(wedge, 'B2', 'E5', scale_factor=0.6)
        wedge.set_color("#FFFFFF")
        
        self.play(FadeIn(wedge))
        self.lecture[0].set_color("#FFFFFF")
        
        # === Animation for Lecture Line 2 ===
        # Draw arc representing path of reflection in #FFC300
        arc = Arc(radius=1.0, start_angle=0, angle=PI/2, color="#FFC300")
        arc.move_to(wedge.get_center())
        
        self.play(Create(arc))
        self.lecture[1].set_color("#FFC300")
        
        # === Animation for Lecture Line 3 ===
        # Highlight final arc length related to Pi in wedge #FF5733
        # Position y_dot_label at C4
        y_dot_label = Text("Pi", font_size=24, color="#FF5733")
        self.place_at_grid(y_dot_label, 'C4', scale_factor=0.7)
        
        self.play(wedge.animate.set_color("#FF5733"), Write(y_dot_label))
        self.lecture[2].set_color("#FF5733")
        self.wait(2)
