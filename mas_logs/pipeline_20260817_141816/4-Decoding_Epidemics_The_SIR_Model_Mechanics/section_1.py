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
        self.setup_layout("The Hook: Why do diseases spread?", ["Diseases spread through population interactions.", "Individuals occupy three distinct states.", "Susceptible, Infectious, and Recovered."])
        
        # Define nodes with assets
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/person.svg
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/microbe.svg
        
        s_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/person.svg", color=WHITE)
        s_label = Text("S", font_size=24)
        s_group = VGroup(s_icon, s_label).arrange(DOWN)
        
        i_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microbe.svg", color=WHITE)
        i_label = Text("I", font_size=24)
        i_group = VGroup(i_icon, i_label).arrange(DOWN)
        
        r_node = Circle(radius=0.4, color=WHITE, fill_opacity=0.5)
        r_label = Text("R", font_size=24)
        r_group = VGroup(r_node, r_label).arrange(DOWN)
        
        # Layout improvement per critiques
        self.place_at_grid(s_group, 'D2', scale_factor=0.6)
        self.place_at_grid(i_group, 'D4', scale_factor=0.6)
        self.place_at_grid(r_group, 'D6', scale_factor=0.6)
        
        transmission_arrow1 = Arrow(start=s_group.get_right(), end=i_group.get_left(), color=WHITE)
        transmission_arrow2 = Arrow(start=i_group.get_right(), end=r_group.get_left(), color=WHITE)
        
        self.add(s_group, i_group, r_group, transmission_arrow1, transmission_arrow2)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(s_group), FadeIn(i_group), FadeIn(r_group))

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(WHITE),
            s_icon.animate.set_color("#FFD700"),
            i_icon.animate.set_color("#FF4500"),
            r_node.animate.set_color("#00CED1")
        )

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            s_label.animate.set_color("#FFD700"),
            i_label.animate.set_color("#FF4500"),
            r_label.animate.set_color("#00CED1")
        )
        self.wait(1)
