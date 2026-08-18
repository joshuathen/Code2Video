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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "SIR defines three key population states.",
            "Susceptible people haven't been infected yet.",
            "Infected individuals are currently spreading the virus.",
            "Recovered individuals are now immune to reinfection.",
            "These states track the entire disease progression."
        ]
        self.setup_layout("The SIR Framework: Defining the States", lecture_lines)
        
        # Asset Loading
        s_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/person.svg")
        i_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/virus.svg")
        r_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/person.svg")
        
        # Elements
        s_box = Square(side_length=1.5, fill_opacity=0.6, color="#0000FF").set_fill("#0000FF", opacity=0.6)
        i_box = Square(side_length=1.5, fill_opacity=0.6, color="#FF0000").set_fill("#FF0000", opacity=0.6)
        r_box = Square(side_length=1.5, fill_opacity=0.6, color="#00FF00").set_fill("#00FF00", opacity=0.6)
        
        s_group = VGroup(s_box, s_icon)
        i_group = VGroup(i_box, i_icon)
        r_group = VGroup(r_box, r_icon)
        
        s_label = Text("S", font_size=48, color="#FFFFFF")
        i_label = Text("I", font_size=48, color="#FFFFFF")
        r_label = Text("R", font_size=48, color="#FFFFFF")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#0000FF")
        self.place_at_grid(s_group, 'B2', scale_factor=0.8)
        self.place_at_grid(s_label, 'C2', scale_factor=1.0)
        self.play(FadeIn(s_group), FadeIn(s_label))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF0000")
        self.place_at_grid(i_group, 'B4', scale_factor=0.8)
        self.place_at_grid(i_label, 'C4', scale_factor=1.0)
        self.play(FadeIn(i_group), FadeIn(i_label))
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FF00")
        self.place_at_grid(r_group, 'B6', scale_factor=0.8)
        self.place_at_grid(r_label, 'C6', scale_factor=1.0)
        self.play(FadeIn(r_group), FadeIn(r_label))
        
        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFFFF")
        arrow1 = Arrow(s_group.get_right(), i_group.get_left(), color="#FFFFFF")
        arrow2 = Arrow(i_group.get_right(), r_group.get_left(), color="#FFFFFF")
        flow_arrows = VGroup(arrow1, arrow2)
        self.place_in_area(flow_arrows, 'B3', 'B5', scale_factor=0.9)
        self.play(Create(flow_arrows))
        self.wait(2)
