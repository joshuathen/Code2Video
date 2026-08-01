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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "The Chain Rule multiplies successive local scaling factors.",
            "First scaling is three and second scaling is two.",
            "The total transformation scales the input by six."
        ]
        self.setup_layout("Synthesis: The Chain Rule as Scaling", lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Horizontal lines for X, Y, Z at rows A, C, E
        line_x = Line(self.grid['A2'], self.grid['A6'], color="#FFFFFF")
        line_y = Line(self.grid['C2'], self.grid['C6'], color="#FFFFFF")
        line_z = Line(self.grid['E2'], self.grid['E6'], color="#FFFFFF")
        
        # Labels for the lines
        label_x = Text("X", color=WHITE)
        self.place_at_grid(label_x, 'A1', scale_factor=0.8)
        label_y = Text("Y", color=WHITE)
        self.place_at_grid(label_y, 'C1', scale_factor=0.8)
        label_z = Text("Z", color=WHITE)
        self.place_at_grid(label_z, 'E1', scale_factor=0.8)
        
        self.play(
            Create(line_x), Create(line_y), Create(line_z),
            Write(label_x), Write(label_y), Write(label_z)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Define segments demonstrating scaling
        # X segment (length 0.5 units)
        x_start = self.grid['A2']
        x_end = (self.grid['A2'] + self.grid['A3']) / 2
        x_seg = Line(x_start, x_end, color=BLUE_B, stroke_width=8)
        
        # Y segment (3x larger than X = 1.5 units)
        y_start = self.grid['C2']
        y_end = (self.grid['C3'] + self.grid['C4']) / 2
        y_seg = Line(y_start, y_end, color=ORANGE, stroke_width=8)
        
        # Z segment (2x larger than Y = 3.0 units)
        z_start = self.grid['E2']
        z_end = self.grid['E5']
        z_seg = Line(z_start, z_end, color=PURPLE_B, stroke_width=8)
        
        # Mapping arrows and labels
        arrow_xy = Arrow(self.grid['B2'], self.grid['B2'] + DOWN * 0.7, color=WHITE, buff=0)
        label_xy = Text("x3", color=WHITE)
        self.place_at_grid(label_xy, 'B2', scale_factor=0.7)
        
        arrow_yz = Arrow(self.grid['D2'], self.grid['D2'] + DOWN * 0.7, color=WHITE, buff=0)
        label_yz = Text("x2", color=WHITE)
        self.place_at_grid(label_yz, 'D2', scale_factor=0.7)
        
        self.play(Create(x_seg), Create(y_seg), Create(z_seg))
        self.play(
            GrowArrow(arrow_xy), Write(label_xy),
            GrowArrow(arrow_yz), Write(label_yz)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Total scaling arrow from X directly to Z
        total_arrow = CurvedArrow(
            self.grid['A6'], 
            self.grid['E6'], 
            angle=-PI/3, 
            color="#00FF00"
        )
        total_label = Text("x6", color="#00FF00")
        self.place_at_grid(total_label, 'C6', scale_factor=0.8)
        
        self.play(Create(total_arrow), Write(total_label))
        self.wait(2)
