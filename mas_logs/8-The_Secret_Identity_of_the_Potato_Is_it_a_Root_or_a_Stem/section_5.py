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
        # Initial layout setup
        title = "The Great Comparison: Potato vs. Sweet Potato"
        lines = [
            "Compare the potato to its cousin, the sweet potato.",
            "The potato is a stem tuber with visible nodes.",
            "The sweet potato is a root tuber without nodes.",
            "X-rays reveal the potato's internal stem-like pith.",
            "Sweet potatoes have a central core and lateral roots."
        ]
        self.setup_layout(title, lines)

        # Create basic vegetable shapes using the grid
        # Potato: Rounder/Oval in area A1-C3
        potato = Ellipse(width=2.5, height=2.0, color="#D2B48C", fill_opacity=1)
        self.place_in_area(potato, "A1", "C3")

        # Sweet Potato: Pointier/Longer in area A4-C6
        sweet_potato = Ellipse(width=3.2, height=1.6, color="#A0522D", fill_opacity=1)
        self.place_in_area(sweet_potato, "A4", "C6")

        # Labels positioned strictly on grid
        potato_label = Text("Potato", font_size=24, color="#D2B48C")
        self.place_at_grid(potato_label, "D2")
        
        sweet_potato_label = Text("Sweet Potato", font_size=24, color="#A0522D")
        self.place_at_grid(sweet_potato_label, "D5")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(
            Create(potato), 
            Create(sweet_potato), 
            Write(potato_label), 
            Write(sweet_potato_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Eyes (nodes) using specific grid points within the potato area
        nodes = VGroup(
            self.place_at_grid(Dot(color=BLACK, radius=0.06), "A2"),
            self.place_at_grid(Dot(color=BLACK, radius=0.06), "B1"),
            self.place_at_grid(Dot(color=BLACK, radius=0.06), "B3"),
            self.place_at_grid(Dot(color=BLACK, radius=0.06), "C2")
        )

        # Blue rings for highlighting nodes, also using grid points
        rings = VGroup(
            self.place_at_grid(Circle(radius=0.18, color="#00BFFF"), "A2"),
            self.place_at_grid(Circle(radius=0.18, color="#00BFFF"), "B1"),
            self.place_at_grid(Circle(radius=0.18, color="#00BFFF"), "B3"),
            self.place_at_grid(Circle(radius=0.18, color="#00BFFF"), "C2")
        )

        self.play(Create(nodes))
        self.play(Create(rings))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Highlight sweet potato surface (no nodes) using a grid-based indicator
        self.play(Indicate(sweet_potato, color="#A0522D"))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Internal pith: vertical stem-like structure in the center of the potato area
        pith = Rectangle(width=0.4, height=1.6, color="#F0E68C", fill_opacity=0.9, stroke_width=0)
        self.place_at_grid(pith, "B2") # Center of potato area
        
        self.play(
            potato.animate.set_fill(opacity=0.3),
            nodes.animate.set_opacity(0.3),
            rings.animate.set_opacity(0.3),
            Create(pith)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Central core (horizontal structure in the sweet potato area)
        core = Rectangle(width=2.5, height=0.4, color="#DEB887", fill_opacity=0.9, stroke_width=0)
        self.place_at_grid(core, "B5") # Center of sweet potato area

        # Lateral roots branching from core area
        roots = VGroup(
            Line(self.grid["B5"], self.grid["A4"], color="#CD853F"),
            Line(self.grid["B5"], self.grid["A6"], color="#CD853F"),
            Line(self.grid["B5"], self.grid["C4"], color="#CD853F"),
            Line(self.grid["B5"], self.grid["C6"], color="#CD853F")
        )

        self.play(
            sweet_potato.animate.set_fill(opacity=0.3),
            Create(core),
            Create(roots)
        )
        self.wait(2)

        # Final state cleanup
        self.lecture[4].set_color(WHITE)
        self.wait(2)
