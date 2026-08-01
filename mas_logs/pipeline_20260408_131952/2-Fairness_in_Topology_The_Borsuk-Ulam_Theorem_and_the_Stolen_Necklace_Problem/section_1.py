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
        title = "The Hook: The Disputed Treasure"
        lines = [
            "Alice and Bob must share a precious stolen necklace.",
            "It contains ten red and eight green beads.",
            "How many cuts ensure each gets a fair split?"
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Alice (#FFD700) and Bob (#87CEEB) icons appear with a necklace
        self.lecture[0].set_color("#FFD700")
        
        # Alice Icon (Asset: person.svg)
        alice_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/person.svg", color="#FFD700")
        alice_label = Text("Alice", font_size=16, color="#FFD700").next_to(alice_icon, UP, buff=0.1)
        alice_group = VGroup(alice_icon, alice_label)
        self.place_at_grid(alice_group, "A2", scale_factor=0.8) # Issue 44: Scale factor 0.8

        # Bob Icon (Asset: person.svg)
        bob_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/person.svg", color="#87CEEB")
        bob_label = Text("Bob", font_size=16, color="#87CEEB").next_to(bob_icon, UP, buff=0.1)
        bob_group = VGroup(bob_icon, bob_label)
        self.place_at_grid(bob_group, "A5", scale_factor=0.8) # Issue 44: Scale factor 0.8

        # Necklace Construction (Asset: necklace.svg used as base decoration)
        necklace_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/necklace.svg", color=GREY_A)
        self.place_in_area(necklace_svg, "B3", "B4", scale_factor=0.4)
        
        necklace_line = Line(self.grid["B1"], self.grid["B6"], color=GREY_A, stroke_width=2)
        # Sequence with 10 Red and 8 Green
        colors = ["#FF0000", "#00FF00", "#FF0000", "#FF0000", "#00FF00", 
                  "#00FF00", "#FF0000", "#FF0000", "#FF0000", "#00FF00", 
                  "#FF0000", "#00FF00", "#FF0000", "#FF0000", "#00FF00", 
                  "#00FF00", "#00FF00", "#FF0000"]
        
        beads = VGroup()
        for i, color in enumerate(colors):
            dot = Dot(radius=0.12, color=color, fill_opacity=1)
            dot.move_to(necklace_line.point_from_proportion(i/17))
            beads.add(dot)
        
        self.play(FadeIn(alice_group), FadeIn(bob_group))
        self.play(Create(necklace_line), FadeIn(necklace_svg), LaggedStart(*[FadeIn(b) for b in beads], lag_ratio=0.05))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The necklace is labeled 'Stolen Necklace' and beads are counted: '10 Red, 8 Green'.
        self.lecture[1].set_color("#00FF00")
        
        stolen_text = Text("Stolen Necklace", font_size=24, color=WHITE)
        self.place_in_area(stolen_text, 'C2', 'C5', scale_factor=0.7) # Issue 45
        
        count_text = Text("10 Red, 8 Green", font_size=24, color=WHITE)
        self.place_in_area(count_text, 'D2', 'D5', scale_factor=0.7) # Issue 45
        
        # Apply colors to counts
        count_text[0:2].set_color("#FF0000") # "10"
        count_text[3:6].set_color("#FF0000") # "Red"
        count_text[8:9].set_color("#00FF00") # "8"
        count_text[10:].set_color("#00FF00") # "Green"

        self.play(Write(stolen_text))
        self.play(Write(count_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # A question mark appears next to a knife: 'How many cuts for a fair split?'.
        self.lecture[2].set_color(WHITE)
        
        # Knife graphic
        knife_handle = Rectangle(height=0.1, width=0.4, color="#8B4513", fill_opacity=1)
        knife_blade = Triangle(color="#C0C0C0", fill_opacity=1).scale(0.3).rotate(-90*DEGREES)
        knife_blade.next_to(knife_handle, RIGHT, buff=0)
        knife = VGroup(knife_handle, knife_blade).rotate(45*DEGREES)
        
        question_mark = Text("?", font_size=60, color=YELLOW)
        knife_group = VGroup(knife, question_mark).arrange(RIGHT, buff=0.8)
        self.place_in_area(knife_group, 'E2', 'F5', scale_factor=0.7) # Issue 43
        
        self.play(FadeIn(knife_group))
        self.wait(3)
