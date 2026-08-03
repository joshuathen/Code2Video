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
            "Rare events make false positives very likely.",
            "Suppose only one percent of people are secret agents.",
            "A robot detector is ninety-five percent accurate.",
            "A positive ping still leaves the probability low.",
            "Most pings come from the large non-agent population."
        ]
        self.setup_layout("Application: The Robot Spy Detector Paradox", lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        
        # 10x10 grid of squares
        squares = VGroup()
        agent_square = None
        for i in range(10):
            for j in range(10):
                sq = Square(side_length=0.3, fill_opacity=0.8, stroke_width=1)
                if i == 0 and j == 0:
                    sq.set_fill("#FF0000") # 1% Agent
                    agent_square = sq
                else:
                    sq.set_fill("#808080") # 99% Normal (Grey per storyboard)
                squares.add(sq)
        
        squares.arrange_in_grid(rows=10, cols=10, buff=0.05)
        # Issue 26 Fix: place_in_area(squares, 'B1', 'E6', scale_factor=0.8)
        self.place_in_area(squares, "B1", "E6", scale_factor=0.8)
        
        self.play(FadeIn(squares))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        
        # Issue 27 Fix: place_in_area(accuracy_label, 'A1', 'A6', scale_factor=0.8)
        accuracy_label = MathTex(r"P(\text{Ping}|\text{Agent}) = 0.95", font_size=24, color=WHITE)
        self.place_in_area(accuracy_label, "A1", "A6", scale_factor=0.8)
        
        self.play(Write(accuracy_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(YELLOW))
        
        # Highlight 1 agent and 5 false positives
        # Storyboard: Apply a yellow glow (#FFFF00) to the red square and to 5 random grey squares.
        pinged_grey = squares[1:6] # First 5 grey squares for simplicity
        glows = VGroup()
        for sq in [agent_square] + list(pinged_grey):
            glow = SurroundingRectangle(sq, color="#FFFF00", buff=0.02, stroke_width=2)
            glows.add(glow)
            
        self.play(Create(glows))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(YELLOW))
        
        # Storyboard: Move the glowing red square and the 5 glowing grey squares to the center, grouped together.
        pinged_group = VGroup()
        for sq, glow in zip([agent_square] + list(pinged_grey), glows):
            # We want to group the square and its glow
            pinged_group.add(VGroup(sq, glow))
            
        # Draw a circle around the group
        enclosing_circle = Circle(color=WHITE, stroke_width=2).surround(pinged_group, buffer_factor=1.2)
        final_cluster = VGroup(pinged_group, enclosing_circle)
        
        # Issue 28 Fix: place_in_area(ratio_text, 'F1', 'F6', scale_factor=0.8)
        ratio_text = MathTex(r"P(\text{Agent}|\text{Ping}) = \frac{1}{1 + 5} = \frac{1}{6}", font_size=32, color=WHITE)
        self.place_in_area(ratio_text, "F1", "F6", scale_factor=0.8)
        
        # Non-pinged squares fade out
        non_pinged = VGroup(*[sq for sq in squares if sq not in [agent_square] + list(pinged_grey)])
        
        # Animation: move pinged to center of their area (C3-D4 roughly)
        target_pos = self.grid["C3"] # Center
