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
        self.setup_layout(
            "Mapping States to Geometry", 
            [
                'Represent each valid disk configuration as a vertex.', 
                'Connect configurations reachable by a single legal move.', 
                'For two disks, states group into three clusters.', 
                'Connecting these clusters forms a larger triangular structure.', 
                'The final graph maps every possible puzzle state.'
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Create 3 dots for 1-disk states - Issue 41: Moved to Row B
        dot_0 = Dot(color="#00FFFF", radius=0.1)
        dot_1 = Dot(color="#00FFFF", radius=0.1)
        dot_2 = Dot(color="#00FFFF", radius=0.1)
        
        self.place_at_grid(dot_0, "B2", scale_factor=0.8)
        self.place_at_grid(dot_1, "B3", scale_factor=0.8)
        self.place_at_grid(dot_2, "B4", scale_factor=0.8)
        
        label_0 = Text("0", font_size=18).next_to(dot_0, DOWN, buff=0.2)
        label_1 = Text("1", font_size=18).next_to(dot_1, DOWN, buff=0.2)
        label_2 = Text("2", font_size=18).next_to(dot_2, DOWN, buff=0.2)
        
        self.play(
            FadeIn(dot_0), FadeIn(dot_1), FadeIn(dot_2),
            Write(label_0), Write(label_1), Write(label_2)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Connect dots with white lines
        line_01 = Line(dot_0.get_center(), dot_1.get_center(), color=WHITE, stroke_width=2)
        line_12 = Line(dot_1.get_center(), dot_2.get_center(), color=WHITE, stroke_width=2)
        
        self.play(Create(line_01), Create(line_12))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Clear the 1-disk visualization
        self.play(
            FadeOut(dot_0, dot_1, dot_2, label_0, label_1, label_2, line_01, line_12)
        )

        # Define 9 vertices for 2-disk states
        # Group 2 (Top): Disk 2 at Peg 2
        v22 = Dot(color="#00FFFF", radius=0.08) # (2,2)
        v20 = Dot(color="#00FFFF", radius=0.08) # (2,0)
        v21 = Dot(color="#00FFFF", radius=0.08) # (2,1)
        
        # Group 0 (Bottom-Left): Disk 2 at Peg 0
        v00 = Dot(color="#00FFFF", radius=0.08) # (0,0)
        v01 = Dot(color="#00FFFF", radius=0.08) # (0,1)
        v02 = Dot(color="#00FFFF", radius=0.08) # (0,2)
        
        # Group 1 (Bottom-Right): Disk 2 at Peg 1 - Issue 42: Shifted inward
        v11 = Dot(color="#00FFFF", radius=0.08) # (1,1)
        v10 = Dot(color="#00FFFF", radius=0.08) # (1,0)
        v12 = Dot(color="#00FFFF", radius=0.08) # (1,2)

        self.place_at_grid(v22, "A3")
        self.place_at_grid(v20, "B3")
        self.place_at_grid(v21, "B4")
        
        self.place_at_grid(v00, "D1")
        self.place_at_grid(v01, "E1")
        self.place_at_grid(v02, "E2")
        
        self.place_at_grid(v11, "D5")
        self.place_at_grid(v10, "E4")
        self.place_at_grid(v12, "E5")

        # Inner connections (Move Disk 1)
        inner_lines = [
            Line(v22.get_center(), v20.get_center(), color=WHITE, stroke_width=1.5),
            Line(v20.get_center(), v21.get_center(), color=WHITE, stroke_width=1.5),
            Line(v00.get_center(), v01.get_center(), color=WHITE, stroke_width=1.5),
            Line(v01.get_center(), v02.get_center(), color=WHITE, stroke_width=1.5),
            Line(v11.get_center(), v10.get_center(), color=WHITE, stroke_width=1.5),
            Line(v10.get_center(), v12.get_center(), color=WHITE, stroke_width=1.5)
        ]

        # Labels for groups (Disk 2 position)
        g2_label = Text("Peg 2 Cluster", font_size=14).next_to(v22, UP, buff=0.1)
        g0_label = Text("Peg 0 Cluster", font_size=14).next_to(v01, DOWN, buff=0.1)
        g1_label = Text("Peg 1 Cluster", font_size=14).next_to(v12, DOWN, buff=0.1)

        self.play(
            AnimationGroup(
                *[FadeIn(v) for v in [v22, v20, v21, v00, v01, v02, v11, v12, v10]],
                *[Create(l) for l in inner_lines],
                Write(g2_label), Write(g0_label), Write(g1_label),
                lag_ratio=0.1
            )
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)

        # Bridge edges (Move Disk 2)
        bridge_lines = [
            Line(v21.get_center(), v01.get_center(), color=WHITE, stroke_width=1.5), # (2,1) to (0,1)
            Line(v02.get_center(), v12.get_center(), color=WHITE, stroke_width=1.5), # (0,2) to (1,2)
            Line(v10.get_center(), v20.get_center(), color=WHITE, stroke_width=1.5)  # (1,0) to (2,0)
        ]

        self.play(*[Create(bl) for bl in bridge_lines])
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)

        # Highlight specific vertex and show configuration
        # Highlight (Disk2=2, Disk1=1) -> v21
        highlight_circle = Circle(radius=0.2, color=YELLOW).move_to(v21.get_center())
        
        # Disk Config Window - Issue 40: Repositioned to C3-D4
        window_bg = Rectangle(width=2.5, height=1.5, color=WHITE, fill_opacity=0.2)
        self.place_in_area(window_bg, "C3", "D4", scale_factor=0.7)
        
        # Simple Pegs and Disks inside the window
        # State: Disk 2 on Peg 2, Disk 1 on Peg 1
        p0 = Line(UP*0.3, DOWN*0.3, color=GRAY).shift(window_bg.get_center() + LEFT*0.6)
        p1 = Line(UP*0.3, DOWN*0.3, color=GRAY).shift(window_bg.get_center())
        p2 = Line(UP*0.3, DOWN*0.3, color=GRAY).shift(window_bg.get_center() + RIGHT*0.6)
        
        d2 = RoundedRectangle(corner_radius=0.05, width=0.4, height=0.1, color=BLUE, fill_opacity=0.8).move_to(p2.get_start() + DOWN*0.5)
        d1 = RoundedRectangle(corner_radius=0.05, width=0.2, height=0.1, color=RED, fill_opacity=0.8).move_to(p1.get_start() + DOWN*0.5)
        
        config_group = VGroup(window_bg, p0, p1, p2, d2, d1)
        
        self.play(Create(highlight_circle), FadeIn(config_group))
        self.wait(2)
        
        # Final Highlight: Clean up and show full structure
        self.play(
            FadeOut(highlight_circle), 
            FadeOut(config_group),
            FadeOut(g2_label, g0_label, g1_label)
        )
        
        # Highlight resulting 9-vertex structure
        all_dots = VGroup(v22, v20, v21, v00, v01, v02, v11, v12, v10)
        all_lines = VGroup(*inner_lines, *bridge_lines)
        
        self.play(
            all_dots.animate.set_color(WHITE),
            all_lines.animate.set_stroke(width=3)
        )
        self.wait(2)
        
        self.lecture[4].set_color(WHITE)
