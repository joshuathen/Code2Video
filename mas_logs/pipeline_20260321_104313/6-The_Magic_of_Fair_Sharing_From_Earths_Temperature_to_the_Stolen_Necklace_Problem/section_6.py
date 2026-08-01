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

class Section6Scene(TeachingScene):
    def construct(self):
        # Initializing the layout with specific section title and lecture lines
        lecture_lines = [
            "Topology provides elegant solutions to discrete problems.",
            "Continuity ensures fairness in complex sharing scenarios.",
            "Mathematical beauty solves the thieves' difficult dilemma."
        ]
        self.setup_layout("Summary and Real-world Intuition", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # "Topology provides elegant solutions to discrete problems."
        self.play(self.lecture[0].animate.set_color("#DA70D6"))
        
        # Necklace representation (Discrete elements)
        necklace_beads = VGroup(*[Dot(radius=0.1, color=WHITE) for _ in range(6)]).arrange(RIGHT, buff=0.15)
        necklace_frame = Rectangle(height=0.6, width=1.5, color="#DA70D6")
        necklace_group = VGroup(necklace_beads, necklace_frame)
        self.place_in_area(necklace_group, "A1", "B2", scale_factor=0.8)
        
        # Sphere representation (Continuous shape in topology)
        sphere_base = Circle(radius=0.6, color="#DA70D6")
        sphere_equator = Ellipse(width=1.2, height=0.3, color="#DA70D6")
        sphere_group = VGroup(sphere_base, sphere_equator)
        self.place_in_area(sphere_group, "A5", "B6", scale_factor=0.8)
        
        # 'Topology' Bridge linking discrete and continuous
        # Fixed positioning of the bridge label (Issue 36)
        bridge_arrow = DoubleArrow(self.grid["B2"], self.grid["B5"], color="#DA70D6", stroke_width=4, tip_length=0.2)
        bridge_label = Text("Topology", font_size=24, color="#DA70D6")
        self.place_in_area(bridge_label, "A3", "B4", scale_factor=0.8)
        
        self.play(Create(necklace_group), Create(sphere_group))
        self.play(GrowArrow(bridge_arrow), Write(bridge_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Continuity ensures fairness in complex sharing scenarios."
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        
        continuity_text = Text("Continuity is the Key", font_size=32, color="#00FFFF")
        # Fixed positioning and scale of continuity text (Issue 37)
        self.place_in_area(continuity_text, "C2", "C5", scale_factor=0.8)
        
        self.play(Write(continuity_text))
        # Flash animation to highlight the "Key" concept
        self.play(Flash(continuity_text, color="#00FFFF", line_length=0.3, flash_radius=1.8))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Mathematical beauty solves the thieves' difficult dilemma."
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        
        def create_stick_figure(color):
            head = Circle(radius=0.15, color=color)
            body = Line(DOWN*0.15, DOWN*0.6, color=color)
            arms = Line(LEFT*0.3+DOWN*0.3, RIGHT*0.3+DOWN*0.3, color=color)
            legs = VGroup(Line(DOWN*0.6, DOWN*1.0+LEFT*0.2), Line(DOWN*0.6, DOWN*1.0+RIGHT*0.2)).set_color(color)
            return VGroup(head, body, arms, legs)

        thief1 = create_stick_figure(WHITE)
        thief2 = create_stick_figure(WHITE)
        
        self.place_at_grid(thief1, "E2", scale_factor=0.7)
        self.place_at_grid(thief2, "E5", scale_factor=0.7)
        
        pile1 = VGroup(
            Dot(color=RED), Dot(color=RED), Dot(color=RED),
            Dot(color=GREEN), Dot(color=GREEN), Dot(color=GREEN)
        ).arrange_in_grid(2, 3, buff=0.1)
        pile2 = pile1.copy()
        
        self.place_at_grid(pile1, "F2", scale_factor=0.6)
        self.place_at_grid(pile2, "F5", scale_factor=0.6)
        
        self.play(FadeIn(thief1), FadeIn(thief2), FadeIn(pile1), FadeIn(pile2))
        
        # Handshake sequence
        t1_handshake = create_stick_figure(WHITE)
        self.place_at_grid(t1_handshake, "E3", scale_factor=0.7)
        t2_handshake = create_stick_figure(WHITE)
        self.place_at_grid(t2_handshake, "E4", scale_factor=0.7)
        
        sparkle = Star(n=6, color="#FFD700", fill_opacity=0.8).scale(0.3)
        # Fixed scale of sparkle (Issue 38)
        self.place_in_area(sparkle, "E3", "E4", scale_factor=0.6)
        
        self.play(
            Transform(thief1, t1_handshake),
            Transform(thief2, t2_handshake),
            run_time=1.5
        )
        self.play(FadeIn(sparkle), sparkle.animate.scale(1.2))
        self.wait(2)
