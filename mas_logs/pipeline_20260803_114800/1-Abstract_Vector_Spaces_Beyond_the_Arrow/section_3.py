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
            "The Definition: The 8 'Club Rules'", 
            [
                "Abstractly, a vector space is a set following rules.",
                "These eight axioms define how elements must behave.",
                "Addition and scaling must always stay inside the set.",
                "We call this fundamental property \"closure\" within the space.",
                "Every space needs a zero element—the \"ghost ingredient.\""
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        
        # Gold 'VIP Club' sign
        vip_sign = Text("VIP CLUB", color="#FFD700", font_size=36)
        vip_border = SurroundingRectangle(vip_sign, color="#FFD700", buff=0.2)
        vip_group = VGroup(vip_sign, vip_border)
        self.place_in_area(vip_group, "A3", "A4", scale_factor=0.8)
        
        # White circle representing the set
        set_circle = Circle(radius=1.5, color=WHITE)
        self.place_in_area(set_circle, "C2", "E4", scale_factor=1.0)
        set_label = Text("Set V", font_size=18, color=WHITE).next_to(set_circle, DOWN, buff=0.1)
        
        self.play(Create(vip_group))
        self.play(Create(set_circle), Write(set_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#00FF7F")
        )
        
        # Rules List - Resolved Issue 26: Expanded area A5-F6
        rules = VGroup(
            Text("1. Commutativity", font_size=16),
            Text("2. Associativity", font_size=16),
            Text("3. Zero Identity", font_size=16),
            Text("4. Inverse", font_size=16),
            Text("5. Distributivity I", font_size=16),
            Text("6. Distributivity II", font_size=16),
            Text("7. Scalar Assoc.", font_size=16),
            Text("8. Unit Identity", font_size=16)
        ).arrange(DOWN, aligned_edge=LEFT)
        self.place_in_area(rules, "A5", "F6", scale_factor=0.9)
        
        # Pulse Label 'A+B = B+A' - Resolved Issue 25: Repositioned to B3-B4
        axiom_label = MathTex("A+B = B+A", color="#00FF7F", font_size=30)
        self.place_in_area(axiom_label, "B3", "B4", scale_factor=0.8)
        
        self.play(Write(rules))
        self.play(FadeIn(axiom_label))
        self.play(axiom_label.animate.scale(1.2), run_time=0.5)
        self.play(axiom_label.animate.scale(1/1.2), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FF69B4")
        )
        
        # Two pink shapes merging inside the set
        shape1 = Star(n=5, color="#FF69B4", fill_opacity=0.8).scale(0.2)
        shape2 = Triangle(color="#FF69B4", fill_opacity=0.8).scale(0.2)
        
        start_pos1 = set_circle.get_center() + LEFT*0.8 + UP*0.4
        start_pos2 = set_circle.get_center() + RIGHT*0.8 + DOWN*0.4
        shape1.move_to(start_pos1)
        shape2.move_to(start_pos2)
        
        target_pos = set_circle.get_center()
        
        self.play(FadeIn(shape1), FadeIn(shape2))
        self.play(
            shape1.animate.move_to(target_pos + LEFT*0.3),
            shape2.animate.move_to(target_pos + RIGHT*0.3)
        )
        
        merged_blob = Circle(radius=0.4, color="#FF69B4", fill_opacity=0.5)
        merged_blob.move_to(target_pos)
        
        self.play(
            ReplacementTransform(VGroup(shape1, shape2), merged_blob)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color("#00FF00")
        )
        
        # Green checkmark - Resolved Issue 24: Moved to B4
        checkmark = Tex(r"\checkmark", color="#00FF00", font_size=60)
        self.place_at_grid(checkmark, "B4", scale_factor=0.8)
        
        self.play(FadeIn(checkmark, scale=0.5))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color("#A9A9A9")
        )
        
        # Ghost Zero Element
        zero_icon = MathTex("0", color="#A9A9A9", font_size=48).set_opacity(0.3)
        self.place_at_grid(zero_icon, "D4")
        
        self.play(FadeIn(zero_icon))
        self.play(zero_icon.animate.set_opacity(0.8), run_time=1)
        self.play(zero_icon.animate.set_opacity(0.2), run_time=1)
        self.play(zero_icon.animate.set_opacity(0.5), run_time=1)
        
        self.wait(2)
