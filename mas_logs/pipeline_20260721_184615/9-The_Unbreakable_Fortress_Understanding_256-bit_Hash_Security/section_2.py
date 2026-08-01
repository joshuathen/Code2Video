from manim import *
import numpy as np

# Use the provided TeachingScene base class without modification.
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

class Section2Scene(TeachingScene):
    def construct(self):
        # --- Metadata ---
        title = "Prerequisite: The Power of Doubling"
        lines = [
            "Each bit added doubles the total possible combinations.",
            "Binary growth creates massive numbers very quickly.",
            "Just 256 bits reach a scale beyond human intuition."
        ]
        self.setup_layout(title, lines)

        # Initial dimming of lecture lines
        for line in self.lecture:
            line.set_color("#444444")

        # Helper to create a light switch
        def create_switch(color="#FFFFFF", is_on=False):
            outer = Rectangle(width=0.5, height=0.8, color=color, stroke_width=2)
            inner = Rectangle(width=0.3, height=0.3, color=color, fill_opacity=0.5, stroke_width=1)
            if is_on:
                inner.shift(UP * 0.15)
            else:
                inner.shift(DOWN * 0.15)
            return VGroup(outer, inner)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1.5)

        # Visual: Single white #FFFFFF light switch (on/off)
        # Note: Using Text instead of MathTex for stability (L022)
        switch1 = create_switch(color="#FFFFFF", is_on=False)
        self.place_in_area(switch1, "B2", "B2", scale_factor=1.0)
        label1 = Text("2^1 = 2", color="#FFFFFF")
        self.place_at_grid(label1, "C2", scale_factor=0.8)

        self.play(FadeIn(switch1), Write(label1))
        self.wait(1.5)

        # Visual: Add a second switch to show 4 total combinations
        switch2_group = VGroup(
            create_switch(color="#FFFFFF", is_on=False),
            create_switch(color="#FFFFFF", is_on=True)
        ).arrange(RIGHT, buff=0.2)
        self.place_in_area(switch2_group, "B4", "B5", scale_factor=1.0)
        label2 = Text("2^2 = 4", color="#FFFFFF")
        # Fix for Issue 22: use area positioning for formula alignment
        self.place_in_area(label2, "C4", "C5", scale_factor=0.8)

        self.play(FadeIn(switch2_group), Write(label2))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color("#444444"),
            self.lecture[1].animate.set_color("#00FF00")
        )
        self.wait(1.5)

        # Clean up line 1
        self.play(FadeOut(switch1, label1, switch2_group, label2))

        # Visual: Show 10 switches and the number 1,024
        ten_switches = VGroup(*[create_switch(color="#00FF00", is_on=(i % 3 == 0)) for i in range(10)])
        ten_switches.arrange_in_grid(rows=2, cols=5, buff=0.1)
        self.place_in_area(ten_switches, "B2", "C4", scale_factor=0.6)
        label10 = Text("2^10 = 1,024", color="#00FF00")
        # Fix for Issue 23: use area positioning to avoid crowding
        self.place_in_area(label10, "B5", "C6", scale_factor=0.8)

        self.play(FadeIn(ten_switches), Write(label10))
        self.wait(1.5)

        # Visual: Grow a green #00FF00 branching tree of combinations
        def build_tree(depth, color="#00FF00"):
            if depth == 0:
                return Dot(radius=0.03, color=color)
            node = Dot(radius=0.04, color=color)
            left = build_tree(depth - 1, color)
            right = build_tree(depth - 1, color)
            spacing = (2 ** (depth - 1)) * 0.15
            left.shift(DOWN * 0.5 + LEFT * spacing)
            right.shift(DOWN * 0.5 + RIGHT * spacing)
            
            l1 = Line(node.get_center(), left.get_top() if hasattr(left, "get_top") else left.get_center(), color=color, stroke_width=1)
            l2 = Line(node.get_center(), right.get_top() if hasattr(right, "get_top") else right.get_center(), color=color, stroke_width=1)
            
            return VGroup(node, left, right, l1, l2)

        tree = build_tree(3, color="#00FF00")
        self.place_in_area(tree, "D3", "F5", scale_factor=1.0)
        self.play(Create(tree))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color("#444444"),
            self.lecture[2].animate.set_color("#FFD700")
        )
        self.wait(1.5)

        # Clean up line 2
        self.play(FadeOut(ten_switches, label10, tree))

        # Visual: Display '2^256' in gold #FFD700
        final_val = Text("2^256", color="#FFD700")
        final_label = Text("Scale beyond human intuition", font_size=24, color="#FFD700")
        final_ui = VGroup(final_val, final_label).arrange(DOWN, buff=0.5)
        # Fix for Issue 24: reduce scale factor to 1.2 to avoid bleeding edges
        self.place_in_area(final_ui, "B2", "E6", scale_factor=1.2)

        self.play(FadeIn(final_ui))
        # Highlight utilizing correct Indicate class (L004)
        self.play(Indicate(final_val, color="#FFD700", scale_factor=1.2))
        self.wait(2.0)
