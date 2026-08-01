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

class Section2Scene(TeachingScene):
    def construct(self):
        # Asset Paths
        SWITCH_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/switch.svg"
        ATOM_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/atom.svg"

        # Retrieve data from shared state
        title = "The Power of Exponents: From 1 to 256"
        lecture_lines = [
            "Computers use bits, where each bit doubles the possibilities.",
            "One bit has two states; two bits have four.",
            "A 256-bit hash has two to the 256th power.",
            "That is a one followed by seventy-seven zeros.",
            "This number is larger than atoms in the universe."
        ]
        
        self.setup_layout(title, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Show a single light switch [Asset: switch.svg] flipping between 0 (#FF0000) and 1 (#00FF00).
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Setup initial state (Off)
        switch_svg = SVGMobject(SWITCH_ASSET).set_color("#FF0000")
        switch_label = Text("0", color="#FF0000", font_size=24)
        
        switch_group = VGroup(switch_svg, switch_label).arrange(DOWN, buff=0.2)
        self.place_in_area(switch_group, "B2", "C3", scale_factor=0.8)
        
        self.play(FadeIn(switch_group))
        self.wait(1)
        
        # Flip (On)
        new_label = Text("1", color="#00FF00", font_size=24).move_to(switch_label.get_center())
        
        self.play(
            switch_svg.animate.set_color("#00FF00").rotate(PI, axis=RIGHT), # Simulate flip
            Transform(switch_label, new_label),
            run_time=0.8
        )
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Show two then three switches [Asset: switch.svg], with counts 2, 4, 8 appearing above.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        self.play(FadeOut(switch_group))
        
        def create_asset_switch(state="0"):
            color = "#FF0000" if state == "0" else "#00FF00"
            sw = SVGMobject(SWITCH_ASSET).set_color(color).scale(0.3)
            if state == "1":
                sw.rotate(PI, axis=RIGHT)
            return sw

        # 1 bit -> 2 states
        g1_sw = create_asset_switch("1")
        g1_lbl = Text("2 states", font_size=20, color=WHITE).next_to(g1_sw, UP, buff=0.2)
        g1 = VGroup(g1_sw, g1_lbl)
        self.place_at_grid(g1, "B2")
        
        # 2 bits -> 4 states
        g2_sw = VGroup(create_asset_switch("1"), create_asset_switch("0")).arrange(RIGHT, buff=0.1)
        g2_lbl = Text("4 states", font_size=20, color=WHITE).next_to(g2_sw, UP, buff=0.2)
        g2 = VGroup(g2_sw, g2_lbl)
        self.place_at_grid(g2, "B4")
        
        # 3 bits -> 8 states (Issue 27: move to B5)
        g3_sw = VGroup(create_asset_switch("1"), create_asset_switch("1"), create_asset_switch("0")).arrange(RIGHT, buff=0.1)
        g3_lbl = Text("8 states", font_size=20, color=WHITE).next_to(g3_sw, UP, buff=0.2)
        g3 = VGroup(g3_sw, g3_lbl)
        self.place_at_grid(g3, "B5")

        self.play(FadeIn(g1))
        self.wait(0.5)
        self.play(FadeIn(g2))
        self.wait(0.5)
        self.play(FadeIn(g3))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # A binary tree (#ADD8E6) grows from the top, branching rapidly downwards.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        self.play(FadeOut(g1, g2, g3))
        
        tree_color = "#ADD8E6"
        tree = VGroup()
        root = Dot(ORIGIN, radius=0.04, color=tree_color)
        tree.add(root)
        
        current_level_nodes = [root]
        for depth in range(4):
            new_level_nodes = []
            for node in current_level_nodes:
                for side in [-1, 1]:
                    spacing_x = 1.2 / (2**depth)
                    spacing_y = 0.6
                    target_pos = node.get_center() + np.array([side * spacing_x, -spacing_y, 0])
                    branch = Line(node.get_center(), target_pos, stroke_width=1.5, color=tree_color)
                    leaf = Dot(target_pos, radius=0.02, color=tree_color)
                    tree.add(branch, leaf)
                    new_level_nodes.append(leaf)
            current_level_nodes = new_level_nodes
            
        self.place_in_area(tree, "B2", "E5", scale_factor=0.8)
        tree.shift(UP * 0.5)
        
        self.play(Create(tree), run_time=2)
        
        # Display 2^256 (Issue 26: move to F3, scale 1.2)
        math_256 = MathTex("2^{256}", color=WHITE)
        self.place_at_grid(math_256, "F3", scale_factor=1.2)
        self.play(Write(math_256))
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        # Reveal its full 77-digit expansion.
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )
        
        full_val = "115792089237316195423570985008687907853269984665640564039457584007913129639936"
        # Split number for better layout
        wrapped_val = "\n".join([full_val[i:i+10] for i in range(0, len(full_val), 10)])
        full_val_text = Text(wrapped_val, font_size=16, color=WHITE, line_spacing=0.8)
        # Issue 28: position in area C3-E5
        self.place_in_area(full_val_text, "C3", "E5")
        
        self.play(
            FadeOut(tree),
            Transform(math_256, full_val_text)
        )
        self.wait(2.5)

        # === Animation for Lecture Line 5 ===
        # Scale the large number down until it is a tiny dot [Asset: atom.svg] in darkness.
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        
        # Use atom.svg as the tiny dot
        atom = SVGMobject(ATOM_ASSET).set_color(WHITE).scale(0.05)
        atom.move_to(math_256.get_center())
        
        self.play(
            math_256.animate.scale(0.01).move_to(atom.get_center()),
            run_time=3
        )
        self.remove(math_256)
        self.play(FadeIn(atom))
        self.wait(2)
        
        # Reset highlight
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(2)
