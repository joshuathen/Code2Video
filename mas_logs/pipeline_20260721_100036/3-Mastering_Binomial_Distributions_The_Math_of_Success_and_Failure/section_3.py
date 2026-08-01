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
        # LECTURE CONTENT
        title = "Visualizing the Probability Tree"
        lines = [
            "A tree diagram shows every possible path.",
            "Each path represents a specific sequence of outcomes.",
            "Multiple paths can lead to the same result."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Draw a probability tree with 3 levels in white #FFFFFF.
        
        # Node Definitions
        n0 = Dot(color=WHITE, radius=0.08)
        self.place_at_grid(n0, "C1")
        
        # Level 1
        n1s = Dot(color=WHITE, radius=0.08)
        n1f = Dot(color=WHITE, radius=0.08)
        self.place_at_grid(n1s, "B2")
        self.place_at_grid(n1f, "D2")
        
        # Level 2
        n2ss = Dot(color=WHITE, radius=0.08)
        n2sf = Dot(color=WHITE, radius=0.08)
        n2fs = Dot(color=WHITE, radius=0.08)
        n2ff = Dot(color=WHITE, radius=0.08)
        self.place_at_grid(n2ss, "A3")
        self.place_at_grid(n2sf, "B3")
        self.place_at_grid(n2fs, "D3")
        self.place_at_grid(n2ff, "E3")
        
        # Level 3
        n3sss = Dot(color=WHITE, radius=0.08)
        n3ssf = Dot(color=WHITE, radius=0.08)
        n3sfs = Dot(color=WHITE, radius=0.08)
        n3sff = Dot(color=WHITE, radius=0.08)
        n3fss = Dot(color=WHITE, radius=0.08)
        n3fsf = Dot(color=WHITE, radius=0.08)
        n3ffs = Dot(color=WHITE, radius=0.08)
        n3fff = Dot(color=WHITE, radius=0.08)
        
        # Grid positioning optimized for logical flow and visibility
        self.place_at_grid(n3sss, "A4")
        self.place_at_grid(n3ssf, "B4")
        self.place_at_grid(n3sfs, "C4")
        self.place_at_grid(n3sff, "D4") # Fixed Issue 30: Binary order
        self.place_at_grid(n3fss, "E4") # Fixed Issue 30: Binary order
        self.place_at_grid(n3fsf, "F4")
        self.place_at_grid(n3ffs, "D5") # Adjusted to avoid label overlap
        self.place_at_grid(n3fff, "F5") # Adjusted to avoid label overlap
        
        # Edges
        edges_list = [
            Line(n0.get_center(), n1s.get_center(), color=WHITE),      # 0 (root-s)
            Line(n0.get_center(), n1f.get_center(), color=WHITE),      # 1 (root-f)
            Line(n1s.get_center(), n2ss.get_center(), color=WHITE),    # 2 (s-ss)
            Line(n1s.get_center(), n2sf.get_center(), color=WHITE),    # 3 (s-sf)
            Line(n1f.get_center(), n2fs.get_center(), color=WHITE),    # 4 (f-fs)
            Line(n1f.get_center(), n2ff.get_center(), color=WHITE),    # 5 (f-ff)
            Line(n2ss.get_center(), n3sss.get_center(), color=WHITE),  # 6 (ss-sss)
            Line(n2ss.get_center(), n3ssf.get_center(), color=WHITE),  # 7 (ss-ssf)
            Line(n2sf.get_center(), n3sfs.get_center(), color=WHITE),  # 8 (sf-sfs)
            Line(n2sf.get_center(), n3sff.get_center(), color=WHITE),  # 9 (sf-sff)
            Line(n2fs.get_center(), n3fss.get_center(), color=WHITE),  # 10 (fs-fss)
            Line(n2fs.get_center(), n3fsf.get_center(), color=WHITE),  # 11 (fs-fsf)
            Line(n2ff.get_center(), n3ffs.get_center(), color=WHITE),  # 12 (ff-ffs)
            Line(n2ff.get_center(), n3fff.get_center(), color=WHITE)   # 13 (ff-fff)
        ]
        edges = VGroup(*edges_list)
        nodes = VGroup(n0, n1s, n1f, n2ss, n2sf, n2fs, n2ff, n3sss, n3ssf, n3sfs, n3sff, n3fss, n3fsf, n3ffs, n3fff)

        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(edges), Create(nodes), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight three paths: (S,S,F), (S,F,S), and (F,S,S) in yellow #FFFF00.
        # Label each highlighted path as 'Successes = 2' in yellow #FFFF00.
        
        # Path Indices based on logical tree structure
        highlight_edges_indices = [0, 1, 2, 3, 4, 7, 8, 10]
        highlight_nodes = VGroup(n0, n1s, n1f, n2ss, n2sf, n2fs, n3ssf, n3sfs, n3fss)
        highlight_edges = VGroup(*[edges[i] for i in highlight_edges_indices])
        
        l_ssf = Text("Successes = 2", font_size=16, color=YELLOW)
        l_sfs = Text("Successes = 2", font_size=16, color=YELLOW)
        l_fss = Text("Successes = 2", font_size=16, color=YELLOW)
        
        # Position labels 1 unit away from leaves in Column 5
        self.place_at_grid(l_ssf, "B5")
        self.place_at_grid(l_sfs, "C5")
        self.place_at_grid(l_fss, "E5") # Fixed Issue 30: Corresponding to n3fss at E4

        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        self.play(
            highlight_edges.animate.set_color(YELLOW),
            highlight_nodes.animate.set_color(YELLOW),
            Write(l_ssf), Write(l_sfs), Write(l_fss),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Group these three paths with a bracket and label them '3 ways to get 2 [Asset: ...]' in light blue #ADD8E6.
        # Show the text '3C2 = 3' appearing next to the bracket in light green #90EE90.
        
        labels_group = VGroup(l_ssf, l_sfs, l_fss)
        bracket = Brace(labels_group, direction=RIGHT, color="#ADD8E6")
        
        # Asset Integration (Issue 20)
        ways_text = Text("3 ways to get 2 ", font_size=20, color="#ADD8E6")
        fish_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fish.svg", height=0.3, color="#ADD8E6")
        ways_label = VGroup(ways_text, fish_icon).arrange(RIGHT, buff=0.1)
        
        # Positioning and Scaling (Issues 28 and 29)
        self.place_at_grid(ways_label, "C6", scale_factor=0.4) # Fixed Issue 28: Scale factor and position
        
        comb_label = MathTex("3C2 = 3", color="#90EE90", font_size=32)
        self.place_at_grid(comb_label, "F6", scale_factor=0.6) # Fixed Issue 29: Scale factor and position

        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        self.play(Create(bracket), FadeIn(ways_label))
        self.play(Write(comb_label))
        self.wait(2)
