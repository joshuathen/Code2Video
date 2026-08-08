from manim import *

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
        # Setup the layout with section-specific title and lines
        title = "Visualizing the 2-adic Space"
        lines = [
            "Imagine numbers on a branching binary tree.",
            "Paths split based on divisibility by two.",
            "Closeness is determined by shared branching points.",
            "Powers of 2 lead deeper into the tree's root.",
            "Thus, 2, 4, 8, 16 approach zero."
        ]
        self.setup_layout(title, lines)

        # Colors from storyboard
        COLOR_TREE = "#FFFFFF"
        COLOR_HIGHLIGHT = "#FF00FF"
        COLOR_DOT = "#00FFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        
        # Spine in column 4 (horizontal center of the right area), branches to column 5
        # The vertical stem is centered in the visual area as requested by critics
        spine_nodes = ["A4", "B4", "C4", "D4", "E4"]
        leaf_nodes = ["B5", "C5", "D5", "E5"] 
        
        tree_edges = VGroup()
        # Draw spine
        for i in range(len(spine_nodes) - 1):
            edge = Line(self.grid[spine_nodes[i]], self.grid[spine_nodes[i+1]], color=COLOR_TREE)
            tree_edges.add(edge)
            
        # Draw branches
        for i in range(len(leaf_nodes)):
            edge = Line(self.grid[spine_nodes[i+1]], self.grid[leaf_nodes[i]], color=COLOR_TREE)
            tree_edges.add(edge)
            
        self.play(Create(tree_edges), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(WHITE)
        # 0 labels on spine, 1 labels on branches to show binary splits
        labels_0 = VGroup()
        for i in range(len(spine_nodes) - 1):
            start = self.grid[spine_nodes[i]]
            end = self.grid[spine_nodes[i+1]]
            mid = (start + end) / 2
            lbl = Text("0", font_size=16, color=WHITE).move_to(mid).shift(LEFT * 0.2)
            labels_0.add(lbl)
            
        labels_1 = VGroup()
        for i in range(len(leaf_nodes)):
            start = self.grid[spine_nodes[i+1]]
            end = self.grid[leaf_nodes[i]]
            mid = (start + end) / 2
            lbl = Text("1", font_size=16, color=WHITE).move_to(mid).shift(UP * 0.2)
            labels_1.add(lbl)
            
        self.play(FadeIn(labels_0), FadeIn(labels_1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHITE)
        # Emphasis on shared branching points determining distance
        shared_segment = Line(self.grid["A4"], self.grid["B4"], color=COLOR_HIGHLIGHT, stroke_width=6)
        self.play(ShowPassingFlash(shared_segment.copy()), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(COLOR_HIGHLIGHT)
        # Highlight path for powers of 2. Powers branch off at deeper levels.
        # Nodes: 2^1 (B5), 2^2 (C5), 2^3 (D5), 2^4 (E5)
        power_nodes = ["B5", "C5", "D5", "E5"]
        power_labels = VGroup(
            MathTex("2^1", font_size=20, color=WHITE),
            MathTex("2^2", font_size=20, color=WHITE),
            MathTex("2^3", font_size=20, color=WHITE),
            MathTex("2^4", font_size=20, color=WHITE)
        )
        for i, node in enumerate(power_nodes):
            self.place_at_grid(power_labels[i], node, scale_factor=1.0)
            power_labels[i].shift(RIGHT * 0.4)
            
        # Highlight the segments connecting the path to power 2^4
        path_to_2_4 = VGroup(
            Line(self.grid["A4"], self.grid["B4"], color=COLOR_HIGHLIGHT, stroke_width=6),
            Line(self.grid["B4"], self.grid["C4"], color=COLOR_HIGHLIGHT, stroke_width=6),
            Line(self.grid["C4"], self.grid["D4"], color=COLOR_HIGHLIGHT, stroke_width=6),
            Line(self.grid["D4"], self.grid["E4"], color=COLOR_HIGHLIGHT, stroke_width=6),
            Line(self.grid["E4"], self.grid["E5"], color=COLOR_HIGHLIGHT, stroke_width=6)
        )
        
        self.play(FadeIn(power_labels), Create(path_to_2_4))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(COLOR_DOT)
        
        # Dot starting at first power of 2
        dot = Dot(color=COLOR_DOT).move_to(self.grid["B5"])
        self.add(dot)
        
        # Apply critic fixes for labels to prevent occlusion and edge proximity
        # Issue 23: Move '0 (Root)' to E4
        zero_label = Text("0 (Root)", font_size=22, color=WHITE)
        self.place_at_grid(zero_label, 'E4', scale_factor=0.8)
        zero_label.shift(LEFT * 0.8) # Slight offset to keep the root point clear
        
        # Issue 24: Move 'Closer to 0' to F4
        closer_text = Text("Closer to 0", font_size=24, color=WHITE)
        self.place_at_grid(closer_text, 'F4', scale_factor=0.7)

        # Animate a cyan dot moving through the powers of 2 then approaching the root (0)
        self.play(dot.animate.move_to(self.grid["C5"]), run_time=0.5)
        self.play(dot.animate.move_to(self.grid["D5"]), run_time=0.5)
        self.play(dot.animate.move_to(self.grid["E5"]), run_time=0.5)
        self.play(
            dot.animate.move_to(self.grid["E4"]), 
            FadeIn(closer_text), 
            FadeIn(zero_label), 
            run_time=1
        )
        
        self.wait(2)
