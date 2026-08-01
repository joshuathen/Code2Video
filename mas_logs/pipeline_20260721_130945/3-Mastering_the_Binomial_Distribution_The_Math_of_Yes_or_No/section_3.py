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
        # Title and Lecture Lines
        title_text = "Visualizing Combinations: The 'How Many Ways' Logic"
        lecture_lines = [
            "Multiple trials create many different paths to the same result.",
            "A tree diagram helps visualize these branching possibilities.",
            "For two shots, there are two ways to hit once.",
            "We use the combination formula to count these unique paths.",
            "This explains why middle outcomes often have higher probabilities."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors from Storyboard and Issues
        COLOR_ROOT = WHITE
        COLOR_TREE = "#AAAAAA" # Grey for branches/nodes
        COLOR_HIGHLIGHT = "#FFFF55" # Yellow for SF/FS paths and formula
        COLOR_CHART = "#FFFF55"
        COLOR_TEXT = WHITE

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_ROOT))
        
        # Issue 23: Load asset for root node
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/shots.svg]
        root_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/shots.svg")
        root_asset.set_color(COLOR_ROOT)
        
        # Issue 31 Fix: Position root at C3 and start label at C2
        self.place_at_grid(root_asset, "C3", scale_factor=0.6)
        start_label = Text("Start", font_size=20, color=COLOR_TEXT)
        self.place_at_grid(start_label, "C2", scale_factor=0.8)
        
        self.play(FadeIn(root_asset), Write(start_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_TREE)
        )
        
        # Issue 32 Fix: Trial 1 Nodes at B4, D4
        nodes_l1 = VGroup(
            Dot(color=COLOR_TREE),
            Dot(color=COLOR_TREE)
        )
        self.place_at_grid(nodes_l1[0], "B4")
        self.place_at_grid(nodes_l1[1], "D4")
        
        l1_lines = VGroup(
            Line(root_asset.get_critical_point(RIGHT), nodes_l1[0].get_center(), color=COLOR_TREE),
            Line(root_asset.get_critical_point(RIGHT), nodes_l1[1].get_center(), color=COLOR_TREE)
        )
        
        l1_labels = VGroup(
            Text("S", font_size=18, color=COLOR_TREE).next_to(l1_lines[0].get_center(), UP, buff=0.1),
            Text("F", font_size=18, color=COLOR_TREE).next_to(l1_lines[1].get_center(), DOWN, buff=0.1)
        )
        
        # Trial 2 Nodes: A5, B5, C5, D5 (Positioned for a clean balanced tree)
        nodes_l2 = VGroup(*[Dot(color=COLOR_TREE) for _ in range(4)])
        pos_l2 = ["A5", "B5", "C5", "D5"]
        for node, pos in zip(nodes_l2, pos_l2):
            self.place_at_grid(node, pos)
            
        l2_lines = VGroup(
            Line(nodes_l1[0].get_center(), nodes_l2[0].get_center(), color=COLOR_TREE), # S -> SS
            Line(nodes_l1[0].get_center(), nodes_l2[1].get_center(), color=COLOR_TREE), # S -> SF
            Line(nodes_l1[1].get_center(), nodes_l2[2].get_center(), color=COLOR_TREE), # F -> FS
            Line(nodes_l1[1].get_center(), nodes_l2[3].get_center(), color=COLOR_TREE)  # F -> FF
        )
        
        self.play(
            Create(l1_lines), Create(l2_lines),
            FadeIn(nodes_l1), FadeIn(nodes_l2),
            Write(l1_labels)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_HIGHLIGHT)
        )
        
        # Outcome labels at Col 6 (scaled 0.7 to avoid clipping - L003)
        outcomes = VGroup(
            Text("SS", font_size=24, color=COLOR_TREE),
            Text("SF", font_size=24, color=COLOR_HIGHLIGHT),
            Text("FS", font_size=24, color=COLOR_HIGHLIGHT),
            Text("FF", font_size=24, color=COLOR_TREE)
        )
        out_pos = ["A6", "B6", "C6", "D6"]
        for out, pos in zip(outcomes, out_pos):
            self.place_at_grid(out, pos, scale_factor=0.7)
            
        bracket = Brace(VGroup(outcomes[1], outcomes[2]), direction=RIGHT, color=COLOR_HIGHLIGHT)
        ways_label = Text("2 Ways", font_size=22, color=COLOR_HIGHLIGHT).next_to(bracket, RIGHT, buff=0.1)
        
        self.play(Write(outcomes))
        self.play(Create(bracket), Write(ways_label))
        
        # Highlight SF and FS paths as per storyboard
        path_sf = VGroup(l1_lines[0], l2_lines[1])
        path_fs = VGroup(l1_lines[1], l2_lines[2])
        
        self.play(
            path_sf.animate.set_stroke(color=COLOR_HIGHLIGHT, width=6),
            path_fs.animate.set_stroke(color=COLOR_HIGHLIGHT, width=6),
            Indicate(outcomes[1]), Indicate(outcomes[2]) # Rule L004
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(COLOR_HIGHLIGHT)
        )
        
        # Issue 33 Fix: formula at E4-E6, scaled appropriately
        formula = MathTex(r"\binom{2}{1} = 2", color=COLOR_HIGHLIGHT)
        self.place_in_area(formula, "E4", "E6", scale_factor=1.0)
        
        # Morphing the ways label and bracket into the formula
        self.play(
            ReplacementTransform(VGroup(bracket, ways_label), formula)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(COLOR_CHART)
        )
        
        # Bar Chart showing probability distribution logic
        chart = BarChart(
            values=[1, 2, 1],
            bar_names=["0", "1", "2"],
            y_range=[0, 3, 1],
            y_length=2.0,
            x_length=3.0,
            bar_colors=[COLOR_TREE, COLOR_CHART, COLOR_TREE]
        )
        # Place in area A4 to C6 (Top-Right of grid)
        self.place_in_area(chart, "A4", "C6", scale_factor=0.8)
        
        # Clean up tree to focus on the bar chart
        self.play(
            FadeOut(VGroup(root_asset, nodes_l1, nodes_l2, l1_lines, l2_lines, outcomes, l1_labels, start_label)),
            FadeIn(chart)
        )
        self.play(Indicate(chart.bars[1]))
        
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(2)
