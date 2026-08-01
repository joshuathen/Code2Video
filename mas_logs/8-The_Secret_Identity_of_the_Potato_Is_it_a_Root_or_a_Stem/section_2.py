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
        # Initialize lecture lines
        lecture_lines = [
            "To solve this, we must identify stem markers.",
            "Nodes are the specific points where leaves grow.",
            "Internodes are the spaces found between these nodes.",
            "Axillary buds wait at nodes to sprout new growth.",
            "Roots lack these organized segments and buds entirely."
        ]
        
        # Setup the layout
        self.setup_layout("Prerequisite Knowledge: Anatomy of a Stem", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # "To solve this, we must identify stem markers."
        self.play(self.lecture[0].animate.set_color("#32CD32"))
        
        # Simple vertical green line representing a plant stem
        # Placed in Column 2
        stem_start = self.grid["B2"]
        stem_end = self.grid["E2"]
        stem = Line(stem_start, stem_end, color="#32CD32", stroke_width=10)
        
        self.play(Create(stem))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Nodes are the specific points where leaves grow."
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color("#00BFFF"))
        
        # Horizontal blue lines at regular intervals along the stem
        node_positions = ["B2", "C2", "D2", "E2"]
        nodes = VGroup()
        for pos in node_positions:
            node_line = Line(
                self.grid[pos] + LEFT * 0.4, 
                self.grid[pos] + RIGHT * 0.4, 
                color="#00BFFF", 
                stroke_width=6
            )
            nodes.add(node_line)
            
        node_label = Text("Nodes", font_size=24, color=WHITE)
        self.place_at_grid(node_label, "B1")
        
        self.play(Create(nodes), Write(node_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Internodes are the spaces found between these nodes."
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color("#FFFFE0"))
        
        # Glowing yellow segments between the nodes on the stem
        internode_highlights = VGroup()
        internode_locs = [("B2", "C2"), ("C2", "D2"), ("D2", "E2")]
        for start_key, end_key in internode_locs:
            glow = Line(
                self.grid[start_key], 
                self.grid[end_key], 
                color="#FFFFE0", 
                stroke_width=14
            ).set_opacity(0.5)
            internode_highlights.add(glow)
            
        internode_label = Text("Internodes", font_size=24, color=WHITE)
        self.place_at_grid(internode_label, "C3")
        
        self.play(FadeIn(internode_highlights), Write(internode_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Axillary buds wait at nodes to sprout new growth."
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color("#00FF00"))
        
        # Small green circles at node intersections
        buds = VGroup()
        for pos in node_positions:
            bud = Dot(radius=0.15, color="#00FF00").move_to(self.grid[pos])
            buds.add(bud)
            
        bud_label = Text("Axillary Buds", font_size=24, color=WHITE)
        self.place_at_grid(bud_label, "E3")
        
        self.play(FadeIn(buds, shift=UP*0.1), Write(bud_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Roots lack these organized segments and buds entirely."
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color("#CD853F"))
        
        # Brownish smooth root shape to the right (Column 5)
        # We define a smooth path using points to show lack of segments
        root_path = VMobject(color="#CD853F").set_stroke(width=10)
        root_points = [
            self.grid["B5"], 
            self.grid["C6"], 
            self.grid["D4"], 
            self.grid["E5"]
        ]
        root_path.set_points_as_corners(root_points).make_smooth()
        
        root_label = Text("Root", font_size=24, color=WHITE)
        self.place_at_grid(root_label, "A5")
        
        self.play(Create(root_path), Write(root_label))
        self.wait(2)
