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
        title = "Multi-Head Attention: Parallel Perspectives"
        lines = [
            "One attention head cannot capture every relationship.",
            "Multi-head attention runs several processes in parallel.",
            "Different heads focus on grammar, logic, or context."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_HEAD1 = "#FF0000"
        COLOR_HEAD2 = "#0000FF"
        COLOR_HEAD3 = "#00FF00"
        COLOR_HEAD4 = "#FFFF00"
        COLOR_SINGLE = "#A9A9A9"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Single Attention Head
        head_box = Rectangle(width=1.5, height=1.0, color=COLOR_SINGLE, fill_opacity=0.5)
        head_label = Text("Head 1", font_size=20, color=WHITE)
        single_head = VGroup(head_box, head_label)
        # Centered at C4 to prepare for expansion
        self.place_at_grid(single_head, "C4")
        
        self.play(FadeIn(single_head))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # 4 Colored Boxes
        box1 = Rectangle(width=1.0, height=0.7, color=COLOR_HEAD1, fill_opacity=0.7)
        lbl1 = Text("Head 1", font_size=16, color=WHITE)
        h1 = VGroup(box1, lbl1)
        
        box2 = Rectangle(width=1.0, height=0.7, color=COLOR_HEAD2, fill_opacity=0.7)
        lbl2 = Text("Head 2", font_size=16, color=WHITE)
        h2 = VGroup(box2, lbl2)
        
        box3 = Rectangle(width=1.0, height=0.7, color=COLOR_HEAD3, fill_opacity=0.7)
        lbl3 = Text("Head 3", font_size=16, color=WHITE)
        h3 = VGroup(box3, lbl3)
        
        box4 = Rectangle(width=1.0, height=0.7, color=COLOR_HEAD4, fill_opacity=0.7)
        lbl4 = Text("Head 4", font_size=16, color=WHITE)
        h4 = VGroup(box4, lbl4)

        # Updated Positions per Issue 38
        self.place_at_grid(h1, "B3")
        self.place_at_grid(h2, "B5")
        self.place_at_grid(h3, "D3")
        self.place_at_grid(h4, "D5")

        self.play(
            ReplacementTransform(single_head, h1),
            FadeIn(h2),
            FadeIn(h3),
            FadeIn(h4)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Labels for roles (flanking the heads)
        role1 = Text("Grammar", font_size=18, color=COLOR_HEAD1)
        role2 = Text("Logic", font_size=18, color=COLOR_HEAD2)
        role3 = Text("Time", font_size=18, color=COLOR_HEAD3)
        role4 = Text("Entities", font_size=18, color=COLOR_HEAD4)

        self.place_at_grid(role1, "B2")
        self.place_at_grid(role2, "B6")
        self.place_at_grid(role3, "D2")
        self.place_at_grid(role4, "D6")

        # Words to connect to (shifted right)
        word_a = Text("Word A", font_size=20)
        word_b = Text("Word B", font_size=20)
        self.place_at_grid(word_a, "A3")
        self.place_at_grid(word_b, "A5")
        
        # Connectors - showing unique perspectives on the same words
        # Each head connects to both words
        lines_vgroup = VGroup()
        for head, color in zip([h1, h2, h3, h4], [COLOR_HEAD1, COLOR_HEAD2, COLOR_HEAD3, COLOR_HEAD4]):
            l_a = Line(head.get_top(), word_a.get_bottom(), color=color, stroke_width=2, stroke_opacity=0.6)
            l_b = Line(head.get_top(), word_b.get_bottom(), color=color, stroke_width=2, stroke_opacity=0.6)
            lines_vgroup.add(l_a, l_b)

        self.play(
            FadeIn(role1), FadeIn(role2), FadeIn(role3), FadeIn(role4),
            FadeIn(word_a), FadeIn(word_b)
        )
        self.play(Create(lines_vgroup, lag_ratio=0.1))
        self.wait(2)

        # Merge back into Multi-Head Output block
        # Create a multi-colored block to represent combined heads
        mh_bg = Rectangle(width=2.5, height=1.5, color=WHITE, stroke_width=2)
        c1 = Rectangle(width=0.5, height=0.5, color=COLOR_HEAD1, fill_opacity=0.8, stroke_width=0)
        c2 = Rectangle(width=0.5, height=0.5, color=COLOR_HEAD2, fill_opacity=0.8, stroke_width=0)
        c3 = Rectangle(width=0.5, height=0.5, color=COLOR_HEAD3, fill_opacity=0.8, stroke_width=0)
        c4 = Rectangle(width=0.5, height=0.5, color=COLOR_HEAD4, fill_opacity=0.8, stroke_width=0)
        colors_grid = VGroup(c1, c2, c3, c4).arrange_in_grid(rows=2, cols=2, buff=0.1)
        multi_head_label = Text("Multi-Head Output", font_size=20, color=WHITE).next_to(colors_grid, DOWN, buff=0.1)
        multi_head_block = VGroup(mh_bg, colors_grid, multi_head_label)
        self.place_at_grid(multi_head_block, "C4")

        self.play(
            FadeOut(lines_vgroup),
            FadeOut(role1), FadeOut(role2), FadeOut(role3), FadeOut(role4),
            FadeOut(word_a), FadeOut(word_b),
            ReplacementTransform(VGroup(h1, h2, h3, h4), multi_head_block)
        )
        self.wait(3)
