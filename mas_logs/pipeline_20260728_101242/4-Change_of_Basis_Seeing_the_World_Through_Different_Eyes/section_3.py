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
        # Setup layout with lecture lines from storyboard
        self.setup_layout("The 'Dictionary': The Transition Matrix", [
            "- A transition matrix translates between different bases.",
            "- Describe the Owl's basis vectors using Human coordinates.",
            "- These Human-relative vectors become the matrix columns.",
            "- Each column translates one basis vector's language.",
            "- This matrix acts as our mathematical dictionary."
        ])

        # Colors
        owl_color = "#FFFF00"
        matrix_label_color = "#00FFFF"
        col1_highlight = "#FF8800"
        col2_highlight = "#FF00FF"

        # Load Assets once
        owl_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/owl.svg")
        self.place_at_grid(owl_icon, "E2", scale_factor=0.4)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(owl_color))
        
        # Create coordinate plane in the grid area
        plane = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=3.5,
            y_length=3.5,
            axis_config={"include_tip": True},
            background_line_style={"stroke_opacity": 0.4}
        )
        self.place_in_area(plane, "A2", "F6", scale_factor=1.0)
        
        # Owl's basis vectors: b1=(1,1) and b2=(-1,1)
        b1_arrow = Arrow(plane.c2p(0,0), plane.c2p(1,1), buff=0, color=owl_color)
        b2_arrow = Arrow(plane.c2p(0,0), plane.c2p(-1,1), buff=0, color=owl_color)
        
        b1_label = MathTex(r"\vec{b}_1 = (1,1)", font_size=20, color=owl_color)
        b2_label = MathTex(r"\vec{b}_2 = (-1,1)", font_size=20, color=owl_color)
        
        self.place_at_grid(b1_label, "B5", scale_factor=1.0)
        self.place_at_grid(b2_label, "B3", scale_factor=1.0)

        self.play(Create(plane), GrowArrow(b1_arrow), GrowArrow(b2_arrow), FadeIn(owl_icon))
        self.play(Write(b1_label), Write(b2_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(owl_color))
        
        # Show specific coordinates for the Owl's vectors
        coord_1 = VGroup(MathTex("1", font_size=32, color=owl_color), MathTex("1", font_size=32, color=owl_color)).arrange(DOWN, buff=0.2)
        coord_2 = VGroup(MathTex("-1", font_size=32, color=owl_color), MathTex("1", font_size=32, color=owl_color)).arrange(DOWN, buff=0.2)
        
        # FIX ISSUE 32: coord_1 (right-hand vector b1) to B6
        self.place_at_grid(coord_1, "B6", scale_factor=1.0)
        # FIX ISSUE 33: coord_2 (left-hand vector b2) to B2
        self.place_at_grid(coord_2, "B2", scale_factor=1.0)
        
        self.play(Write(coord_1), Write(coord_2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(owl_color))

        # Create the transition matrix P
        matrix_p = Matrix([[1, -1], [1, 1]], h_buff=1.0, v_buff=0.7).set_color(WHITE)
        self.place_at_grid(matrix_p, "D4", scale_factor=1.0)
        
        entries = matrix_p.get_entries() # Indexing: [0,0]=0, [0,1]=1, [1,0]=2, [1,1]=3
        
        # Animate coordinates moving into matrix columns
        self.play(
            FadeOut(plane), FadeOut(b1_arrow), FadeOut(b2_arrow), FadeOut(b1_label), FadeOut(b2_label), FadeOut(owl_icon),
            ReplacementTransform(coord_1[0], entries[0]),
            ReplacementTransform(coord_1[1], entries[2]),
            ReplacementTransform(coord_2[0], entries[1]),
            ReplacementTransform(coord_2[1], entries[3]),
            Write(matrix_p.get_brackets())
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(owl_color))

        # Label the matrix
        p_label = MathTex("P =", font_size=32, color=matrix_label_color)
        p_title = Text("Transition Matrix", font_size=20, color=matrix_label_color)
        
        # FIX ISSUE 31: p_label to D3
        self.place_at_grid(p_label, "D3", scale_factor=1.0)
        self.place_at_grid(p_title, "C4", scale_factor=1.0)
        
        self.play(Write(p_label), Write(p_title))

        # Highlight the columns
        col1_rect = SurroundingRectangle(VGroup(entries[0], entries[2]), color=col1_highlight, buff=0.1)
        col2_rect = SurroundingRectangle(VGroup(entries[1], entries[3]), color=col2_highlight, buff=0.1)

        self.play(Create(col1_rect))
        self.wait(0.5)
        self.play(ReplacementTransform(col1_rect, col2_rect))
        self.wait(1)
        self.play(FadeOut(col2_rect))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(owl_color))

        # Emphasize that columns are basis vectors
        # Reposition owl
        self.place_at_grid(owl_icon, "C5", scale_factor=0.4)
        
        # Show mini vectors next to matrix
        b1_mini = Arrow(ORIGIN, 0.4*(RIGHT+UP), buff=0, color=owl_color)
        b2_mini = Arrow(ORIGIN, 0.4*(LEFT+UP), buff=0, color=owl_color)
        
        self.place_at_grid(b1_mini, "E3", scale_factor=1.0)
        self.place_at_grid(b2_mini, "E5", scale_factor=1.0)

        self.play(
            FadeIn(owl_icon),
            Indicate(VGroup(entries[0], entries[2]), color=col1_highlight),
            Indicate(VGroup(entries[1], entries[3]), color=col2_highlight),
            Create(b1_mini), Create(b2_mini)
        )
        
        self.play(Flash(b1_mini, color=col1_highlight), Flash(b2_mini, color=col2_highlight))

        self.wait(2)
        self.play(FadeOut(b1_mini), FadeOut(b2_mini), FadeOut(owl_icon), self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
