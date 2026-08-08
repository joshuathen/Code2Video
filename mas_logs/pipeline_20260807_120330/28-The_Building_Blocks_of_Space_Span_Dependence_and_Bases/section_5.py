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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup layout with specific title and lecture lines for Section 5
        title_text = "Bases: The Minimalist Toolkit"
        lecture_lines = [
            "A Basis is a minimal set of independent directions.",
            "It must span the space without any redundant arrows.",
            "Many different bases can describe the same 2D world."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Define colors for consistency
        GREEN = "#00FF00"
        GREY = "#808080"
        YELLOW = "#FFFF00"
        
        # === Animation for Lecture Line 1 ===
        # The words 'Linearly Independent' and 'Spans Space' appear in green #00FF00.
        # Two arrows i and j appear at right angles in #00FF00 to form a basis.
        self.play(self.lecture[0].animate.set_color(GREEN))
        
        self.labels_vgroup = VGroup(
            Text("Linearly Independent", color=GREEN, font_size=18),
            Text("Spans Space", color=GREEN, font_size=18)
        ).arrange(RIGHT, buff=0.5)
        
        # Using place_in_area for multi-word labels (B002) - Updated per Issue 25
        self.place_in_area(self.labels_vgroup, "A2", "A5")
        
        # Create coordinate axes in the central-right area (B2 to E5)
        axes = Axes(
            x_range=[-1, 3], y_range=[-1, 3], 
            axis_config={"include_tip": True, "stroke_width": 2},
            x_length=4, y_length=4
        )
        self.place_in_area(axes, "B2", "E5")
        
        # Standard basis arrows i and j
        vec_i = Arrow(axes.c2p(0,0), axes.c2p(1,0), color=GREEN, buff=0)
        vec_j = Arrow(axes.c2p(0,0), axes.c2p(0,1), color=GREEN, buff=0)
        label_i = MathTex("\\vec{i}", color=GREEN, font_size=24).next_to(vec_i, DOWN, buff=0.1)
        label_j = MathTex("\\vec{j}", color=GREEN, font_size=24).next_to(vec_j, LEFT, buff=0.1)
        
        self.play(FadeIn(self.labels_vgroup))
        self.play(Create(axes))
        self.play(GrowArrow(vec_i), GrowArrow(vec_j), FadeIn(label_i), FadeIn(label_j))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # Remove arrow j; the painted span area on the grid shrinks from a plane to a single line.
        # Show i, j, and a diagonal vector; the diagonal vector turns grey #808080 to indicate it is extra.
        # Highlight the two independent vectors i and j in #00FF00 and label them 'Basis'.
        
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(GREEN)
        )
        
        # Painted span area (green rectangle)
        span_plane = Rectangle(
            width=4, height=4, 
            fill_color=GREEN, fill_opacity=0.2, stroke_width=0
        ).move_to(axes.c2p(1,1))
        
        self.play(FadeIn(span_plane))
        self.wait(0.5)
        
        # Remove j and show the span shrinking to a line along i
        self.play(
            FadeOut(vec_j), 
            FadeOut(label_j),
            span_plane.animate.stretch_to_fit_height(0.05).move_to(axes.c2p(1,0)),
            run_time=2
        )
        self.wait(1)
        
        # Restore j and show a redundant diagonal vector
        vec_extra = Arrow(axes.c2p(0,0), axes.c2p(1,1), color=WHITE, buff=0)
        label_extra = MathTex("\\vec{v}", color=WHITE, font_size=24).next_to(vec_extra, UR, buff=0.1)
        
        self.play(
            FadeIn(vec_j), 
            FadeIn(label_j), 
            GrowArrow(vec_extra), 
            FadeIn(label_extra),
            FadeOut(span_plane)
        )
        self.wait(0.5)
        
        # The extra vector turns grey to indicate redundancy
        self.play(
            vec_extra.animate.set_color(GREY),
            label_extra.animate.set_color(GREY)
        )
        self.wait(1)
        
        # Define and place "Basis" label - Updated per Issue 26
        self.basis_label = Text("Basis", color=GREEN, font_size=26)
        self.place_in_area(self.basis_label, "F3", "F4")
        
        # Highlight the minimal independent set
        self.play(
            Indicate(vec_i, color=GREEN), 
            Indicate(vec_j, color=GREEN),
            FadeIn(self.basis_label),
            FadeOut(vec_extra),
            FadeOut(label_extra)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Many different bases can describe the same 2D world.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(GREEN)
        )
        
        # Transform the standard basis into an alternative non-orthogonal basis
        vec_b1 = Arrow(axes.c2p(0,0), axes.c2p(1.5, 0.5), color=YELLOW, buff=0)
        vec_b2 = Arrow(axes.c2p(0,0), axes.c2p(0.5, 1.5), color=YELLOW, buff=0)
        label_b1 = MathTex("\\vec{b}_1", color=YELLOW, font_size=24).next_to(vec_b1, DR, buff=0.1)
        label_b2 = MathTex("\\vec{b}_2", color=YELLOW, font_size=24).next_to(vec_b2, UL, buff=0.1)
        
        self.play(
            ReplacementTransform(vec_i, vec_b1),
            ReplacementTransform(vec_j, vec_b2),
            ReplacementTransform(label_i, label_b1),
            ReplacementTransform(label_j, label_b2),
            self.basis_label.animate.set_color(YELLOW)
        )
        self.wait(2)
        
        # Final cleanup for the section
        self.play(
            FadeOut(axes),
            FadeOut(vec_b1),
            FadeOut(vec_b2),
            FadeOut(label_b1),
            FadeOut(label_b2),
            FadeOut(self.basis_label),
            FadeOut(self.labels_vgroup),
            self.lecture[2].animate.set_color(WHITE)
        )
        self.wait(1)
