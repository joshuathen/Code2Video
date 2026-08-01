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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "For k bead types, we use a k-dimensional sphere.",
            "A point on this sphere defines k cut locations.",
            "Borsuk-Ulam guarantees antipodal points with identical bead shares.",
            "If Thief A gets half, Thief B must also.",
            "Thus, only k cuts ensure a perfectly fair split."
        ]
        self.setup_layout("The Mathematical Solution: The k-cut Proof", lecture_lines)

        # Colors for highlighting
        HL_COLOR = YELLOW

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(HL_COLOR)
        # Necklace at C4 (Issue 38 resolution)
        necklace = Annulus(inner_radius=0.9, outer_radius=1.0, color=GREY_B)
        self.place_at_grid(necklace, "C4", scale_factor=1.1)
        
        # 2 types of beads: 5 Red, 5 Blue
        bead_colors = [RED, RED, BLUE, RED, BLUE, BLUE, RED, BLUE, RED, BLUE]
        beads = VGroup(*[
            Dot(radius=0.12, color=color).move_to(necklace.point_from_proportion(i/10))
            for i, color in enumerate(bead_colors)
        ])
        
        self.play(Create(necklace), FadeIn(beads))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HL_COLOR)
        
        cut_marker_1 = Line(UP, DOWN, color=YELLOW).scale(0.3)
        cut_marker_2 = Line(UP, DOWN, color=YELLOW).scale(0.3)
        
        # Initial cut positions
        cut_marker_1.move_to(necklace.point_from_proportion(0.1))
        cut_marker_2.move_to(necklace.point_from_proportion(0.6))
        
        self.play(FadeIn(cut_marker_1), FadeIn(cut_marker_2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HL_COLOR)
        
        # Vector showing difference (Issue 39 resolution: E4)
        vector_label = Text("(Diff Red: 2, Diff Blue: 1)", font_size=20, color=WHITE)
        self.place_at_grid(vector_label, "E4", scale_factor=0.9)
        
        self.play(Write(vector_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(HL_COLOR)
        
        # New vector state
        vector_zero = Text("(Diff Red: 0, Diff Blue: 0)", font_size=20, color="#00FF00")
        self.place_at_grid(vector_zero, "E4", scale_factor=0.9)
        
        # Adjust cuts to "ideal" positions (conceptually)
        self.play(
            cut_marker_1.animate.move_to(necklace.point_from_proportion(0.2)),
            cut_marker_2.animate.move_to(necklace.point_from_proportion(0.7)),
            Transform(vector_label, vector_zero),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(HL_COLOR)
        
        # Transition to balance scale
        self.play(
            FadeOut(necklace),
            FadeOut(cut_marker_1),
            FadeOut(cut_marker_2),
            FadeOut(vector_label),
            beads.animate.scale(0.5)
        )
        
        # Create Balance Scale at D4 area
        scale_base = Line(LEFT, RIGHT, color=GREY_A).scale(1.5)
        scale_pivot = Triangle(color=GREY_A).scale(0.2).rotate(PI)
        scale_arm = Line(LEFT, RIGHT, color=GREY_B).scale(1.5)
        
        scale_group = VGroup(scale_base, scale_pivot, scale_arm)
        self.place_at_grid(scale_group, "D4", scale_factor=1.0)
        scale_pivot.next_to(scale_base, UP, buff=0)
        scale_arm.next_to(scale_pivot, UP, buff=0)
        
        # Sorting beads into piles on the scale
        pile_a = VGroup(*[Dot(radius=0.08, color=RED) for _ in range(3)] + [Dot(radius=0.08, color=BLUE) for _ in range(2)]).arrange_in_grid(rows=2)
        pile_b = VGroup(*[Dot(radius=0.08, color=RED) for _ in range(2)] + [Dot(radius=0.08, color=BLUE) for _ in range(3)]).arrange_in_grid(rows=2)
        # (Simplified sorting visual for the scale)
        
        pile_a.next_to(scale_arm.get_left(), UP, buff=0.1)
        pile_b.next_to(scale_arm.get_right(), UP, buff=0.1)
        
        self.play(Create(scale_group))
        self.play(
            ReplacementTransform(beads[:5], pile_a),
            ReplacementTransform(beads[5:], pile_b)
        )
        
        # Final leveling indication
        success_text = Text("Perfect Split!", font_size=24, color="#00FF00")
        self.place_at_grid(success_text, "F4")
        self.play(Write(success_text))
        
        self.wait(2)
        self.lecture[4].set_color(WHITE)
