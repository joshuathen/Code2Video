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
        title = "Prerequisite: The Coordinate System of Language"
        lines = [
            "Computers represent words as numerical vectors.",
            "Similar meanings sit closer in high-dimensional space.",
            "Mathematical distances reveal relationships between words."
        ]
        self.setup_layout(title, lines)

        # Colors
        color_king = "#FFD700"
        color_queen = "#FF69B4"
        color_man = "#1E90FF"
        color_woman = "#EE82EE"
        color_apple = "#FF4500"
        highlight_color = YELLOW

        # Assets
        path_king = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/king.svg"
        path_queen = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/queen.svg"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(highlight_color)
        
        # Create Coordinate System (Axes)
        axes = Axes(
            x_range=[-1, 6, 1],
            y_range=[-1, 5, 1],
            x_length=5,
            y_length=4,
            axis_config={"color": GREY_C},
            tips=True
        )
        self.place_in_area(axes, "B2", "F6", scale_factor=0.8)
        self.play(Create(axes))

        # Initial word positions (Spread out)
        p_king_init = axes.c2p(1, 4)
        p_queen_init = axes.c2p(5, 4)
        p_man_init = axes.c2p(1, 1)
        p_woman_init = axes.c2p(5, 1)
        p_apple_init = axes.c2p(5.5, 2.5)

        # King (Icon + Label + Dot)
        dot_king = Dot(p_king_init, color=color_king)
        icon_king = SVGMobject(path_king).scale(0.25).set_color(color_king)
        label_king = Text("King", font_size=16, color=color_king)
        king_tag = VGroup(icon_king, label_king).arrange(UP, buff=0.05).next_to(dot_king, UP, buff=0.1)
        king_group = VGroup(dot_king, king_tag)

        # Queen (Icon + Label + Dot)
        dot_queen = Dot(p_queen_init, color=color_queen)
        icon_queen = SVGMobject(path_queen).scale(0.25).set_color(color_queen)
        label_queen = Text("Queen", font_size=16, color=color_queen)
        queen_tag = VGroup(icon_queen, label_queen).arrange(UP, buff=0.05).next_to(dot_queen, UP, buff=0.1)
        queen_group = VGroup(dot_queen, queen_tag)

        # Man (Label + Dot)
        dot_man = Dot(p_man_init, color=color_man)
        label_man = Text("Man", font_size=16, color=color_man).next_to(dot_man, DOWN, buff=0.1)
        man_group = VGroup(dot_man, label_man)

        # Woman (Label + Dot)
        dot_woman = Dot(p_woman_init, color=color_woman)
        label_woman = Text("Woman", font_size=16, color=color_woman).next_to(dot_woman, DOWN, buff=0.1)
        woman_group = VGroup(dot_woman, label_woman)

        # Apple (Label + Dot)
        dot_apple = Dot(p_apple_init, color=color_apple)
        label_apple = Text("Apple", font_size=16, color=color_apple).next_to(dot_apple, RIGHT, buff=0.1)
        apple_group = VGroup(dot_apple, label_apple)

        self.play(
            LaggedStart(
                FadeIn(king_group), FadeIn(queen_group),
                FadeIn(man_group), FadeIn(woman_group), FadeIn(apple_group),
                lag_ratio=0.1
            )
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(highlight_color)

        # Clustered positions
        c_king = axes.c2p(1.5, 3)
        c_man = axes.c2p(1.5, 2.2)
        c_queen = axes.c2p(2.5, 3)
        c_woman = axes.c2p(2.5, 2.2)

        self.play(
            dot_king.animate.move_to(c_king),
            king_tag.animate.next_to(c_king, UP, buff=0.1),
            dot_man.animate.move_to(c_man),
            label_man.animate.next_to(c_man, DOWN, buff=0.1),
            dot_queen.animate.move_to(c_queen),
            queen_tag.animate.next_to(c_queen, UP, buff=0.1),
            dot_woman.animate.move_to(c_woman),
            label_woman.animate.next_to(c_woman, DOWN, buff=0.1),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(highlight_color)

        # Equation Group
        eq_king_icon = SVGMobject(path_king).scale(0.2).set_color(color_king)
        eq_king_text = Text("King", font_size=18, color=color_king)
        eq_king = VGroup(eq_king_text, eq_king_icon).arrange(RIGHT, buff=0.1)

        eq_minus = Text("-", font_size=24)
        eq_man = Text("Man", font_size=18, color=color_man)
        eq_plus = Text("+", font_size=24)
        eq_woman = Text("Woman", font_size=18, color=color_woman)
        eq_equals = Text("=", font_size=24)

        eq_queen_icon = SVGMobject(path_queen).scale(0.2).set_color(color_queen)
        eq_queen_text = Text("Queen", font_size=18, color=color_queen)
        eq_queen = VGroup(eq_queen_text, eq_queen_icon).arrange(RIGHT, buff=0.1)
        
        equation = VGroup(eq_king, eq_minus, eq_man, eq_plus, eq_woman, eq_equals, eq_queen).arrange(RIGHT, buff=0.15)
        
        # Apply fix for Issue 24: place in area A2-A6
        self.place_in_area(equation, 'A2', 'A6', scale_factor=0.8)
        self.play(Write(equation))

        # Vector Visuals
        v_man = Arrow(axes.c2p(0,0), dot_man.get_center(), buff=0, color=color_man, stroke_width=2)
        v_king = Arrow(axes.c2p(0,0), dot_king.get_center(), buff=0, color=color_king, stroke_width=2)
        v_diff = Arrow(dot_man.get_center(), dot_king.get_center(), buff=0, color=WHITE, stroke_width=3)
        
        self.play(Create(v_man))
        self.play(Create(v_king))
        self.play(Create(v_diff))
        self.wait(0.5)

        # Move the diff vector to Woman to show King - Man + Woman = Queen
        target_queen_pos = dot_woman.get_center() + (dot_king.get_center() - dot_man.get_center())
        
        self.play(
            v_diff.animate.put_start_and_end_on(dot_woman.get_center(), target_queen_pos),
            run_time=2
        )
        self.play(Indicate(queen_group))
        self.wait(3)
