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
        # Initialize Scene with the updated teaching script lines
        lecture_lines = [
            'Curl your fingers from vector v towards vector w.', 
            "Your thumb points in the resulting vector's direction.", 
            'Reversing the order curls your fingers the other way.', 
            'Consequently, the resulting vector flips to point downward.', 
            'This anti-commutative property means that order matters.'
        ]
        self.setup_layout("The Direction: The Right-Hand Rule", lecture_lines)

        # Colors
        COLOR_V = "#58C4DD"
        COLOR_W = "#83C167"
        COLOR_N = "#F8B195"
        COLOR_ACCENT = "#FFFF00"

        # Asset path
        HAND_ASSET = "/mmfs1/data/home/jthen/Code2Video/assets/icon/hand.svg"

        # Visual Anchor System (Issue 31: Origin shifted to C4)
        origin = self.grid["C4"]
        v_end = self.grid["D5"]
        w_end = self.grid["B5"]
        n_up_end = self.grid["A4"]
        n_down_end = self.grid["E4"]
        
        # Vector and Base Label Definitions
        v_vec = Arrow(start=origin, end=v_end, color=COLOR_V, buff=0)
        w_vec = Arrow(start=origin, end=w_end, color=COLOR_W, buff=0)
        
        v_label = Text("v", color=COLOR_V, font_size=24, slant=ITALIC)
        self.place_at_grid(v_label, "D6", scale_factor=0.8)
        
        w_label = Text("w", color=COLOR_W, font_size=24, slant=ITALIC)
        self.place_at_grid(w_label, "B6", scale_factor=0.8)

        # Hand Asset
        hand = SVGMobject(HAND_ASSET).set_color(WHITE).scale(0.3)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_ACCENT)
        
        # Calculate arc for rotation from v to w
        v_unit = (v_end - origin) / np.linalg.norm(v_end - origin)
        w_unit = (w_end - origin) / np.linalg.norm(w_end - origin)
        rot_arc_1 = CurvedArrow(
            start_point=origin + 0.6 * v_unit,
            end_point=origin + 0.6 * w_unit,
            color=COLOR_ACCENT,
            angle=PI/2
        )
        
        self.add(v_vec, w_vec, v_label, w_label)
        self.play(Create(rot_arc_1), FadeIn(hand.move_to(rot_arc_1.get_start())))
        self.play(MoveAlongPath(hand, rot_arc_1), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_ACCENT)
        
        n_vec_up = Arrow(start=origin, end=n_up_end, color=COLOR_N, buff=0)
        n_label_up = Text("v x w", color=COLOR_N, font_size=24)
        # Issue 30: up_label at B4
        self.place_at_grid(n_label_up, "B4", scale_factor=0.8)
        
        self.play(GrowArrow(n_vec_up), Write(n_label_up))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_ACCENT)
        
        # Reverse curved arrow (w to v) and move hand along it
        rot_arc_2 = CurvedArrow(
            start_point=origin + 0.6 * w_unit,
            end_point=origin + 0.6 * v_unit,
            color=COLOR_ACCENT,
            angle=-PI/2
        )
        
        self.play(
            ReplacementTransform(rot_arc_1, rot_arc_2),
            MoveAlongPath(hand, rot_arc_2),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_ACCENT)
        
        # Invert result vector to point downwards
        n_vec_down = Arrow(start=origin, end=n_down_end, color=COLOR_N, buff=0)
        n_label_down = Text("-v x w", color=COLOR_N, font_size=24)
        # Positioning label within rows D and E per Issue 31
        self.place_at_grid(n_label_down, "E5", scale_factor=0.8)
        
        self.play(
            ReplacementTransform(n_vec_up, n_vec_down),
            ReplacementTransform(n_label_up, n_label_down),
            FadeOut(hand),
            FadeOut(rot_arc_2)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_ACCENT)
        
        # Anti-commutative Equation
        eqn = Text("v x w = -(w x v)", color=COLOR_ACCENT, font_size=32)
        # Issue 29: Fixed placement and scale
        self.place_in_area(eqn, 'F2', 'F5', scale_factor=0.8)
        
        self.play(Write(eqn))
        self.wait(2)
